"""Headless mesh -> IMAGE renderer.

Pure numpy z-buffer rasterizer, so it needs no OpenGL / pyrender / display.
Renders a TRIMESH (or a core MESH via the bridge) as a turntable batch of
images that can go straight into SaveImage / PreviewImage.

Handles, in order of preference:
  * UV + baseColorTexture / SimpleMaterial.image  (textured GLB)
  * vertex colors
  * flat base color (untextured mesh)
with a simple headlight lambert term on top.
"""

import numpy as np
import trimesh as Trimesh


def _texture_array(tm):
    """Return (uv[V,2], texture[H,W,3] float 0..1) or (None, None)."""
    visual = getattr(tm, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None, None
    img = None
    mat = getattr(visual, "material", None)
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        return None, None
    tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.asarray(uv, dtype=np.float32), tex


def _vertex_colors(tm):
    visual = getattr(tm, "visual", None)
    # `visual.vertex_colors` fabricates a flat default grey when the mesh has no
    # real colors — kind tells us whether they are genuine.
    if getattr(visual, "kind", None) not in ("vertex", "face"):
        return None
    try:
        vc = visual.vertex_colors
    except Exception:
        return None
    if vc is None or len(vc) == 0:
        return None
    vc = np.asarray(vc, dtype=np.float32)
    if vc.shape[0] == 0:
        return None
    return vc[:, :3] / 255.0


def _rot(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    ca, sa = np.cos(az), np.sin(az)
    ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)
    ce, se = np.cos(el), np.sin(el)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    return rx @ ry


def _render_one(v, faces, vn, W, H, zoom, base_rgb, uv, tex, vcol, ambient, bg_rgb):
    """Rasterize one camera. v is already rotated into camera space (look down -Z)."""
    img = np.empty((H, W, 3), dtype=np.float32)
    img[:] = bg_rgb
    mask = np.zeros((H, W), dtype=bool)
    zbuf = np.full((H, W), -np.inf, dtype=np.float32)

    px = (v[:, 0] * zoom * 0.5 + 0.5) * (W - 1)
    py = (0.5 - v[:, 1] * zoom * 0.5) * (H - 1)
    pz = v[:, 2]

    fx = px[faces]  # [F,3]
    fy = py[faces]
    fz = pz[faces]

    x0 = np.floor(fx.min(1)).astype(np.int32)
    x1 = np.ceil(fx.max(1)).astype(np.int32)
    y0 = np.floor(fy.min(1)).astype(np.int32)
    y1 = np.ceil(fy.max(1)).astype(np.int32)
    keep = (x1 >= 0) & (x0 <= W - 1) & (y1 >= 0) & (y0 <= H - 1)

    area = (fx[:, 1] - fx[:, 0]) * (fy[:, 2] - fy[:, 0]) - (fx[:, 2] - fx[:, 0]) * (
        fy[:, 1] - fy[:, 0]
    )
    keep &= np.abs(area) > 1e-9

    idx = np.nonzero(keep)[0]
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # per-pixel attribute buffers (barycentric weights + winning face)
    for i in idx:
        ax, bx, cx = fx[i]
        ay, by, cy = fy[i]
        xs = np.arange(x0[i], x1[i] + 1, dtype=np.float32)
        ys = np.arange(y0[i], y1[i] + 1, dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            continue
        gx = xs[None, :]
        gy = ys[:, None]
        inv = 1.0 / area[i]
        w1 = ((gx - ax) * (cy - ay) - (gy - ay) * (cx - ax)) * inv
        w2 = ((bx - ax) * (gy - ay) - (by - ay) * (gx - ax)) * inv
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * fz[i, 0] + w1 * fz[i, 1] + w2 * fz[i, 2]
        sub_z = zbuf[y0[i] : y1[i] + 1, x0[i] : x1[i] + 1]
        win = inside & (z > sub_z)
        if not win.any():
            continue
        sub_z[win] = z[win]

        yy, xx = np.nonzero(win)
        yy = yy + y0[i]
        xx = xx + x0[i]
        b0 = w0[win][:, None]
        b1 = w1[win][:, None]
        b2 = w2[win][:, None]

        tri = faces[i]
        if uv is not None:
            t = b0 * uv[tri[0]] + b1 * uv[tri[1]] + b2 * uv[tri[2]]
            th, tw = tex.shape[:2]
            u = np.clip((t[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(np.int32)
            vv = np.clip((1.0 - (t[:, 1] % 1.0)) * (th - 1), 0, th - 1).astype(np.int32)
            albedo = tex[vv, u]
        elif vcol is not None:
            albedo = b0 * vcol[tri[0]] + b1 * vcol[tri[1]] + b2 * vcol[tri[2]]
        else:
            albedo = np.broadcast_to(base_rgb, (yy.size, 3)).copy()

        n = b0 * vn[tri[0]] + b1 * vn[tri[1]] + b2 * vn[tri[2]]
        nz = np.abs(n[:, 2:3]) / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        shade = ambient + (1.0 - ambient) * nz
        img[yy, xx] = np.clip(albedo * shade, 0.0, 1.0)
        mask[yy, xx] = True

    return img, mask


class TrimeshRenderToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "views": ("INT", {"default": 4, "min": 1, "max": 24}),
                "start_azimuth": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0}),
                "elevation": ("FLOAT", {"default": 15.0, "min": -89.0, "max": 89.0, "step": 5.0}),
                "zoom": ("FLOAT", {"default": 0.9, "min": 0.2, "max": 3.0, "step": 0.05}),
                "use_texture": ("BOOLEAN", {"default": True}),
                "ambient": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_color": ("STRING", {"default": "#B4B4B4"}),
                "background": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "render"
    CATEGORY = "3d"
    DESCRIPTION = (
        "Renders a TRIMESH to an IMAGE batch (turntable) with a pure-numpy "
        "z-buffer rasterizer — no OpenGL. Uses the GLB texture when present, "
        "else vertex colors, else a flat base color. Feed SaveImage to store "
        "a picture of the mesh / textured mesh."
    )

    @staticmethod
    def _hex(s, default=(0.7, 0.7, 0.7)):
        s = (s or "").strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6:
            return np.array(default, dtype=np.float32)
        try:
            return np.array([int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
        except ValueError:
            return np.array(default, dtype=np.float32)

    def render(
        self,
        trimesh,
        width,
        height,
        views,
        start_azimuth,
        elevation,
        zoom,
        use_texture,
        ambient,
        base_color,
        background,
    ):
        import torch

        tm = trimesh
        if isinstance(tm, Trimesh.Scene):
            tm = tm.dump(concatenate=True)

        verts = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces, dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            empty = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty, torch.zeros((1, height, width), dtype=torch.float32))

        try:
            vn = np.asarray(tm.vertex_normals, dtype=np.float32)
            if vn.shape != verts.shape:
                raise ValueError
        except Exception:
            vn = np.zeros_like(verts)
            fn = np.cross(
                verts[faces[:, 1]] - verts[faces[:, 0]],
                verts[faces[:, 2]] - verts[faces[:, 0]],
            )
            for k in range(3):
                np.add.at(vn, faces[:, k], fn)

        # center + normalize to unit sphere so the framing is view independent
        center = (verts.min(0) + verts.max(0)) * 0.5
        v = verts - center
        radius = float(np.linalg.norm(v, axis=1).max()) or 1.0
        v = v / radius

        uv, tex = (None, None)
        if use_texture:
            uv, tex = _texture_array(tm)
        vcol = None if uv is not None else _vertex_colors(tm)

        base_rgb = self._hex(base_color, (0.706, 0.706, 0.706))
        bg_rgb = self._hex(background, (1.0, 1.0, 1.0))

        imgs, masks = [], []
        for i in range(views):
            az = start_azimuth + i * (360.0 / views)
            R = _rot(az, elevation)
            img, mask = _render_one(
                v @ R.T,
                faces,
                vn @ R.T,
                width,
                height,
                zoom,
                base_rgb,
                uv,
                tex,
                vcol,
                float(ambient),
                bg_rgb,
            )
            imgs.append(img)
            masks.append(mask.astype(np.float32))

        return (
            torch.from_numpy(np.stack(imgs)),
            torch.from_numpy(np.stack(masks)),
        )


class MeshRenderToImage(TrimeshRenderToImage):
    @classmethod
    def INPUT_TYPES(s):
        types = TrimeshRenderToImage.INPUT_TYPES()
        req = {"mesh": ("MESH",), "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095})}
        for k, val in types["required"].items():
            if k != "trimesh":
                req[k] = val
        return {"required": req}

    DESCRIPTION = (
        "Renders a core ComfyUI MESH to an IMAGE batch (turntable) with a "
        "pure-numpy z-buffer rasterizer — no OpenGL. Feed SaveImage to store "
        "a picture of the untextured mesh."
    )

    def render(self, mesh, batch_index, **kwargs):
        from . import MeshToTrimesh

        (tm,) = MeshToTrimesh().convert(mesh, batch_index, True)
        return super().render(tm, **kwargs)
