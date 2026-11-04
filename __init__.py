"""Bridge between ComfyUI core MESH (comfy_api geometry_types.MESH) and
trimesh.Trimesh (TRIMESH), the type used by ComfyUI-Hunyuan3d-2-1.

Lets a core shape pipeline (VoxelToMesh) feed the Hunyuan3D 2.1 Paint nodes
without loading the wrapper's own DiT checkpoint.
"""

import numpy as np
import trimesh as Trimesh


def _to_numpy(t):
    return t.detach().cpu().numpy()


class MeshToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                "merge_vertices": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "3d"
    DESCRIPTION = "Converts a core ComfyUI MESH into a trimesh.Trimesh (TRIMESH)."

    def convert(self, mesh, batch_index, merge_vertices):
        vertices = mesh.vertices
        faces = mesh.faces

        # MESH stores batched tensors: [B, V, 3] / [B, F, 3]
        if vertices.ndim == 3:
            idx = min(batch_index, vertices.shape[0] - 1)
            vertices = vertices[idx]
            faces = faces[idx]

        v = _to_numpy(vertices).astype(np.float32)
        f = _to_numpy(faces).astype(np.int64)

        # variable-size batches pad with -1 face indices
        if f.size:
            f = f[(f >= 0).all(axis=1)]

        tm = Trimesh.Trimesh(vertices=v, faces=f, process=False)
        if merge_vertices:
            tm.merge_vertices()
            tm.remove_unreferenced_vertices()
        return (tm,)


class TrimeshToMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"trimesh": ("TRIMESH",)}}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "convert"
    CATEGORY = "3d"
    DESCRIPTION = "Converts a trimesh.Trimesh (TRIMESH) back into a core ComfyUI MESH."

    def convert(self, trimesh):
        import torch
        from comfy_api.latest._util.geometry_types import MESH

        v = torch.from_numpy(np.asarray(trimesh.vertices, dtype=np.float32))[None]
        f = torch.from_numpy(np.asarray(trimesh.faces, dtype=np.int64))[None]

        # Carry UVs and PBR textures through so bake/texture nodes can use them.
        uvs = texture = mr = None
        visual = getattr(trimesh, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv) == len(trimesh.vertices):
            uv = np.asarray(uv, dtype=np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # core MESH UV v is flipped vs trimesh
            uvs = torch.from_numpy(uv)[None]
            material = getattr(visual, "material", None)

            def _img(pil):
                if pil is None:
                    return None
                arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
                return torch.from_numpy(arr)[None]

            if material is not None:
                texture = _img(getattr(material, "baseColorTexture", None) or getattr(material, "image", None))
                mr = _img(getattr(material, "metallicRoughnessTexture", None))

        return (MESH(v, f, uvs=uvs, texture=texture, metallic_roughness=mr),)


class MeshToTrimeshPBR:
    """MESH -> TRIMESH that keeps UVs and every PBR map (base color, ORM, normal, AO).

    Goes through the same GLB serializer SaveGLB uses, so the result matches the saved file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",)}}

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "3d"
    DESCRIPTION = "Converts a textured core MESH into a TRIMESH, keeping UVs and all PBR textures."

    def convert(self, mesh):
        from io import BytesIO
        from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

        glb = mesh_item_to_glb_bytes(mesh, 0)
        if glb is None:
            raise ValueError("MeshToTrimeshPBR: mesh is empty.")
        tm = Trimesh.load(BytesIO(glb), file_type="glb", force="mesh", process=False)
        return (tm,)


def _channel(img, h, w):
    """IMAGE [B,H,W,C] -> single channel [H,W] float, resized to h x w."""
    import torch
    import torch.nn.functional as F

    x = img[0].float()
    x = x[..., :3].mean(dim=-1) if x.shape[-1] >= 3 else x[..., 0]
    if x.shape[0] != h or x.shape[1] != w:
        x = F.interpolate(x[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return x.clamp(0.0, 1.0)


class UnityMaskMapPack:
    """Unity mask map: R=metallic, G=occlusion, B=detail mask, A=smoothness (1-roughness).

    HDRP Lit uses it as the Mask Map. URP Lit: assign it to both Metallic Map
    (R + smoothness from alpha) and Occlusion Map (reads G).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metallic": ("IMAGE",),
                "roughness": ("IMAGE",),
            },
            "optional": {
                "occlusion": ("IMAGE",),
                "detail_mask": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_map",)
    FUNCTION = "pack"
    CATEGORY = "3d/unity"

    def pack(self, metallic, roughness, occlusion=None, detail_mask=0.0):
        import torch

        maps = [m for m in (metallic, roughness, occlusion) if m is not None]
        h = max(int(m.shape[1]) for m in maps)
        w = max(int(m.shape[2]) for m in maps)
        r = _channel(metallic, h, w)
        g = _channel(occlusion, h, w) if occlusion is not None else torch.ones_like(r)
        b = torch.full_like(r, float(detail_mask))
        a = 1.0 - _channel(roughness, h, w)
        return (torch.stack([r, g, b, a], dim=-1)[None],)


class NormalMapFlipGreen:
    """Inverts the G channel: OpenGL (+Y, glTF/Unity/Blender) <-> DirectX (-Y, Unreal)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"normal_map": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "flip"
    CATEGORY = "3d/unity"

    def flip(self, normal_map):
        out = normal_map.clone()
        out[..., 1] = 1.0 - out[..., 1]
        return (out,)


class NormalMapStrength:
    """Scales tangent-space normal map detail: XY * strength, Z re-derived so every texel stays unit length."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.05,
                                       "tooltip": "1 = unchanged, >1 = deeper bumps, <1 = flatter."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "apply"
    CATEGORY = "3d/unity"

    def apply(self, normal_map, strength):
        import torch

        n = normal_map[..., :3].float() * 2.0 - 1.0
        xy = n[..., :2] * strength
        z = torch.sqrt(torch.clamp(1.0 - (xy * xy).sum(-1, keepdim=True), min=0.0))
        v = torch.cat([xy, z], dim=-1)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # where |xy| > 1 the length is pulled back to 1
        out = (v * 0.5 + 0.5).clamp(0.0, 1.0)
        if normal_map.shape[-1] > 3:
            out = torch.cat([out, normal_map[..., 3:]], dim=-1)
        return (out,)


def _unity_target(out_root, filename_prefix):
    import os

    target = os.path.abspath(os.path.join(out_root, filename_prefix))
    if os.path.commonpath([out_root, target]) != out_root:
        raise ValueError("Unity export: prefix must stay inside the output folder.")
    folder, base = os.path.split(target)
    os.makedirs(folder, exist_ok=True)
    return folder, base


def _next_asset_name(folder, base):
    import os
    import re

    pat = re.compile(re.escape(base) + r"_v(\d{3})")
    runs = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))]
    return f"{base}_v{max(runs, default=0) + 1:03d}"


class UnityAssetName:
    """Next free <base>_vNNN name under output/<prefix>, shared by the rigger and the exporter."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filename_prefix": ("STRING", {"default": "unity/monster"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_name",)
    FUNCTION = "name"
    CATEGORY = "3d/unity"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")  # the next free version depends on the disk

    def name(self, filename_prefix):
        import os
        import folder_paths

        folder, base = _unity_target(os.path.abspath(folder_paths.get_output_directory()), filename_prefix)
        return (_next_asset_name(folder, base),)


class UnityRigExport:
    """Collects the rigged FBX (+ mask map) into output/<prefix>_vNNN_XX_*.

    Removes the rigger's side files (.fbm texture folder, preview .glb) so only
    the deliverables are left. The FBX embeds base color + normal map.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fbx_path": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "unity/monster"}),
            },
            "optional": {
                "asset_name": ("STRING", {"forceInput": True,
                                          "tooltip": "From Unity Asset Name; also feed it to the rigger's fbx_name so "
                                                     "the embedded textures are named <asset>_albedo / <asset>_normal."}),
                "cleanup_files": ("STRING", {"default": "hy3d21_textured.glb",
                                             "tooltip": "Comma-separated files (relative to output) that upstream nodes "
                                                        "always write; deleted so only the deliverables remain."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_relative_path",)
    FUNCTION = "export"
    CATEGORY = "3d/unity"
    OUTPUT_NODE = True

    def export(self, fbx_path, filename_prefix, asset_name="", cleanup_files=""):
        import os
        import shutil
        import folder_paths

        out_root = os.path.abspath(folder_paths.get_output_directory())
        folder, base = _unity_target(out_root, filename_prefix)
        stem = asset_name or _next_asset_name(folder, base)

        src = fbx_path
        if not os.path.isabs(src):
            src = os.path.join(out_root, src)
        fbx_name = f"{stem}.fbx"
        shutil.move(src, os.path.join(folder, fbx_name))
        files = [fbx_name]

        # rigger side files: <name>.fbm/ (textures, already embedded) and <name>.glb (preview)
        side = os.path.splitext(src)[0]
        shutil.rmtree(side + ".fbm", ignore_errors=True)
        if os.path.exists(side + ".glb"):
            os.remove(side + ".glb")
        for name in (n.strip() for n in cleanup_files.split(",")):
            p = os.path.abspath(os.path.join(out_root, name)) if name else ""
            if p and os.path.commonpath([out_root, p]) == out_root and os.path.isfile(p):
                os.remove(p)

        rel_folder = os.path.relpath(folder, out_root).replace("\\", "/")
        rel_fbx = f"{rel_folder}/{fbx_name}" if rel_folder != "." else fbx_name
        return {"ui": {"text": ["\n".join(f"{rel_folder}/{f}" for f in files)]}, "result": (rel_fbx,)}


from .render import TrimeshRenderToImage, MeshRenderToImage  # noqa: E402

NODE_CLASS_MAPPINGS = {
    "MeshToTrimesh": MeshToTrimesh,
    "TrimeshToMesh": TrimeshToMesh,
    "MeshToTrimeshPBR": MeshToTrimeshPBR,
    "UnityMaskMapPack": UnityMaskMapPack,
    "NormalMapFlipGreen": NormalMapFlipGreen,
    "UnityRigExport": UnityRigExport,
    "UnityAssetName": UnityAssetName,
    "NormalMapStrength": NormalMapStrength,
    "TrimeshRenderToImage": TrimeshRenderToImage,
    "MeshRenderToImage": MeshRenderToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshToTrimesh": "MESH → TRIMESH (bridge)",
    "TrimeshToMesh": "TRIMESH → MESH (bridge)",
    "MeshToTrimeshPBR": "MESH → TRIMESH (PBR 텍스처 유지)",
    "UnityMaskMapPack": "유니티 마스크맵 패킹 (R메탈 G AO A스무스)",
    "NormalMapFlipGreen": "노멀맵 G 반전 (OpenGL↔DirectX)",
    "UnityRigExport": "유니티 리깅 FBX 최종 출력",
    "UnityAssetName": "유니티 에셋 이름 (다음 버전 번호)",
    "NormalMapStrength": "노멀맵 강도 (굴곡 깊이)",
    "TrimeshRenderToImage": "TRIMESH → 이미지 렌더",
    "MeshRenderToImage": "MESH → 이미지 렌더",
}
