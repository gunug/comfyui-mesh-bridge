"""SVG (vtracer-style filled paths) -> relief-on-base-plate mesh. No ComfyUI imports."""
import re

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|" + _NUM)
_PATH = re.compile(r"<path\b[^>]*>", re.S)
_ATTR = lambda name: re.compile(r'\b' + name + r'\s*=\s*"([^"]*)"')


def _bez(p0, p1, p2, p3, n=8):
    t = np.linspace(0, 1, n + 1)[1:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _subpaths(d):
    toks = _TOKEN.findall(d)
    i, cmd = 0, None
    cur = np.zeros(2)
    start = np.zeros(2)
    pts, out = [], []

    def num():
        nonlocal i
        v = float(toks[i]); i += 1
        return v

    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]; i += 1
            if cmd in "Zz":
                if len(pts) >= 3:
                    out.append(np.array(pts))
                pts = []
                cur = start.copy()
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            p = np.array([num(), num()]) + (cur if rel else 0)
            if len(pts) >= 3:
                out.append(np.array(pts))
            pts = [p]; cur = p; start = p.copy()
            cmd = "l" if rel else "L"
        elif c == "L":
            p = np.array([num(), num()]) + (cur if rel else 0)
            pts.append(p); cur = p
        elif c == "H":
            x = num(); p = np.array([x + (cur[0] if rel else 0), cur[1]])
            pts.append(p); cur = p
        elif c == "V":
            y = num(); p = np.array([cur[0], y + (cur[1] if rel else 0)])
            pts.append(p); cur = p
        elif c == "C":
            off = cur if rel else 0
            p1 = np.array([num(), num()]) + off
            p2 = np.array([num(), num()]) + off
            p3 = np.array([num(), num()]) + off
            pts.extend(_bez(cur, p1, p2, p3)); cur = p3
        else:
            raise ValueError(f"unsupported SVG path command: {cmd}")
    if len(pts) >= 3:
        out.append(np.array(pts))
    return out


def _luma(fill):
    fill = (fill or "#000000").lstrip("#")
    if len(fill) == 3:
        fill = "".join(ch * 2 for ch in fill)
    r, g, b = (int(fill[k:k + 2], 16) / 255 for k in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def svg_dark_region(svg, threshold=0.5, invert=False):
    """Paint paths in order (stacked vtracer output): dark adds, light removes."""
    w = float(_ATTR("width").search(svg).group(1))
    h = float(_ATTR("height").search(svg).group(1))
    region = Polygon()
    for tag in _PATH.findall(svg):
        dm = _ATTR("d").search(tag)
        if not dm:
            continue
        fm = _ATTR("fill").search(tag)
        tm = re.search(r'translate\(\s*(' + _NUM + r')[\s,]+(' + _NUM + r')\s*\)', tag)
        off = np.array([float(tm.group(1)), float(tm.group(2))]) if tm else np.zeros(2)
        shape = Polygon()
        for ring in _subpaths(dm.group(1)):
            poly = Polygon(ring + off).buffer(0)
            if not poly.is_empty:
                shape = shape.symmetric_difference(poly)  # even-odd: inner rings become holes
        if shape.is_empty:
            continue
        dark = _luma(fm.group(1) if fm else None) < threshold
        if dark != invert:
            region = region.union(shape)
        else:
            region = region.difference(shape)
    return region.buffer(0), w, h


def build_relief_mesh(svg, width_mm=50.0, height_mm=50.0, relief_mm=2.0, base_mm=5.0, margin_mm=0.0,
                      threshold=0.5, invert=False, mirror=False, simplify_mm=0.05, draft_deg=0.0,
                      layer_mm=0.1):
    region, w, h = svg_dark_region(svg, threshold, invert)
    sx, sy = width_mm / w, height_mm / h
    # SVG y points down; flip so the design reads correctly from above. Mirror for stamps.
    def xf(geom):
        from shapely import affinity
        g = affinity.scale(geom, xfact=-sx if mirror else sx, yfact=-sy, origin=(0, 0))
        return affinity.translate(g, xoff=width_mm if mirror else 0, yoff=height_mm)
    region = xf(region)
    if simplify_mm > 0:
        region = region.simplify(simplify_mm, preserve_topology=True).buffer(0)
    W, H = width_mm, height_mm
    parts = []
    if base_mm > 0:
        plate = box(-margin_mm, -margin_mm, W + margin_mm, H + margin_mm)
        parts.append(trimesh.creation.extrude_polygon(plate, base_mm))
    polys = region.geoms if isinstance(region, MultiPolygon) else [region]
    relief = []
    import math
    grow_total = relief_mm * math.tan(math.radians(draft_deg)) if draft_deg > 0 else 0.0
    # Draft: the TIP face (first to touch the clay) is exactly the SVG outline, so the stamped
    # mark is the same however deep it is pressed. Walls widen towards the base plate, built as
    # thin stacked slices (bottom slice = widest), so it releases cleanly from clay.
    n = max(1, math.ceil(relief_mm / layer_mm)) if grow_total > 0 else 1
    dz = relief_mm / n
    plate_box = box(-margin_mm, -margin_mm, W + margin_mm, H + margin_mm)
    for k in range(n):
        grow = grow_total * (n - 1 - k) / n  # k = 0 at the plate (widest) ... n-1 at the tip (= SVG)
        layer = region if grow == 0 else region.buffer(grow, join_style=2, mitre_limit=2.0)
        layer = layer.intersection(plate_box)  # never overhang the plate edge
        for q in (layer.geoms if isinstance(layer, MultiPolygon) else [layer]):
            if q.is_empty or q.area < 1e-4:
                continue
            m = trimesh.creation.extrude_polygon(q, dz)
            m.apply_translation([0, 0, base_mm + k * dz])
            relief.append(m)
    parts += relief
    if not parts:
        raise ValueError("SVG produced no geometry (check threshold / invert).")
    try:
        mesh = trimesh.boolean.union(parts, engine="manifold") if len(parts) > 1 else parts[0]
    except Exception:
        mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    return mesh, len(relief), (W + 2 * margin_mm, H + 2 * margin_mm, base_mm + relief_mm)
