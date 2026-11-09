import os

import numpy as np
import torch

import folder_paths
from comfy_api.latest import Types

from .svg_geometry import build_relief_mesh


class SVGReliefToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "width_mm": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1000.0, "step": 0.5, "tooltip": "Stamp width (X) in mm."}),
                "height_mm": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1000.0, "step": 0.5, "tooltip": "Stamp height (Y) in mm. Differs from the SVG aspect = design is stretched."}),
                "relief_mm": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "Height of the raised (dark) shapes."}),
                "base_mm": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Base plate thickness. 0 = no plate."}),
                "margin_mm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5, "tooltip": "Extra plate border around the design."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Colors darker than this are raised."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Raise the light areas instead of the dark ones."}),
                "mirror": ("BOOLEAN", {"default": False, "tooltip": "Mirror left-right (for stamps that print the right way round)."}),
                "simplify_mm": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Outline simplification; higher = smoother, fewer triangles."}),
                "draft_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip": "Draft angle of the raised walls (deg). Walls lean inwards so the stamp releases from clay. 3-5 is typical."}),
                "layer_mm": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 1.0, "step": 0.01, "tooltip": "Step height used to build the draft; match your printer layer height."}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "run"
    CATEGORY = "3d/svg"

    def run(self, svg_string, width_mm, height_mm, relief_mm, base_mm, margin_mm, threshold, invert, mirror, simplify_mm, draft_deg=0.0, layer_mm=0.1):
        if isinstance(svg_string, list):
            svg_string = svg_string[0]
        mesh, islands, (w, h, z) = build_relief_mesh(svg_string, width_mm, height_mm, relief_mm, base_mm, margin_mm,
                                                     threshold, invert, mirror, simplify_mm, draft_deg, layer_mm)
        v = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).unsqueeze(0)
        f = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).unsqueeze(0)
        info = (f"{w:.1f} x {h:.1f} x {z:.1f} mm | relief shapes: {islands} | "
                f"draft: {draft_deg} deg | triangles: {len(mesh.faces)} | watertight: {mesh.is_watertight}")
        return (Types.MESH(v, f), info)


class SaveMeshSTL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "filename_prefix": ("STRING", {"default": "stl/relief"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_folder",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "3d/svg"

    def save(self, mesh, filename_prefix):
        import trimesh
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        os.makedirs(full_dir, exist_ok=True)
        paths = []
        for b in range(mesh.vertices.shape[0]):
            nv = int(mesh.vertex_counts[b]) if mesh.vertex_counts is not None else mesh.vertices.shape[1]
            nf = int(mesh.face_counts[b]) if mesh.face_counts is not None else mesh.faces.shape[1]
            tm = trimesh.Trimesh(mesh.vertices[b, :nv].cpu().numpy(), mesh.faces[b, :nf].cpu().numpy(), process=False)
            path = os.path.join(full_dir, f"{name}_{counter + b:05}_.stl")
            tm.export(path)
            paths.append(path)
        folder = os.path.normpath(full_dir)
        return {"ui": {"text": [folder]}, "result": (folder,)}


NODE_CLASS_MAPPINGS = {"SVGReliefToMesh": SVGReliefToMesh, "SaveMeshSTL": SaveMeshSTL}
NODE_DISPLAY_NAME_MAPPINGS = {"SVGReliefToMesh": "SVG Relief to Mesh (extrude + base)", "SaveMeshSTL": "Save Mesh as STL"}
