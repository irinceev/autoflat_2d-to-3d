"""
Геометрический движок.

Строит 3D-геометрию (стены, полы, потолки, простые двери/окна, мебель)
из данных, которые отдаёт parse_floorplan_image.parse().
"""
import re
import numpy as np

DEFAULT_MATERIAL_COLORS = {
    "wall": (0.80, 0.78, 0.74),
    "floor": (0.82, 0.74, 0.58),
    "ceiling": (0.95, 0.95, 0.93),
}
try:
    from furniture import FURNITURE_TYPES as _FURNITURE_TYPES
    FURNITURE_MATERIAL_COLORS_HEX = {name: hexcol for name, (hexcol, _h) in _FURNITURE_TYPES.items()}
except ImportError:
    FURNITURE_MATERIAL_COLORS_HEX = {}
FURNITURE_MATERIAL_COLORS_HEX.setdefault("other", "#b5651d")

DEFAULT_WALL_HEIGHT_M = 2.70


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def sanitize_group_name(name):
    """OBJ 'g' — токены через пробел; пробелы/спецсимволы ломают группировку в Blender."""
    return re.sub(r"[^0-9A-Za-zА-Яа-яёЁ_\-]+", "_", str(name))


def material_for_group(group):
    """Определяет материал по имени группы."""
    if group == "Walls" or group.startswith("Walls"):
        return "wall", DEFAULT_MATERIAL_COLORS["wall"]
    if group.startswith("Floor"):
        return "floor", DEFAULT_MATERIAL_COLORS["floor"]
    if group.startswith("Ceiling"):
        return "ceiling", DEFAULT_MATERIAL_COLORS["ceiling"]
    if group.startswith("Door") and group.endswith("_Handle"):
        return "door_handle", (0.75, 0.75, 0.78)            # ручка - просто серый кубик
    if group.startswith("Door"):
        return "wall", DEFAULT_MATERIAL_COLORS["wall"]     # полотно красим как стену, просто
    if group.startswith("Window"):
        return "wall", DEFAULT_MATERIAL_COLORS["wall"]     # окно (глухие полосы) - тоже стена
    if group.startswith("Furniture"):
        parts = group.split("_")
        ftype = parts[2] if len(parts) > 2 else "other"
        hexcol = FURNITURE_MATERIAL_COLORS_HEX.get(ftype, FURNITURE_MATERIAL_COLORS_HEX["other"])
        return f"furn_{ftype}", _hex_to_rgb01(hexcol)
    return "default", (0.7, 0.7, 0.7)


class Mesh:
    """Простой контейнер треугольных мешей с группами (для OBJ-экспорта)."""

    def __init__(self):
        self.vertices = []
        self.groups = []

    def add_box(self, xmin, xmax, ymin, ymax, zmin, zmax, group):
        v0 = len(self.vertices)
        pts = [
            (xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
            (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax),
        ]
        self.vertices.extend(pts)
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 5, 1), (0, 4, 5),
            (1, 6, 2), (1, 5, 6),
            (2, 7, 3), (2, 6, 7),
            (3, 4, 0), (3, 7, 4),
        ]
        faces = [tuple(v0 + i + 1 for i in f) for f in faces]
        self.groups.append((group, faces))

    def write_obj(self, path):
        mtl_path = path.rsplit(".", 1)[0] + ".mtl"
        mtl_name = mtl_path.split("/")[-1].split("\\")[-1]

        used_materials = {}
        for group, _ in self.groups:
            mat_name, color = material_for_group(group)
            used_materials[mat_name] = color

        with open(mtl_path, "w", encoding="utf-8") as f:
            for mat_name, (r, g, b) in used_materials.items():
                f.write(f"newmtl {mat_name}\n")
                f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")
                f.write("Ka 0.1 0.1 0.1\nKs 0.05 0.05 0.05\nd 1.0\nillum 1\n\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Упрощённая 3D-модель квартиры\n")
            f.write(f"mtllib {mtl_name}\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.4f} {v[2]:.4f} {v[1]:.4f}\n")  # Y-up
            cur_group = None
            cur_mat = None
            for group, faces in self.groups:
                safe_group = sanitize_group_name(group)
                mat_name, _ = material_for_group(group)
                if safe_group != cur_group:
                    f.write(f"g {safe_group}\n")
                    cur_group = safe_group
                if mat_name != cur_mat:
                    f.write(f"usemtl {mat_name}\n")
                    cur_mat = mat_name
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")


# ------------------------------------------------------------------
# Двери и окна — максимально просто, без рамы/стекла/ручки.
# ------------------------------------------------------------------
DOOR_LEAF_THICKNESS_M = 0.05
DOOR_HANDLE_SIZE_M = 0.08     # просто кубик-ручка
DOOR_HANDLE_HEIGHT_M = 1.0    # высота ручки от пола
WINDOW_STRIP_M = 0.5   # глухая часть стены сверху и снизу окна


def add_door_geometry(mesh, o, px_per_m, img_h, wall_h, group_id):
    """Дверь = плоская плита прямо в проёме + кубик-ручка с обеих сторон полотна."""
    cx_m = o["cx"] / px_per_m
    cy_m = (img_h - o["cy"]) / px_per_m
    ys, xs = np.where(o["mask"])
    horiz = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    w, h = o["width_m"], o["height_m"]
    hs = DOOR_HANDLE_SIZE_M
    lt = DOOR_LEAF_THICKNESS_M
    if horiz:
        mesh.add_box(cx_m - w / 2, cx_m + w / 2,
                     cy_m - lt / 2, cy_m + lt / 2,
                     0.0, h, f"Door_{group_id}")
        hx = cx_m + w / 2 - 2 * hs   # ручка ближе к краю полотна
        # сторона 1
        mesh.add_box(hx, hx + hs, cy_m - lt / 2 - hs, cy_m - lt / 2,
                     DOOR_HANDLE_HEIGHT_M, DOOR_HANDLE_HEIGHT_M + hs, f"Door_{group_id}_Handle")
        # сторона 2
        mesh.add_box(hx, hx + hs, cy_m + lt / 2, cy_m + lt / 2 + hs,
                     DOOR_HANDLE_HEIGHT_M, DOOR_HANDLE_HEIGHT_M + hs, f"Door_{group_id}_Handle")
    else:
        mesh.add_box(cx_m - lt / 2, cx_m + lt / 2,
                     cy_m - w / 2, cy_m + w / 2,
                     0.0, h, f"Door_{group_id}")
        hy = cy_m + w / 2 - 2 * hs
        mesh.add_box(cx_m - lt / 2 - hs, cx_m - lt / 2, hy, hy + hs,
                     DOOR_HANDLE_HEIGHT_M, DOOR_HANDLE_HEIGHT_M + hs, f"Door_{group_id}_Handle")
        mesh.add_box(cx_m + lt / 2, cx_m + lt / 2 + hs, hy, hy + hs,
                     DOOR_HANDLE_HEIGHT_M, DOOR_HANDLE_HEIGHT_M + hs, f"Door_{group_id}_Handle")


def add_window_geometry(mesh, o, px_per_m, img_h, wall_h, group_id):
    """Окно = глухая полоса стены 0.5 м снизу и 0.5 м сверху, посередине - пусто."""
    cx_m = o["cx"] / px_per_m
    cy_m = (img_h - o["cy"]) / px_per_m
    ys, xs = np.where(o["mask"])
    horiz = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    dx = (xs.max() - xs.min() + 1) / px_per_m
    dy = (ys.max() - ys.min() + 1) / px_per_m
    t = dy if horiz else dx
    w = o["width_m"]

    def box(a0, a1, z0, z1, suffix):
        if horiz:
            mesh.add_box(a0, a1, cy_m - t / 2, cy_m + t / 2, z0, z1, f"Window_{group_id}{suffix}")
        else:
            mesh.add_box(cx_m - t / 2, cx_m + t / 2, a0, a1, z0, z1, f"Window_{group_id}{suffix}")

    a0 = (cx_m - w / 2) if horiz else (cy_m - w / 2)
    a1 = (cx_m + w / 2) if horiz else (cy_m + w / 2)

    box(a0, a1, 0.0, WINDOW_STRIP_M, "_Bottom")
    box(a0, a1, wall_h - WINDOW_STRIP_M, wall_h, "_Top")