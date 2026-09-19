"""
Парсер цветного растрового чертежа (см. SPEC_разметка_чертежа.md) в
структурированную 2D/3D-модель квартиры.

Использование:
    python parse_floorplan_image.py plan.png
        -> debug_rooms.png (картинка с номерами комнат)
        -> rooms_legend_template.csv (шаблон для заполнения названий)

    python parse_floorplan_image.py plan.png --legend rooms_legend.csv
        -> apartment_model.obj, smeta_obemy_rabot.xlsx, visualization_3d.png, floorplan_2d.png
"""
import os
import sys
import csv
import argparse
import numpy as np
import cv2
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from geometry_engine import Mesh
from furniture import FURNITURE_COLORS_RGB, DEFAULT_FURNITURE_HEIGHT

# ------------------------------------------------------------------
# Палитра (см. SPEC) и константы
# ------------------------------------------------------------------
PALETTE = {
    "ext_wall":  (255, 0, 0),
    "int_wall":  (0, 0, 255),
    "door":      (0, 255, 0),
    "window":    (255, 255, 0),
    "window_sm": (255, 165, 0),
    "calib":     (255, 0, 255),
    "label":     (0, 0, 0),
    "zone_split": (0, 255, 255),   # цвет-разделитель зон БЕЗ физической стены (открытая планировка)
}
# добавляем по одному цвету на каждый тип мебели (furn_bed, furn_sofa, ...) —
# цвет прямоугольника на чертеже сам определяет тип предмета
for _ftype, _rgb in FURNITURE_COLORS_RGB.items():
    PALETTE[f"furn_{_ftype}"] = _rgb
COLOR_TOL = 40          # допуск классификации пикселя по цвету (евклидова дистанция)
CALIB_LENGTH_M = 1.000  # реальная длина калибровочного отрезка
DEFAULT_WALL_HEIGHT_M = 2.70
DOOR_HEIGHT_M = 2.10
WINDOW_HEIGHT_M = 1.50
WINDOW_SM_HEIGHT_M = 0.90
MIN_ROOM_AREA_M2 = 0.5   # фильтр шумовых компонент
MIN_FURNITURE_AREA_M2 = 0.03  # фильтр шумовых компонент мебели (маленький прямоугольник)
# DEFAULT_FURNITURE_HEIGHT и цвета мебели — теперь в furniture.py (см. импорт выше)


# ------------------------------------------------------------------
# 1. Классификация пикселей
# ------------------------------------------------------------------
def classify(img):
    """Возвращает dict: имя_цвета -> boolean-маска (H,W)."""
    masks = {}
    img_f = img.astype(np.int16)
    for name, rgb in PALETTE.items():
        d = np.sqrt(((img_f - np.array(rgb)) ** 2).sum(axis=2))
        masks[name] = d < COLOR_TOL
    return masks


# ------------------------------------------------------------------
# 2. Калибровка масштаба
# ------------------------------------------------------------------
def calibrate(masks):
    lbl, n = ndimage.label(masks["calib"], structure=np.ones((3, 3)))
    if n == 0:
        raise ValueError("Калибровочный отрезок (magenta #FF00FF) не найден на чертеже.")
    # берём крупнейшую компоненту (на случай шумовых пикселей того же цвета)
    sizes = ndimage.sum(masks["calib"], lbl, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    ys, xs = np.where(lbl == biggest)
    px_len = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    px_per_m = px_len / CALIB_LENGTH_M
    print(f"Калибровка: отрезок {px_len}px = {CALIB_LENGTH_M}м -> {px_per_m:.2f} px/м")
    return px_per_m


# ------------------------------------------------------------------
# 3. Разбиение бинарной маски на прямоугольники (scanline run-merge)
#    Работает для ЛЮБОЙ ортогональной формы (углы, Г-образные комнаты и т.д.)
# ------------------------------------------------------------------
def regularize_mask(mask, block=8):
    """Снимает пиксельный шум (антиалиасинг, растровое дрожание границы) —
    привязывает границу маски к грубой сетке block x block, чтобы
    mask_to_rects не резал сотни тонких прямоугольников на каждый
    однопиксельный скачок границы."""
    H, W = mask.shape
    Hc, Wc = (H + block - 1) // block, (W + block - 1) // block
    padded = np.zeros((Hc * block, Wc * block), dtype=bool)
    padded[:H, :W] = mask
    blocks = padded.reshape(Hc, block, Wc, block)
    coarse = blocks.mean(axis=(1, 3)) > 0.5
    regular = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
    return regular[:H, :W]


def mask_to_rects(mask):
    H, W = mask.shape
    rects = []
    active = {}  # (col0,col1) -> row_start
    for row in range(H + 1):
        runs = set()
        if row < H:
            r = mask[row]
            diff = np.diff(np.concatenate(([0], r.astype(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            runs = set(zip(starts.tolist(), ends.tolist()))
        # закрываем те, что не продолжились
        for key in list(active.keys()):
            if key not in runs:
                row0 = active.pop(key)
                rects.append((key[0], row0, key[1], row))
        # открываем новые
        for key in runs:
            if key not in active:
                active[key] = row
    return rects  # список (col0,row0,col1,row1) в пикселях


def rects_to_m(rects, px_per_m, img_h):
    """Пиксельные прямоугольники -> метры, с переворотом оси Y (низ картинки = y=0)."""
    out = []
    for (c0, r0, c1, r1) in rects:
        x0, x1 = c0 / px_per_m, c1 / px_per_m
        y0, y1 = (img_h - r1) / px_per_m, (img_h - r0) / px_per_m
        out.append((x0, y0, x1, y1))
    return out


# ------------------------------------------------------------------
# 4. Поиск комнат (заливка areas, не являющихся стеной/проёмом)
# ------------------------------------------------------------------
def find_rooms(masks, px_per_m):
    barrier = (masks["ext_wall"] | masks["int_wall"] | masks["door"] |
               masks["window"] | masks["window_sm"] | masks["zone_split"])
    # морфологическое замыкание: убирает однопиксельные "протечки" на стыке
    # двух цветов палитры (антиалиасинг создаёт смешанный пиксель вне допуска
    # обоих цветов -> дырка в барьере -> комнаты сливаются через неё)
    barrier = cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8)).astype(bool)
    candidate = ~barrier
    lbl, n = ndimage.label(candidate, structure=np.ones((3, 3)))
    H, W = candidate.shape

    rooms = []
    block = max(4, round(0.12 * px_per_m))  # сетка ~12 см — снимает пиксельный шум границ
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        touches_border = r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1
        area_px = len(ys)
        area_m2 = area_px / px_per_m ** 2
        if touches_border or area_m2 < MIN_ROOM_AREA_M2:
            continue
        mask_i = regularize_mask(lbl == i, block)
        rooms.append(dict(mask=mask_i, area_m2=round(area_m2, 2),
                           cx=xs.mean(), cy=ys.mean(),
                           bbox=(c0, r0, c1, r1)))

    # сортировка "как читаем": сверху-вниз, слева-направо
    rooms.sort(key=lambda r: (round(r["cy"] / (0.6 * px_per_m)), r["cx"]))
    for idx, r in enumerate(rooms, start=1):
        r["id"] = idx
    return rooms


# ------------------------------------------------------------------
# 5. Проёмы: компоненты door/window масок -> ширина, центр, тип
# ------------------------------------------------------------------
def find_openings(masks, px_per_m, kind, height_m):
    lbl, n = ndimage.label(masks[kind], structure=np.ones((3, 3)))
    openings = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) < 3:
            continue
        w_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        openings.append(dict(kind=kind, width_m=round(w_px / px_per_m, 2),
                              height_m=height_m, cx=xs.mean(), cy=ys.mean(),
                              mask=(lbl == i)))
    return openings


def find_furniture(masks, px_per_m, img_h):
    """Каждый закрашенный прямоугольник = один предмет мебели; ЦВЕТ определяет
    тип (furn_bed, furn_sofa, ...), CSV с типом больше не нужен."""
    items = []
    for key in masks:
        if not key.startswith("furn_"):
            continue
        ftype = key[len("furn_"):]
        lbl, n = ndimage.label(masks[key], structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            c0, c1, r0, r1 = xs.min(), xs.max(), ys.min(), ys.max()
            area_m2 = len(ys) / px_per_m ** 2
            if area_m2 < MIN_FURNITURE_AREA_M2:
                continue
            x0, x1 = c0 / px_per_m, (c1 + 1) / px_per_m
            y0, y1 = (img_h - (r1 + 1)) / px_per_m, (img_h - r0) / px_per_m
            items.append(dict(mask=(lbl == i), bbox=(c0, r0, c1, r1), type=ftype,
                               height=DEFAULT_FURNITURE_HEIGHT[ftype],
                               x0=x0, y0=y0, x1=x1, y1=y1,
                               cx=xs.mean(), cy=ys.mean(),
                               w_m=round(x1 - x0, 2), d_m=round(y1 - y0, 2)))
    items.sort(key=lambda f: (round(f["cy"] / (0.6 * px_per_m)), f["cx"]))
    for idx, it in enumerate(items, start=1):
        it["id"] = idx
    return items


def attribute_opening_to_rooms(opening, rooms, px_per_m, img_h, pad_m=0.3):
    """Проём принадлежит комнате(ам), чей bbox лежит рядом с проёмом (в пределах pad_m)."""
    ys, xs = np.where(opening["mask"])
    oc0, oc1, or0, or1 = xs.min(), xs.max(), ys.min(), ys.max()
    ox0, ox1 = oc0 / px_per_m, oc1 / px_per_m
    oy0, oy1 = (img_h - or1) / px_per_m, (img_h - or0) / px_per_m
    owners = []
    for r in rooms:
        bc0, br0, bc1, br1 = r["bbox"]
        bx0, bx1 = bc0 / px_per_m, bc1 / px_per_m
        by0, by1 = (img_h - br1) / px_per_m, (img_h - br0) / px_per_m
        if not (ox1 < bx0 - pad_m or ox0 > bx1 + pad_m or oy1 < by0 - pad_m or oy0 > by1 + pad_m):
            owners.append(r["id"])
    return owners


# ------------------------------------------------------------------
# 6. Основной пайплайн разбора
# ------------------------------------------------------------------
def parse(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    masks = classify(img)
    px_per_m = calibrate(masks)

    rooms = find_rooms(masks, px_per_m)
    doors = find_openings(masks, px_per_m, "door", DOOR_HEIGHT_M)
    windows = (find_openings(masks, px_per_m, "window", WINDOW_HEIGHT_M) +
               find_openings(masks, px_per_m, "window_sm", WINDOW_SM_HEIGHT_M))

    dilate_px = max(3, int(0.15 * px_per_m))
    for o in doors + windows:
        o["rooms"] = attribute_opening_to_rooms(o, rooms, px_per_m, H)

    furniture = find_furniture(masks, px_per_m, H)
    for f in furniture:
        f["rooms"] = attribute_opening_to_rooms(f, rooms, px_per_m, H, pad_m=0.05)
        f["blocks_door"] = []
        dx0, dx1 = f["x0"] - 0.05, f["x1"] + 0.05
        dy0, dy1 = f["y0"] - 0.05, f["y1"] + 0.05
        for d in doors:
            ys_d, xs_d = np.where(d["mask"])
            oc0, oc1, or0, or1 = xs_d.min(), xs_d.max(), ys_d.min(), ys_d.max()
            ox0, ox1 = oc0 / px_per_m, oc1 / px_per_m
            oy0, oy1 = (H - or1) / px_per_m, (H - or0) / px_per_m
            overlap = not (dx1 < ox0 or dx0 > ox1 or dy1 < oy0 or dy0 > oy1)
            if overlap:
                f["blocks_door"].append(d)

    wall_mask = masks["ext_wall"] | masks["int_wall"]
    wall_block = max(4, round(0.12 * px_per_m))
    wall_mask = regularize_mask(wall_mask, wall_block)
    wall_rects_px = mask_to_rects(wall_mask)
    wall_rects_m = rects_to_m(wall_rects_px, px_per_m, H)

    for f in furniture:
        if f["blocks_door"]:
            print(f"ВНИМАНИЕ: мебель №{f['id']} перекрывает дверной проём "
                  f"(ширина {f['w_m']}x{f['d_m']}м) — сдвинь её на чертеже.")

    return dict(img=img, H=H, W=W, px_per_m=px_per_m, masks=masks,
                rooms=rooms, doors=doors, windows=windows, furniture=furniture,
                wall_rects_m=wall_rects_m)


# ------------------------------------------------------------------
# 7. Отладочная картинка + шаблон CSV (первый проход)
# ------------------------------------------------------------------
def write_debug_and_template(parsed, out_dir):
    img = parsed["img"].copy()
    for r in parsed["rooms"]:
        cx, cy = int(r["cx"]), int(r["cy"])
        cv2.circle(img, (cx, cy), 4, (0, 0, 0), -1)
        cv2.putText(img, str(r["id"]), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(r["id"]), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
    for f in parsed["furniture"]:
        cx, cy = int(f["cx"]), int(f["cy"])
        label = f["type"]
        color = (200, 0, 0) if f["blocks_door"] else (0, 120, 0)  # img в RGB: (R,G,B)
        cv2.putText(img, label, (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    debug_path = os.path.join(out_dir, "debug_rooms.png")
    # cv2.imwrite молча возвращает False на путях с кириллицей/пробелами в Windows —
    # поэтому кодируем в память и пишем через обычный Python-файл (Unicode-safe)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Не удалось закодировать debug_rooms.png (cv2.imencode)")
    with open(debug_path, "wb") as f:
        f.write(buf.tobytes())
    if not os.path.exists(debug_path):
        raise RuntimeError(f"Файл не появился на диске: {debug_path}")

    csv_path = os.path.join(out_dir, "rooms_legend_template.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "name", "height_m", "flooring"])
        for r in parsed["rooms"]:
            w.writerow([r["id"], "", DEFAULT_WALL_HEIGHT_M, ""])
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Файл не появился на диске: {csv_path}")

    furn_csv_path = os.path.join(out_dir, "furniture_overrides_template.csv")
    with open(furn_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["furniture_id", "type_(автоопределён)", "height_override_m",
                     "room_id(ы), через ;", "габарит_wxd_м"])
        for it in parsed["furniture"]:
            rooms_str = ";".join(str(x) for x in it["rooms"])
            w.writerow([it["id"], it["type"], "", rooms_str, f"{it['w_m']}x{it['d_m']}"])
        f.write("# тип определён по цвету автоматически, этот CSV нужен, только\n")
        f.write("# если хочешь переопределить высоту конкретного предмета (необязательно)\n")
    if not os.path.exists(furn_csv_path):
        raise RuntimeError(f"Файл не появился на диске: {furn_csv_path}")

    print(f"Найдено комнат: {len(parsed['rooms'])}")
    print(f"Найдено предметов мебели: {len(parsed['furniture'])}")
    blocked = [f for f in parsed["furniture"] if f["blocks_door"]]
    if blocked:
        print(f"ВНИМАНИЕ: {len(blocked)} предмет(ов) перекрывают дверные проёмы: "
              f"{[f['id'] for f in blocked]} — см. красные подписи на debug_rooms.png")
    print(f"Отладочная картинка: {debug_path}  ({os.path.getsize(debug_path)} байт)")
    print(f"Шаблон легенды комнат: {csv_path}  ({os.path.getsize(csv_path)} байт)")
    print(f"(опционально) переопределение высоты мебели: {furn_csv_path}")
    print("<- заполни rooms_legend.csv (название/высоту/покрытие) и запусти с --legend")


def load_legend(path):
    legend = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = int(row["room_id"])
            legend[rid] = dict(
                name=(row["name"].strip() or f"Комната №{rid}"),
                height=float(row["height_m"]) if row["height_m"] else DEFAULT_WALL_HEIGHT_M,
                flooring=(row["flooring"].strip() or "не указано"),
            )
    return legend


def load_furniture_legend(path):
    """Необязательный override — только высота конкретного предмета по его id.
    Тип определяется автоматически по цвету и здесь не меняется."""
    overrides = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid_raw = (row.get("furniture_id") or "").strip()
            if not fid_raw.isdigit():
                continue  # пропускаем строки-комментарии
            fid = int(fid_raw)
            height_raw = (row.get("height_override_m") or "").strip()
            if height_raw:
                overrides[fid] = float(height_raw)
    return overrides


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--legend", default=None)
    ap.add_argument("--furniture", default=None)
    
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs") 
    ap.add_argument("--out", default=DEFAULT_OUTPUT_DIR) 
    args = ap.parse_args() 
    os.makedirs(args.out, exist_ok=True)

    parsed = parse(args.image)

    if args.legend is None:
        write_debug_and_template(parsed, args.out)
    else:
        legend = load_legend(args.legend)
        print(f"Легенда комнат загружена: {len(legend)} комнат")
        if args.furniture:
            flegend = load_furniture_legend(args.furniture)
            print(f"Легенда мебели загружена: {len(flegend)} предметов")
        # экспорт делает build_from_parsed.py (следующий шаг пайплайна)