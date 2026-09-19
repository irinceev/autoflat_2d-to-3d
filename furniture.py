"""
Заготовленная мебель: простые нетекстурированные "болванки" (только боксы,
без деталей) для каждого типа, плюс уникальный цвет на 2D-чертеже.
"""

# type -> (HEX-цвет прямоугольника на чертеже, высота предмета по умолчанию, м)
FURNITURE_TYPES = {
    "bed":        ("#CD853F", 0.55),  # кровать
    "sofa":       ("#4682B4", 0.80),  # диван
    "wardrobe":   ("#8B4513", 2.10),  # шкаф
    "kitchen":    ("#2E8B57", 0.90),  # кухонный гарнитур
    "table":      ("#800080", 0.75),  # стол
    "chair":      ("#DB7093", 0.90),  # стул
    "nightstand": ("#006400", 0.50),  # тумба
    "toilet":     ("#B0E0E6", 0.40),  # унитаз
    "sink":       ("#191970", 0.85),  # раковина
    "bathtub":    ("#F0E68C", 0.50),  # ванна
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# готовые RGB-тройки для парсера (классификация пикселей чертежа)
FURNITURE_COLORS_RGB = {name: hex_to_rgb(hexcol) for name, (hexcol, _h) in FURNITURE_TYPES.items()}
# высота по типу — то, что раньше называлось DEFAULT_FURNITURE_HEIGHT
DEFAULT_FURNITURE_HEIGHT = {name: h for name, (_hexcol, h) in FURNITURE_TYPES.items()}
DEFAULT_FURNITURE_HEIGHT["other"] = 0.80  # на случай нераспознанного цвета


def build_furniture_box(mesh, item, group_name):
    """Простая недетализированная 'болванка' — один прямоугольный параллелепипед
    от пола до высоты предмета, габарит = как нарисовано на плане."""
    mesh.add_box(item["x0"], item["x1"], item["y0"], item["y1"],
                 0.0, item["height"], group_name)