"""Layout geometry and style tokens can be combined independently."""

LAYOUTS = {
    "single-hero": ((1, 1), ((0, 0, 1, 1),)),
    "hero-1plus3": ((3, 5), ((0, 0, 3, 3), (0, 3, 1, 2), (1, 3, 1, 2), (2, 3, 1, 2))),
    "grid-2x2": ((2, 2), ((0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1))),
    "three-step-story": ((1, 3), ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1))),
    "asymmetric-mosaic": ((5, 2), ((0, 0, 3, 2), (3, 0, 2, 1), (3, 1, 2, 1))),
}

PRESETS = {
    "canteen-hero": "single-hero",
    "food-hero": "single-hero",
    "food-grid-4": "grid-2x2",
    "hero-1plus3": "hero-1plus3",
    "kitchen-process": "three-step-story",
    "ingredient-mosaic": "asymmetric-mosaic",
}

STYLES = {
    "real-photo": {"background": "#F6F4EE", "foreground": "#222B25", "accent": "#3A5447",
                   "line": "#D2D7CF", "margin": 48, "gap": 14, "radius": 0, "border": 1, "shadow": 0,
                   "headline_size": 60, "serif": False},
    "editorial": {"background": "#F5E9D9", "foreground": "#25211F", "accent": "#9C3B2F",
                  "line": "#D3B9A3", "margin": 64, "gap": 16, "radius": 0, "border": 2, "shadow": 0,
                  "headline_size": 76, "serif": True},
    "corporate-clean": {"background": "#EAF1EF", "foreground": "#142D29", "accent": "#176358",
                        "line": "#ACC5BB", "margin": 54, "gap": 22, "radius": 22, "border": 1, "shadow": 6,
                        "headline_size": 60, "serif": False},
}


def rectangles(layout, x, y, width, height, gap):
    """Allocate gutters only between cells. Return Pillow half-open boxes."""
    (cols, rows), cells = LAYOUTS[layout]
    unit_x = (width - (cols - 1) * gap) / cols
    unit_y = (height - (rows - 1) * gap) / rows
    return [(round(x + col * (unit_x + gap)), round(y + row * (unit_y + gap)),
             round(x + (col + w) * (unit_x + gap) - gap),
             round(y + (row + h) * (unit_y + gap) - gap))
            for col, row, w, h in cells]
