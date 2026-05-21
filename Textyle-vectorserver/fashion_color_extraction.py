"""Fashion color extraction module for Textyle vector search server.

Provides clothing-oriented color classification using named-color matching,
CIELab distance, RGB heuristics, and k-means clustering.  The main entry
point is ``extract_dominant_color_result`` which accepts a PIL image (and an
optional segmentation mask) and returns a ``ColorExtractionResult`` with the
dominant fashion color, confidence level, and search-color weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from PIL import Image, ImageColor


# ---------------------------------------------------------------------------
# Text-based context helpers (unchanged public API)
# ---------------------------------------------------------------------------

DENIM_CONTEXT_TERMS = (
    "denim",
    "jean",
    "jeans",
    "raw denim",
    "dark denim",
    "\ub370\ub2d8",
    "\uccad\ubc14\uc9c0",
    "\ud751\uccad",
    "\uc9c4\uccad",
    "\uc911\uccad",
    "\uc5f0\uccad",
)

PATTERN_CONTEXT_TERMS = (
    "check",
    "checked",
    "checkered",
    "plaid",
    "tartan",
    "gingham",
    "stripe",
    "striped",
    "pattern",
    "patterns",
    "dot",
    "dotted",
    "\uccb4\ud06c",
    "\uccb4\ud06c\ubb34\ub2ac",
    "\uc2a4\ud2b8\ub77c\uc774\ud504",
    "\uc904\ubb34\ub2ac",
    "\ud328\ud134",
    "\ub3c4\ud2b8",
)


# ---------------------------------------------------------------------------
# Color constants (ported from verify_groundingdino_sam_color_extraction.py)
# ---------------------------------------------------------------------------

COLOR_CENTROIDS = {
    "black": (30, 30, 32),
    "white": (235, 235, 225),
    "gray": (125, 125, 125),
    "navy": (22, 34, 72),
    "blue": (45, 100, 185),
    "indigo": (38, 58, 95),
    "red": (175, 45, 45),
    "green": (55, 120, 70),
    "khaki": (105, 110, 70),
    "yellow": (220, 190, 65),
    "beige": (205, 180, 135),
    "brown": (105, 70, 45),
    "pink": (215, 120, 155),
    "purple": (110, 70, 145),
    "orange": (210, 115, 45),
}

FINAL_COLOR_CATEGORIES = {
    "white", "black", "red", "yellow", "green",
    "blue", "purple", "gray", "orange", "brown", "pink",
}

FASHION_TO_FINAL_COLOR = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "navy": "blue",
    "blue": "blue",
    "indigo": "blue",
    "red": "red",
    "green": "green",
    "khaki": "green",
    "yellow": "yellow",
    "beige": "white",
    "brown": "brown",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
}

NAMED_COLOR_GROUPS = {
    "white": {
        "white", "snow", "honeydew", "mintcream", "azure", "aliceblue",
        "ghostwhite", "whitesmoke", "seashell", "beige", "oldlace",
        "floralwhite", "ivory", "antiquewhite", "linen", "lavenderblush",
        "mistyrose",
    },
    "gray": {
        "gray", "gainsboro", "lightgray", "silver", "darkgray", "dimgray",
        "lightslategray", "slategray", "darkslategray", "black",
    },
    "red": {
        "red", "lightsalmon", "salmon", "darksalmon", "lightcoral",
        "indianred", "crimson", "firebrick", "darkred",
    },
    "pink": {
        "pink", "lightpink", "hotpink", "deeppink", "palevioletred",
        "mediumvioletred",
    },
    "orange": {"orange", "darkorange", "coral", "tomato", "orangered"},
    "yellow": {
        "yellow", "lightyellow", "lemonchiffon", "lightgoldenrodyellow",
        "papayawhip", "moccasin", "peachpuff", "palegoldenrod", "khaki",
        "darkkhaki", "gold",
    },
    "brown": {
        "brown", "cornsilk", "blanchedalmond", "bisque", "navajowhite",
        "wheat", "burlywood", "tan", "rosybrown", "sandybrown", "goldenrod",
        "darkgoldenrod", "peru", "chocolate", "saddlebrown", "sienna",
        "maroon",
    },
    "green": {
        "green", "palegreen", "lightgreen", "yellowgreen", "greenyellow",
        "chartreuse", "lawngreen", "lime", "limegreen",
        "mediumspringgreen", "springgreen", "mediumaquamarine",
        "aquamarine", "lightseagreen", "mediumseagreen", "seagreen",
        "darkseagreen", "forestgreen", "darkgreen", "olivedrab", "olive",
        "darkolivegreen", "teal",
    },
    "blue": {
        "blue", "lightblue", "powderblue", "paleturquoise", "turquoise",
        "mediumturquoise", "darkturquoise", "lightcyan", "cyan", "aqua",
        "darkcyan", "cadetblue", "lightsteelblue", "steelblue",
        "lightskyblue", "skyblue", "deepskyblue", "dodgerblue",
        "cornflowerblue", "royalblue", "mediumblue", "darkblue", "navy",
        "midnightblue",
    },
    "purple": {
        "purple", "lavender", "thistle", "plum", "violet", "orchid",
        "fuchsia", "magenta", "mediumorchid", "mediumpurple", "amethyst",
        "blueviolet", "darkviolet", "darkorchid", "darkmagenta",
        "slateblue", "darkslateblue", "mediumslateblue", "indigo",
    },
}

NAMED_GROUP_TO_FASHION_COLOR = {
    "white": "white", "gray": "gray", "red": "red", "pink": "pink",
    "orange": "orange", "yellow": "yellow", "brown": "brown",
    "green": "green", "blue": "blue", "purple": "purple",
}

NAMED_COLOR_TO_FASHION_COLOR = {
    "black": "black",
    "dimgray": "gray", "darkslategray": "gray",
    "beige": "white", "antiquewhite": "white", "linen": "white",
    "navy": "blue", "midnightblue": "blue", "darkblue": "blue",
    "mediumblue": "blue", "blue": "blue", "royalblue": "blue",
    "dodgerblue": "blue", "cornflowerblue": "blue", "indigo": "blue",
    "khaki": "green", "darkkhaki": "green", "olive": "green",
    "olivedrab": "green", "darkolivegreen": "green",
    "tan": "brown", "wheat": "brown", "burlywood": "brown",
    "bisque": "brown", "navajowhite": "brown", "blanchedalmond": "brown",
    "saddlebrown": "brown", "sienna": "brown", "chocolate": "brown",
    "peru": "brown", "maroon": "brown",
}

AMBIGUOUS_NEUTRAL_NAMED_COLORS = {
    "darkslategray", "darkslategrey", "dimgray", "dimgrey",
    "slategray", "slategrey", "lightslategray", "lightslategrey",
    "gray", "grey", "darkgray", "darkgrey",
}

SEARCH_COLOR_ALIASES = {
    "black": {"black": 1.0, "gray": 0.35, "blue": 0.20, "brown": 0.20},
    "white": {"white": 1.0, "gray": 0.35, "brown": 0.25},
    "gray": {"gray": 1.0, "white": 0.35, "brown": 0.25, "blue": 0.20},
    "blue": {"blue": 1.0, "gray": 0.25, "purple": 0.20},
    "green": {"green": 1.0, "brown": 0.30, "yellow": 0.25},
    "brown": {"brown": 1.0, "green": 0.35, "yellow": 0.25, "orange": 0.20},
    "red": {"red": 1.0, "brown": 0.40, "pink": 0.35, "purple": 0.25},
    "pink": {"pink": 1.0, "red": 0.45, "purple": 0.35},
    "purple": {"purple": 1.0, "pink": 0.35, "red": 0.30, "blue": 0.25},
    "yellow": {"yellow": 1.0, "orange": 0.45, "brown": 0.25},
    "orange": {"orange": 1.0, "yellow": 0.45, "brown": 0.40, "red": 0.30},
}

COLOR_FAMILIES = (
    {"black", "gray"},
    {"blue", "gray"},
    {"brown", "green", "yellow", "orange"},
    {"white", "gray", "brown"},
    {"red", "pink", "purple", "brown"},
)

# Pixel extraction constants
_MIN_PIXEL_COUNT = 50
_MAX_KMEANS_PIXELS = 12000
_BACKGROUND_DISTANCE_SQ = 48 ** 2
_SEGMENTATION_MASK_THRESHOLD = 24
_BORDER_RATIO = 0.08


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorExtractionResult:
    color: str = ""
    confidence: str = "low"
    reason: str = ""
    dominant_ratio: float = 0.0
    second_ratio: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    secondary_colors: list[str] = field(default_factory=list)
    is_mixed_color: bool = False
    pattern: str = ""
    search_color_weights: dict[str, float] = field(default_factory=dict)
    segmentation_used: bool = False
    candidate_pixel_count: int = 0
    ignored_pixel_count: int = 0
    background_pixel_count: int = 0


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _compact_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    compact = []
    for char in text:
        if char.isascii() and char.isalnum():
            compact.append(char)
        elif "\uac00" <= char <= "\ud7a3":
            compact.append(char)
    return "".join(compact)


def is_denim_context_from_text(*values: Any) -> bool:
    compact_text = _compact_text(*values)
    return any(_compact_text(term) in compact_text for term in DENIM_CONTEXT_TERMS)


def should_run_pattern_classifier(text: str) -> bool:
    compact_text = _compact_text(text)
    return any(_compact_text(term) in compact_text for term in PATTERN_CONTEXT_TERMS)


# ---------------------------------------------------------------------------
# CIELab colour-space utilities
# ---------------------------------------------------------------------------

def _rgb_to_xyz_component(value: float) -> float:
    value = value / 255.0
    if value > 0.04045:
        return ((value + 0.055) / 1.055) ** 2.4
    return value / 12.92


def _xyz_to_lab_component(value: float) -> float:
    if value > 0.008856:
        return value ** (1 / 3)
    return (7.787 * value) + (16 / 116)


def rgb_to_lab(rgb: tuple) -> tuple[float, float, float]:
    r, g, b = [_rgb_to_xyz_component(float(v)) for v in rgb]
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    x = _xyz_to_lab_component(x)
    y = _xyz_to_lab_component(y)
    z = _xyz_to_lab_component(z)
    return (116 * y) - 16, 500 * (x - y), 200 * (y - z)


def lab_distance(left_rgb: tuple, right_rgb: tuple) -> float:
    left = rgb_to_lab(left_rgb)
    right = rgb_to_lab(right_rgb)
    return sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))


def brightness(rgb: tuple) -> float:
    return sum(int(v) for v in rgb) / 3


def rgb_stddev(rgb: tuple) -> float:
    mean = brightness(rgb)
    return sqrt(sum((int(v) - mean) ** 2 for v in rgb) / 3)


# ---------------------------------------------------------------------------
# Named colour matching (PIL ImageColor palette)
# ---------------------------------------------------------------------------

_NAMED_COLOR_PALETTE_CACHE: list[tuple[str, str, tuple]] | None = None


def _named_color_palette() -> list[tuple[str, str, tuple]]:
    global _NAMED_COLOR_PALETTE_CACHE
    if _NAMED_COLOR_PALETTE_CACHE is not None:
        return _NAMED_COLOR_PALETTE_CACHE
    palette = []
    for name, value in ImageColor.colormap.items():
        try:
            rgb = ImageColor.getrgb(value)
        except ValueError:
            continue
        palette.append((name, named_color_group(name), rgb))
    _NAMED_COLOR_PALETTE_CACHE = palette
    return palette


def named_color_group(name: str) -> str:
    normalized = name.replace("grey", "gray")
    for group, names in NAMED_COLOR_GROUPS.items():
        if normalized in names:
            return group
    return ""


def nearest_named_color(rgb: tuple) -> tuple[str, str, tuple]:
    name, group, named_rgb = min(
        _named_color_palette(),
        key=lambda item: lab_distance(rgb, item[2]),
    )
    return name, group, named_rgb


# ---------------------------------------------------------------------------
# RGB heuristic classifiers
# ---------------------------------------------------------------------------

def _is_light_blue_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        135 <= brightness(rgb) <= 225
        and b >= g + 8
        and g >= r + 5
        and b >= r + 18
    )


def _is_denim_blue_gray_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        70 <= brightness(rgb) <= 170
        and b >= r + 16
        and g >= r + 5
    )


def _is_washed_denim_gray_blue_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    spread = max(r, g, b) - min(r, g, b)
    return (
        105 <= brightness(rgb) <= 180
        and 8 <= spread <= 42
        and b >= r + 9
        and g >= r + 4
        and b >= g + 2
    )


def _is_sage_khaki_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        95 <= brightness(rgb) <= 180
        and abs(r - g) <= 22
        and r >= b + 14
        and g >= b + 14
    )


def _is_muted_green_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        55 <= brightness(rgb) <= 155
        and g >= r + 10
        and g >= b + 6
        and b >= r - 2
    )


def _is_warm_neutral_beige_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        115 <= brightness(rgb) <= 205
        and abs(r - g) <= 18
        and r >= b + 5
        and g >= b + 3
    )


def _is_light_warm_neutral_white_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    spread = max(r, g, b) - min(r, g, b)
    return (
        170 <= brightness(rgb) <= 235
        and spread <= 48
        and abs(r - g) <= 24
        and r >= b + 4
        and g >= b
    )


def _is_dark_navy_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        20 <= brightness(rgb) <= 90
        and b >= r + 12
        and b >= g + 6
    )


def _is_dark_brown_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        28 <= brightness(rgb) <= 95
        and r >= b + 8
        and g >= b + 4
        and abs(r - g) <= 18
    )


def _is_blue_leaning_neutral_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        30 <= brightness(rgb) <= 170
        and b >= r + 8
        and b >= g + 3
    )


def _is_warm_leaning_neutral_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        35 <= brightness(rgb) <= 170
        and r >= b + 5
        and g >= b
    )


def _is_olive_leaning_neutral_rgb(rgb: tuple) -> bool:
    r, g, b = [int(v) for v in rgb]
    return (
        35 <= brightness(rgb) <= 170
        and g >= b + 4
        and r >= b + 3
        and abs(r - g) <= 28
    )


# ---------------------------------------------------------------------------
# Fashion colour classification
# ---------------------------------------------------------------------------

def classify_achromatic_rgb(rgb: tuple, color_hint: str = "") -> tuple[str, str]:
    value = brightness(rgb)
    deviation = rgb_stddev(rgb)
    if deviation > 10:
        return "", ""
    if color_hint in {"black", "gray", "white"}:
        return color_hint, f"hint_{color_hint}_achromatic"
    if value <= 70:
        return "black", "rgb_achromatic_black"
    if value >= 188:
        return "white", "rgb_achromatic_white"
    return "gray", "rgb_achromatic_gray"


def hinted_chromatic_neutral_color(rgb: tuple, color_hint: str) -> tuple[str, str]:
    if color_hint == "blue" and (
        _is_blue_leaning_neutral_rgb(rgb)
        or _is_dark_navy_rgb(rgb)
        or _is_denim_blue_gray_rgb(rgb)
        or _is_washed_denim_gray_blue_rgb(rgb)
    ):
        return "blue", "hint_blue_neutral"
    if color_hint == "white" and _is_light_warm_neutral_white_rgb(rgb):
        return "white", "hint_white_warm_neutral"
    if color_hint == "green" and (
        _is_olive_leaning_neutral_rgb(rgb)
        or _is_muted_green_rgb(rgb)
        or _is_sage_khaki_rgb(rgb)
    ):
        return "green", "hint_green_neutral"
    if color_hint == "brown" and (
        _is_warm_leaning_neutral_rgb(rgb)
        or _is_dark_brown_rgb(rgb)
        or _is_warm_neutral_beige_rgb(rgb)
    ):
        return "brown", "hint_brown_neutral"
    return "", ""


def reinterpret_fashion_color(
    named_color: str,
    rgb: tuple,
    default_color: str,
    color_hint: str = "",
) -> tuple[str, str]:
    normalized = named_color.replace("grey", "gray")
    default_color = FASHION_TO_FINAL_COLOR.get(default_color, default_color)
    color_hint = FASHION_TO_FINAL_COLOR.get(color_hint, color_hint)

    if normalized not in AMBIGUOUS_NEUTRAL_NAMED_COLORS:
        return default_color, "named_color"

    hinted, hinted_reason = hinted_chromatic_neutral_color(rgb, color_hint)
    if hinted:
        return hinted, hinted_reason

    if color_hint in {"black", "gray", "white"}:
        achromatic, achromatic_reason = classify_achromatic_rgb(rgb, color_hint)
        if achromatic:
            return achromatic, achromatic_reason
    if color_hint == "black" and brightness(rgb) <= 95:
        return "black", "hint_black_dark_neutral"
    if color_hint == "gray" and brightness(rgb) <= 120:
        return "gray", "hint_gray_dark_neutral"
    if color_hint == "gray":
        return "gray", "hint_gray_neutral"

    achromatic, achromatic_reason = classify_achromatic_rgb(rgb)
    if achromatic:
        return achromatic, achromatic_reason
    if _is_dark_navy_rgb(rgb):
        return "blue", "rgb_dark_navy_neutral"
    if _is_denim_blue_gray_rgb(rgb):
        return "blue", "rgb_denim_neutral"
    if _is_dark_brown_rgb(rgb):
        return "brown", "rgb_dark_brown_neutral"
    if _is_light_warm_neutral_white_rgb(rgb):
        return "white", "rgb_warm_light_neutral"
    if _is_olive_leaning_neutral_rgb(rgb):
        return "green", "rgb_olive_neutral"
    return default_color, "named_color"


def classify_named_and_fashion_color(
    rgb: tuple, color_hint: str = "",
) -> tuple[str, str, tuple, str, str]:
    """Return (named_color, named_group, named_rgb, fashion_color, fashion_reason)."""
    name, group, named_rgb = nearest_named_color(rgb)
    normalized = name.replace("grey", "gray")
    default_color = NAMED_COLOR_TO_FASHION_COLOR.get(
        normalized,
        NAMED_GROUP_TO_FASHION_COLOR.get(group, group),
    )
    fashion_color, fashion_reason = reinterpret_fashion_color(
        name, rgb, default_color, color_hint,
    )
    return name, group, named_rgb, fashion_color, fashion_reason


def nearest_color(rgb: tuple) -> str:
    """Classify *rgb* into one of the 11 final fashion colour categories."""
    if _is_light_blue_rgb(rgb):
        return "blue"
    if _is_washed_denim_gray_blue_rgb(rgb):
        return "blue"
    if _is_denim_blue_gray_rgb(rgb):
        return "blue"
    if _is_dark_navy_rgb(rgb):
        return "blue"
    if _is_muted_green_rgb(rgb):
        return "green"
    if _is_sage_khaki_rgb(rgb):
        return "green"
    if _is_dark_brown_rgb(rgb):
        return "brown"
    if _is_light_warm_neutral_white_rgb(rgb):
        return "white"
    if _is_warm_neutral_beige_rgb(rgb):
        return "brown"
    name, group, _named_rgb = nearest_named_color(rgb)
    normalized = name.replace("grey", "gray")
    default_color = NAMED_COLOR_TO_FASHION_COLOR.get(
        normalized,
        NAMED_GROUP_TO_FASHION_COLOR.get(group, group),
    )
    fashion_color, _reason = reinterpret_fashion_color(name, rgb, default_color)
    return fashion_color


# ---------------------------------------------------------------------------
# K-means clustering + candidate generation
# ---------------------------------------------------------------------------

def _squared_distance(a: tuple, b: tuple) -> float:
    return sum((int(a[i]) - int(b[i])) ** 2 for i in range(3))


def kmeans_color_candidates(
    pixels: list[tuple],
    n_clusters: int = 5,
    color_hint: str = "",
) -> list[dict]:
    if not pixels:
        return []
    if len(pixels) > _MAX_KMEANS_PIXELS:
        step = max(1, len(pixels) // _MAX_KMEANS_PIXELS)
        pixels = pixels[::step]

    # Initialise centres from quantised buckets
    buckets: dict[tuple, list] = {}
    for r, g, b in pixels:
        key = (round(r / 32) * 32, round(g / 32) * 32, round(b / 32) * 32)
        buckets.setdefault(key, [0, 0, 0, 0])
        buckets[key][0] += 1
        buckets[key][1] += r
        buckets[key][2] += g
        buckets[key][3] += b

    centres: list[tuple] = []
    for _key, (count, r_sum, g_sum, b_sum) in sorted(
        buckets.items(), key=lambda row: row[1][0], reverse=True,
    ):
        centres.append((r_sum / count, g_sum / count, b_sum / count))
        if len(centres) >= min(n_clusters, len(pixels)):
            break
    if not centres:
        return []

    labels = [0] * len(pixels)
    for _ in range(8):
        changed = False
        sums = [[0, 0, 0, 0] for _ in centres]
        for idx, pixel in enumerate(pixels):
            label = min(
                range(len(centres)),
                key=lambda ci: _squared_distance(pixel, centres[ci]),
            )
            if labels[idx] != label:
                changed = True
            labels[idx] = label
            sums[label][0] += 1
            sums[label][1] += pixel[0]
            sums[label][2] += pixel[1]
            sums[label][3] += pixel[2]
        for idx, (count, r_sum, g_sum, b_sum) in enumerate(sums):
            if count:
                centres[idx] = (r_sum / count, g_sum / count, b_sum / count)
        if not changed:
            break

    # Rebuild counts from final labels
    counts = [0] * len(centres)
    rgb_sums = [[0, 0, 0] for _ in centres]
    for pixel, label in zip(pixels, labels):
        counts[label] += 1
        rgb_sums[label][0] += pixel[0]
        rgb_sums[label][1] += pixel[1]
        rgb_sums[label][2] += pixel[2]

    total = sum(counts)
    candidates = []
    for count, rs in zip(counts, rgb_sums):
        if not count:
            continue
        rgb = tuple(int(round(v / count)) for v in rs)
        named_color, named_group, named_rgb, fashion_color, fashion_reason = (
            classify_named_and_fashion_color(rgb, color_hint)
        )
        candidates.append({
            "rgb": rgb,
            "named_color": named_color,
            "named_group": named_group,
            "named_rgb": named_rgb,
            "fashion_color": fashion_color,
            "fashion_reason": fashion_reason,
            "color": fashion_color,
            "ratio": count / total,
        })
    candidates.sort(key=lambda c: c["ratio"], reverse=True)
    return candidates


def merge_named_color_candidates(
    candidates: list[dict], color_hint: str = "",
) -> list[dict]:
    merged: dict[str, dict] = {}
    for c in candidates:
        color = c["named_color"]
        ratio = c["ratio"]
        rgb = c["rgb"]
        if color not in merged:
            merged[color] = {
                "named_color": color,
                "named_group": c.get("named_group", ""),
                "named_rgb": c.get("named_rgb"),
                "fashion_color": c.get("fashion_color", ""),
                "color": c.get("fashion_color", ""),
                "ratio": 0.0,
                "rgb_sum": [0.0, 0.0, 0.0],
            }
        merged[color]["ratio"] += ratio
        for i in range(3):
            merged[color]["rgb_sum"][i] += rgb[i] * ratio

    result = []
    for item in merged.values():
        ratio = item["ratio"]
        rgb = (
            tuple(int(round(v / ratio)) for v in item["rgb_sum"])
            if ratio
            else (0, 0, 0)
        )
        (
            named_color, named_group, named_rgb,
            fashion_color, fashion_reason,
        ) = classify_named_and_fashion_color(rgb, color_hint)
        result.append({
            "color": fashion_color,
            "fashion_color": fashion_color,
            "fashion_reason": fashion_reason,
            "named_color": named_color,
            "named_group": named_group,
            "named_rgb": named_rgb,
            "ratio": ratio,
            "rgb": rgb,
        })
    result.sort(key=lambda c: c["ratio"], reverse=True)
    return result


def merge_fashion_color_candidates(named_candidates: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for c in named_candidates:
        color = c["fashion_color"]
        ratio = c["ratio"]
        rgb = c["rgb"]
        if color not in merged:
            merged[color] = {
                "color": color,
                "fashion_color": color,
                "ratio": 0.0,
                "rgb_sum": [0.0, 0.0, 0.0],
                "named_colors": [],
                "fashion_reasons": {},
            }
        merged[color]["ratio"] += ratio
        merged[color]["named_colors"].append({
            "named_color": c["named_color"],
            "ratio": ratio,
            "rgb": c["rgb"],
        })
        reason = c.get("fashion_reason", "named_color")
        merged[color]["fashion_reasons"][reason] = (
            merged[color]["fashion_reasons"].get(reason, 0.0) + ratio
        )
        for i in range(3):
            merged[color]["rgb_sum"][i] += rgb[i] * ratio

    result = []
    for item in merged.values():
        ratio = item["ratio"]
        rgb = (
            tuple(int(round(v / ratio)) for v in item["rgb_sum"])
            if ratio
            else (0, 0, 0)
        )
        result.append({
            "color": item["color"],
            "fashion_color": item["fashion_color"],
            "ratio": ratio,
            "rgb": rgb,
            "named_colors": sorted(
                item["named_colors"],
                key=lambda n: n["ratio"],
                reverse=True,
            ),
            "fashion_reasons": item["fashion_reasons"],
        })
    result.sort(key=lambda c: c["ratio"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Confidence & search-colour weights
# ---------------------------------------------------------------------------

def colors_are_related(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    return any(left in family and right in family for family in COLOR_FAMILIES)


def infer_color_confidence(
    candidates: list[dict],
    color_hint: str = "",
    pattern_hint: str = "",
) -> tuple[str, str]:
    if not candidates:
        return "low", "no_color_candidates"

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    top_ratio = top["ratio"]
    second_ratio = second["ratio"] if second else 0.0
    margin = top_ratio - second_ratio
    hint_disagrees = bool(
        color_hint and not colors_are_related(top["color"], color_hint),
    )

    if pattern_hint:
        if top_ratio >= 0.75 and second_ratio < 0.18 and not hint_disagrees:
            return "medium", "pattern_hint_needs_vit"
        return "low", "pattern_hint_needs_vit"

    if hint_disagrees:
        return "low", "color_hint_disagrees"
    if second_ratio >= 0.30 or margin < 0.25:
        return (
            ("medium" if top_ratio >= 0.55 else "low"),
            "ambiguous_second_color",
        )
    if top_ratio >= 0.75:
        return "high", "dominant_image_color"
    if top_ratio >= 0.55:
        return "medium", "moderate_image_color"
    return "low", "weak_image_color"


def _candidate_confidence(ratio: float) -> str:
    if ratio >= 0.65:
        return "high"
    if ratio >= 0.25:
        return "medium"
    return "low"


def _add_search_color(
    search_colors: dict,
    color: str,
    score: float,
    source: str,
    confidence: str,
    base_color: str | None = None,
) -> None:
    if not color or score <= 0:
        return
    item: dict[str, Any] = {
        "color": color,
        "score": score,
        "source": source,
        "confidence": confidence,
    }
    if base_color and base_color != color:
        item["base_color"] = base_color
    existing = search_colors.get(color)
    if existing is None or score > existing["score"]:
        search_colors[color] = item


def search_color_candidates(
    candidates: list[dict],
    color_confidence: str,
    color_hint: str = "",
    min_ratio: float = 0.15,
) -> list[dict]:
    search_colors: dict[str, dict] = {}
    if color_hint in FINAL_COLOR_CATEGORIES:
        _add_search_color(search_colors, color_hint, 1.0, "product_name", "high")
    for idx, c in enumerate(candidates):
        if idx > 0 and c["ratio"] < min_ratio:
            continue
        base_color = c["color"]
        base_score = c["ratio"]
        confidence = color_confidence if idx == 0 else _candidate_confidence(base_score)
        for color, weight in SEARCH_COLOR_ALIASES.get(
            base_color, {base_color: 1.0},
        ).items():
            source = "image" if color == base_color else "family"
            family_confidence = confidence if color == base_color else "low"
            _add_search_color(
                search_colors,
                color,
                base_score * weight,
                source,
                family_confidence,
                base_color=base_color,
            )
    return sorted(
        search_colors.values(), key=lambda item: item["score"], reverse=True,
    )


# ---------------------------------------------------------------------------
# Pixel extraction helpers
# ---------------------------------------------------------------------------

def _is_skin_like(r: int, g: int, b: int) -> bool:
    brt = (r + g + b) / 3
    return (
        brt >= 90
        and r > 95 and g > 40 and b > 20
        and (max(r, g, b) - min(r, g, b)) > 15
        and abs(r - g) > 15
        and r > g and r > b
    )


def _is_ignored_pixel(r: int, g: int, b: int) -> bool:
    if r > 242 and g > 242 and b > 242:
        return True
    if r < 12 and g < 12 and b < 12:
        return True
    return _is_skin_like(r, g, b)


def _extract_mask_pixels(
    image_obj: Image.Image,
    mask: Image.Image,
    threshold: int = _SEGMENTATION_MASK_THRESHOLD,
) -> tuple[list[tuple], int, int]:
    """Extract non-ignored pixels within *mask*.

    Returns (pixels, ignored_count, total_mask_pixels).
    """
    image = image_obj.convert("RGB").resize((224, 224))
    mask_resized = mask.convert("L").resize((224, 224))
    pixels = []
    ignored = 0
    total_mask = 0
    for y in range(224):
        for x in range(224):
            if mask_resized.getpixel((x, y)) < threshold:
                continue
            total_mask += 1
            r, g, b = image.getpixel((x, y))
            if _is_ignored_pixel(r, g, b):
                ignored += 1
                continue
            pixels.append((r, g, b))
    return pixels, ignored, total_mask


def _collect_border_pixels(
    image_obj: Image.Image,
    border_ratio: float = _BORDER_RATIO,
) -> list[tuple]:
    image = image_obj.convert("RGB").resize((224, 224))
    w, h = image.size
    bx = max(1, int(w * border_ratio))
    by = max(1, int(h * border_ratio))
    pixels = []
    for y in range(h):
        for x in range(w):
            if bx <= x < w - bx and by <= y < h - by:
                continue
            r, g, b = image.getpixel((x, y))
            if _is_ignored_pixel(r, g, b):
                continue
            pixels.append((r, g, b))
    return pixels


def _simple_kmeans_centres(pixels: list[tuple], n: int = 3) -> list[tuple]:
    """Quick kmeans to find *n* background centre colours."""
    if not pixels:
        return []
    buckets: dict[tuple, list] = {}
    for r, g, b in pixels:
        key = (round(r / 32) * 32, round(g / 32) * 32, round(b / 32) * 32)
        buckets.setdefault(key, [0, 0, 0, 0])
        buckets[key][0] += 1
        buckets[key][1] += r
        buckets[key][2] += g
        buckets[key][3] += b
    centres = []
    for _, (count, rs, gs, bs) in sorted(
        buckets.items(), key=lambda row: row[1][0], reverse=True,
    ):
        centres.append((rs / count, gs / count, bs / count))
        if len(centres) >= n:
            break
    return centres


def _estimate_background_rgbs(image_obj: Image.Image) -> list[tuple]:
    border = _collect_border_pixels(image_obj)
    if len(border) < _MIN_PIXEL_COUNT:
        return []
    centres = _simple_kmeans_centres(border, n=3)
    # Keep only centres that represent >= 12% of border pixels
    if not centres:
        return []
    # Simplified: return all centres (border already filtered)
    return centres


def _extract_center_filtered_pixels(
    image_obj: Image.Image,
) -> tuple[list[tuple], int, int]:
    """Center-crop + background-filter pixel extraction.

    Returns (pixels, ignored_count, background_removed_count).
    """
    w, h = image_obj.size
    background_rgbs = _estimate_background_rgbs(image_obj)
    cropped = image_obj.convert("RGB").crop((
        int(w * 0.12),
        int(h * 0.08),
        int(w * 0.88),
        int(h * 0.92),
    )).resize((224, 224))

    pixels = []
    fallback = []
    ignored = 0
    bg_removed = 0
    for pixel in cropped.getdata():
        r, g, b = pixel[0], pixel[1], pixel[2]
        if _is_ignored_pixel(r, g, b):
            ignored += 1
            continue
        p = (r, g, b)
        fallback.append(p)
        if background_rgbs and any(
            _squared_distance(p, bg) <= _BACKGROUND_DISTANCE_SQ
            for bg in background_rgbs
        ):
            bg_removed += 1
            continue
        pixels.append(p)

    if len(pixels) >= _MIN_PIXEL_COUNT:
        return pixels, ignored, bg_removed
    return fallback, ignored, 0


# ---------------------------------------------------------------------------
# Denim-specific pixel classification
# ---------------------------------------------------------------------------

_DENIM_COLOR_CENTROIDS = {
    "black": (32, 32, 35),
    "gray": (95, 95, 100),
    "blue": (70, 115, 175),
}


def _classify_denim_from_pixels(pixels: list[tuple]) -> tuple[str, str, tuple[int, int, int] | None]:
    if not pixels:
        return "", "", None
    neutral_dark = 0
    indigo = 0
    blue_count = 0
    light_blue = 0
    blue_bias_sum = 0.0
    brt_sum = 0.0
    r_sum = g_sum = b_sum = 0.0

    for r, g, b in pixels:
        brt = (r + g + b) / 3
        spread = max(r, g, b) - min(r, g, b)
        blue_bias = b - max(r, g)
        brt_sum += brt
        blue_bias_sum += blue_bias
        r_sum += r
        g_sum += g
        b_sum += b
        if brt < 95 and spread < 34:
            neutral_dark += 1
        if brt < 135 and b >= r + 10 and b >= g - 8:
            indigo += 1
        if b >= r + 18 and b >= g + 2:
            blue_count += 1
            if brt >= 145:
                light_blue += 1

    total = len(pixels)
    neutral_ratio = neutral_dark / total
    indigo_ratio = indigo / total
    blue_ratio = blue_count / total
    light_ratio = light_blue / total
    avg_rgb = (r_sum / total, g_sum / total, b_sum / total)
    avg_rgb_int = tuple(max(0, min(255, int(round(value)))) for value in avg_rgb)
    avg_brt = brt_sum / total
    avg_blue_bias = blue_bias_sum / total

    if light_ratio >= 0.20 or blue_ratio >= 0.18:
        return "blue", "light" if light_ratio >= 0.20 or avg_brt >= 145 else "medium", avg_rgb_int
    if indigo_ratio >= 0.22 or (
        avg_rgb[2] >= avg_rgb[0] + 8
        and avg_rgb[2] >= avg_rgb[1] - 6
        and avg_brt < 135
    ):
        return "blue", "dark" if avg_brt < 118 else "medium", avg_rgb_int
    if neutral_ratio >= 0.30 and avg_brt < 78 and avg_blue_bias < 8:
        return "black", "dark", avg_rgb_int
    if neutral_ratio >= 0.24 and avg_brt < 125 and avg_blue_bias < 14:
        return "gray", "dark", avg_rgb_int

    color = min(
        _DENIM_COLOR_CENTROIDS,
        key=lambda c: sum(
            (avg_rgb[i] - _DENIM_COLOR_CENTROIDS[c][i]) ** 2 for i in range(3)
        ),
    )
    if color == "blue":
        tone = "light" if avg_brt >= 145 else ("dark" if avg_brt < 118 else "medium")
    elif color in {"black", "gray"}:
        tone = "dark"
    else:
        tone = ""
    return color, tone, avg_rgb_int


# ---------------------------------------------------------------------------
# Main extraction entry points
# ---------------------------------------------------------------------------

def extract_dominant_color_result(
    image_obj: Image.Image,
    denim_context: bool = False,
    pattern_context_text: str = "",
    product_id: str = "",
    product_name: str = "",
    segmentation_mask: Image.Image | None = None,
) -> ColorExtractionResult:
    """Analyse *image_obj* and return the dominant clothing colour.

    If *segmentation_mask* (a PIL ``"L"`` image) is provided, pixels are
    extracted from the masked region.  Otherwise a centre-crop with
    background filtering is used.
    """
    # --- 1. Extract pixels ---------------------------------------------------
    segmentation_used = False
    ignored_count = 0
    bg_count = 0

    if segmentation_mask is not None:
        pixels, ignored_count, _total_mask = _extract_mask_pixels(
            image_obj, segmentation_mask,
        )
        segmentation_used = True
        if len(pixels) < _MIN_PIXEL_COUNT:
            # Mask produced too few pixels – fall back to centre crop
            pixels, ignored_count, bg_count = _extract_center_filtered_pixels(
                image_obj,
            )
            segmentation_used = False
    else:
        pixels, ignored_count, bg_count = _extract_center_filtered_pixels(
            image_obj,
        )

    if len(pixels) < _MIN_PIXEL_COUNT:
        return ColorExtractionResult(
            color="",
            confidence="low",
            reason="not_enough_valid_pixels",
            candidate_pixel_count=len(pixels),
            ignored_pixel_count=ignored_count,
            background_pixel_count=bg_count,
            segmentation_used=segmentation_used,
        )

    # --- 2. Denim shortcut ----------------------------------------------------
    if denim_context:
        denim_color, denim_tone, denim_rgb = _classify_denim_from_pixels(pixels)
        if denim_color:
            named_color = named_group = ""
            named_rgb = None
            if denim_rgb:
                named_color, named_group, named_rgb = nearest_named_color(denim_rgb)
            search_weights = {denim_color: 1.0}
            for alias_color, alias_weight in SEARCH_COLOR_ALIASES.get(
                denim_color, {},
            ).items():
                if alias_color != denim_color:
                    search_weights[alias_color] = max(
                        search_weights.get(alias_color, 0.0), alias_weight,
                    )
            return ColorExtractionResult(
                color=denim_color,
                confidence="high",
                reason=f"denim_context_{denim_tone}" if denim_tone else "denim_context",
                dominant_ratio=1.0,
                second_ratio=0.0,
                candidates=[{
                    "color": denim_color,
                    "score": 1.0,
                    "source": "image",
                    "confidence": "high",
                    "rgb": denim_rgb,
                    "named_color": named_color,
                    "named_group": named_group,
                    "named_rgb": named_rgb,
                }],
                search_color_weights=search_weights,
                segmentation_used=segmentation_used,
                candidate_pixel_count=len(pixels),
                ignored_pixel_count=ignored_count,
                background_pixel_count=bg_count,
            )

    # --- 3. K-means + named colour classification ----------------------------
    pattern_hint = ""
    if pattern_context_text and should_run_pattern_classifier(pattern_context_text):
        pattern_hint = "pattern_detected"

    raw_candidates = kmeans_color_candidates(pixels, color_hint="")
    named_candidates = merge_named_color_candidates(raw_candidates, color_hint="")
    fashion_candidates = merge_fashion_color_candidates(named_candidates)

    if not fashion_candidates:
        return ColorExtractionResult(
            color="",
            confidence="low",
            reason="no_color_candidates",
            candidate_pixel_count=len(pixels),
            ignored_pixel_count=ignored_count,
            background_pixel_count=bg_count,
            segmentation_used=segmentation_used,
        )

    # --- 4. Confidence --------------------------------------------------------
    color_confidence, color_reason = infer_color_confidence(
        fashion_candidates, color_hint="", pattern_hint=pattern_hint,
    )

    # --- 5. Search colour weights ---------------------------------------------
    search_candidates = search_color_candidates(
        fashion_candidates, color_confidence, color_hint="",
    )
    search_weights: dict[str, float] = {}
    result_candidates: list[dict[str, Any]] = []
    for sc in search_candidates[:3]:
        search_weights[sc["color"]] = max(
            search_weights.get(sc["color"], 0.0), sc["score"],
        )
        result_candidates.append(sc)

    # --- 6. Build result ------------------------------------------------------
    top = fashion_candidates[0]
    second = fashion_candidates[1] if len(fashion_candidates) > 1 else None
    secondary_colors = [
        c["color"]
        for c in fashion_candidates[1:]
        if c["color"] != top["color"]
    ]
    # Remove duplicates while keeping order
    seen: set[str] = set()
    unique_secondary: list[str] = []
    for sc in secondary_colors:
        if sc not in seen:
            seen.add(sc)
            unique_secondary.append(sc)

    return ColorExtractionResult(
        color=top["color"],
        confidence=color_confidence,
        reason=color_reason,
        dominant_ratio=top["ratio"],
        second_ratio=second["ratio"] if second else 0.0,
        candidates=result_candidates,
        secondary_colors=unique_secondary,
        is_mixed_color=bool(
            unique_secondary and second and second["ratio"] >= 0.18,
        ),
        pattern=pattern_hint,
        search_color_weights=search_weights,
        segmentation_used=segmentation_used,
        candidate_pixel_count=len(pixels),
        ignored_pixel_count=ignored_count,
        background_pixel_count=bg_count,
    )


def extract_dominant_color(
    image_obj: Image.Image,
    denim_context: bool = False,
    pattern_context_text: str = "",
    product_id: str = "",
    product_name: str = "",
) -> str:
    result = extract_dominant_color_result(
        image_obj,
        denim_context=denim_context,
        pattern_context_text=pattern_context_text,
        product_id=product_id,
        product_name=product_name,
    )
    return result.color
