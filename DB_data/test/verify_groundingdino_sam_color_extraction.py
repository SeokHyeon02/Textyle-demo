import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from math import sqrt

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_IMAGE_DIR = os.path.join(ROOT_DIR, "DB_data", "image_jpg_700")
DEFAULT_OUTPUT_CSV = os.path.join(ROOT_DIR, "DB_data", "test", "groundingdino_sam_color_report.csv")
DEFAULT_OUTPUT_SHEET = os.path.join(ROOT_DIR, "DB_data", "test", "groundingdino_sam_debug_sheet.jpg")
DEFAULT_DINO_MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
DEFAULT_SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
DEFAULT_SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "")
DEFAULT_PROMPT = (
    "shirt. t-shirt. blouse. sweater. sweatshirt. hoodie. jacket. coat. "
    "pants. jeans. shorts. skirt. dress. clothing."
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEBUG_CELL_SIZE = (160, 150)
DEBUG_LABEL_HEIGHT = 80
MAX_KMEANS_PIXELS = 12000


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
    "white",
    "black",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "gray",
    "orange",
    "brown",
    "pink",
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
    "beige": "brown",
    "brown": "brown",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
}


COLOR_KEYWORDS = {
    "black": {"black", "blk", "\ube14\ub799", "\uac80\uc815", "\uac80\uc815\uc0c9", "\uae4c\ub9cc", "\ud751\uccad"},
    "white": {"white", "wht", "ivory", "\ud654\uc774\ud2b8", "\ud770\uc0c9", "\uc544\uc774\ubcf4\ub9ac"},
    "gray": {"gray", "grey", "charcoal", "\uadf8\ub808\uc774", "\ud68c\uc0c9", "\ucc28\ucf5c"},
    "navy": {"navy", "nvy", "\ub124\uc774\ube44", "\ube48\ud2f0\uc9c0\ub124\uc774\ube44", "\ub0a8\uc0c9"},
    "blue": {"blue", "sax", "sky blue", "\ube14\ub8e8", "\ud30c\ub791", "\uc2a4\uce74\uc774\ube14\ub8e8", "\uc911\uccad", "\uc5f0\uccad"},
    "indigo": {"indigo", "raw denim", "dark denim", "dark blue", "\uc778\ub514\uace0", "\uc0dd\uc9c0", "\uc9c4\uccad", "\ub2e4\ud06c\ube14\ub8e8"},
    "green": {"green", "mint", "teal", "\uadf8\ub9b0", "\ucd08\ub85d", "\uc62c\ub9ac\ube0c\uadf8\ub9b0", "\ubbfc\ud2b8"},
    "khaki": {"khaki", "olive", "sage", "\uce74\ud0a4", "\uc62c\ub9ac\ube0c", "\uc138\uc774\uc9c0"},
    "beige": {"beige", "cream", "sand", "oatmeal", "mud beige", "mud.beige", "\ubca0\uc774\uc9c0", "\ud06c\ub9bc", "\uc0cc\ub4dc"},
    "brown": {"brown", "camel", "mocha", "mud", "\ube0c\ub77c\uc6b4", "\uac08\uc0c9", "\uce74\uba5c"},
    "red": {"red", "burgundy", "wine", "\ub808\ub4dc", "\ubc84\uac74\ub514", "\uc640\uc778"},
    "pink": {"pink", "\ud551\ud06c", "\ubd84\ud64d"},
    "purple": {"purple", "violet", "lavender", "\ud37c\ud50c", "\ubcf4\ub77c"},
    "yellow": {"yellow", "mustard", "\uc610\ub85c\uc6b0", "\ub178\ub791"},
    "orange": {"orange", "\uc624\ub80c\uc9c0", "\uc8fc\ud669"},
}

COLOR_PRIORITY = (
    "black",
    "white",
    "gray",
    "navy",
    "indigo",
    "blue",
    "green",
    "khaki",
    "beige",
    "brown",
    "red",
    "pink",
    "purple",
    "yellow",
    "orange",
)

FINAL_COLOR_PRIORITY = (
    "white",
    "black",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "gray",
    "orange",
    "brown",
    "pink",
)

BLOCKED_COLOR_HINT_TERMS = {
    "blue": {"\ube14\ub8e8\uc885"},
}

COLOR_FAMILIES = (
    {"black", "gray"},
    {"blue", "gray"},
    {"brown", "green", "yellow", "orange"},
    {"white", "gray", "brown"},
    {"red", "pink", "purple", "brown"},
)

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

NAMED_COLOR_GROUPS = {
    "white": {
        "white", "snow", "honeydew", "mintcream", "azure", "aliceblue", "ghostwhite",
        "whitesmoke", "seashell", "beige", "oldlace", "floralwhite", "ivory",
        "antiquewhite", "linen", "lavenderblush", "mistyrose",
    },
    "gray": {
        "gray", "gainsboro", "lightgray", "silver", "darkgray", "dimgray",
        "lightslategray", "slategray", "darkslategray", "black",
    },
    "red": {
        "red", "lightsalmon", "salmon", "darksalmon", "lightcoral", "indianred",
        "crimson", "firebrick", "darkred",
    },
    "pink": {
        "pink", "lightpink", "hotpink", "deeppink", "palevioletred", "mediumvioletred",
    },
    "orange": {
        "orange", "darkorange", "coral", "tomato", "orangered",
    },
    "yellow": {
        "yellow", "lightyellow", "lemonchiffon", "lightgoldenrodyellow",
        "papayawhip", "moccasin", "peachpuff", "palegoldenrod", "khaki",
        "darkkhaki", "gold",
    },
    "brown": {
        "brown", "cornsilk", "blanchedalmond", "bisque", "navajowhite", "wheat",
        "burlywood", "tan", "rosybrown", "sandybrown", "goldenrod",
        "darkgoldenrod", "peru", "chocolate", "saddlebrown", "sienna", "maroon",
    },
    "green": {
        "green", "palegreen", "lightgreen", "yellowgreen", "greenyellow",
        "chartreuse", "lawngreen", "lime", "limegreen", "mediumspringgreen",
        "springgreen", "mediumaquamarine", "aquamarine", "lightseagreen",
        "mediumseagreen", "seagreen", "darkseagreen", "forestgreen", "darkgreen",
        "olivedrab", "olive", "darkolivegreen", "teal",
    },
    "blue": {
        "blue", "lightblue", "powderblue", "paleturquoise", "turquoise",
        "mediumturquoise", "darkturquoise", "lightcyan", "cyan", "aqua",
        "darkcyan", "cadetblue", "lightsteelblue", "steelblue", "lightskyblue",
        "skyblue", "deepskyblue", "dodgerblue", "cornflowerblue", "royalblue",
        "mediumblue", "darkblue", "navy", "midnightblue",
    },
    "purple": {
        "purple", "lavender", "thistle", "plum", "violet", "orchid", "fuchsia",
        "magenta", "mediumorchid", "mediumpurple", "amethyst", "blueviolet",
        "darkviolet", "darkorchid", "darkmagenta", "slateblue", "darkslateblue",
        "mediumslateblue", "indigo",
    },
}

NAMED_GROUP_TO_FASHION_COLOR = {
    "white": "white",
    "gray": "gray",
    "red": "red",
    "pink": "pink",
    "orange": "orange",
    "yellow": "yellow",
    "brown": "brown",
    "green": "green",
    "blue": "blue",
    "purple": "purple",
}

NAMED_COLOR_TO_FASHION_COLOR = {
    "black": "black",
    "dimgray": "gray",
    "darkslategray": "gray",
    "navy": "blue",
    "midnightblue": "blue",
    "darkblue": "blue",
    "mediumblue": "blue",
    "blue": "blue",
    "royalblue": "blue",
    "dodgerblue": "blue",
    "cornflowerblue": "blue",
    "indigo": "blue",
    "khaki": "green",
    "darkkhaki": "green",
    "olive": "green",
    "olivedrab": "green",
    "darkolivegreen": "green",
    "tan": "brown",
    "wheat": "brown",
    "burlywood": "brown",
    "bisque": "brown",
    "navajowhite": "brown",
    "blanchedalmond": "brown",
    "antiquewhite": "brown",
    "linen": "brown",
    "saddlebrown": "brown",
    "sienna": "brown",
    "chocolate": "brown",
    "peru": "brown",
    "maroon": "brown",
}

AMBIGUOUS_NEUTRAL_NAMED_COLORS = {
    "darkslategray",
    "darkslategrey",
    "dimgray",
    "dimgrey",
    "slategray",
    "slategrey",
    "lightslategray",
    "lightslategrey",
    "gray",
    "grey",
    "darkgray",
    "darkgrey",
}

PATTERN_KEYWORDS = {
    "check": {"check", "checked", "tartan", "plaid", "\uccb4\ud06c", "\ud0c0\ud0c4"},
    "stripe": {"stripe", "striped", "\uc2a4\ud2b8\ub77c\uc774\ud504", "\uc2a4\ud2b8\ub77c\uc774\ud37c"},
    "camo": {"camo", "camouflage", "\uce74\ubaa8", "\uce74\ubb34\ud50c\ub77c\uc8fc"},
    "herringbone": {"herringbone", "\ud5e4\ub9c1\ubcf8"},
    "checkerboard": {"checkerboard", "\uccb4\ucee4\ubcf4\ub4dc"},
    "graphic": {"graphic", "logo", "\uadf8\ub798\ud53d", "\ub85c\uace0"},
}


@dataclass
class VerificationResult:
    filename: str
    image_path: str
    status: str = "not_run"
    label: str = ""
    detection_score: float = 0.0
    sam_score: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    mask_ratio: float = 0.0
    mask_bbox_ratio: float = 0.0
    mask_pixel_count: int = 0
    product_name: str = ""
    color_hint: str = ""
    pattern_hint: str = ""
    should_run_pattern_vit: bool = False
    pre_hint_color: str = ""
    hint_applied: bool = False
    extracted_color: str = ""
    extracted_named_color: str = ""
    color_confidence: str = ""
    color_reason: str = ""
    nearest_named_color: str = ""
    nearest_named_group: str = ""
    dominant_named_rgb: tuple[int, int, int] | None = None
    dominant_rgb: tuple[int, int, int] | None = None
    dominant_ratio: float = 0.0
    second_color: str = ""
    second_ratio: float = 0.0
    sam_candidates_json: str = ""
    named_candidates_json: str = ""
    candidates_json: str = ""
    search_colors_json: str = ""
    error: str = ""


def list_images(image_dir):
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]


def choose_sample(image_paths, limit, seed, filenames):
    if filenames:
        wanted = set(filenames)
        return [path for path in image_paths if os.path.basename(path) in wanted]
    if limit >= len(image_paths):
        return image_paths
    return random.Random(seed).sample(image_paths, limit)


def normalize_product_id(value):
    text = os.path.splitext(os.path.basename(str(value or "").strip()))[0]
    return text


def compact_text(value):
    text = str(value or "").lower()
    compact = []
    for char in text:
        if char.isascii() and char.isalnum():
            compact.append(char)
        elif "\uac00" <= char <= "\ud7a3":
            compact.append(char)
    return "".join(compact)


def keyword_matches(keyword, compact_value):
    return compact_text(keyword) in compact_value


def blocked_color_hint(color, compact_value):
    return any(compact_text(term) in compact_value for term in BLOCKED_COLOR_HINT_TERMS.get(color, ()))


def infer_color_hint(text):
    compact_value = compact_text(text)
    if not compact_value:
        return ""
    for color in COLOR_PRIORITY:
        if blocked_color_hint(color, compact_value):
            continue
        if any(keyword_matches(keyword, compact_value) for keyword in COLOR_KEYWORDS.get(color, ())):
            return FASHION_TO_FINAL_COLOR.get(color, color)
    return ""


def infer_pattern_hint(text):
    compact_value = compact_text(text)
    if not compact_value:
        return ""
    matches = []
    for pattern, keywords in PATTERN_KEYWORDS.items():
        if any(keyword_matches(keyword, compact_value) for keyword in keywords):
            matches.append(pattern)
    return ",".join(matches)


def load_product_names(metadata_csv, id_column, name_column):
    if not metadata_csv:
        return {}
    names = {}
    with open(metadata_csv, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            product_id = normalize_product_id(row.get(id_column, ""))
            product_name = str(row.get(name_column, "")).strip()
            if product_id and product_name:
                names[product_id] = product_name
    return names


def load_env_file(env_path):
    if not env_path or not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_product_names_from_supabase(product_ids, args):
    if not args.use_supabase_product_names:
        return {}

    load_env_file(args.supabase_env)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing. Check --supabase-env.")

    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("Missing package: install supabase first.") from exc

    ids = [product_id for product_id in dict.fromkeys(product_ids) if product_id]
    if not ids:
        return {}

    client = create_client(supabase_url, supabase_key)
    response = (
        client.table(args.supabase_table)
        .select(f"{args.id_column},{args.name_column}")
        .in_(args.id_column, ids)
        .execute()
    )

    names = {}
    for row in response.data or []:
        product_id = normalize_product_id(row.get(args.id_column, ""))
        product_name = str(row.get(args.name_column, "")).strip()
        if product_id and product_name:
            names[product_id] = product_name
    return names


def squared_distance(a, b):
    return sum((int(a[index]) - int(b[index])) ** 2 for index in range(3))


def rgb_to_xyz_component(value):
    value = value / 255.0
    if value > 0.04045:
        return ((value + 0.055) / 1.055) ** 2.4
    return value / 12.92


def xyz_to_lab_component(value):
    if value > 0.008856:
        return value ** (1 / 3)
    return (7.787 * value) + (16 / 116)


def rgb_to_lab(rgb):
    r, g, b = [rgb_to_xyz_component(float(value)) for value in rgb]
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    x = xyz_to_lab_component(x)
    y = xyz_to_lab_component(y)
    z = xyz_to_lab_component(z)
    return (116 * y) - 16, 500 * (x - y), 200 * (y - z)


def lab_distance(left_rgb, right_rgb):
    left = rgb_to_lab(left_rgb)
    right = rgb_to_lab(right_rgb)
    return sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def named_color_group(name):
    normalized_name = name.replace("grey", "gray")
    for group, names in NAMED_COLOR_GROUPS.items():
        if normalized_name in names:
            return group
    return ""


def named_color_palette():
    palette = []
    for name, value in ImageColor.colormap.items():
        try:
            rgb = ImageColor.getrgb(value)
        except ValueError:
            continue
        palette.append((name, named_color_group(name), rgb))
    return palette


def nearest_named_color(rgb):
    name, group, named_rgb = min(
        named_color_palette(),
        key=lambda item: lab_distance(rgb, item[2]),
    )
    return name, group, named_rgb


def brightness(rgb):
    return sum(int(value) for value in rgb) / 3


def rgb_stddev(rgb):
    mean = brightness(rgb)
    return sqrt(sum((int(value) - mean) ** 2 for value in rgb) / 3)


def classify_achromatic_rgb(rgb, color_hint=""):
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


def is_light_blue_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        135 <= brightness(rgb) <= 225
        and b >= g + 8
        and g >= r + 5
        and b >= r + 18
    )


def is_denim_blue_gray_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        70 <= brightness(rgb) <= 170
        and b >= r + 16
        and g >= r + 5
    )


def is_washed_denim_gray_blue_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    spread = max(r, g, b) - min(r, g, b)
    return (
        105 <= brightness(rgb) <= 180
        and 8 <= spread <= 42
        and b >= r + 9
        and g >= r + 4
        and b >= g + 2
    )


def is_sage_khaki_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        95 <= brightness(rgb) <= 180
        and abs(r - g) <= 22
        and r >= b + 14
        and g >= b + 14
    )


def is_muted_green_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        55 <= brightness(rgb) <= 155
        and g >= r + 10
        and g >= b + 6
        and b >= r - 2
    )


def is_warm_neutral_beige_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        115 <= brightness(rgb) <= 205
        and abs(r - g) <= 18
        and r >= b + 5
        and g >= b + 3
    )


def is_dark_navy_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        20 <= brightness(rgb) <= 90
        and b >= r + 12
        and b >= g + 6
    )


def is_dark_brown_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        28 <= brightness(rgb) <= 95
        and r >= b + 8
        and g >= b + 4
        and abs(r - g) <= 18
    )


def is_blue_leaning_neutral_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        30 <= brightness(rgb) <= 170
        and b >= r + 8
        and b >= g + 3
    )


def is_warm_leaning_neutral_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        35 <= brightness(rgb) <= 170
        and r >= b + 5
        and g >= b
    )


def is_olive_leaning_neutral_rgb(rgb):
    r, g, b = [int(value) for value in rgb]
    return (
        35 <= brightness(rgb) <= 170
        and g >= b + 4
        and r >= b + 3
        and abs(r - g) <= 28
    )


def hinted_chromatic_neutral_color(rgb, color_hint):
    if color_hint == "blue" and (
        is_blue_leaning_neutral_rgb(rgb)
        or is_dark_navy_rgb(rgb)
        or is_denim_blue_gray_rgb(rgb)
        or is_washed_denim_gray_blue_rgb(rgb)
    ):
        return "blue", "hint_blue_neutral"
    if color_hint == "green" and (
        is_olive_leaning_neutral_rgb(rgb)
        or is_muted_green_rgb(rgb)
        or is_sage_khaki_rgb(rgb)
    ):
        return "green", "hint_green_neutral"
    if color_hint == "brown" and (
        is_warm_leaning_neutral_rgb(rgb)
        or is_dark_brown_rgb(rgb)
        or is_warm_neutral_beige_rgb(rgb)
    ):
        return "brown", "hint_brown_neutral"
    return "", ""


def reinterpret_fashion_color(named_color, rgb, default_color, color_hint=""):
    normalized_name = named_color.replace("grey", "gray")
    default_color = FASHION_TO_FINAL_COLOR.get(default_color, default_color)
    color_hint = FASHION_TO_FINAL_COLOR.get(color_hint, color_hint)
    if normalized_name not in AMBIGUOUS_NEUTRAL_NAMED_COLORS:
        return default_color, "named_color"

    hinted_color, hinted_reason = hinted_chromatic_neutral_color(rgb, color_hint)
    if hinted_color:
        return hinted_color, hinted_reason

    if color_hint in {"black", "gray", "white"}:
        achromatic_color, achromatic_reason = classify_achromatic_rgb(rgb, color_hint)
        if achromatic_color:
            return achromatic_color, achromatic_reason
    if color_hint == "black" and brightness(rgb) <= 95:
        return "black", "hint_black_dark_neutral"
    if color_hint == "gray" and brightness(rgb) <= 120:
        return "gray", "hint_gray_dark_neutral"
    if color_hint == "gray":
        return "gray", "hint_gray_neutral"

    achromatic_color, achromatic_reason = classify_achromatic_rgb(rgb)
    if achromatic_color:
        return achromatic_color, achromatic_reason
    if is_dark_navy_rgb(rgb):
        return "blue", "rgb_dark_navy_neutral"
    if is_denim_blue_gray_rgb(rgb):
        return "blue", "rgb_denim_neutral"
    if is_dark_brown_rgb(rgb):
        return "brown", "rgb_dark_brown_neutral"
    if is_olive_leaning_neutral_rgb(rgb):
        return "green", "rgb_olive_neutral"
    return default_color, "named_color"


def nearest_color(rgb):
    if is_light_blue_rgb(rgb):
        return "blue"
    if is_washed_denim_gray_blue_rgb(rgb):
        return "blue"
    if is_denim_blue_gray_rgb(rgb):
        return "blue"
    if is_dark_navy_rgb(rgb):
        return "blue"
    if is_muted_green_rgb(rgb):
        return "green"
    if is_sage_khaki_rgb(rgb):
        return "green"
    if is_dark_brown_rgb(rgb):
        return "brown"
    if is_warm_neutral_beige_rgb(rgb):
        return "brown"
    name, group, _named_rgb = nearest_named_color(rgb)
    normalized_name = name.replace("grey", "gray")
    default_color = NAMED_COLOR_TO_FASHION_COLOR.get(
        normalized_name,
        NAMED_GROUP_TO_FASHION_COLOR.get(group, group),
    )
    fashion_color, _reason = reinterpret_fashion_color(name, rgb, default_color)
    return fashion_color


def classify_named_and_fashion_color(rgb, color_hint=""):
    name, group, named_rgb = nearest_named_color(rgb)
    normalized_name = name.replace("grey", "gray")
    default_color = NAMED_COLOR_TO_FASHION_COLOR.get(
        normalized_name,
        NAMED_GROUP_TO_FASHION_COLOR.get(group, group),
    )
    fashion_color, fashion_reason = reinterpret_fashion_color(name, rgb, default_color, color_hint)
    return name, group, named_rgb, fashion_color, fashion_reason


def merge_named_color_candidates(candidates, color_hint=""):
    merged = {}
    for candidate in candidates:
        color = candidate["named_color"]
        ratio = candidate["ratio"]
        rgb = candidate["rgb"]
        if color not in merged:
            merged[color] = {
                "named_color": color,
                "named_group": candidate.get("named_group", ""),
                "named_rgb": candidate.get("named_rgb"),
                "fashion_color": candidate.get("fashion_color", ""),
                "color": candidate.get("fashion_color", ""),
                "ratio": 0.0,
                "rgb_sum": [0.0, 0.0, 0.0],
            }
        merged[color]["ratio"] += ratio
        for index in range(3):
            merged[color]["rgb_sum"][index] += rgb[index] * ratio

    result = []
    for item in merged.values():
        ratio = item["ratio"]
        rgb = tuple(int(round(value / ratio)) for value in item["rgb_sum"]) if ratio else (0, 0, 0)
        named_color, named_group, named_rgb, fashion_color, fashion_reason = classify_named_and_fashion_color(
            rgb,
            color_hint,
        )
        result.append(
            {
                "color": fashion_color,
                "fashion_color": fashion_color,
                "fashion_reason": fashion_reason,
                "named_color": named_color,
                "named_group": named_group,
                "named_rgb": named_rgb,
                "ratio": ratio,
                "rgb": rgb,
            }
        )
    result.sort(key=lambda candidate: candidate["ratio"], reverse=True)
    return result


def merge_fashion_color_candidates(named_candidates):
    merged = {}
    for candidate in named_candidates:
        color = candidate["fashion_color"]
        ratio = candidate["ratio"]
        rgb = candidate["rgb"]
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
        merged[color]["named_colors"].append(
            {
                "named_color": candidate["named_color"],
                "ratio": ratio,
                "rgb": candidate["rgb"],
            }
        )
        reason = candidate.get("fashion_reason", "named_color")
        merged[color]["fashion_reasons"][reason] = merged[color]["fashion_reasons"].get(reason, 0.0) + ratio
        for index in range(3):
            merged[color]["rgb_sum"][index] += rgb[index] * ratio

    result = []
    for item in merged.values():
        ratio = item["ratio"]
        rgb = tuple(int(round(value / ratio)) for value in item["rgb_sum"]) if ratio else (0, 0, 0)
        result.append(
            {
                "color": item["color"],
                "fashion_color": item["fashion_color"],
                "ratio": ratio,
                "rgb": rgb,
                "named_colors": sorted(
                    item["named_colors"],
                    key=lambda named: named["ratio"],
                    reverse=True,
                ),
                "fashion_reasons": item["fashion_reasons"],
            }
        )
    result.sort(key=lambda candidate: candidate["ratio"], reverse=True)
    return result


def colors_are_related(left, right):
    if not left or not right:
        return False
    if left == right:
        return True
    return any(left in family and right in family for family in COLOR_FAMILIES)


def infer_color_confidence(candidates, color_hint="", pattern_hint=""):
    if not candidates:
        return "low", "no_color_candidates"

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    top_ratio = top["ratio"]
    second_ratio = second["ratio"] if second else 0.0
    margin = top_ratio - second_ratio
    hint_disagrees = bool(color_hint and not colors_are_related(top["color"], color_hint))

    if pattern_hint:
        if top_ratio >= 0.75 and second_ratio < 0.18 and not hint_disagrees:
            return "medium", "pattern_hint_needs_vit"
        return "low", "pattern_hint_needs_vit"

    if hint_disagrees:
        return "low", "color_hint_disagrees"
    if second_ratio >= 0.30 or margin < 0.25:
        return "medium" if top_ratio >= 0.55 else "low", "ambiguous_second_color"
    if top_ratio >= 0.75:
        return "high", "dominant_image_color"
    if top_ratio >= 0.55:
        return "medium", "moderate_image_color"
    return "low", "weak_image_color"


def candidate_confidence(ratio):
    if ratio >= 0.65:
        return "high"
    if ratio >= 0.25:
        return "medium"
    return "low"


def add_search_color(search_colors, color, score, source, confidence, base_color=None):
    if not color or score <= 0:
        return
    existing = search_colors.get(color)
    item = {
        "color": color,
        "score": score,
        "source": source,
        "confidence": confidence,
    }
    if base_color and base_color != color:
        item["base_color"] = base_color
    if existing is None or score > existing["score"]:
        search_colors[color] = item


def search_color_candidates(candidates, color_confidence, color_hint="", min_ratio=0.15):
    search_colors = {}
    if color_hint in FINAL_COLOR_CATEGORIES:
        add_search_color(
            search_colors,
            color_hint,
            1.0,
            "product_name",
            "high",
        )
    for index, candidate in enumerate(candidates):
        if index > 0 and candidate["ratio"] < min_ratio:
            continue
        base_color = candidate["color"]
        base_score = candidate["ratio"]
        confidence = color_confidence if index == 0 else candidate_confidence(base_score)
        for color, weight in SEARCH_COLOR_ALIASES.get(base_color, {base_color: 1.0}).items():
            source = "image" if color == base_color else "family"
            family_confidence = confidence if color == base_color else "low"
            add_search_color(
                search_colors,
                color,
                base_score * weight,
                source,
                family_confidence,
                base_color=base_color,
            )
    return sorted(search_colors.values(), key=lambda item: item["score"], reverse=True)


def kmeans_color_candidates(pixels, n_clusters=5, color_hint=""):
    if not pixels:
        return []
    if len(pixels) > MAX_KMEANS_PIXELS:
        step = max(1, len(pixels) // MAX_KMEANS_PIXELS)
        pixels = pixels[::step]

    buckets = {}
    for r, g, b in pixels:
        key = (round(r / 32) * 32, round(g / 32) * 32, round(b / 32) * 32)
        buckets.setdefault(key, [0, 0, 0, 0])
        buckets[key][0] += 1
        buckets[key][1] += r
        buckets[key][2] += g
        buckets[key][3] += b

    centers = []
    for _key, (count, r_sum, g_sum, b_sum) in sorted(buckets.items(), key=lambda row: row[1][0], reverse=True):
        centers.append((r_sum / count, g_sum / count, b_sum / count))
        if len(centers) >= min(n_clusters, len(pixels)):
            break

    labels = [0] * len(pixels)
    for _ in range(8):
        changed = False
        sums = [[0, 0, 0, 0] for _center in centers]
        for index, pixel in enumerate(pixels):
            label = min(range(len(centers)), key=lambda center_index: squared_distance(pixel, centers[center_index]))
            if labels[index] != label:
                changed = True
            labels[index] = label
            sums[label][0] += 1
            sums[label][1] += pixel[0]
            sums[label][2] += pixel[1]
            sums[label][3] += pixel[2]
        for index, (count, r_sum, g_sum, b_sum) in enumerate(sums):
            if count:
                centers[index] = (r_sum / count, g_sum / count, b_sum / count)
        if not changed:
            break

    counts = [0] * len(centers)
    sums = [[0, 0, 0] for _center in centers]
    for pixel, label in zip(pixels, labels):
        counts[label] += 1
        sums[label][0] += pixel[0]
        sums[label][1] += pixel[1]
        sums[label][2] += pixel[2]

    total = sum(counts)
    candidates = []
    for count, rgb_sum in zip(counts, sums):
        if not count:
            continue
        rgb = tuple(int(round(value / count)) for value in rgb_sum)
        named_color, named_group, named_rgb, fashion_color, fashion_reason = classify_named_and_fashion_color(
            rgb,
            color_hint,
        )
        candidates.append(
            {
                "rgb": rgb,
                "named_color": named_color,
                "named_group": named_group,
                "named_rgb": named_rgb,
                "fashion_color": fashion_color,
                "fashion_reason": fashion_reason,
                "color": fashion_color,
                "ratio": count / total,
            }
        )
    candidates.sort(key=lambda candidate: candidate["ratio"], reverse=True)
    return candidates


def mask_bbox(mask):
    height = len(mask)
    width = len(mask[0]) if height else 0
    xs = []
    ys = []
    for y in range(height):
        for x in range(width):
            if mask[y][x]:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def collect_mask_pixels(image, mask):
    rgb = image.convert("RGB")
    pixels = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                pixels.append(rgb.getpixel((x, y)))
    return pixels


def load_models(dino_model_id, sam_checkpoint, sam_model_type, device):
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Missing packages. Install torch, transformers, and segment-anything first.") from exc

    if not sam_checkpoint:
        raise RuntimeError("Set SAM_CHECKPOINT or pass --sam-checkpoint.")
    if not os.path.exists(sam_checkpoint):
        raise RuntimeError(f"SAM checkpoint does not exist: {sam_checkpoint}")

    processor = AutoProcessor.from_pretrained(dino_model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)
    dino_model.eval()

    if sam_model_type not in sam_model_registry:
        raise RuntimeError(f"Unsupported SAM model type: {sam_model_type}")
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam_model)
    return torch, processor, dino_model, sam_predictor


def post_process_grounding_dino(processor, outputs, inputs, image, box_threshold, text_threshold):
    target_sizes = [image.size[::-1]]
    call_variants = (
        {
            "args": (outputs, inputs.input_ids),
            "kwargs": {
                "threshold": box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs,),
            "kwargs": {
                "threshold": box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs, inputs.input_ids),
            "kwargs": {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs,),
            "kwargs": {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            },
        },
    )
    last_error = None
    for variant in call_variants:
        try:
            return processor.post_process_grounded_object_detection(
                *variant["args"],
                **variant["kwargs"],
            )[0]
        except TypeError as exc:
            last_error = exc
    raise last_error


def detect_best_box(torch, processor, model, image, prompt, device, box_threshold, text_threshold):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = post_process_grounding_dino(processor, outputs, inputs, image, box_threshold, text_threshold)
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])
    if len(boxes) == 0:
        return None, "", 0.0

    width, height = image.size
    best_index = 0
    best_rank = -1.0
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(value) for value in box.tolist()]
        area_ratio = max(0.0, (x1 - x0) * (y1 - y0)) / max(1, width * height)
        score = float(scores[index])
        rank = score + min(area_ratio, 0.8)
        if rank > best_rank:
            best_index = index
            best_rank = rank

    box = tuple(int(round(value)) for value in boxes[best_index].tolist())
    label = str(labels[best_index]) if len(labels) > best_index else ""
    score = float(scores[best_index])
    return box, label, score


def mask_color_candidates(image, mask, color_hint=""):
    pixels = collect_mask_pixels(image, mask)
    named_candidates = merge_named_color_candidates(
        kmeans_color_candidates(pixels, color_hint=color_hint),
        color_hint=color_hint,
    )
    return merge_fashion_color_candidates(named_candidates), len(pixels)


def mask_bbox_area_ratio(mask, image_size):
    bbox_from_mask = mask_bbox(mask)
    if not bbox_from_mask:
        return 0.0
    width, height = image_size
    x0, y0, x1, y1 = bbox_from_mask
    return ((x1 - x0) * (y1 - y0)) / max(1, width * height)


def rank_sam_mask(image, mask, sam_score, color_hint="", return_details=False):
    candidates, pixel_count = mask_color_candidates(image, mask, color_hint)
    width, height = image.size
    mask_ratio = pixel_count / max(1, width * height)
    mask_bbox_ratio = mask_bbox_area_ratio(mask, image.size)
    rank = float(sam_score)
    adjustments = []

    if mask_ratio < 0.005:
        rank -= 0.25
        adjustments.append("tiny_mask")
    elif mask_ratio < 0.25:
        rank -= 0.25
        adjustments.append("small_mask")
    elif mask_ratio < 0.30:
        rank -= 0.12
        adjustments.append("somewhat_small_mask")
    elif 0.35 <= mask_ratio <= 0.85:
        rank += 0.08
        adjustments.append("normal_mask_bonus")
    if mask_bbox_ratio > 0.95:
        rank -= 0.10
        adjustments.append("huge_bbox")

    if not candidates:
        rank -= 0.50
        adjustments.append("no_color_candidates")
        if return_details:
            return rank, {
                "rank": rank,
                "sam_score": float(sam_score),
                "mask_ratio": mask_ratio,
                "mask_bbox_ratio": mask_bbox_ratio,
                "pixel_count": pixel_count,
                "top_color": "",
                "top_ratio": 0.0,
                "top_rgb": None,
                "adjustments": adjustments,
            }
        return rank

    top = candidates[0]
    top_color = top["color"]
    top_ratio = top["ratio"]
    top_rgb = top["rgb"]

    pure_white_background = (
        top_color == "white"
        and top_ratio > 0.92
        and brightness(top_rgb) >= 248
        and mask_bbox_ratio > 0.85
    )

    if color_hint:
        if top_color == color_hint:
            rank += 0.55 * min(1.0, top_ratio)
            adjustments.append("hint_match")
        elif colors_are_related(top_color, color_hint):
            rank += 0.20 * min(1.0, top_ratio)
            adjustments.append("hint_related")
        else:
            rank -= 0.15
            adjustments.append("hint_mismatch")

        if top_color != color_hint and top_ratio > 0.90 and mask_ratio < 0.30:
            rank -= 0.40
            adjustments.append("small_single_color_hint_mismatch")
        if color_hint != "white" and top_color == "white" and top_ratio > 0.75:
            rank -= 0.85
            adjustments.append("white_against_nonwhite_hint")
        if color_hint != "white" and mask_bbox_ratio > 0.85 and top_color == "white" and top_ratio > 0.60:
            rank -= 0.35
            adjustments.append("large_white_against_nonwhite_hint")
        if color_hint == "white" and pure_white_background:
            rank -= 0.75
            adjustments.append("pure_white_background")

    if return_details:
        second = candidates[1] if len(candidates) > 1 else None
        return rank, {
            "rank": rank,
            "sam_score": float(sam_score),
            "mask_ratio": mask_ratio,
            "mask_bbox_ratio": mask_bbox_ratio,
            "pixel_count": pixel_count,
            "top_color": top_color,
            "top_ratio": top_ratio,
            "top_rgb": top_rgb,
            "second_color": second["color"] if second else "",
            "second_ratio": second["ratio"] if second else 0.0,
            "adjustments": adjustments,
        }
    return rank


def segment_with_sam(sam_predictor, image, bbox, color_hint=""):
    import numpy as np

    sam_predictor.set_image(np.array(image.convert("RGB")))
    masks, scores, _logits = sam_predictor.predict(box=np.array(bbox), multimask_output=True)
    best_index = 0
    best_rank = None
    candidate_details = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        candidate_mask = mask.astype(bool).tolist()
        rank, details = rank_sam_mask(image, candidate_mask, float(score), color_hint, return_details=True)
        details["index"] = index
        candidate_details.append(details)
        if best_rank is None or rank > best_rank:
            best_index = index
            best_rank = rank
    return masks[best_index].astype(bool).tolist(), float(scores[best_index]), candidate_details


def verify_image(image_path, models, args, product_names):
    torch, processor, dino_model, sam_predictor = models
    image = Image.open(image_path).convert("RGB")
    result = VerificationResult(os.path.basename(image_path), image_path)
    product_id = normalize_product_id(image_path)
    result.product_name = product_names.get(product_id, "")
    result.color_hint = infer_color_hint(result.product_name)
    result.pattern_hint = infer_pattern_hint(result.product_name)
    result.should_run_pattern_vit = bool(result.pattern_hint)
    try:
        bbox, label, detection_score = detect_best_box(
            torch,
            processor,
            dino_model,
            image,
            args.prompt,
            args.device,
            args.box_threshold,
            args.text_threshold,
        )
        if bbox is None:
            result.status = "no_detection"
            return result, image, None

        mask, sam_score, sam_candidate_details = segment_with_sam(
            sam_predictor,
            image,
            bbox,
            result.color_hint,
        )
        result.sam_candidates_json = json.dumps(sam_candidate_details, ensure_ascii=False)
        pixels = collect_mask_pixels(image, mask)
        named_candidates = merge_named_color_candidates(
            kmeans_color_candidates(pixels, color_hint=result.color_hint),
            color_hint=result.color_hint,
        )
        candidates = merge_fashion_color_candidates(named_candidates)
        result.pre_hint_color = candidates[0]["color"] if candidates else ""
        width, height = image.size
        result.status = "ok" if candidates else "no_mask_pixels"
        result.label = label
        result.detection_score = detection_score
        result.sam_score = sam_score
        result.bbox = bbox
        result.mask_pixel_count = len(pixels)
        result.mask_ratio = len(pixels) / max(1, width * height)
        bbox_from_mask = mask_bbox(mask)
        if bbox_from_mask:
            x0, y0, x1, y1 = bbox_from_mask
            result.mask_bbox_ratio = ((x1 - x0) * (y1 - y0)) / max(1, width * height)
        if candidates:
            result.named_candidates_json = json.dumps(named_candidates, ensure_ascii=False)
            result.candidates_json = json.dumps(candidates, ensure_ascii=False)
            result.color_confidence, result.color_reason = infer_color_confidence(
                candidates,
                result.color_hint,
                result.pattern_hint,
            )
            result.search_colors_json = json.dumps(
                search_color_candidates(candidates, result.color_confidence, result.color_hint),
                ensure_ascii=False,
            )
            top = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            top_named = named_candidates[0] if named_candidates else None
            result.hint_applied = False
            if result.color_hint in FINAL_COLOR_CATEGORIES:
                result.hint_applied = result.color_hint != top["color"]
                result.extracted_color = result.color_hint
                if result.hint_applied:
                    result.color_confidence = "high"
                    result.color_reason = "product_name_priority"
            else:
                result.extracted_color = top["color"]
            result.extracted_named_color = top_named["named_color"] if top_named else ""
            result.dominant_rgb = top["rgb"]
            result.nearest_named_color = top_named["named_color"] if top_named else ""
            result.nearest_named_group = top_named["named_group"] if top_named else ""
            result.dominant_named_rgb = top_named["rgb"] if top_named else None
            result.dominant_ratio = top["ratio"]
            result.second_color = second["color"] if second else ""
            result.second_ratio = second["ratio"] if second else 0.0
        return result, image, mask
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        return result, image, None


def fit_image(image, size):
    fitted = image.copy()
    fitted.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def mask_to_l_image(mask, size):
    mask_image = Image.new("L", size, 0)
    if not mask:
        return mask_image
    pixels = mask_image.load()
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                pixels[x, y] = 180
    return mask_image


def overlay_mask(image, mask, bbox=None):
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    if mask:
        overlay = Image.new("RGB", preview.size, (40, 210, 90))
        mask_image = mask_to_l_image(mask, preview.size)
        blended = Image.blend(preview, overlay, 0.45)
        preview.paste(blended, mask=mask_image)
    if bbox:
        draw.rectangle(bbox, outline=(230, 50, 50), width=3)
    return preview


def masked_image(image, mask):
    rgb = image.convert("RGB")
    preview = Image.new("RGB", rgb.size, (218, 218, 218))
    preview.paste(rgb, mask=mask_to_l_image(mask, rgb.size))
    return preview


def swatch(rgb, size):
    return Image.new("RGB", size, rgb or (245, 245, 245))


def write_csv(results, output_csv):
    fieldnames = [
        "filename",
        "image_path",
        "status",
        "label",
        "detection_score",
        "sam_score",
        "bbox",
        "mask_ratio",
        "mask_bbox_ratio",
        "mask_pixel_count",
        "product_name",
        "color_hint",
        "pattern_hint",
        "should_run_pattern_vit",
        "pre_hint_color",
        "hint_applied",
        "extracted_color",
        "extracted_named_color",
        "color_confidence",
        "color_reason",
        "nearest_named_color",
        "nearest_named_group",
        "dominant_named_rgb",
        "dominant_rgb",
        "dominant_ratio",
        "second_color",
        "second_ratio",
        "sam_candidates_json",
        "named_candidates_json",
        "candidates_json",
        "search_colors_json",
        "error",
    ]
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.__dict__.copy()
            row["bbox"] = list(result.bbox) if result.bbox else ""
            row["dominant_named_rgb"] = list(result.dominant_named_rgb) if result.dominant_named_rgb else ""
            row["dominant_rgb"] = list(result.dominant_rgb) if result.dominant_rgb else ""
            writer.writerow(row)


def write_debug_sheet(items, output_sheet):
    if not items:
        return
    headers = ["original", "dino+sam", "masked", "swatch"]
    cols = len(headers)
    cell_w, cell_h = DEBUG_CELL_SIZE
    row_h = cell_h + DEBUG_LABEL_HEIGHT
    sheet = Image.new("RGB", (cols * cell_w, len(items) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for row_index, (result, image, mask) in enumerate(items):
        y0 = row_index * row_h
        panels = [
            fit_image(image, DEBUG_CELL_SIZE),
            fit_image(overlay_mask(image, mask, result.bbox), DEBUG_CELL_SIZE),
            fit_image(masked_image(image, mask), DEBUG_CELL_SIZE),
            swatch(result.dominant_rgb, DEBUG_CELL_SIZE),
        ]
        for col_index, panel in enumerate(panels):
            x0 = col_index * cell_w
            sheet.paste(panel, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + row_h - 1), outline=(220, 220, 220))
            draw.text((x0 + 4, y0 + 4), headers[col_index], fill="black", font=font)

        label = (
            f"{result.filename} {result.status} color={result.extracted_color or '-'} "
            f"named={result.extracted_named_color or '-'} "
            f"conf={result.color_confidence or '-'} hint={result.color_hint or '-'} "
            f"pattern={result.pattern_hint or '-'} "
            f"dino={result.detection_score:.2f} sam={result.sam_score:.2f} "
            f"mask={result.mask_ratio:.3f} bbox={result.mask_bbox_ratio:.3f}"
        )
        draw.text((4, y0 + cell_h + 4), label[:110], fill="black", font=font)
        if result.product_name:
            draw.text((4, y0 + cell_h + 22), result.product_name[:95], fill="black", font=font)
        if result.error:
            draw.text((4, y0 + cell_h + 40), result.error[:110], fill="red", font=font)

    sheet.save(output_sheet, quality=92)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify clothing color extraction with GroundingDINO + SAM")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-sheet", default=DEFAULT_OUTPUT_SHEET)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--filenames", nargs="*", default=[])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dino-model-id", default=DEFAULT_DINO_MODEL_ID)
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default=DEFAULT_SAM_MODEL_TYPE)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--use-supabase-product-names", action="store_true")
    parser.add_argument("--supabase-env", default=os.path.join(ROOT_DIR, "DB_data", ".env"))
    parser.add_argument("--supabase-table", default="clothes")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default=os.environ.get("DINO_SAM_DEVICE", "cpu"))
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = choose_sample(list_images(args.image_dir), args.limit, args.seed, args.filenames)
    product_names = load_product_names(args.metadata_csv, args.id_column, args.name_column)
    product_ids = [normalize_product_id(image_path) for image_path in image_paths]
    product_names.update(load_product_names_from_supabase(product_ids, args))
    models = load_models(args.dino_model_id, args.sam_checkpoint, args.sam_model_type, args.device)

    results = []
    debug_items = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {os.path.basename(image_path)}")
        result, image, mask = verify_image(image_path, models, args, product_names)
        results.append(result)
        debug_items.append((result, image, mask))

    write_csv(results, args.output_csv)
    write_debug_sheet(debug_items, args.output_sheet)
    print(f"Wrote {len(results)} rows to {args.output_csv}")
    print(f"Wrote debug sheet to {args.output_sheet}")


if __name__ == "__main__":
    main()
