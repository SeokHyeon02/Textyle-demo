import os
import re
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client
from dotenv import load_dotenv
from math import sqrt
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# 1. Environment and database connection
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # service_role key required

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OVERWRITE_ATTRIBUTES = os.environ.get("OVERWRITE_ATTRIBUTES", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

# -------------------------------------------------------------
# 2. Model configuration
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
TARGET_EMBEDDING_COLUMN = "fashion_embedding"
ORDER_COLUMN = os.environ.get("FASHION_EMBEDDING_ORDER_COLUMN", "image_url")
PAGE_SIZE = int(os.environ.get("ATTRIBUTE_UPDATE_PAGE_SIZE", "1000"))
SCRIPT_VERSION = "update.py fashion-attribute-update-2026-05-04"
MATERIAL_DB_COLUMN = "material"
FIT_DB_COLUMN = "fit"
DELETE_OPTION_COUNT_ROWS = os.environ.get("DELETE_OPTION_COUNT_ROWS", "false").strip().lower() in {
    "1", "true", "yes", "y"
}
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

model = None
processor = None

# -------------------------------------------------------------
# 3. Attribute labels and keyword settings
# -------------------------------------------------------------
ATTRIBUTE_LABELS = {
    "pattern": {
        "solid": "solid plain",
        "stripe": "striped",
        "vertical_stripe": "vertical striped",
        "horizontal_stripe": "horizontal striped",
        "check": "checkered",
        "plaid": "plaid",
        "houndstooth": "houndstooth",
        "dot": "polka dot",
        "floral": "floral",
        "graphic": "graphic print",
        "logo": "logo print",
        "camouflage": "camouflage",
        "paisley": "paisley pattern",
        "argyle": "argyle pattern",
        "animal_print": "animal print",
        "color_block": "color block",
        "tie_dye": "tie dye",
        "washed": "washed vintage",
        "distressed": "distressed",
    },
    "fit": {
        "skinny": "skinny",
        "slim": "slim",
        "regular": "regular",
        "relaxed": "relaxed",
        "loose": "loose",
        "oversized": "oversized",
        "wide": "wide",
        "straight": "straight",
        "tapered": "tapered",
        "cropped": "cropped",
        "bootcut": "bootcut",
        "flare": "flare",
        "bell_bottom": "bell bottom",
        "jogger_fit": "jogger",
        "boxy": "boxy",
        "semi_oversized": "semi oversized",
    },
    "material": {
        "cotton": "cotton",
        "denim": "denim",
        "wool": "wool",
        "leather": "leather",
        "faux_leather": "faux leather",
        "nylon": "nylon",
        "polyester": "polyester",
        "linen": "linen",
        "fleece": "fleece",
        "corduroy": "corduroy",
        "jersey": "jersey fabric",
        "silk": "silk",
        "satin": "satin",
        "chiffon": "chiffon",
        "tweed": "tweed",
        "suede": "suede",
        "cashmere": "cashmere",
        "spandex": "stretch spandex",
        "rayon": "rayon",
        "mesh": "mesh",
        "canvas": "canvas fabric",
    },
}

# Thresholds are kept for legacy attribute helpers.
ATTRIBUTE_THRESHOLDS = {
    "pattern": 0.18,
    "fit": 0.16,
    "material": 0.16,
}

ATTRIBUTE_MIN_MARGIN = {
    "pattern": 0.035,
    "fit": 0.025,
    "material": 0.025,
}

COLOR_REFERENCES = {
    "black": (25, 25, 25),
    "white": (235, 235, 235),
    "gray": (140, 140, 140),
    "red": (190, 45, 55),
    "orange": (220, 120, 40),
    "yellow": (220, 190, 50),
    "green": (65, 140, 70),
    "khaki": (95, 105, 65),
    "blue": (55, 95, 180),
    "navy": (35, 55, 95),
    "purple": (120, 80, 150),
    "pink": (220, 150, 170),
    "brown": (120, 85, 55),
    "beige": (205, 185, 150),
    "indigo": (45, 70, 115),
    "camouflage": (70, 75, 50),
}

DENIM_COLOR_REFERENCES = {
    "black": (32, 32, 35),
    "gray": (95, 95, 100),
    "indigo": (38, 58, 95),
    "blue": (70, 115, 175),
}

COLOR_KEYWORDS = {
    "black": ["black", "blk", "bk", "black denim", "washed black"],
    "white": ["white", "wht", "wh", "ivory", "off white", "offwhite", "cream white"],
    "gray": ["gray", "grey", "charcoal", "melange", "ash gray"],
    "red": ["red", "burgundy", "wine"],
    "orange": ["orange"],
    "yellow": ["yellow", "mustard"],
    "green": ["green", "mint"],
    "khaki": ["khaki", "olive"],
    "blue": ["blue", "sky blue", "sax", "light denim", "mid blue", "medium blue"],
    "navy": ["navy", "nvy"],
    "purple": ["purple", "violet", "lavender"],
    "pink": ["pink"],
    "brown": ["brown", "camel", "mocha", "brn"],
    "beige": ["beige", "cream", "sand", "oatmeal"],
    "indigo": ["indigo", "raw denim", "deep blue denim", "dark indigo"],
    "camouflage": ["camouflage", "camo"],
}

MATERIAL_KEYWORDS = {
    "faux_leather": ["faux leather", "fake leather", "synthetic leather", "pu leather", "vegan leather", "eco leather", "artificial leather"],
    "leather": ["leather", "real leather", "genuine leather", "goat leather", "goat skin", "goatskin", "cowhide", "cow leather", "lambskin", "lamb leather", "sheep leather", "sheepskin"],
    "denim": ["denim", "jean", "jeans", "raw denim", "black denim", "washed denim", "selvedge", "selvage"],
    "cotton": ["cotton", "cotton 100", "100 cotton"],
    "wool": ["wool", "wool blend", "merino", "knit"],
    "linen": ["linen"],
    "fleece": ["fleece", "boa fleece"],
    "corduroy": ["corduroy"],
    "suede": ["suede"],
    "nylon": ["nylon"],
    "polyester": ["polyester", "poly"],
    "cashmere": ["cashmere"],
    "silk": ["silk"],
    "tweed": ["tweed"],
}

EXTRA_COLOR_KEYWORDS = {
    "black": ["\\ube14\\ub799", "\\uac80\\uc815", "\\uac80\\uc815\\uc0c9", "\\uae4c\\ub9cc\\uc0c9", "\\ud751\\uc0c9", "\\ud751\\uccad"],
    "white": ["\\ud654\\uc774\\ud2b8", "\\ud770\\uc0c9", "\\ud558\\uc580\\uc0c9", "\\ubc31\\uc0c9", "\\uc544\\uc774\\ubcf4\\ub9ac"],
    "gray": ["\\uadf8\\ub808\\uc774", "\\ud68c\\uc0c9", "\\ucc28\\ucf5c", "\\uba5c\\ub780\\uc9c0"],
    "indigo": ["\\uc778\\ub514\\uace0", "\\uc0dd\\uc9c0", "\\uc9c4\\uccad"],
    "blue": ["\\ube14\\ub8e8", "\\ud30c\\ub791", "\\ud30c\\ub780\\uc0c9", "\\uc18c\\ub77c", "\\uc2a4\\uce74\\uc774\\ube14\\ub8e8", "\\uc911\\uccad", "\\uc5f0\\uccad"],
    "navy": ["\\ub124\\uc774\\ube44", "\\ub0a8\\uc0c9"],
    "khaki": ["\\uce74\\ud0a4", "\\uc62c\\ub9ac\\ube0c"],
    "beige": ["\\ubca0\\uc774\\uc9c0", "\\uc624\\ud2b8\\ubc00", "\\uc0cc\\ub4dc", "\\ud06c\\ub9bc"],
    "brown": ["\\ube0c\\ub77c\\uc6b4", "\\uac08\\uc0c9", "\\uce74\\uba5c", "\\ubaa8\\uce74"],
    "red": ["\\ub808\\ub4dc", "\\ube68\\uac15", "\\ube68\\uac04\\uc0c9", "\\ubc84\\uac74\\ub514", "\\uc640\\uc778"],
    "green": ["\\uadf8\\ub9b0", "\\ucd08\\ub85d", "\\ub179\\uc0c9", "\\ubbfc\\ud2b8"],
    "yellow": ["\\uc610\\ub85c\\uc6b0", "\\ub178\\ub791", "\\ub178\\ub780\\uc0c9", "\\uba38\\uc2a4\\ud0c0\\ub4dc"],
    "pink": ["\\ud551\\ud06c", "\\ubd84\\ud64d", "\\ubd84\\ud64d\\uc0c9"],
    "purple": ["\\ud37c\\ud50c", "\\ubcf4\\ub77c", "\\ubcf4\\ub77c\\uc0c9", "\\ubc14\\uc774\\uc62c\\ub81b", "\\ub77c\\ubca4\\ub354"],
    "orange": ["\\uc624\\ub80c\\uc9c0", "\\uc8fc\\ud669", "\\uc8fc\\ud669\\uc0c9"],
    "camouflage": ["\\uce74\\ubaa8", "\\uce74\\ubaa8\\ud50c\\ub77c\\uc8fc", "\\uc704\\uc7a5", "\\ubc00\\ub9ac\\ud130\\ub9ac"],
}

BASIC_COLOR_KEYWORDS = {
    "black": [
        "black", "blk", "bk", "블랙", "검정", "검정색", "검은색", "까만색", "흑색", "흑청",
    ],
    "white": [
        "white", "wht", "wh", "ivory", "off white", "offwhite",
        "화이트", "흰색", "하얀색", "백색", "아이보리", "오프 화이트", "오프화이트",
    ],
    "gray": [
        "gray", "grey", "charcoal", "charcole", "melange", "ash gray",
        "그레이", "회색", "차콜", "챠콜", "멜란지", "애쉬그레이",
    ],
    "red": [
        "red", "burgundy", "wine", "레드", "빨강", "빨간색", "버건디", "와인",
    ],
    "orange": [
        "orange", "오렌지", "주황", "주황색",
    ],
    "yellow": [
        "yellow", "mustard", "옐로우", "노랑", "노란색", "머스타드",
    ],
    "green": [
        "green", "mint", "그린", "초록", "초록색", "녹색", "민트",
    ],
    "khaki": [
        "khaki", "olive", "카키", "올리브",
    ],
    "blue": [
        "blue", "sky blue", "light blue", "sax", "블루", "파랑", "파란색",
        "소라", "하늘색", "스카이블루", "연청", "중청",
    ],
    "navy": [
        "navy", "nvy", "네이비", "남색",
    ],
    "purple": [
        "purple", "violet", "lavender", "퍼플", "보라", "보라색", "바이올렛", "라벤더",
    ],
    "pink": [
        "pink", "핑크", "분홍", "분홍색",
    ],
    "brown": [
        "brown", "camel", "mocha", "brn", "브라운", "갈색", "카멜", "모카",
    ],
    "beige": [
        "beige", "cream", "sand", "oatmeal", "베이지", "크림", "샌드", "오트밀",
    ],
    "indigo": [
        "indigo", "raw denim", "deep blue denim", "dark indigo",
        "인디고", "생지", "진청",
    ],
    "camouflage": [
        "camouflage", "camo", "카모", "카모플라주", "위장", "밀리터리",
    ],
}

EXTRA_MATERIAL_KEYWORDS = {
    "faux_leather": ["\\ube44\\uac74\\ub808\\ub354", "\\uc778\\uc870\\uac00\\uc8fd", "\\ud569\\uc131\\uac00\\uc8fd", "\\uc5d0\\ucf54\\ub808\\ub354"],
    "leather": ["\\ub808\\ub354", "\\uac00\\uc8fd", "\\uace0\\ud2b8", "\\uc591\\uac00\\uc8fd", "\\uc18c\\uac00\\uc8fd", "\\ub7a8\\uc2a4\\ud0a8", "\\uce74\\uc6b0\\ud558\\uc774\\ub4dc"],
    "denim": ["\\ub370\\ub2d8", "\\uccad\\ubc14\\uc9c0", "\\uccad\\uc790\\ucf13", "\\ud751\\uccad", "\\uc0dd\\uc9c0", "\\uc9c4\\uccad", "\\uc911\\uccad", "\\uc5f0\\uccad"],
    "cotton": ["\\ucf54\\ud2bc", "\\uba74", "\\uba74 100", "\\uba74100"],
    "wool": ["\\uc6b8", "\\uc6b8\\ube14\\ub80c\\ub4dc", "\\uba54\\ub9ac\\ub178", "\\ub2c8\\ud2b8", "\\ubaa8\\uc9c1"],
    "linen": ["\\ub9b0\\ub128", "\\ub9ac\\ub128"],
    "fleece": ["\\ud50c\\ub9ac\\uc2a4", "\\ud6c4\\ub9ac\\uc2a4", "\\ubcf4\\uc544"],
    "corduroy": ["\\ucf54\\ub4c0\\ub85c\\uc774", "\\uace8\\ub374"],
    "suede": ["\\uc2a4\\uc6e8\\uc774\\ub4dc"],
    "nylon": ["\\ub098\\uc77c\\ub860"],
    "polyester": ["\\ud3f4\\ub9ac\\uc5d0\\uc2a4\\ud130", "\\ud3f4\\ub9ac"],
    "cashmere": ["\\uce90\\uc2dc\\ubbf8\\uc5b4"],
    "silk": ["\\uc2e4\\ud06c"],
    "tweed": ["\\ud2b8\\uc704\\ub4dc"],
}

FIT_KEYWORDS = {
    "oversized": ["oversized", "over fit", "overfit", "loose fit", "\\uc624\\ubc84\\ud54f", "\\uc624\\ubc84\\uc0ac\\uc774\\uc988", "\\ub8e8\\uc988\\ud54f", "\\ubc15\\uc2a4\\ud54f"],
    "wide": ["wide", "balloon", "\\uc640\\uc774\\ub4dc", "\\ubc8c\\ub8ec"],
    "slim": ["slim", "skinny", "\\uc2ac\\ub9bc\\ud54f", "\\uc2ac\\ub9bc", "\\uc2a4\\ud0a4\\ub2c8"],
    "regular": ["regular", "standard", "basic fit", "\\ub808\\uade4\\ub7ec\\ud54f", "\\ub808\\uade4\\ub7ec", "\\uc2a4\\ud0e0\\ub2e4\\ub4dc", "\\uae30\\ubcf8\\ud54f"],
    "relaxed": ["relaxed", "comfort", "\\ucef4\\ud3ec\\ud2b8\\ud54f", "\\ub9b4\\ub799\\uc2a4\\ud54f"],
    "cropped": ["cropped", "crop", "\\ud06c\\ub86d", "\\ud06c\\ub86d\\ud54f"],
}

FIT_PRIORITY_BY_MAIN_CATEGORY = {
    "\\ud558\\uc758": ["wide", "slim", "regular", "relaxed", "cropped", "oversized"],
    "\\uc0c1\\uc758": ["oversized", "regular", "slim", "cropped", "relaxed", "wide"],
    "\\uc544\\uc6b0\\ud130": ["oversized", "regular", "slim", "cropped", "relaxed", "wide"],
}

MATERIAL_BY_SUB_CATEGORY = {
    "\\ub370\\ub2d8\\uc790\\ucf13": "denim",
    "\\ub2c8\\ud2b8/\\uc2a4\\uc6e8\\ud130": "wool",
    "\\ub808\\ub354\\uc790\\ucf13": "leather",
    "\\ucf54\\ud2bc \\uc790\\ucf13": "cotton",
}

IMAGE_URL_PATTERN = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
DIRECT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
COLOR_DB_COLUMN = "dominant_color"
COLOR_CONFIDENCE_DB_COLUMN = "color_confidence"
HIGH_COLOR_RATIO = 0.45
MEDIUM_COLOR_RATIO = 0.30
MIN_COLOR_PIXEL_COUNT = 80
IMAGE_REQUEST_TIMEOUT = (5, 10)
UPDATE_CLEANED_PRODUCT_NAME = os.environ.get("UPDATE_CLEANED_PRODUCT_NAME", "true").strip().lower() in {
    "1", "true", "yes", "y"
}

PRODUCT_NAME_NOISE_PATTERNS = [
    re.compile(r"(?<![a-z0-9])\d+\s*[-_/]?\s*colors?(?![a-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![a-z0-9])n\s*[-_/]?\s*colors?(?![a-z0-9])", re.IGNORECASE),
    re.compile(r"\d+\s*[-_/]?\s*\uceec\ub7ec"),
    re.compile(r"n\s*[-_/]?\s*\uceec\ub7ec", re.IGNORECASE),
    re.compile(r"\d+\s*[-_/]?\s*\uc885"),
    re.compile(r"n\s*[-_/]?\s*\uc885", re.IGNORECASE),
]

WORN_IMAGE_LABELS = [
    "a person wearing clothes",
    "a model wearing clothes",
    "a full body outfit photo",
    "a person wearing a fashion item",
]
PRODUCT_ONLY_IMAGE_LABELS = [
    "a clothing product photo without a person",
    "a flat lay clothing product",
    "a garment on a white background",
    "a close up product image of clothing",
]
WORN_IMAGE_SCORE_THRESHOLD = 0.54
WORN_IMAGE_MARGIN_THRESHOLD = 0.08


def build_image_request_headers(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if "msscdn.net" in (url or ""):
        headers["Referer"] = "https://www.musinsa.com/"

    return headers


# -------------------------------------------------------------
# 4. Image preprocessing helpers
def crop_center_region(image: Image.Image, width_ratio: float = 0.78, height_ratio: float = 0.90):
    width, height = image.size
    crop_width = max(1, int(width * width_ratio))
    crop_height = max(1, int(height * height_ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    return image.crop((left, top, right, bottom))


def prepare_clip_image(image: Image.Image):
    """
    Prepare a center crop for legacy CLIP helpers.
    Delete-only mode does not call this function.
    """
    return crop_center_region(image.convert("RGB"), width_ratio=0.82, height_ratio=0.92)


def ensure_clip_model_loaded():
    global model, processor
    if model is not None and processor is not None:
        return

    print(f"Loading FashionCLIP... model={model_id}, device={device}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()


def classify_image_by_prompts(image: Image.Image, labels):
    ensure_clip_model_loaded()
    clip_image = prepare_clip_image(image)
    inputs = processor(
        images=clip_image,
        text=labels,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits_per_image[0], dim=0)

    return {label: probs[index].item() for index, label in enumerate(labels)}


def detect_worn_image(image: Image.Image):
    scores = classify_image_by_prompts(image, [*WORN_IMAGE_LABELS, *PRODUCT_ONLY_IMAGE_LABELS])
    worn_score = max(scores[label] for label in WORN_IMAGE_LABELS)
    product_score = max(scores[label] for label in PRODUCT_ONLY_IMAGE_LABELS)

    return {
        "is_worn": (
            worn_score >= WORN_IMAGE_SCORE_THRESHOLD
            and (worn_score - product_score) >= WORN_IMAGE_MARGIN_THRESHOLD
        ),
        "worn_score": worn_score,
        "product_score": product_score,
    }

# -------------------------------------------------------------
# 5. CLIP embedding helpers
def get_image_embedding(image: Image.Image):
    ensure_clip_model_loaded()
    clip_image = prepare_clip_image(image)

    inputs = processor(
        images=clip_image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.squeeze().tolist()

# -------------------------------------------------------------
# 6. CLIP prompt and attribute classification helpers
# -------------------------------------------------------------
def build_attribute_prompts(attribute_name: str, item_name: str):
    labels_map = ATTRIBUTE_LABELS[attribute_name]
    normalized_name = " ".join((clean_product_name_noise(item_name) or "clothing item").strip().lower().split())

    prompt_templates = {
        "pattern": [
            "a close-up product photo of a {item} with {label} design",
            "a studio product image of a {item} with {label} pattern",
            "a fashion product photo showing {label} detail on a {item}",
        ],
        "fit": [
            "a product photo of a {item} with {label} fit",
            "a fashion photo of a {label} fit {item}",
            "a clothing product with a {label} silhouette",
        ],
        "material": [
            "a close-up photo of {label} fabric",
            "a product photo of a {item} made of {label}",
            "a detailed texture photo of {label} clothing material",
        ],
    }

    templates = prompt_templates.get(attribute_name, [
        "a product photo of a {item} with {label}"
    ])

    prompts = []
    label_keys = []

    for label_key, base_prompt in labels_map.items():
        for template in templates:
            prompts.append(template.format(item=normalized_name, label=base_prompt))
            label_keys.append(label_key)

    return label_keys, prompts


def classify_attribute(image: Image.Image, attribute_name: str, item_name: str):
    ensure_clip_model_loaded()
    clip_image = prepare_clip_image(image)
    label_keys, prompts = build_attribute_prompts(attribute_name, item_name)

    inputs = processor(
        text=prompts,
        images=clip_image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]

    # Average multiple prompts into one score per label.
    label_logit_map = {}
    for label_key, logit in zip(label_keys, logits):
        label_logit_map.setdefault(label_key, [])
        label_logit_map[label_key].append(logit)

    labels = list(label_logit_map.keys())
    label_logits = torch.stack([
        torch.stack(label_logit_map[label]).mean()
        for label in labels
    ])

    label_probs = torch.softmax(label_logits, dim=0)
    sorted_indices = torch.argsort(label_probs, descending=True)

    best_idx = sorted_indices[0].item()
    second_idx = sorted_indices[1].item() if len(sorted_indices) > 1 else best_idx

    best_label = labels[best_idx]
    best_score = label_probs[best_idx].item()
    second_score = label_probs[second_idx].item() if len(sorted_indices) > 1 else 0.0

    if best_score < ATTRIBUTE_THRESHOLDS[attribute_name]:
        return None

    if (best_score - second_score) < ATTRIBUTE_MIN_MARGIN[attribute_name]:
        return None

    return best_label

# -------------------------------------------------------------
# 7. Product-name matching helpers
# -------------------------------------------------------------
def clean_product_name_noise(item_name: str):
    cleaned = item_name or ""
    for pattern in PRODUCT_NAME_NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    cleaned = re.sub(r"\[\s*\]", " ", cleaned)
    cleaned = re.sub(r"\{\s*\}", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def should_delete_product_name(item_name: str):
    name = item_name or ""
    return any(pattern.search(name) for pattern in PRODUCT_NAME_NOISE_PATTERNS)


def normalize_product_name_for_match(item_name: str):
    lowered = clean_product_name_noise(item_name).strip().lower()
    spaced = re.sub(r"[^\w\uac00-\ud7a3]+", " ", lowered)
    spaced = f" {' '.join(spaced.split())} "
    compact = re.sub(r"[^a-z0-9\uac00-\ud7a3]+", "", lowered)
    return spaced, compact


def merged_keyword_list(base_map, extra_map, key):
    return [*base_map.get(key, []), *extra_map.get(key, [])]


def decode_keyword_escapes(keyword: str):
    if "\\u" not in keyword:
        return keyword

    try:
        return keyword.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return keyword


def keyword_matches_product_name(keyword: str, spaced_name: str, compact_name: str):
    keyword = decode_keyword_escapes(keyword or "").strip().lower()
    if not keyword:
        return False

    keyword_spaced = re.sub(r"[^\w\uac00-\ud7a3]+", " ", keyword)
    keyword_spaced = " ".join(keyword_spaced.split())
    keyword_compact = re.sub(r"[^a-z0-9\uac00-\ud7a3]+", "", keyword)

    if not keyword_spaced or not keyword_compact:
        return False

    if re.search(rf"(?<![a-z0-9]){re.escape(keyword_spaced)}(?![a-z0-9])", spaced_name):
        return True

    has_korean = bool(re.search(r"[\uac00-\ud7a3]", keyword_compact))
    if has_korean and len(keyword_compact) >= 2 and keyword_compact in compact_name:
        return True

    if not has_korean and len(keyword_compact) >= 4 and keyword_compact in compact_name:
        return True

    return False


def classify_color_from_name(item_name: str):
    spaced_name, compact_name = normalize_product_name_for_match(item_name)
    if not compact_name:
        return None

    # More specific colors should be matched before broad tokens.
    color_priority = [
        "camouflage", "indigo", "khaki", "navy", "beige", "gray", "black", "white", "brown", "green",
        "blue", "red", "pink", "purple", "orange", "yellow",
    ]

    for color in color_priority:
        for keyword in BASIC_COLOR_KEYWORDS.get(color, []):
            if keyword_matches_product_name(keyword, spaced_name, compact_name):
                return color

    for color in color_priority:
        for keyword in merged_keyword_list(COLOR_KEYWORDS, EXTRA_COLOR_KEYWORDS, color):
            if keyword_matches_product_name(keyword, spaced_name, compact_name):
                return color

    return None


def classify_material_from_name(item_name: str):
    spaced_name, compact_name = normalize_product_name_for_match(item_name)
    if not compact_name:
        return None

    # Match faux leather before broad leather keywords.
    material_priority = [
        "faux_leather", "leather", "denim", "suede", "corduroy", "fleece",
        "cashmere", "wool", "linen", "cotton", "nylon", "polyester", "silk", "tweed",
    ]

    for material in material_priority:
        for keyword in merged_keyword_list(MATERIAL_KEYWORDS, EXTRA_MATERIAL_KEYWORDS, material):
            if keyword_matches_product_name(keyword, spaced_name, compact_name):
                return material

    return None


def classify_fit_from_name(item_name: str, main_category: str = None):
    normalized_name = (item_name or "").strip().lower()
    if not normalized_name:
        return None

    priority = FIT_PRIORITY_BY_MAIN_CATEGORY.get(
        main_category,
        ["oversized", "wide", "slim", "regular", "relaxed", "cropped"]
    )

    for fit in priority:
        for keyword in FIT_KEYWORDS[fit]:
            keyword = keyword.lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", normalized_name):
                if fit == "wide" and main_category in {"\uc0c1\uc758", "\uc544\uc6b0\ud130"}:
                    return "oversized"
                return fit

    return None


def classify_material_from_category(sub_category: str):
    return MATERIAL_BY_SUB_CATEGORY.get(sub_category)


def is_empty_attribute(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"", "null", "none", "unknown"}


def is_direct_image_url(url: str):
    lower_url = (url or "").split("?")[0].lower()
    return lower_url.endswith(DIRECT_IMAGE_EXTENSIONS)


def extract_first_image_url_from_html(html: str, base_url: str):
    meta_patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]

    for pattern in meta_patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(1))

    for match in IMAGE_URL_PATTERN.finditer(html):
        candidate = match.group(0)
        if is_direct_image_url(candidate):
            return candidate

    return None


def download_product_image(image_or_product_url: str):
    if not image_or_product_url:
        raise ValueError("image_url is empty.")

    headers = build_image_request_headers(image_or_product_url)

    first_response = requests.get(image_or_product_url, timeout=IMAGE_REQUEST_TIMEOUT, headers=headers)
    first_response.raise_for_status()
    content_type = first_response.headers.get("content-type", "").lower()

    if "image/" in content_type or is_direct_image_url(image_or_product_url):
        return Image.open(BytesIO(first_response.content)).convert("RGB")

    image_url = extract_first_image_url_from_html(first_response.text, image_or_product_url)
    if not image_url:
        raise ValueError("Could not find an image URL from the product page.")

    image_response = requests.get(image_url, timeout=IMAGE_REQUEST_TIMEOUT, headers=build_image_request_headers(image_url))
    image_response.raise_for_status()
    return Image.open(BytesIO(image_response.content)).convert("RGB")

# -------------------------------------------------------------
# 8. Color extraction helpers
# -------------------------------------------------------------
def is_skin_like(red: int, green: int, blue: int):
    return (
        red > 95 and green > 40 and blue > 20
        and (max(red, green, blue) - min(red, green, blue)) > 15
        and abs(red - green) > 15
        and red > green and red > blue
    )


def rgb_to_xyz_component(value: float):
    value = value / 255.0
    if value > 0.04045:
        return ((value + 0.055) / 1.055) ** 2.4
    return value / 12.92


def rgb_to_lab(rgb):
    """
    Convert RGB to CIE Lab without skimage.
    This is kept for legacy color extraction helpers.
    """
    r, g, b = rgb
    r = rgb_to_xyz_component(float(r))
    g = rgb_to_xyz_component(float(g))
    b = rgb_to_xyz_component(float(b))

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        return (7.787 * t) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return (l, a, b)


def lab_distance(color_a, color_b):
    return sqrt(sum((a - b) ** 2 for a, b in zip(color_a, color_b)))


def classify_color_by_lab(rgb_color):
    input_lab = rgb_to_lab(rgb_color)

    best_name = None
    best_distance = float("inf")

    for color_name, ref_rgb in COLOR_REFERENCES.items():
        ref_lab = rgb_to_lab(ref_rgb)
        distance = lab_distance(input_lab, ref_lab)

        if distance < best_distance:
            best_distance = distance
            best_name = color_name

    return best_name


def classify_color_name_for_rgb(rgb_color):
    return classify_color_by_lab(rgb_color)


def is_denim_context(item_name: str = "", sub_category: str = "", material: str = None):
    text = f"{item_name or ''} {sub_category or ''} {material or ''}".lower()
    denim_terms = (
        "denim", "jean", "jeans", "raw denim",
        "\ub370\ub2d8", "\uccad\ubc14\uc9c0", "\uccad\uc790\ucf13",
        "\ud751\uccad", "\uc9c4\uccad", "\uc911\uccad", "\uc5f0\uccad",
    )
    return material == "denim" or any(term in text for term in denim_terms)


def classify_denim_color_by_lab(rgb_color):
    input_lab = rgb_to_lab(rgb_color)
    return min(
        DENIM_COLOR_REFERENCES,
        key=lambda color_name: lab_distance(input_lab, rgb_to_lab(DENIM_COLOR_REFERENCES[color_name])),
    )


def classify_denim_color_from_pixels(pixels):
    if pixels is None or len(pixels) < 20:
        return None

    pixels = np.asarray(pixels, dtype=np.float32)
    brightness = pixels.mean(axis=1)
    channel_spread = pixels.max(axis=1) - pixels.min(axis=1)
    blue_bias = pixels[:, 2] - np.maximum(pixels[:, 0], pixels[:, 1])

    dark_mask = brightness < 95
    neutral_dark_mask = dark_mask & (channel_spread < 34)
    indigo_mask = (brightness < 135) & (pixels[:, 2] >= pixels[:, 0] + 10) & (pixels[:, 2] >= pixels[:, 1] - 8)
    blue_mask = (pixels[:, 2] >= pixels[:, 0] + 18) & (pixels[:, 2] >= pixels[:, 1] + 2)
    light_blue_mask = blue_mask & (brightness >= 145)

    neutral_dark_ratio = float(np.mean(neutral_dark_mask))
    indigo_ratio = float(np.mean(indigo_mask))
    blue_ratio = float(np.mean(blue_mask))
    light_blue_ratio = float(np.mean(light_blue_mask))
    avg_rgb = tuple(int(x) for x in pixels.mean(axis=0))
    avg_blue_bias = float(np.mean(blue_bias))

    if neutral_dark_ratio >= 0.24 and avg_blue_bias < 18:
        return "black"
    if indigo_ratio >= 0.22 or (avg_rgb[2] >= avg_rgb[0] + 8 and avg_rgb[2] >= avg_rgb[1] - 6 and brightness.mean() < 135):
        return "indigo"
    if light_blue_ratio >= 0.20:
        return "blue"
    if blue_ratio >= 0.18:
        return "blue"

    return classify_denim_color_by_lab(avg_rgb)


def is_camouflage_cluster_mix(candidates):
    if len(candidates) < 3:
        return False

    total_count = sum(count for count, _ in candidates)
    if total_count <= 0:
        return False

    earthy_colors = []
    for count, rgb in candidates:
        color_name = classify_color_name_for_rgb(rgb)
        ratio = count / total_count
        if ratio >= 0.12 and color_name in {"black", "brown", "green", "gray", "beige", "camouflage"}:
            earthy_colors.append(color_name)

    return len(set(earthy_colors)) >= 3


def is_indigo_denim_like_color(rgb_color):
    r, g, b = rgb_color
    return (
        45 <= r <= 130
        and 65 <= g <= 150
        and 85 <= b <= 185
        and b >= r + 20
        and abs(b - g) <= 70
    )


def extract_pixels_from_mask(image: Image.Image, mask: Image.Image = None):
    image = image.convert("RGB").resize((224, 224))
    image_np = np.array(image)

    if mask is not None:
        mask = mask.convert("L").resize((224, 224))
        mask_np = np.array(mask)
        pixels = image_np[mask_np > 128]
    else:
        # Fallback to the central product region when no mask is available.
        h, w, _ = image_np.shape
        top = int(h * 0.08)
        bottom = int(h * 0.92)
        left = int(w * 0.12)
        right = int(w * 0.88)
        pixels = image_np[top:bottom, left:right].reshape(-1, 3)

    filtered_pixels = []

    for r, g, b in pixels:
        r = int(r)
        g = int(g)
        b = int(b)

        # Drop near-white background pixels.
        if r > 242 and g > 242 and b > 242:
            continue

        # Drop near-black shadow pixels.
        if r < 12 and g < 12 and b < 12:
            continue

        # Drop skin-like pixels from model photos.
        if is_skin_like(r, g, b):
            continue

        filtered_pixels.append([r, g, b])

    return np.array(filtered_pixels, dtype=np.float32)


def validate_basic_image_quality(image: Image.Image):
    width, height = image.size
    if min(width, height) < 80:
        return False, "image_too_small"

    resized = image.convert("RGB").resize((96, 96))
    image_np = np.asarray(resized, dtype=np.float32)
    brightness = image_np.mean(axis=2)
    avg_brightness = float(brightness.mean())
    pixel_std = float(brightness.std())

    if pixel_std < 8:
        return False, "image_has_too_little_detail"
    if avg_brightness < 25:
        return False, "image_too_dark"
    if avg_brightness > 245:
        return False, "image_too_bright"

    return True, ""


def classify_color_confidence_from_candidates(candidates):
    if not candidates:
        return {
            "color": None,
            "confidence": "low",
            "reason": "no_color_candidates",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
        }

    total_count = sum(count for count, _ in candidates)
    if total_count <= 0:
        return {
            "color": None,
            "confidence": "low",
            "reason": "no_valid_color_pixels",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
        }

    valid_candidates = [
        (count, rgb)
        for count, rgb in candidates
        if count / total_count >= 0.08
    ]
    if not valid_candidates:
        return {
            "color": None,
            "confidence": "low",
            "reason": "only_tiny_color_clusters",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
        }

    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    top_count, dominant_rgb = valid_candidates[0]
    top_ratio = top_count / total_count
    second_ratio = valid_candidates[1][0] / total_count if len(valid_candidates) > 1 else 0.0

    if is_camouflage_cluster_mix(valid_candidates):
        return {
            "color": "camouflage",
            "confidence": "medium",
            "reason": "camouflage_cluster_mix",
            "dominant_ratio": top_ratio,
            "second_ratio": second_ratio,
        }

    if second_ratio >= 0.18 and abs(top_ratio - second_ratio) < 0.15:
        return {
            "color": "multi_color",
            "confidence": "low",
            "reason": "mixed_color_clusters",
            "dominant_ratio": top_ratio,
            "second_ratio": second_ratio,
        }

    color = "indigo" if is_indigo_denim_like_color(dominant_rgb) else classify_color_by_lab(dominant_rgb)
    if top_ratio >= HIGH_COLOR_RATIO and top_ratio - second_ratio >= 0.18:
        confidence = "high"
    elif top_ratio >= MEDIUM_COLOR_RATIO:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "color": color,
        "confidence": confidence,
        "reason": "",
        "dominant_ratio": top_ratio,
        "second_ratio": second_ratio,
    }


def extract_dominant_color_result(image: Image.Image, mask: Image.Image = None, n_clusters: int = 5, denim_context: bool = False):
    ok, quality_reason = validate_basic_image_quality(image)
    if not ok:
        return {
            "color": None,
            "confidence": "low",
            "reason": quality_reason,
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
        }

    pixels = extract_pixels_from_mask(image, mask)
    if len(pixels) < MIN_COLOR_PIXEL_COUNT:
        return {
            "color": None,
            "confidence": "low",
            "reason": "not_enough_valid_pixels",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
        }

    if denim_context:
        denim_color = classify_denim_color_from_pixels(pixels)
        if denim_color:
            return {
                "color": denim_color,
                "confidence": "high",
                "reason": "denim_context",
                "dominant_ratio": 1.0,
                "second_ratio": 0.0,
            }

    n_clusters = min(n_clusters, len(pixels))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(pixels)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total_count = len(kmeans.labels_)
    candidates = []

    for label, count in zip(labels, counts):
        center = kmeans.cluster_centers_[label]
        r, g, b = [int(x) for x in center]

        if r > 240 and g > 240 and b > 240:
            continue

        candidates.append((int(count), (r, g, b)))

    if not candidates and total_count:
        dominant_index = labels[np.argmax(counts)]
        dominant_rgb = tuple(int(x) for x in kmeans.cluster_centers_[dominant_index])
        candidates = [(int(max(counts)), dominant_rgb)]

    return classify_color_confidence_from_candidates(candidates)


def extract_dominant_color(image: Image.Image, mask: Image.Image = None, n_clusters: int = 3, denim_context: bool = False):
    return extract_dominant_color_result(image, mask, n_clusters, denim_context).get("color")

    pixels = extract_pixels_from_mask(image, mask)

    if len(pixels) < 20:
        return None

    if denim_context:
        denim_color = classify_denim_color_from_pixels(pixels)
        if denim_color:
            return denim_color

    n_clusters = min(n_clusters, len(pixels))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(pixels)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Drop tiny clusters that are likely logos or prints.
    min_ratio = 0.12
    valid_candidates = []
    total_count = len(kmeans.labels_)

    for label, count in zip(labels, counts):
        ratio = count / total_count
        center = kmeans.cluster_centers_[label]
        r, g, b = [int(x) for x in center]

        if ratio < min_ratio:
            continue

        # Exclude mostly white background clusters.
        if r > 240 and g > 240 and b > 240:
            continue

        valid_candidates.append((count, (r, g, b)))

    if not valid_candidates:
        dominant_index = labels[np.argmax(counts)]
        dominant_rgb = tuple(int(x) for x in kmeans.cluster_centers_[dominant_index])
    else:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        if is_camouflage_cluster_mix(valid_candidates):
            return "camouflage"

        dominant_rgb = valid_candidates[0][1]

    if is_indigo_denim_like_color(dominant_rgb):
        return "indigo"

    return classify_color_by_lab(dominant_rgb)

# -------------------------------------------------------------
# 9. Combined attribute extraction helpers
# -------------------------------------------------------------
def extract_fashion_attributes(
    image: Image.Image,
    item_name: str,
    main_category: str = None,
    sub_category: str = None,
    mask: Image.Image = None
):
    material = classify_material_from_name(item_name)
    if material is None:
        material = classify_material_from_category(sub_category)

    denim_context = is_denim_context(item_name, sub_category, material)
    color = classify_color_from_name(item_name)
    if color is not None:
        color_result = {
            "color": color,
            "confidence": "high",
            "reason": "product_name",
        }
    else:
        color_result = extract_dominant_color_result(image, mask, denim_context=denim_context)
        color = color_result["color"]

    fit = classify_fit_from_name(item_name, main_category)

    return {
        "color": color,
        "color_confidence": color_result["confidence"],
        "color_reason": color_result.get("reason", ""),
        "fit": fit,
        "material": material,
    }


def extract_name_based_attributes(item_name: str):
    color = classify_color_from_name(item_name)
    return {
        "color": color,
        "color_confidence": "high" if color else "low",
        "material": classify_material_from_name(item_name),
    }

# -------------------------------------------------------------
# 10. Supabase attribute update job
# -------------------------------------------------------------
def should_update_db_value(current_value, new_value):
    if is_empty_attribute(new_value):
        return False
    return OVERWRITE_ATTRIBUTES or is_empty_attribute(current_value)


def build_attribute_update_payload(item, attributes):
    payload = {}

    if should_update_db_value(item.get(COLOR_DB_COLUMN), attributes.get("color")):
        payload[COLOR_DB_COLUMN] = attributes["color"]
        payload[COLOR_CONFIDENCE_DB_COLUMN] = attributes.get("color_confidence") or "low"
    elif (
        should_update_db_value(item.get(COLOR_CONFIDENCE_DB_COLUMN), attributes.get("color_confidence"))
        and not is_empty_attribute(item.get(COLOR_DB_COLUMN))
    ):
        payload[COLOR_CONFIDENCE_DB_COLUMN] = attributes["color_confidence"]

    if should_update_db_value(item.get(MATERIAL_DB_COLUMN), attributes.get("material")):
        payload[MATERIAL_DB_COLUMN] = attributes["material"]

    if should_update_db_value(item.get(FIT_DB_COLUMN), attributes.get("fit")):
        payload[FIT_DB_COLUMN] = attributes["fit"]

    return payload


def write_job_log(file_name: str, rows):
    if not rows:
        return

    log_path = os.path.join(BASE_DIR, file_name)
    with open(log_path, "w", encoding="utf-8") as log_file:
        for row in rows:
            log_file.write("\t".join(str(value) for value in row) + "\n")
    print(f"Log saved: {log_path}")


def update_all_attributes():
    print(f"Running {SCRIPT_VERSION}")
    print(f"Script path: {os.path.abspath(__file__)}")
    print(f"Loading rows from Supabase in pages of {PAGE_SIZE}...")
    print(f"Order column: {ORDER_COLUMN}")
    print(f"Overwrite existing attributes: {OVERWRITE_ATTRIBUTES}")
    print(f"Delete option-count rows: {DELETE_OPTION_COUNT_ROWS}")
    print(f"Dry run: {DRY_RUN}")

    all_items = []
    last_order_value = None
    limit = PAGE_SIZE
    select_columns = [
        "name",
        "image_url",
        "main_category",
        "sub_category",
        COLOR_DB_COLUMN,
        COLOR_CONFIDENCE_DB_COLUMN,
        MATERIAL_DB_COLUMN,
        FIT_DB_COLUMN,
    ]
    if ORDER_COLUMN not in select_columns:
        select_columns.append(ORDER_COLUMN)

    while True:
        query = (
            supabase
            .table("clothes")
            .select(", ".join(select_columns))
            .order(ORDER_COLUMN, desc=False)
            .limit(limit)
        )

        if last_order_value is not None:
            query = query.gt(ORDER_COLUMN, last_order_value)

        response = query.execute()
        data = response.data
        if not data:
            break

        all_items.extend(data)
        print(f"Loaded rows: {len(all_items)}")

        if len(data) < limit:
            break

        next_order_value = data[-1].get(ORDER_COLUMN)
        if next_order_value is None or next_order_value == last_order_value:
            raise RuntimeError(f"Cannot continue pagination with ORDER_COLUMN={ORDER_COLUMN}")
        last_order_value = next_order_value

    if not all_items:
        print("No rows to check.")
        return

    print(f"\nStarting fashion attribute update for {len(all_items)} rows.\n")

    failed_items = []
    deleted_items = []
    updated_items = []
    skipped_items = []

    for index, item in enumerate(all_items, 1):
        name = item.get("name") or "NO_NAME"
        image_url = item.get("image_url")
        raw_name = name

        if not image_url:
            failed_items.append((index, raw_name, "image_url missing"))
            print(f"[{index}/{len(all_items)}] skipped: {raw_name} reason=image_url_missing")
            continue

        if DELETE_OPTION_COUNT_ROWS and should_delete_product_name(raw_name):
            try:
                delete_response = None
                if not DRY_RUN:
                    delete_response = (
                        supabase
                        .table("clothes")
                        .delete()
                        .eq("image_url", image_url)
                        .execute()
                    )
                deleted_items.append((index, raw_name, "name_option_count"))
                deleted_rows = len(delete_response.data or []) if delete_response else 0
                print(f"[{index}/{len(all_items)}] deleted: {raw_name} reason=name_option_count rows={deleted_rows}")
            except Exception as e:
                failed_items.append((index, raw_name, str(e)))
                print(f"[{index}/{len(all_items)}] delete failed: {raw_name} error={e}")
            continue

        try:
            image = download_product_image(image_url)
            attributes = extract_fashion_attributes(
                image,
                raw_name,
                item.get("main_category"),
                item.get("sub_category"),
            )
            payload = build_attribute_update_payload(item, attributes)

            if not payload:
                skipped_items.append((index, raw_name, "attributes already filled"))
                print(f"[{index}/{len(all_items)}] skipped: {raw_name} attributes={attributes}")
                continue

            if not DRY_RUN:
                supabase.table("clothes").update(payload).eq("image_url", image_url).execute()

            updated_items.append((index, raw_name, payload))
            print(f"[{index}/{len(all_items)}] updated: {raw_name} payload={payload}")
        except Exception as e:
            failed_items.append((index, raw_name, str(e)))
            print(f"[{index}/{len(all_items)}] failed: {raw_name} error={e}")

        if index % 100 == 0:
            print(f"[{index}/{len(all_items)}] processed rows")

    print("\nFashion attribute update finished.")


    print(f"Updated items: {len(updated_items)}")
    print(f"Skipped items: {len(skipped_items)}")
    print(f"Deleted items: {len(deleted_items)}")
    print(f"Failed items: {len(failed_items)}")

    write_job_log("update_attributes.log", updated_items)
    write_job_log("update_skipped.log", skipped_items)
    write_job_log("update_deleted.log", deleted_items)
    write_job_log("update_failed.log", failed_items)


if __name__ == "__main__":
    update_all_attributes()

