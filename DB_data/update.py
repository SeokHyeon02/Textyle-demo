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
# 1. 설정 및 DB 연결
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # service_role 키 확인 필수

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL 또는 SUPABASE_KEY가 .env에 없습니다.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OVERWRITE_ATTRIBUTES = os.environ.get("OVERWRITE_ATTRIBUTES", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

# -------------------------------------------------------------
# 2. AI 모델 로드
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

print(f"⏳ AI 모델을 불러오는 중입니다... (사용 장치: {device})")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

# -------------------------------------------------------------
# 3. 라벨 설정
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

# 여러 프롬프트를 평균낸 뒤 label 단위 softmax를 사용하므로 기존보다 threshold를 조금 높게 둔다.
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
    "black": [
        "black", "blk", "블랙", "검정", "검정색", "흑색",
    ],
    "white": [
        "white", "wht", "화이트", "흰색", "백색", "아이보리", "ivory",
    ],
    "gray": [
        "gray", "grey", "charcoal", "melange", "그레이", "회색", "차콜", "멜란지",
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
        "green", "mint", "그린", "초록", "녹색", "민트",
    ],
    "khaki": [
        "khaki", "olive", "카키", "올리브",
    ],
    "blue": [
        "blue", "sky blue", "sax", "블루", "파랑", "파란색", "스카이블루", "소라",
    ],
    "navy": [
        "navy", "네이비", "남색",
    ],
    "purple": [
        "purple", "violet", "lavender", "퍼플", "보라", "보라색", "바이올렛", "라벤더",
    ],
    "pink": [
        "pink", "핑크", "분홍", "분홍색",
    ],
    "brown": [
        "brown", "camel", "mocha", "브라운", "갈색", "카멜", "모카",
    ],
    "beige": [
        "beige", "cream", "sand", "oatmeal", "베이지", "크림", "샌드", "오트밀",
    ],
    "indigo": [
        "indigo", "raw denim", "deep blue denim", "인디고", "생지", "진청",
    ],
    "camouflage": [
        "camouflage", "camo", "카모", "카모플라주", "위장", "밀리터리",
    ],
}

MATERIAL_KEYWORDS = {
    "faux_leather": [
        "faux leather", "fake leather", "synthetic leather", "pu leather",
        "비건 레더", "인조가죽", "합성가죽"
    ],
    "leather": [
        "leather", "real leather", "genuine leather", "레더", "가죽"
    ],
    "denim": [
        "denim", "jean", "jeans", "raw denim", "데님", "청자켓", "청바지", "청", "생지", "진청", "중청", "연청", "흑청"
    ],
    "cotton": ["cotton", "코튼", "면"],
    "wool": ["wool", "울", "모직"],
    "linen": ["linen", "린넨"],
    "fleece": ["fleece", "플리스", "후리스"],
    "corduroy": ["corduroy", "코듀로이", "골덴"],
    "suede": ["suede", "스웨이드"],
    "nylon": ["nylon", "나일론"],
    "polyester": ["polyester", "폴리에스터", "폴리"],
    "cashmere": ["cashmere", "캐시미어"],
    "silk": ["silk", "실크"],
    "tweed": ["tweed", "트위드"],
}

EXTRA_COLOR_KEYWORDS = {
    "black": [
        "bk", "black denim", "blackdenim", "washed black", "dark black",
        "\ube14\ub799", "\uac80\uc815", "\uac80\uc815\uc0c9", "\uae4c\ub9cc\uc0c9", "\ud751\uc0c9", "\ud751\uccad",
    ],
    "white": [
        "wh", "ivory", "off white", "offwhite", "cream white",
        "\ud654\uc774\ud2b8", "\ud770\uc0c9", "\ud558\uc580\uc0c9", "\ubc31\uc0c9", "\uc544\uc774\ubcf4\ub9ac",
    ],
    "gray": [
        "gry", "grey", "charcoal", "melange", "ash gray",
        "\uadf8\ub808\uc774", "\ud68c\uc0c9", "\ucc28\ucf5c", "\uba5c\ub780\uc9c0",
    ],
    "indigo": [
        "raw denim", "rawdenim", "dark denim", "deep blue denim", "dark indigo",
        "\uc778\ub514\uace0", "\uc0dd\uc9c0", "\uc9c4\uccad",
    ],
    "blue": [
        "light denim", "light blue denim", "mid blue", "medium blue", "sax", "sky blue",
        "\ube14\ub8e8", "\ud30c\ub791", "\ud30c\ub780\uc0c9", "\uc18c\ub77c", "\uc2a4\uce74\uc774\ube14\ub8e8", "\uc911\uccad", "\uc5f0\uccad",
    ],
    "navy": ["nvy", "\ub124\uc774\ube44", "\ub0a8\uc0c9"],
    "khaki": ["olive", "\uce74\ud0a4", "\uc62c\ub9ac\ube0c"],
    "beige": ["oatmeal", "sand", "cream", "\ubca0\uc774\uc9c0", "\uc624\ud2b8\ubc00", "\uc0cc\ub4dc", "\ud06c\ub9bc"],
    "brown": ["brn", "camel", "mocha", "\ube0c\ub77c\uc6b4", "\uac08\uc0c9", "\uce74\uba5c", "\ubaa8\uce74"],
    "red": ["burgundy", "wine", "\ub808\ub4dc", "\ube68\uac15", "\ube68\uac04\uc0c9", "\ubc84\uac74\ub514", "\uc640\uc778"],
    "green": ["mint", "\uadf8\ub9b0", "\ucd08\ub85d", "\ub179\uc0c9", "\ubbfc\ud2b8"],
    "yellow": ["mustard", "\uc610\ub85c\uc6b0", "\ub178\ub791", "\ub178\ub780\uc0c9", "\uba38\uc2a4\ud0c0\ub4dc"],
    "pink": ["\ud551\ud06c", "\ubd84\ud64d", "\ubd84\ud64d\uc0c9"],
    "purple": ["violet", "lavender", "\ud37c\ud50c", "\ubcf4\ub77c", "\ubcf4\ub77c\uc0c9", "\ubc14\uc774\uc62c\ub81b", "\ub77c\ubca4\ub354"],
    "orange": ["\uc624\ub80c\uc9c0", "\uc8fc\ud669", "\uc8fc\ud669\uc0c9"],
    "camouflage": ["camo", "\uce74\ubaa8", "\uce74\ubaa8\ud50c\ub77c\uc8fc", "\uc704\uc7a5", "\ubc00\ub9ac\ud130\ub9ac"],
}

EXTRA_MATERIAL_KEYWORDS = {
    "faux_leather": [
        "vegan leather", "eco leather", "artificial leather",
        "\ube44\uac74\ub808\ub354", "\uc778\uc870\uac00\uc8fd", "\ud569\uc131\uac00\uc8fd", "\uc5d0\ucf54\ub808\ub354",
    ],
    "leather": [
        "goat leather", "goat skin", "goatskin", "cowhide", "cow leather",
        "lambskin", "lamb leather", "sheep leather", "sheepskin", "real leather",
        "\ub808\ub354", "\uac00\uc8fd", "\uace0\ud2b8", "\uc591\uac00\uc8fd", "\uc18c\uac00\uc8fd", "\ub7a8\uc2a4\ud0a8", "\uce74\uc6b0\ud558\uc774\ub4dc",
    ],
    "denim": [
        "black denim", "raw denim", "washed denim", "selvedge", "selvage",
        "\ub370\ub2d8", "\uccad\ubc14\uc9c0", "\uccad\uc790\ucf13", "\ud751\uccad", "\uc0dd\uc9c0", "\uc9c4\uccad", "\uc911\uccad", "\uc5f0\uccad",
    ],
    "cotton": ["cotton 100", "100 cotton", "\ucf54\ud2bc", "\uba74", "\uba74 100", "\uba74100"],
    "wool": ["wool blend", "merino", "knit", "\uc6b8", "\uc6b8\ube14\ub80c\ub4dc", "\uba54\ub9ac\ub178", "\ub2c8\ud2b8", "\ubaa8\uc9c1"],
    "linen": ["\ub9b0\ub128", "\ub9ac\ub128"],
    "fleece": ["boa fleece", "\ud50c\ub9ac\uc2a4", "\ud6c4\ub9ac\uc2a4", "\ubcf4\uc544"],
    "corduroy": ["\ucf54\ub4c0\ub85c\uc774", "\uace8\ub374"],
    "suede": ["\uc2a4\uc6e8\uc774\ub4dc"],
    "nylon": ["\ub098\uc77c\ub860"],
    "polyester": ["poly", "\ud3f4\ub9ac\uc5d0\uc2a4\ud130", "\ud3f4\ub9ac"],
    "cashmere": ["\uce90\uc2dc\ubbf8\uc5b4"],
    "silk": ["\uc2e4\ud06c"],
    "tweed": ["\ud2b8\uc704\ub4dc"],
}

FIT_KEYWORDS = {
    "oversized": [
        "oversized", "over fit", "overfit", "loose fit",
        "오버핏", "오버사이즈", "루즈핏", "박스핏", "빅실루엣"
    ],
    "wide": [
        "wide", "와이드", "벌룬", "balloon"
    ],
    "slim": [
        "slim", "skinny", "슬림핏", "슬림", "스키니"
    ],
    "regular": [
        "regular", "standard", "basic fit",
        "레귤러핏", "레귤러", "스탠다드핏", "스탠다드", "기본핏"
    ],
    "relaxed": [
        "relaxed", "comfort", "컴포트핏", "릴렉스핏", "테이퍼드"
    ],
    "cropped": [
        "cropped", "crop", "크롭", "크롭트"
    ],
}

FIT_PRIORITY_BY_MAIN_CATEGORY = {
    "하의": ["wide", "slim", "regular", "relaxed", "cropped", "oversized"],
    "상의": ["oversized", "regular", "slim", "cropped", "relaxed", "wide"],
    "아우터": ["oversized", "regular", "slim", "cropped", "relaxed", "wide"],
}

MATERIAL_BY_SUB_CATEGORY = {
    "데님팬츠": "denim",
    "니트/스웨터": "wool",
    "레더자켓": "leather",
    "코튼 팬츠": "cotton",
}

IMAGE_URL_PATTERN = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
DIRECT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
COLOR_DB_COLUMN = "dominant_color"

# -------------------------------------------------------------
# 4. 이미지 전처리
# -------------------------------------------------------------
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
    CLIP이 배경보다 의류에 집중하도록 중앙 영역을 사용한다.
    정확도를 더 높이려면 segmentation mask로 옷 영역만 crop해서 넣는 것이 가장 좋다.
    """
    return crop_center_region(image.convert("RGB"), width_ratio=0.82, height_ratio=0.92)

# -------------------------------------------------------------
# 5. CLIP 임베딩
# -------------------------------------------------------------
def get_image_embedding(image: Image.Image):
    clip_image = prepare_clip_image(image)

    inputs = processor(
        text=[""],
        images=clip_image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.squeeze().tolist()

# -------------------------------------------------------------
# 6. CLIP 프롬프트 생성 및 속성 분류
# -------------------------------------------------------------
def build_attribute_prompts(attribute_name: str, item_name: str):
    labels_map = ATTRIBUTE_LABELS[attribute_name]
    normalized_name = " ".join((item_name or "clothing item").strip().lower().split())

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

    # 같은 label에 대해 여러 prompt를 만들고, logit 평균을 label 점수로 사용한다.
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
# 7. 상품명 기반 색상/소재 추출
# -------------------------------------------------------------
def normalize_product_name_for_match(item_name: str):
    lowered = (item_name or "").strip().lower()
    spaced = re.sub(r"[^\w\uac00-\ud7a3]+", " ", lowered)
    spaced = f" {' '.join(spaced.split())} "
    compact = re.sub(r"[^a-z0-9\uac00-\ud7a3]+", "", lowered)
    return spaced, compact


def merged_keyword_list(base_map, extra_map, key):
    return [*base_map.get(key, []), *extra_map.get(key, [])]


def keyword_matches_product_name(keyword: str, spaced_name: str, compact_name: str):
    keyword = (keyword or "").strip().lower()
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
        for keyword in merged_keyword_list(COLOR_KEYWORDS, EXTRA_COLOR_KEYWORDS, color):
            if keyword_matches_product_name(keyword, spaced_name, compact_name):
                return color

    return None


def classify_material_from_name(item_name: str):
    spaced_name, compact_name = normalize_product_name_for_match(item_name)
    if not compact_name:
        return None

    # faux leather가 leather보다 먼저 검사되어야 한다.
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
                if fit == "wide" and main_category in {"상의", "아우터"}:
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
        raise ValueError("image_url이 비어 있습니다.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    first_response = requests.get(image_or_product_url, timeout=15, headers=headers)
    first_response.raise_for_status()
    content_type = first_response.headers.get("content-type", "").lower()

    if "image/" in content_type or is_direct_image_url(image_or_product_url):
        return Image.open(BytesIO(first_response.content)).convert("RGB")

    image_url = extract_first_image_url_from_html(first_response.text, image_or_product_url)
    if not image_url:
        raise ValueError("상품 페이지에서 대표 이미지 URL을 찾지 못했습니다.")

    image_response = requests.get(image_url, timeout=15, headers=headers)
    image_response.raise_for_status()
    return Image.open(BytesIO(image_response.content)).convert("RGB")

# -------------------------------------------------------------
# 8. 색상 추출: fallback 전경 추정 + KMeans + Lab 거리
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
    skimage 없이 RGB를 CIE Lab으로 변환한다.
    RGB 유클리드 거리보다 사람이 느끼는 색 차이에 더 가깝다.
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
        # segmentation mask가 없을 때 fallback: 중앙 의류 영역을 넓게 사용한다.
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

        # 흰색/밝은 회색 배경 제거
        if r > 242 and g > 242 and b > 242:
            continue

        # 검은 그림자 제거
        if r < 12 and g < 12 and b < 12:
            continue

        # 피부색 제거
        if is_skin_like(r, g, b):
            continue

        filtered_pixels.append([r, g, b])

    return np.array(filtered_pixels, dtype=np.float32)


def extract_dominant_color(image: Image.Image, mask: Image.Image = None, n_clusters: int = 3, denim_context: bool = False):
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

    # 너무 작은 군집은 로고/그림자일 가능성이 있어 제외한다.
    min_ratio = 0.12
    valid_candidates = []
    total_count = len(kmeans.labels_)

    for label, count in zip(labels, counts):
        ratio = count / total_count
        center = kmeans.cluster_centers_[label]
        r, g, b = [int(x) for x in center]

        if ratio < min_ratio:
            continue

        # 거의 흰 배경성 군집은 제외
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
# 9. 전체 속성 추출
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
    if color is None:
        color = extract_dominant_color(image, mask, denim_context=denim_context)

    fit = classify_fit_from_name(item_name, main_category)

    return {
        "color": color,
        "fit": fit,
        "material": material,
    }

# -------------------------------------------------------------
# 10. Supabase 업데이트
# -------------------------------------------------------------
def update_all_embeddings():
    print("🔍 Supabase에서 데이터를 가져옵니다 (1000개씩 분할 로드)...")
    print("⚙️ 모든 상품의 embedding과 추출 가능한 속성을 image_url 기준으로 업데이트합니다.")

    all_items = []
    start = 0
    limit = 1000

    while True:
        response = (
            supabase
            .table("clothes")
            .select("name, image_url, main_category, sub_category")
            .range(start, start + limit - 1)
            .execute()
        )

        data = response.data
        if not data:
            break

        all_items.extend(data)
        print(f" 📥 누적 {len(all_items)}개 데이터 로드 완료...")

        if len(data) < limit:
            break

        start += limit

    if not all_items:
        print("업데이트할 데이터가 없습니다.")
        return

    print(f"\n🚀 총 {len(all_items)}개의 데이터 업데이트를 시작합니다!\n")

    for index, item in enumerate(all_items, 1):
        name = item.get("name") or "이름 없음"
        image_url = item.get("image_url")
        main_category = item.get("main_category")
        sub_category = item.get("sub_category")

        if not image_url:
            print(f"❌ [{name}] image_url이 없어 건너뜁니다.")
            continue

        print(f"[{index}/{len(all_items)}] 업데이트 중: {name}")

        try:
            image = download_product_image(image_url)

            embedding_list = get_image_embedding(image)
            attributes = extract_fashion_attributes(
                image=image,
                item_name=name,
                main_category=main_category,
                sub_category=sub_category,
                mask=None
            )

            update_payload = {
                "embedding": embedding_list,
            }

            if attributes["color"] is not None:
                update_payload[COLOR_DB_COLUMN] = attributes["color"]

            if attributes["fit"] is not None:
                update_payload["fit"] = attributes["fit"]

            if attributes["material"] is not None:
                update_payload["material"] = attributes["material"]

            update_response = (
                supabase
                .table("clothes")
                .update(update_payload)
                .eq("image_url", image_url)
                .execute()
            )

            verify_response = (
                supabase
                .table("clothes")
                .select(f"{COLOR_DB_COLUMN}, fit, material")
                .eq("image_url", image_url)
                .limit(1)
                .execute()
            )
            verified_item = verify_response.data[0] if verify_response.data else {}

            print(
                "   -> "
                f"extracted_{COLOR_DB_COLUMN}={update_payload.get(COLOR_DB_COLUMN)}, "
                f"db_{COLOR_DB_COLUMN}={verified_item.get(COLOR_DB_COLUMN)}, "
                f"extracted_fit={attributes['fit']}, "
                f"db_fit={verified_item.get('fit')}, "
                f"extracted_material={attributes['material']}, "
                f"db_material={verified_item.get('material')}, "
                f"returned_rows={len(update_response.data or [])}, "
                f"verified_rows={len(verify_response.data or [])}"
            )

        except Exception as e:
            print(f"❌ [{name}] 처리 중 오류 발생: {e}")

    print("\n✅ 임베딩과 속성 메타데이터 업데이트가 모두 완료되었습니다!")


if __name__ == "__main__":
    update_all_embeddings()
