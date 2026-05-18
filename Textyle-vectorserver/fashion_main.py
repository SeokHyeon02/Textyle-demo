import io
import importlib
import json
import os
import re
import tempfile
import traceback
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from supabase import Client, create_client
from transformers import CLIPModel, CLIPProcessor
from fashion_color_extraction import (
    extract_dominant_color_result as extract_dominant_color_result_v2,
    is_denim_context_from_text,
    should_run_pattern_classifier,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
FASHION_CLIP_MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
FASHION_CLIP_API_MODEL_ID = os.environ.get("FASHION_CLIP_API_MODEL_ID", "fashion-clip")
SEGMENTATION_MODEL_NAME = os.environ.get("SEGMENTATION_MODEL_NAME", "u2net_cloth_seg")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and genai else None
rembg_new_session = None
rembg_remove = None
rembg_loaded = False
segmentation_session = None
segmentation_failed = False

app = FastAPI(title="TexTyle FashionCLIP Search Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading FashionCLIP... model={FASHION_CLIP_MODEL_ID}, device={device}")
fclip = FashionCLIP(FASHION_CLIP_API_MODEL_ID)
print(f"FashionCLIP API loaded: {FASHION_CLIP_API_MODEL_ID}")
model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL_ID)
model.eval()
print("FashionCLIP loaded")


class QueryIntent(BaseModel):
    reasoning: str = Field(description="query analysis reasoning")
    color: str = Field(description="target color, empty string if absent")
    color_mode: str = Field(description="target, same, different, or ignore")
    design: str = Field(description="style, length, detail, or design phrase")


@dataclass
class ImageQualityResult:
    is_usable: bool
    reason: str = ""


@dataclass
class FashionImageValidationResult:
    is_fashion: bool
    fashion_score: float
    non_fashion_score: float
    best_label: str
    face_score: float = 0.0
    reason: str = ""


@dataclass
class ColorExtractionResult:
    color: str
    confidence: str
    reason: str = ""
    dominant_ratio: float = 0.0
    second_ratio: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    secondary_colors: list[str] = field(default_factory=list)
    is_mixed_color: bool = False
    pattern: str = ""
    search_color_weights: dict[str, float] = field(default_factory=dict)


CATEGORY_KEYWORDS = {
    "후드티": ("상의", "후드티"),
    "후디": ("상의", "후드티"),
    "맨투맨": ("상의", "맨투맨"),
    "short sleeve": ("상의", "반소매 티셔츠"),
    "반소매": ("상의", "반소매 티셔츠"),
    "반팔": ("상의", "반소매 티셔츠"),
    "long sleeve": ("상의", "긴소매 티셔츠"),
    "긴소매": ("상의", "긴소매 티셔츠"),
    "긴팔": ("상의", "긴소매 티셔츠"),
    "티셔츠": ("상의", None),
    "셔츠": ("상의", None),
    "니트": ("상의", "니트/스웨터"),
    "스웨터": ("상의", "니트/스웨터"),
    "가디건": ("아우터", "가디건"),
    "자켓": ("아우터", None),
    "재킷": ("아우터", None),
    "레더": ("아우터", "레더자켓"),
    "가죽": ("아우터", "레더자켓"),
    "블루종": ("아우터", "블루종/MA-1"),
    "야상": ("아우터", "사파리/헌팅자켓"),
    "사파리": ("아우터", "사파리/헌팅자켓"),
    "헌팅": ("아우터", "사파리/헌팅자켓"),
    "필드자켓": ("아우터", "사파리/헌팅자켓"),
    "필드 자켓": ("아우터", "사파리/헌팅자켓"),
    "코트": ("아우터", None),
    "패딩": ("아우터", None),
    "바지": ("하의", None),
    "팬츠": ("하의", None),
    "청바지": ("하의", "데님팬츠"),
    "데님": ("하의", "데님팬츠"),
    "슬랙스": ("하의", "슬랙스/정장 팬츠"),
    "조거": ("하의", "트레이닝/조거 팬츠"),
    "카고": ("하의", "카고팬츠"),
    "반바지": ("하의", "숏팬츠"),
}

LABEL_TO_EN = {
    "상의": "top",
    "하의": "pants",
    "아우터": "outerwear",
    "후드티": "hoodie",
    "맨투맨": "sweatshirt",
    "반소매 티셔츠": "short sleeve t-shirt",
    "긴소매 티셔츠": "long sleeve t-shirt",
    "니트/스웨터": "knit sweater",
    "가디건": "cardigan",
    "레더자켓": "leather jacket",
    "블루종/MA-1": "blouson jacket",
    "사파리/헌팅자켓": "safari hunting jacket",
    "데님팬츠": "denim jeans",
    "슬랙스/정장 팬츠": "slacks trousers",
    "트레이닝/조거 팬츠": "jogger pants",
    "카고팬츠": "cargo pants",
    "숏팬츠": "shorts",
}

COLOR_ALIASES = {
    "black": {"black", "블랙", "검정", "검정색", "검은색", "검은", "까만색", "까만", "흑색", "흑청"},
    "white": {"white", "화이트", "흰색", "하얀색", "백색", "아이보리", "ivory"},
    "gray": {"gray", "grey", "그레이", "회색", "차콜", "charcoal"},
    "navy": {"navy", "네이비", "남색"},
    "blue": {"blue", "블루", "파랑", "파란색", "청색", "중청", "연청"},
    "indigo": {"indigo", "인디고", "생지", "진청", "raw denim", "dark denim"},
    "red": {"red", "레드", "빨강", "빨간색", "버건디", "burgundy", "와인"},
    "green": {"green", "그린", "초록", "초록색"},
    "khaki": {"khaki", "카키", "olive", "올리브"},
    "yellow": {"yellow", "옐로우", "노랑", "노란색"},
    "beige": {"beige", "베이지", "크림", "cream", "오트밀", "oatmeal"},
    "brown": {"brown", "브라운", "갈색", "카멜", "camel"},
    "pink": {"pink", "핑크", "분홍", "분홍색"},
    "purple": {"purple", "퍼플", "보라", "보라색"},
    "orange": {"orange", "오렌지", "주황", "주황색"},
}

MATERIAL_ALIASES = {
    "denim": {"denim", "jean", "jeans", "데님", "청바지", "흑청", "진청", "중청", "연청"},
    "leather": {"leather", "goat leather", "lambskin", "cowhide", "레더", "가죽", "고트", "램스킨"},
    "faux_leather": {"faux leather", "vegan leather", "pu leather", "비건레더", "인조가죽", "합성가죽"},
    "cotton": {"cotton", "코튼", "면"},
    "wool": {"wool", "knit", "merino", "울", "니트", "메리노"},
    "nylon": {"nylon", "나일론"},
    "polyester": {"polyester", "poly", "폴리에스터", "폴리"},
    "linen": {"linen", "린넨", "리넨"},
    "fleece": {"fleece", "플리스", "후리스"},
    "corduroy": {"corduroy", "코듀로이", "골덴"},
    "suede": {"suede", "스웨이드"},
}

FIT_ALIASES = {
    "wide": {"wide", "balloon", "와이드", "벌룬"},
    "slim": {"slim", "skinny", "슬림", "스키니"},
    "regular": {"regular", "standard", "straight", "레귤러", "스탠다드", "스트레이트"},
    "relaxed": {"relaxed", "loose", "tapered", "릴렉스", "루즈", "테이퍼드"},
    "oversized": {"oversized", "overfit", "over fit", "오버핏", "오버사이즈"},
    "cropped": {"cropped", "crop", "크롭", "크롭트"},
}

DESIGN_DETAIL_ALIASES = {
    "a2": {"a-2", "a2", "에이투", "에이 투"},
    "rider": {"rider", "riders", "라이더", "라이더스"},
    "blouson": {"blouson", "블루종", "bomber", "봄버", "ma-1", "ma1", "항공점퍼"},
    "field": {"field", "필드", "야상", "m65", "m-65"},
    "safari": {"safari", "hunting", "사파리", "헌팅"},
    "china_collar": {"china", "차이나", "stand collar", "스탠드카라"},
    "double": {"double", "더블"},
    "single": {"single", "싱글"},
    "hooded": {"hood", "hooded", "후드"},
    "wide": {"wide", "와이드", "벌룬", "balloon"},
    "straight": {"straight", "스트레이트", "일자"},
    "curved": {"curved", "curve", "커브드", "커브"},
    "cargo": {"cargo", "카고"},
    "shorts": {"shorts", "short pants", "반바지", "숏팬츠"},
    "cropped": {"cropped", "crop", "크롭", "크롭트"},
}

DESIGN_CONFLICT_GROUPS = (
    {"a2", "rider", "blouson", "field", "safari", "china_collar"},
    {"wide", "straight", "curved", "cargo", "shorts"},
    {"single", "double"},
)

DIFFERENT_COLOR_PATTERNS = ("색상이 다른", "색상 다른", "색이 다른", "다른 색", "색만 다른", "컬러가 다른", "컬러 다른")
SAME_COLOR_PATTERNS = (
    "같은 색",
    "비슷한 색",
    "비슷한 색상",
    "유사한 색",
    "유사한 색상",
    "색상은 그대로",
    "색은 그대로",
    "동일한 색",
    "같은 컬러",
    "비슷한 컬러",
    "유사한 컬러",
    "컬러는 그대로",
)
DESIGN_SIMILARITY_PATTERNS = (
    "비슷한 디자인",
    "유사한 디자인",
    "디자인 비슷",
    "디자인이 비슷",
    "비슷한 핏",
    "유사한 핏",
    "핏이 비슷",
    "비슷한 실루엣",
    "유사한 실루엣",
)
GENERIC_DESIGN_TOKENS_BY_LABEL = {
    "상의": {"top", "shirt", "clothing"},
    "하의": {"pants", "trousers", "bottoms", "clothing"},
    "아우터": {"outerwear", "jacket", "coat", "clothing"},
    "데님팬츠": {"denim", "jean", "jeans", "pants", "trousers", "denim jeans"},
    "숏팬츠": {"short", "shorts", "pants"},
    "레더자켓": {"leather", "jacket", "leather jacket"},
    "사파리/헌팅자켓": {"field", "safari", "hunting", "jacket", "field jacket", "safari hunting jacket"},
}
STRICT_SUB_CATEGORIES = {
    "반소매 티셔츠",
    "긴소매 티셔츠",
    "데님팬츠",
    "숏팬츠",
    "레더자켓",
    "블루종/MA-1",
    "사파리/헌팅자켓",
    "카고팬츠",
    "슬랙스/정장 팬츠",
    "트레이닝/조거 팬츠",
}

COLOR_RGB_CENTROIDS = {
    "black": (25, 25, 25),
    "white": (235, 235, 225),
    "gray": (125, 125, 125),
    "navy": (20, 35, 80),
    "blue": (40, 95, 180),
    "indigo": (45, 70, 115),
    "red": (175, 45, 45),
    "green": (55, 120, 70),
    "khaki": (95, 105, 65),
    "yellow": (220, 190, 65),
    "beige": (205, 180, 135),
    "brown": (105, 70, 45),
    "pink": (215, 120, 155),
    "purple": (110, 70, 145),
    "orange": (210, 115, 45),
}

DENIM_COLOR_CENTROIDS = {
    "black": (32, 32, 35),
    "gray": (95, 95, 100),
    "indigo": (38, 58, 95),
    "blue": (70, 115, 175),
}

FASHION_IMAGE_LABELS = [
    "clothing",
    "shirt",
    "pants",
    "jacket",
    "dress",
    "skirt",
    "hoodie",
    "sweater",
    "coat",
    "fashion item",
]

NON_FASHION_IMAGE_LABELS = [
    "food",
    "car",
    "landscape",
    "animal",
    "room",
    "furniture",
    "building",
    "face",
    "human face",
    "portrait",
    "selfie",
    "headshot",
    "person",
]

FACE_DOMINANT_LABELS = {"face", "human face", "portrait", "selfie", "headshot", "person"}
MIN_FASHION_IMAGE_SCORE = 0.33
MIN_FASHION_NON_FASHION_MARGIN = 0.08
MIN_FACE_REJECT_SCORE = 0.18
MIN_COLOR_PIXEL_COUNT = 80
HIGH_COLOR_RATIO = 0.45
MEDIUM_COLOR_RATIO = 0.30
DARK_DENIM_COLORS = {"black", "indigo", "navy"}
DENIM_DARK_COLOR_GROUP = {"black", "gray", "indigo", "navy"}
DENIM_BLUE_COLOR_GROUP = {"blue"}
DIFFERENT_COLOR_GROUP_LIMIT = 3
BACKGROUND_COLOR_DISTANCE = 48
SEGMENTATION_MASK_THRESHOLD = 24
MIN_SEGMENTED_PIXEL_RATIO = 0.03


def normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def infer_color_mode(query: str) -> str:
    normalized_query = normalize_text(query).replace(" ", "")
    if any(pattern.replace(" ", "") in normalized_query for pattern in DIFFERENT_COLOR_PATTERNS):
        return "different"
    if any(pattern.replace(" ", "") in normalized_query for pattern in SAME_COLOR_PATTERNS):
        return "same"
    return "ignore"


def normalize_color(value: Optional[str]) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    for canonical, aliases in COLOR_ALIASES.items():
        if text == canonical or any(alias.lower() in text for alias in aliases):
            return canonical
    return text


def normalize_attribute(value: Optional[str], aliases: dict) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    for canonical, alias_set in aliases.items():
        if text == canonical or any(alias.lower() in text for alias in alias_set):
            return canonical
    return text


def infer_attribute_from_text(text: Optional[str], aliases: dict) -> str:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    if not compact:
        return ""
    for canonical, alias_set in aliases.items():
        for alias in alias_set | {canonical}:
            alias_text = alias.lower()
            alias_compact = re.sub(r"[^a-z0-9가-힣]+", "", alias_text)
            if alias_text in normalized or (len(alias_compact) >= 2 and alias_compact in compact):
                return canonical
    return ""


def infer_design_details(text: Optional[str]) -> set[str]:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    if not compact:
        return set()

    matched = set()
    for canonical, alias_set in DESIGN_DETAIL_ALIASES.items():
        for alias in alias_set | {canonical}:
            alias_text = alias.lower()
            alias_compact = re.sub(r"[^a-z0-9가-힣]+", "", alias_text)
            if alias_text in normalized or (len(alias_compact) >= 2 and alias_compact in compact):
                matched.add(canonical)
                break
    return matched


def design_detail_score(item, query_attrs) -> tuple[float, list[str], list[str]]:
    query_details = set(query_attrs.get("design_details") or [])
    if not query_details:
        return 0.0, [], []

    candidate_text = " ".join(
        str(item.get(field) or "")
        for field in ("name", "brand_name", "main_category", "sub_category")
    )
    candidate_details = infer_design_details(candidate_text)
    if not candidate_details:
        return 0.0, [], []

    matched = sorted(query_details & candidate_details)
    conflicts = []
    for group in DESIGN_CONFLICT_GROUPS:
        query_group = query_details & group
        candidate_group = candidate_details & group
        if query_group and candidate_group and not (query_group & candidate_group):
            conflicts.extend(sorted(candidate_group))

    score = 0.0
    if matched:
        score += min(0.18, 0.10 + (0.04 * (len(matched) - 1)))
    if conflicts:
        score -= min(0.18, 0.10 + (0.04 * (len(set(conflicts)) - 1)))

    return score, matched, sorted(set(conflicts))


def color_matches_target(item_color: str, target_color: str, query_attrs=None) -> bool:
    item_color = normalize_color(item_color)
    target_color = normalize_color(target_color)
    if not item_color or not target_color:
        return False
    if item_color == target_color:
        return True

    query_attrs = query_attrs or {}
    color_confidence = normalize_text(query_attrs.get("color_confidence"))
    if (
        query_attrs.get("is_denim_context")
        and color_confidence in {"high", "medium", ""}
        and target_color in DARK_DENIM_COLORS
        and item_color in DARK_DENIM_COLORS
    ):
        return True

    return False


def color_group(color: str, query_attrs=None) -> str:
    normalized_color = normalize_color(color)
    if not normalized_color:
        return ""

    query_attrs = query_attrs or {}
    if query_attrs.get("is_denim_context"):
        if normalized_color in DENIM_DARK_COLOR_GROUP:
            return "denim_dark"
        if normalized_color in DENIM_BLUE_COLOR_GROUP:
            return "denim_blue"
        return f"denim_{normalized_color}"

    return normalized_color


def color_group_matches_target(item_color: str, target_color: str, query_attrs=None) -> bool:
    item_group = color_group(item_color, query_attrs)
    target_group = color_group(target_color, query_attrs)
    return bool(item_group and target_group and item_group == target_group)


def normalize_color_candidates(item) -> list[dict]:
    raw_candidates = item.get("color_candidates")
    if isinstance(raw_candidates, str):
        try:
            raw_candidates = json.loads(raw_candidates)
        except json.JSONDecodeError:
            raw_candidates = []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    normalized_candidates = []
    seen_colors = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        color = normalize_color(candidate.get("color"))
        if not color or color in seen_colors:
            continue
        try:
            score = float(candidate.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = min(max(score, 0.0), 1.0)
        normalized_candidates.append({
            "color": color,
            "score": score,
            "source": normalize_text(candidate.get("source")) or "unknown",
            "confidence": normalize_text(candidate.get("confidence")) or "medium",
        })
        seen_colors.add(color)

    return sorted(normalized_candidates, key=lambda candidate: candidate["score"], reverse=True)[:3]


def color_candidate_match_score(color_candidates, target_color: str, query_attrs=None) -> tuple[float, bool, bool]:
    target_color = normalize_color(target_color)
    if not target_color:
        return 0.0, False, False

    best_score = 0.0
    exact_match = False
    group_match = False
    for candidate in color_candidates or []:
        candidate_color = normalize_color(candidate.get("color"))
        if not candidate_color:
            continue
        candidate_score = float(candidate.get("score", 0.0) or 0.0)
        if color_matches_target(candidate_color, target_color, query_attrs):
            best_score = max(best_score, candidate_score)
            exact_match = True
        elif color_group_matches_target(candidate_color, target_color, query_attrs):
            best_score = max(best_score, candidate_score * 0.7)
            group_match = True

    return best_score, exact_match, group_match


def dominant_color_match_score(item_color: str, target_color: str, query_attrs=None) -> tuple[float, bool, bool]:
    if not item_color or not target_color:
        return 0.0, False, False
    if color_matches_target(item_color, target_color, query_attrs):
        return 0.7, True, False
    if color_group_matches_target(item_color, target_color, query_attrs):
        return 0.45, False, True
    return 0.0, False, False


def image_color_confidence_weight(query_attrs) -> float:
    confidence = normalize_text((query_attrs or {}).get("color_confidence"))
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.85
    if confidence == "low":
        return 0.55
    return 1.0


def extract_category_from_query(query: str):
    normalized_query = normalize_text(query)
    main_categories = []
    sub_categories = []
    for keyword, (main_category, sub_category) in CATEGORY_KEYWORDS.items():
        if keyword in normalized_query:
            if main_category and main_category not in main_categories:
                main_categories.append(main_category)
            if sub_category and sub_category not in sub_categories:
                sub_categories.append(sub_category)
    return main_categories, sub_categories


def is_denim_query_context(query: str = "", main_categories=None, sub_categories=None) -> bool:
    return is_denim_context_from_text(query, *(main_categories or []), *(sub_categories or []))


def is_design_similarity_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(pattern in normalized_query for pattern in DESIGN_SIMILARITY_PATTERNS)


def sanitize_design_terms(design: str, clothing_label: str, main_categories, sub_categories) -> str:
    design_text = normalize_text(design)
    if not design_text:
        return ""

    generic_tokens = set()
    labels = [clothing_label, *(main_categories or []), *(sub_categories or [])]
    for label in labels:
        generic_tokens.update(GENERIC_DESIGN_TOKENS_BY_LABEL.get(label, set()))

    tokens = [token for token in re.split(r"[\s,/]+", design_text) if token]
    filtered_tokens = [token for token in tokens if token not in generic_tokens]
    filtered_design = " ".join(filtered_tokens)

    if filtered_design in generic_tokens:
        return ""
    return filtered_design


def build_prompt_label(design: str, en_clothing_label: str) -> str:
    design_tokens = [token for token in re.split(r"[\s,/]+", normalize_text(design)) if token]
    label_tokens = [token for token in re.split(r"[\s,/]+", normalize_text(en_clothing_label)) if token]
    label_token_set = set(label_tokens)
    filtered_design_tokens = [token for token in design_tokens if token not in label_token_set]
    prompt_tokens = []
    for token in [*filtered_design_tokens, *label_tokens]:
        if token not in prompt_tokens:
            prompt_tokens.append(token)
    return " ".join(prompt_tokens) if prompt_tokens else "fashion item"


def classify_denim_color_from_pixels(pixels) -> str:
    if not pixels:
        return ""

    neutral_dark_count = 0
    indigo_count = 0
    blue_count = 0
    light_blue_count = 0
    blue_bias_sum = 0.0
    brightness_sum = 0.0
    r_sum = g_sum = b_sum = 0.0

    for r, g, b in pixels:
        brightness = (r + g + b) / 3
        spread = max(r, g, b) - min(r, g, b)
        blue_bias = b - max(r, g)
        brightness_sum += brightness
        blue_bias_sum += blue_bias
        r_sum += r
        g_sum += g
        b_sum += b
        if brightness < 95 and spread < 34:
            neutral_dark_count += 1
        if brightness < 135 and b >= r + 10 and b >= g - 8:
            indigo_count += 1
        if b >= r + 18 and b >= g + 2:
            blue_count += 1
            if brightness >= 145:
                light_blue_count += 1

    total = len(pixels)
    neutral_dark_ratio = neutral_dark_count / total
    indigo_ratio = indigo_count / total
    blue_ratio = blue_count / total
    light_blue_ratio = light_blue_count / total
    avg_rgb = (r_sum / total, g_sum / total, b_sum / total)
    avg_brightness = brightness_sum / total
    avg_blue_bias = blue_bias_sum / total

    if neutral_dark_ratio >= 0.24 and avg_blue_bias < 18:
        return "black"
    if indigo_ratio >= 0.22 or (avg_rgb[2] >= avg_rgb[0] + 8 and avg_rgb[2] >= avg_rgb[1] - 6 and avg_brightness < 135):
        return "indigo"
    if light_blue_ratio >= 0.20 or blue_ratio >= 0.18:
        return "blue"

    return min(
        DENIM_COLOR_CENTROIDS,
        key=lambda color: sum((avg_rgb[idx] - DENIM_COLOR_CENTROIDS[color][idx]) ** 2 for idx in range(3)),
    )


def rgb_to_xyz_component(value: float):
    value = value / 255.0
    if value > 0.04045:
        return ((value + 0.055) / 1.055) ** 2.4
    return value / 12.92


def rgb_to_lab(rgb):
    r, g, b = rgb
    r = rgb_to_xyz_component(float(r))
    g = rgb_to_xyz_component(float(g))
    b = rgb_to_xyz_component(float(b))

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(value):
        if value > 0.008856:
            return value ** (1 / 3)
        return (7.787 * value) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)
    return ((116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz))


def lab_distance(color_a, color_b):
    return sqrt(sum((a - b) ** 2 for a, b in zip(color_a, color_b)))


def classify_color_by_lab(rgb_color):
    input_lab = rgb_to_lab(rgb_color)
    return min(
        COLOR_RGB_CENTROIDS,
        key=lambda color: lab_distance(input_lab, rgb_to_lab(COLOR_RGB_CENTROIDS[color])),
    )


def is_skin_like(r: int, g: int, b: int) -> bool:
    brightness = (r + g + b) / 3
    return (
        brightness >= 90
        and r > 95 and g > 40 and b > 20
        and (max(r, g, b) - min(r, g, b)) > 15
        and abs(r - g) > 15
        and r > g and r > b
    )


def is_ignored_color_pixel(r: int, g: int, b: int) -> bool:
    if r > 242 and g > 242 and b > 242:
        return True
    if r < 12 and g < 12 and b < 12:
        return True
    return is_skin_like(r, g, b)


def iter_image_pixels(image_obj: Image.Image):
    if hasattr(image_obj, "get_flattened_data"):
        return image_obj.get_flattened_data()
    return image_obj.getdata()


def validate_basic_image_quality(image_obj: Image.Image) -> ImageQualityResult:
    width, height = image_obj.size
    if min(width, height) < 80:
        return ImageQualityResult(False, "image_too_small")

    sample = image_obj.convert("RGB").resize((96, 96))
    pixels = list(iter_image_pixels(sample))
    brightness_values = [(r + g + b) / 3 for r, g, b in pixels]
    avg_brightness = sum(brightness_values) / len(brightness_values)
    variance = sum((value - avg_brightness) ** 2 for value in brightness_values) / len(brightness_values)
    pixel_std = sqrt(variance)

    if pixel_std < 8:
        return ImageQualityResult(False, "image_has_too_little_detail")
    if avg_brightness < 25:
        return ImageQualityResult(False, "image_too_dark")
    if avg_brightness > 245:
        return ImageQualityResult(False, "image_too_bright")

    return ImageQualityResult(True)


def build_fashion_validation_result(labels, scores) -> FashionImageValidationResult:
    fashion_scores = scores[:len(FASHION_IMAGE_LABELS)]
    non_fashion_scores = scores[len(FASHION_IMAGE_LABELS):]
    fashion_score = max(fashion_scores) if fashion_scores else 0.0
    non_fashion_score = max(non_fashion_scores) if non_fashion_scores else 0.0
    face_scores = [
        score
        for label, score in zip(labels, scores)
        if label in FACE_DOMINANT_LABELS
    ]
    face_score = max(face_scores) if face_scores else 0.0
    best_index = max(range(len(scores)), key=lambda index: scores[index]) if scores else 0
    best_label = labels[best_index] if labels else ""
    margin = fashion_score - non_fashion_score
    face_dominant = (
        face_score >= MIN_FACE_REJECT_SCORE
        and face_score >= fashion_score * 0.65
    )

    reason = ""
    if face_dominant:
        reason = "face_dominant"
    elif fashion_score < MIN_FASHION_IMAGE_SCORE:
        reason = "low_fashion_score"
    elif margin < MIN_FASHION_NON_FASHION_MARGIN:
        reason = "low_fashion_margin"

    is_fashion = not reason

    return FashionImageValidationResult(
        is_fashion,
        fashion_score,
        non_fashion_score,
        best_label,
        face_score,
        reason,
    )


def validate_fashion_image(image_obj: Image.Image) -> FashionImageValidationResult:
    labels = FASHION_IMAGE_LABELS + NON_FASHION_IMAGE_LABELS
    clip_image = crop_center_region(image_obj.convert("RGB"))
    inputs = processor(images=clip_image, text=labels, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    scores = [float(prob.item()) if hasattr(prob, "item") else float(prob) for prob in probs]
    return build_fashion_validation_result(labels, scores)


def collect_border_pixels(image_obj: Image.Image, border_ratio: float = 0.08):
    image = image_obj.convert("RGB").resize((224, 224))
    width, height = image.size
    border_x = max(1, int(width * border_ratio))
    border_y = max(1, int(height * border_ratio))
    pixels = []

    for y in range(height):
        for x in range(width):
            if x >= border_x and x < width - border_x and y >= border_y and y < height - border_y:
                continue
            r, g, b = image.getpixel((x, y))
            if is_ignored_color_pixel(r, g, b):
                continue
            pixels.append((r, g, b))

    return pixels


def estimate_background_colors(image_obj: Image.Image):
    border_pixels = collect_border_pixels(image_obj)
    if len(border_pixels) < MIN_COLOR_PIXEL_COUNT:
        return []
    return [candidate["rgb"] for candidate in kmeans_color_candidates(border_pixels, n_clusters=3) if candidate["ratio"] >= 0.12]


def is_near_background_color(pixel, background_colors) -> bool:
    if not background_colors:
        return False
    threshold = BACKGROUND_COLOR_DISTANCE ** 2
    return any(squared_rgb_distance(pixel, background_color) <= threshold for background_color in background_colors)


def load_segmentation_backend() -> bool:
    global rembg_loaded, rembg_new_session, rembg_remove, segmentation_failed
    if rembg_loaded:
        return rembg_remove is not None
    rembg_loaded = True

    try:
        rembg_module = importlib.import_module("rembg")
        rembg_new_session = getattr(rembg_module, "new_session", None)
        rembg_remove = getattr(rembg_module, "remove", None)
        return rembg_remove is not None
    except ImportError:
        segmentation_failed = True
        print("Segmentation backend unavailable, install rembg to enable it.")
        return False


def get_segmentation_session():
    global segmentation_session, segmentation_failed
    if segmentation_failed or not load_segmentation_backend():
        return None
    if segmentation_session is not None:
        return segmentation_session

    try:
        segmentation_session = rembg_new_session(SEGMENTATION_MODEL_NAME) if rembg_new_session else None
        return segmentation_session
    except Exception as exc:
        segmentation_failed = True
        print(f"Segmentation session unavailable, fallback color extraction used: {exc}")
        return None


def build_segmentation_mask(image_obj: Image.Image):
    session = get_segmentation_session()
    if session is None:
        return None

    try:
        mask = rembg_remove(
            image_obj.convert("RGB"),
            only_mask=True,
            session=session,
        )
        return mask.convert("L")
    except Exception as exc:
        print(f"Segmentation failed, fallback color extraction used: {exc}")
        return None


def extract_segmented_pixels(image_obj: Image.Image):
    image = image_obj.convert("RGB").resize((224, 224))
    mask = build_segmentation_mask(image_obj)
    if mask is None:
        return []

    mask = mask.resize((224, 224))
    pixels = []
    for y in range(224):
        for x in range(224):
            if mask.getpixel((x, y)) < SEGMENTATION_MASK_THRESHOLD:
                continue
            r, g, b = image.getpixel((x, y))
            if is_ignored_color_pixel(r, g, b):
                continue
            pixels.append((r, g, b))

    min_pixels = int(224 * 224 * MIN_SEGMENTED_PIXEL_RATIO)
    return pixels if len(pixels) >= max(MIN_COLOR_PIXEL_COUNT, min_pixels) else []


def extract_background_filtered_pixels(image_obj: Image.Image):
    width, height = image_obj.size
    background_colors = estimate_background_colors(image_obj)
    cropped = image_obj.convert("RGB").crop((
        int(width * 0.12),
        int(height * 0.08),
        int(width * 0.88),
        int(height * 0.92),
    )).resize((224, 224))

    pixels = []
    fallback_pixels = []
    for r, g, b in iter_image_pixels(cropped):
        if is_ignored_color_pixel(r, g, b):
            continue

        pixel = (r, g, b)
        fallback_pixels.append(pixel)
        if is_near_background_color(pixel, background_colors):
            continue
        pixels.append(pixel)

    if len(pixels) >= MIN_COLOR_PIXEL_COUNT:
        return pixels
    return fallback_pixels


def extract_candidate_pixels(image_obj: Image.Image):
    segmented_pixels = extract_segmented_pixels(image_obj)
    if segmented_pixels:
        return segmented_pixels
    return extract_background_filtered_pixels(image_obj)


def squared_rgb_distance(left, right):
    return sum((left[index] - right[index]) ** 2 for index in range(3))


def initial_kmeans_centers(pixels, n_clusters: int):
    buckets = {}
    for r, g, b in pixels:
        key = (round(r / 32) * 32, round(g / 32) * 32, round(b / 32) * 32)
        if key not in buckets:
            buckets[key] = [0, 0, 0, 0]
        buckets[key][0] += 1
        buckets[key][1] += r
        buckets[key][2] += g
        buckets[key][3] += b

    centers = []
    for _, (count, r_sum, g_sum, b_sum) in sorted(buckets.items(), key=lambda row: row[1][0], reverse=True):
        centers.append((r_sum / count, g_sum / count, b_sum / count))
        if len(centers) >= n_clusters:
            break

    if not centers:
        return []
    while len(centers) < n_clusters:
        centers.append(centers[-1])
    return centers


def kmeans_color_candidates(pixels, n_clusters: int = 5):
    if not pixels:
        return []

    if len(pixels) > 5000:
        step = max(1, len(pixels) // 5000)
        pixels = pixels[::step]

    n_clusters = min(n_clusters, len(pixels))
    centers = initial_kmeans_centers(pixels, n_clusters)
    labels = [0] * len(pixels)

    for _ in range(8):
        changed = False
        cluster_sums = [[0, 0, 0, 0] for _ in centers]
        for index, pixel in enumerate(pixels):
            label = min(range(len(centers)), key=lambda center_index: squared_rgb_distance(pixel, centers[center_index]))
            if labels[index] != label:
                changed = True
            labels[index] = label
            cluster_sums[label][0] += 1
            cluster_sums[label][1] += pixel[0]
            cluster_sums[label][2] += pixel[1]
            cluster_sums[label][3] += pixel[2]

        for index, (count, r_sum, g_sum, b_sum) in enumerate(cluster_sums):
            if count:
                centers[index] = (r_sum / count, g_sum / count, b_sum / count)
        if not changed:
            break

    counts = [0] * len(centers)
    for label in labels:
        counts[label] += 1

    candidates = []
    total = len(labels)
    for count, center in zip(counts, centers):
        if count <= 0:
            continue
        rgb = tuple(int(round(channel)) for channel in center)
        candidates.append({"count": count, "ratio": count / total, "rgb": rgb})

    candidates.sort(key=lambda candidate: candidate["count"], reverse=True)
    return candidates


def classify_color_confidence(candidates):
    if not candidates:
        return ColorExtractionResult("", "low", "no_color_candidates")

    valid_candidates = [candidate for candidate in candidates if candidate["ratio"] >= 0.08]
    if not valid_candidates:
        return ColorExtractionResult("", "low", "only_tiny_color_clusters")

    top = valid_candidates[0]
    second_ratio = valid_candidates[1]["ratio"] if len(valid_candidates) > 1 else 0.0
    top_ratio = top["ratio"]

    if second_ratio >= 0.18 and abs(top_ratio - second_ratio) < 0.15:
        return ColorExtractionResult("multi_color", "low", "mixed_color_clusters", top_ratio, second_ratio)

    color = classify_color_by_lab(top["rgb"])
    if top_ratio >= HIGH_COLOR_RATIO and top_ratio - second_ratio >= 0.18:
        confidence = "high"
    elif top_ratio >= MEDIUM_COLOR_RATIO:
        confidence = "medium"
    else:
        confidence = "low"

    return ColorExtractionResult(color, confidence, "", top_ratio, second_ratio)


def extract_dominant_color_result(image_obj: Image.Image, denim_context: bool = False) -> ColorExtractionResult:
    quality = validate_basic_image_quality(image_obj)
    if not quality.is_usable:
        return ColorExtractionResult("", "low", quality.reason)

    pixels = extract_candidate_pixels(image_obj)
    if len(pixels) < MIN_COLOR_PIXEL_COUNT:
        return ColorExtractionResult("", "low", "not_enough_valid_pixels")

    if denim_context:
        denim_color = classify_denim_color_from_pixels(pixels)
        if denim_color:
            return ColorExtractionResult(denim_color, "high", "denim_context", 1.0, 0.0)

    candidates = kmeans_color_candidates(pixels)
    return classify_color_confidence(candidates)


def extract_dominant_color(image_obj: Image.Image, denim_context: bool = False) -> str:
    result = extract_dominant_color_result(image_obj, denim_context=denim_context)
    return result.color if result.confidence in {"high", "medium"} else ""


async def analyze_query_intent(user_query: str) -> QueryIntent:
    fallback_color = infer_attribute_from_text(user_query, COLOR_ALIASES)
    fallback_color_mode = infer_color_mode(user_query)
    if fallback_color and fallback_color_mode == "ignore":
        fallback_color_mode = "target"
    fallback = QueryIntent(
        reasoning="rule based fallback",
        color=fallback_color,
        color_mode=fallback_color_mode,
        design=" ".join(
            value for value in [
                infer_attribute_from_text(user_query, FIT_ALIASES),
                infer_attribute_from_text(user_query, MATERIAL_ALIASES),
            ] if value
        ),
    )

    if not gemini_client or genai_types is None:
        return fallback

    system_prompt = """\
You are a fashion search query analyzer for a Korean fashion app.
The user will provide a Korean query about clothing. Your job is to extract structured intent.

Fields to extract:
- reasoning: Brief explanation of your analysis (1~2 sentences in English).
- color: The target color explicitly mentioned (canonical English lowercase, e.g. "black", "navy", "beige"). Use "" if no color is mentioned.
- color_mode:
    - "target"    -> user wants items of a specific color (e.g. "검정 후드티 보여줘")
    - "same"      -> user wants the same color as a reference item (e.g. "같은 색으로 보여줘")
    - "different" -> user wants a different color (e.g. "색상이 다른 걸로 보여줘")
    - "ignore"    -> color is not relevant to the query
- design: Style, silhouette, length, fabric, or fit keywords in English, space-separated (e.g. "oversized wide-leg denim"). Use "" if absent.

Rules:
- If a color word appears but the user is not requesting that color (e.g. describing a reference), set color_mode to "same" or "ignore" appropriately.
- Do NOT invent colors not mentioned in the query.
- Return valid JSON only, no markdown.

Example output:
{"reasoning": "User wants a black oversized hoodie.", "color": "black", "color_mode": "target", "design": "oversized"}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"User query: {user_query}",
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=QueryIntent,
            ),
        )
        if getattr(response, "parsed", None):
            parsed = response.parsed
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            data = json.loads(response.text)

        intent = QueryIntent(
            reasoning=data.get("reasoning", ""),
            color=data.get("color", ""),
            color_mode=data.get("color_mode", "ignore"),
            design=data.get("design", ""),
        )
        if intent.color_mode == "ignore":
            inferred_mode = infer_color_mode(user_query)
            if inferred_mode != "ignore":
                intent.color_mode = inferred_mode
        if normalize_color(intent.color) and intent.color_mode == "ignore":
            intent.color_mode = "target"
        return intent
    except Exception as exc:
        print(f"LLM analysis failed, fallback used: {exc}")
        return fallback


def crop_center_region(image: Image.Image, width_ratio: float = 0.82, height_ratio: float = 0.92):
    width, height = image.size
    crop_width = max(1, int(width * width_ratio))
    crop_height = max(1, int(height * height_ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    return image.crop((left, top, right, bottom))


def extract_feature_tensor(model_output):
    if torch.is_tensor(model_output):
        return model_output
    for attr_name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(model_output, attr_name, None)
        if value is not None:
            if attr_name == "last_hidden_state" and value.ndim == 3:
                return value[:, 0, :]
            return value
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise TypeError(f"Cannot find feature tensor from {type(model_output)}")


def l2_normalize_array(embeddings):
    array = np.asarray(embeddings)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array / np.linalg.norm(array, ord=2, axis=-1, keepdims=True)


def encode_image_with_fashion_clip_api(image_obj: Image.Image):
    rgb_image = image_obj.convert("RGB")
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        rgb_image.save(temp_path, format="PNG")
        images = [temp_path]
        image_embeddings = fclip.encode_images(images, batch_size=32)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return torch.from_numpy(image_embeddings).to(device)


def encode_text_with_fashion_clip_api(text: str):
    texts = [text]
    text_embeddings = fclip.encode_text(texts, batch_size=32)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
    return torch.from_numpy(text_embeddings).to(device)


def get_image_embedding(image_obj: Image.Image):
    return encode_image_with_fashion_clip_api(image_obj)


def get_text_embedding(text: str):
    return encode_text_with_fashion_clip_api(text)


def attribute_similarity(left: Optional[str], right: Optional[str]) -> float:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if left_text in right_text or right_text in left_text:
        return 0.7
    left_tokens = set(re.split(r"[\s,/]+", left_text))
    right_tokens = set(re.split(r"[\s,/]+", right_text))
    left_tokens.discard("")
    right_tokens.discard("")
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def score_exact_or_unknown(candidate_value, query_value, match_bonus, mismatch_penalty=0.0):
    candidate = normalize_text(candidate_value)
    query = normalize_text(query_value)
    if not candidate or not query:
        return 0.0
    similarity = attribute_similarity(candidate, query)
    if similarity >= 1.0:
        return match_bonus
    if similarity > 0:
        return match_bonus * 0.6
    return mismatch_penalty


def build_query_attrs(
    main_categories,
    sub_categories,
    image_color: str,
    query: str,
    intent: QueryIntent,
    color_confidence: str = "",
    detected_color: str = "",
    secondary_colors=None,
    is_mixed_color: bool = False,
    pattern: str = "",
    search_color_weights=None,
):
    text_for_attributes = f"{query or ''} {intent.design or ''}"
    material = infer_attribute_from_text(text_for_attributes, MATERIAL_ALIASES)
    is_denim_context = material == "denim" or is_denim_query_context(query, main_categories, sub_categories)
    return {
        "main_category": main_categories[0] if main_categories else "",
        "sub_category": sub_categories[0] if sub_categories else "",
        "color": image_color,
        "detected_color": detected_color or image_color,
        "secondary_colors": secondary_colors or [],
        "is_mixed_color": is_mixed_color,
        "pattern": pattern,
        "search_color_weights": search_color_weights or {},
        "color_confidence": color_confidence,
        "is_denim_context": is_denim_context,
        "design_similarity_mode": is_design_similarity_query(query),
        "design_details": sorted(infer_design_details(f"{query or ''} {intent.design or ''}")),
    }


def build_enhanced_query(en_clothing_label: str, query_image_color: str, intent: QueryIntent, design_similarity_mode: bool = False):
    has_color_request = bool(intent.color.strip()) if intent.color else False
    has_color_condition = intent.color_mode in {"target", "same", "different"}
    has_design_request = bool(intent.design.strip()) if intent.design else False
    is_specific_query = has_color_request or has_color_condition or has_design_request

    if design_similarity_mode and not has_color_condition:
        enhanced_query = f"a photo of {en_clothing_label}"
        text_weight = 0.05
        image_weight = 0.95
    elif not is_specific_query:
        enhanced_query = f"a photo of {query_image_color} {en_clothing_label}" if query_image_color else f"a photo of {en_clothing_label}"
        text_weight = 0.10
        image_weight = 0.90
    elif intent.color_mode == "target" and has_color_request and not has_design_request:
        enhanced_query = f"a photo of {intent.color} {en_clothing_label}"
        text_weight = 0.45
        image_weight = 0.55
    elif intent.color_mode == "same" and not has_design_request:
        enhanced_query = f"a photo of {query_image_color} {en_clothing_label}" if query_image_color else f"a photo of {en_clothing_label}"
        text_weight = 0.25 if query_image_color else 0.15
        image_weight = 0.75 if query_image_color else 0.85
    elif intent.color_mode == "different" and not has_design_request:
        enhanced_query = f"a photo of {en_clothing_label}"
        text_weight = 0.15
        image_weight = 0.85
    elif has_design_request and not has_color_request:
        enhanced_query = f"a photo of {intent.design} {en_clothing_label}"
        text_weight = 0.25
        image_weight = 0.75
    else:
        color_prompt = intent.color if intent.color_mode == "target" else ""
        enhanced_query = f"a photo of {color_prompt} {intent.design} {en_clothing_label}".strip()
        text_weight = 0.35
        image_weight = 0.65

    return {
        "enhanced_query": enhanced_query,
        "text_weight": text_weight,
        "image_weight": image_weight,
        "is_specific_query": is_specific_query,
        "design_similarity_mode": design_similarity_mode,
    }


def should_exclude_candidate(item, item_color: str, target_color: str, color_mode: str, query_attrs):
    return False


def sub_category_score(item_sub_category, query_sub_category) -> float:
    query_text = normalize_text(query_sub_category)
    item_text = normalize_text(item_sub_category)
    if not query_text:
        return 0.0
    if item_text == query_text:
        return 0.18
    if query_text in item_text or item_text in query_text:
        return 0.10
    if query_text in STRICT_SUB_CATEGORIES:
        return -0.24
    return -0.10


def rerank_results(results, intent: QueryIntent, query_attrs, limit: int = 10):
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    target_color = normalize_color(intent.color) or normalize_color(query_attrs.get("color"))
    design_similarity_mode = bool(query_attrs.get("design_similarity_mode"))
    reranked = []

    for item in results or []:
        base_similarity = float(item.get("similarity", item.get("score", 0.0)) or 0.0)
        color_candidates = normalize_color_candidates(item)
        item_color = normalize_color(item.get("dominant_color") or item.get("color"))
        display_color = item_color or (color_candidates[0]["color"] if color_candidates else "")
        has_color_signal = bool(item_color or color_candidates)

        if should_exclude_candidate(item, item_color, target_color, color_mode, query_attrs):
            continue

        category_bonus = score_exact_or_unknown(item.get("main_category"), query_attrs.get("main_category"), 0.14, -0.18)
        sub_category_bonus = sub_category_score(item.get("sub_category"), query_attrs.get("sub_category"))
        design_adjustment, design_matches, design_conflicts = design_detail_score(item, query_attrs)

        color_adjustment = 0.0
        candidate_match_score, color_matched, color_group_matched = color_candidate_match_score(
            color_candidates,
            target_color,
            query_attrs,
        )
        dominant_match_score, dominant_color_matched, dominant_color_group_matched = dominant_color_match_score(
            item_color,
            target_color,
            query_attrs,
        )
        effective_match_score = max(candidate_match_score, dominant_match_score)
        effective_color_matched = color_matched or dominant_color_matched
        effective_group_matched = color_group_matched or dominant_color_group_matched
        if not normalize_color(intent.color):
            effective_match_score *= image_color_confidence_weight(query_attrs)
        item_color_group = color_group(display_color, query_attrs)
        if color_mode == "target" and target_color:
            color_adjustment = 0.20 * effective_match_score if effective_color_matched or effective_group_matched else (-0.12 if has_color_signal else 0.0)
        elif color_mode == "same" and target_color:
            color_adjustment = 0.18 * effective_match_score if effective_color_matched or effective_group_matched else (-0.16 if has_color_signal else 0.0)
        elif color_mode == "different" and target_color:
            if effective_color_matched or effective_group_matched:
                color_adjustment = -0.32 * max(effective_match_score, 0.5)
            else:
                color_adjustment = 0.22 if has_color_signal else 0.0
        elif color_mode == "ignore" and target_color and effective_color_matched and not design_similarity_mode:
            color_adjustment = 0.04 * effective_match_score

        final_score = base_similarity + category_bonus + sub_category_bonus + color_adjustment + design_adjustment
        item["_ranking"] = {
            "base_similarity": round(base_similarity, 4),
            "final_score": round(final_score, 4),
            "color_mode": color_mode,
            "target_color": target_color,
            "candidate_color": display_color,
            "candidate_dominant_color": item_color,
            "candidate_color_candidates": color_candidates,
            "candidate_color_group": item_color_group,
            "target_color_group": color_group(target_color, query_attrs),
            "color_candidate_match_score": round(candidate_match_score, 4),
            "dominant_color_match_score": round(dominant_match_score, 4),
            "effective_color_match_score": round(effective_match_score, 4),
            "color_matched_by_candidates": color_matched,
            "color_matched_by_dominant": dominant_color_matched,
            "category_bonus": round(category_bonus, 4),
            "sub_category_bonus": round(sub_category_bonus, 4),
            "color_adjustment": round(color_adjustment, 4),
            "design_adjustment": round(design_adjustment, 4),
            "design_matches": design_matches,
            "design_conflicts": design_conflicts,
            "design_similarity_mode": design_similarity_mode,
        }
        reranked.append((final_score, item))

    reranked.sort(key=lambda row: row[0], reverse=True)
    if color_mode != "different" or not target_color:
        return [item for _, item in reranked[:limit]]

    selected = []
    skipped = []
    group_counts = {}
    for _, item in reranked:
        item_group = item.get("_ranking", {}).get("candidate_color_group") or "unknown"
        if item_group != "unknown" and group_counts.get(item_group, 0) >= DIFFERENT_COLOR_GROUP_LIMIT:
            skipped.append(item)
            continue
        selected.append(item)
        group_counts[item_group] = group_counts.get(item_group, 0) + 1
        if len(selected) >= limit:
            return selected

    for item in skipped:
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def log_search_debug(
    query: str,
    intent: QueryIntent,
    main_categories,
    sub_categories,
    query_image_color: str,
    color_confidence: str,
    enhanced_query: str,
    image_weight: float,
    text_weight: float,
    design_similarity_mode: bool,
    threshold: float,
    rpc_filters,
    raw_result_count: int,
    results,
):
    intent_payload = intent.model_dump() if hasattr(intent, "model_dump") else intent.dict()
    print("\n[FashionCLIP Search Debug]")
    print(f"original_query={query}")
    print(f"intent={json.dumps(intent_payload, ensure_ascii=False)}")
    print(f"main_categories={main_categories}")
    print(f"sub_categories={sub_categories}")
    print(f"query_image_color={query_image_color}")
    print(f"color_confidence={color_confidence}")
    print(f"enhanced_query={enhanced_query}")
    print(f"design_similarity_mode={design_similarity_mode}")
    print(f"image_weight={image_weight}, text_weight={text_weight}, threshold={threshold}")
    print(f"rpc_filters={json.dumps(rpc_filters, ensure_ascii=False)}")
    print(f"raw_result_count={raw_result_count}")
    for index, item in enumerate(results or [], 1):
        ranking = item.get("_ranking", {})
        print(
            f"result[{index}] "
            f"name={item.get('name')}, "
            f"similarity={item.get('similarity', item.get('score'))}, "
            f"final_score={ranking.get('final_score')}, "
            f"design_adjustment={ranking.get('design_adjustment')}, "
            f"design_matches={ranking.get('design_matches')}, "
            f"design_conflicts={ranking.get('design_conflicts')}, "
            f"candidate_color={ranking.get('candidate_color')}, "
            f"main_category={item.get('main_category')}, "
            f"sub_category={item.get('sub_category')}"
        )


@app.post("/search")
async def search_clothes(file: UploadFile = File(None), query: str = Form(None)):
    if not file or not query:
        raise HTTPException(status_code=400, detail="image and query are required")

    try:
        content = await file.read()
        try:
            image_obj = Image.open(io.BytesIO(content)).convert("RGB")
        except UnidentifiedImageError:
            print(
                "Invalid upload image: "
                f"filename={getattr(file, 'filename', '')}, "
                f"content_type={getattr(file, 'content_type', '')}, "
                f"size={len(content)}, "
                f"head={content[:16].hex()}"
            )
            raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다. JPG 또는 PNG 이미지로 다시 선택해주세요.")

        fashion_validation = validate_fashion_image(image_obj)
        if not fashion_validation.is_fashion:
            raise HTTPException(status_code=400, detail="\uc758\ub958\uac00 \uba85\ud655\ud788 \ubcf4\uc774\ub294 \uc774\ubbf8\uc9c0\ub97c \uc5c5\ub85c\ub4dc\ud574\uc8fc\uc138\uc694.")

        main_categories, sub_categories = extract_category_from_query(query)

        if sub_categories:
            clothing_label = sub_categories[0]
        elif main_categories:
            clothing_label = main_categories[0]
        else:
            clothing_label = "clothing"

        en_clothing_label = LABEL_TO_EN.get(clothing_label, "fashion item")
        intent = await analyze_query_intent(query)
        intent.design = sanitize_design_terms(intent.design, clothing_label, main_categories, sub_categories)
        design_similarity_mode = is_design_similarity_query(query)
        color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"

        query_image_color = ""
        color_result = ColorExtractionResult("", "skipped", "color_not_requested")
        pattern_context_text = f"{query or ''} {intent.design or ''}"
        should_extract_image_attributes = (
            color_mode in {"same", "different"}
            or should_run_pattern_classifier(pattern_context_text)
        )
        if should_extract_image_attributes:
            denim_context = is_denim_query_context(query, main_categories, sub_categories)
            color_result = extract_dominant_color_result_v2(
                image_obj,
                denim_context=denim_context,
                pattern_context_text=pattern_context_text,
            )
            if color_mode in {"same", "different"}:
                query_image_color = color_result.color if color_result.confidence in {"high", "medium"} else ""

        query_build = build_enhanced_query(
            en_clothing_label,
            query_image_color if color_result.confidence in {"high", "medium"} else "",
            intent,
            design_similarity_mode,
        )
        enhanced_query = query_build["enhanced_query"]
        text_weight = query_build["text_weight"]
        image_weight = query_build["image_weight"]
        is_specific_query = query_build["is_specific_query"]

        image_features = get_image_embedding(image_obj)
        text_features = get_text_embedding(enhanced_query)
        query_embedding = F.normalize((image_features * image_weight) + (text_features * text_weight), p=2, dim=-1)
        query_embedding_list = query_embedding.squeeze().tolist()

        threshold = 0.25 if design_similarity_mode else (0.30 if is_specific_query else 0.35)
        match_count = 200 if color_mode == "different" else 100
        rpc_filters = {
            "filter_main_categories": main_categories if main_categories else None,
            "filter_sub_categories": sub_categories if sub_categories else None,
        }
        response = supabase.rpc("match_clothes_fashion", {
            "query_embedding": query_embedding_list,
            "match_threshold": threshold,
            "match_count": match_count,
            **rpc_filters,
        }).execute()
        if design_similarity_mode and not response.data and sub_categories:
            threshold = 0.20
            response = supabase.rpc("match_clothes_fashion", {
                "query_embedding": query_embedding_list,
                "match_threshold": threshold,
                "match_count": match_count,
                **rpc_filters,
            }).execute()
        if design_similarity_mode and not response.data and sub_categories and main_categories:
            rpc_filters = {
                "filter_main_categories": main_categories,
                "filter_sub_categories": None,
            }
            response = supabase.rpc("match_clothes_fashion", {
                "query_embedding": query_embedding_list,
                "match_threshold": threshold,
                "match_count": match_count,
                **rpc_filters,
            }).execute()

        ranking_image_color = color_result.color if color_mode in {"same", "different"} else query_image_color
        query_attrs = build_query_attrs(
            main_categories,
            sub_categories,
            ranking_image_color,
            query,
            intent,
            color_result.confidence,
            color_result.color,
            color_result.secondary_colors,
            color_result.is_mixed_color,
            color_result.pattern,
            color_result.search_color_weights,
        )
        results = rerank_results(response.data, intent, query_attrs, limit=10)
        log_search_debug(
            query=query,
            intent=intent,
            main_categories=main_categories,
            sub_categories=sub_categories,
            query_image_color=query_image_color,
            color_confidence=color_result.confidence,
            enhanced_query=enhanced_query,
            image_weight=image_weight,
            text_weight=text_weight,
            design_similarity_mode=design_similarity_mode,
            threshold=threshold,
            rpc_filters=rpc_filters,
            raw_result_count=len(response.data or []),
            results=results,
        )

        return {
            "message": "Success",
            "model": FASHION_CLIP_MODEL_ID,
            "enhanced_query": enhanced_query,
            "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.dict(),
            "query_image_attributes": query_attrs,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("FashionCLIP search server error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
