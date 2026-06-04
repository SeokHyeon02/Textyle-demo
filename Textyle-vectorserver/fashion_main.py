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
    lab_distance as named_lab_distance,
    nearest_named_color,
    should_run_pattern_classifier,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")
FASHION_CLIP_MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
FASHION_CLIP_API_MODEL_ID = os.environ.get("FASHION_CLIP_API_MODEL_ID", "fashion-clip")
SEGMENTATION_MODEL_NAME = os.environ.get("SEGMENTATION_MODEL_NAME", "u2net_cloth_seg")
GROUNDING_DINO_MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "")
DINO_SAM_DEVICE = os.environ.get("DINO_SAM_DEVICE", "")
DINO_SAM_BOX_THRESHOLD = float(os.environ.get("DINO_SAM_BOX_THRESHOLD", "0.28"))
DINO_SAM_TEXT_THRESHOLD = float(os.environ.get("DINO_SAM_TEXT_THRESHOLD", "0.25"))
DINO_SAM_ENABLED = os.environ.get("TEXTYLE_ENABLE_DINO_SAM", "1").lower() not in {"0", "false", "no"}

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and genai else None
rembg_new_session = None
rembg_remove = None
rembg_loaded = False
segmentation_session = None
segmentation_failed = False
dino_sam_backend = None
dino_sam_failed = False

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


class QueryAnalysis(BaseModel):
    reasoning: str = Field(description="query analysis reasoning")
    color: str = Field(description="target color, empty string if absent")
    color_mode: str = Field(description="target, same, different, or ignore")
    design: str = Field(description="style, length, detail, or design phrase in English")
    main_category: str = Field(description="상의 | 하의 | 아우터, empty string if unclear")
    sub_category: str = Field(description="세부 카테고리 (e.g. 후드티, 긴소매 티셔츠), empty string if unclear")


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


@dataclass
class ImageCategoryResult:
    category: str = ""
    confidence: str = "skipped"
    scores: list[dict] = field(default_factory=list)


@dataclass
class QueryImagePreprocessResult:
    image: Image.Image
    source: str = "original"
    prompt: str = ""
    bbox: tuple[int, int, int, int] | None = None
    detection_score: float = 0.0
    sam_score: float = 0.0
    mask_ratio: float = 0.0
    error: str = ""


CATEGORY_KEYWORDS = {
    "후드티": ("상의", "후드티"),
    "후디": ("상의", "후드티"),
    "맨투맨": ("상의", "스웻셔츠"),
    "스웻셔츠": ("상의", "스웻셔츠"),
    "스웨트셔츠": ("상의", "스웻셔츠"),
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
    "후드집업": ("아우터", "후드집업"),
    "바지": ("하의", None),
    "팬츠": ("하의", None),
    "청바지": ("하의", "데님팬츠"),
    "데님": ("하의", "데님팬츠"),
    "슬랙스": ("하의", "슬랙스/슈트 팬츠"),
    "조거": ("하의", "트레이닝/조거 팬츠"),
    "코튼 팬츠": ("하의", "코튼 팬츠"),
    "코튼팬츠": ("하의", "코튼 팬츠"),
    "면바지": ("하의", "코튼 팬츠"),
    "카고": ("하의", None),
    "반바지": ("하의", "숏팬츠"),
}

LABEL_TO_EN = {
    "상의": "top",
    "하의": "pants",
    "아우터": "outerwear",
    "후드티": "hoodie",
    "스웻셔츠": "sweatshirt",
    "맨투맨": "sweatshirt",
    "반소매 티셔츠": "short sleeve t-shirt",
    "긴소매 티셔츠": "long sleeve t-shirt",
    "니트/스웨터": "knit sweater",
    "가디건": "cardigan",
    "레더자켓": "leather jacket",
    "블루종/MA-1": "blouson jacket",
    "사파리/헌팅자켓": "safari hunting jacket",
    "데님팬츠": "denim jeans",
    "슬랙스/슈트 팬츠": "slacks trousers",
    "슬랙스/정장 팬츠": "slacks trousers",
    "트레이닝/조거 팬츠": "jogger pants",
    "코튼 팬츠": "cotton pants",
    "카고팬츠": "cargo pants",
    "숏팬츠": "shorts",
    "후드집업": "hooded zip-up jacket",
}

COLOR_ALIASES = {
    "black": {"black", "블랙", "검정", "검정색", "검은색", "검은", "까만색", "까만", "흑색", "흑청"},
    "white": {"white", "화이트", "흰색", "하얀색", "백색", "아이보리", "ivory", "cream", "크림", "beige", "베이지", "ecru", "에크루", "oatmeal", "오트밀"},
    "gray": {"gray", "grey", "그레이", "회색", "차콜", "charcoal"},
    "blue": {"blue", "블루", "파랑", "파란", "파란색", "청색", "중청", "연청", "navy", "네이비", "남색", "indigo", "인디고", "생지", "진청", "raw denim", "dark denim", "dark blue", "다크블루"},
    "red": {"red", "레드", "빨강", "빨간색", "버건디", "burgundy", "와인"},
    "green": {"green", "그린", "초록", "초록색", "khaki", "카키", "olive", "올리브", "sage", "세이지"},
    "yellow": {"yellow", "옐로우", "노랑", "노란색"},
    "brown": {"brown", "브라운", "갈색", "카멜", "camel"},
    "pink": {"pink", "핑크", "분홍", "분홍색"},
    "purple": {"purple", "퍼플", "보라", "보라색"},
    "orange": {"orange", "오렌지", "주황", "주황색"},
}

COLOR_BLOCKED_TERMS = {
    "blue": {"블루종"},
    "yellow": {"yellowtin", "옐로우틴"},
}

MATERIAL_ALIASES = {
    "denim": {"denim", "jean", "jeans", "데님", "청바지", "흑청", "진청", "중청", "연청", "생지", "인디고"},
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

DESIGN_DETAIL_PROMPTS = {
    "wide": ("wide leg pants", "baggy wide pants", "loose wide denim jeans"),
    "straight": ("straight fit pants", "straight leg jeans", "regular straight pants"),
    "curved": ("curved leg pants", "barrel leg jeans", "curved silhouette pants"),
    "cargo": ("cargo pants with pockets", "utility cargo pants"),
    "shorts": ("short pants", "denim shorts"),
    "cropped": ("cropped pants", "ankle length cropped pants"),
    "hooded": ("hooded jacket", "hoodie with hood"),
    "blouson": ("blouson jacket", "bomber jacket"),
    "rider": ("rider leather jacket", "biker jacket"),
    "field": ("field jacket", "military field jacket"),
    "safari": ("safari jacket", "hunting jacket"),
    "double": ("double breasted jacket", "double button coat"),
    "single": ("single breasted jacket", "single button jacket"),
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
EXCLUDED_COLOR_PATTERNS = ("말고", "빼고", "제외", "제외하고", "아닌", "not", "except", "without")
DESIGN_SIMILARITY_PATTERNS = (
    "이 디자인",
    "이미지의 디자인",
    "사진의 디자인",
    "이미지 디자인",
    "이런 디자인",
    "해당 디자인",
    "같은 디자인",
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
    "코튼 팬츠": {"cotton", "pants", "cotton pants"},
    "스웻셔츠": {"sweatshirt", "sweat shirt"},
    "후드집업": {"hood", "hooded", "zip", "zip-up", "jacket"},
    "레더자켓": {"leather", "jacket", "leather jacket"},
    "사파리/헌팅자켓": {"field", "safari", "hunting", "jacket", "field jacket", "safari hunting jacket"},
}
GENERIC_DESIGN_INSTRUCTION_TOKENS = {"similar"}
STRICT_SUB_CATEGORIES = {
    "반소매 티셔츠",
    "긴소매 티셔츠",
    "데님팬츠",
    "숏팬츠",
    "레더자켓",
    "블루종/MA-1",
    "사파리/헌팅자켓",
    "슬랙스/슈트 팬츠",
    "트레이닝/조거 팬츠",
    "코튼 팬츠",
    "스웻셔츠",
    "후드집업",
}

IMAGE_CATEGORY_PROMPTS = {
    "데님팬츠": (
        "a photo of denim jeans",
        "a photo of black denim jeans",
        "a photo of wide denim pants",
    ),
    "슬랙스/슈트 팬츠": (
        "a photo of slacks trousers",
        "a photo of suit trousers",
        "a photo of wide slacks",
    ),
    "코튼 팬츠": (
        "a photo of cotton pants",
        "a photo of chino pants",
    ),
    "트레이닝/조거 팬츠": (
        "a photo of jogger pants",
        "a photo of sweatpants",
    ),
    "숏팬츠": (
        "a photo of shorts",
        "a photo of short pants",
    ),
}
IMAGE_CATEGORY_PROMPT_EMBEDDING_CACHE = {}

GROUNDING_DINO_PROMPTS_BY_LABEL = {
    "데님팬츠": "denim jeans",
    "슬랙스/슈트 팬츠": "slacks trousers",
    "슬랙스/정장 팬츠": "slacks trousers",
    "코튼 팬츠": "cotton pants",
    "트레이닝/조거 팬츠": "jogger pants",
    "숏팬츠": "shorts",
    "레더자켓": "leather jacket",
    "블루종/MA-1": "bomber jacket",
    "사파리/헌팅자켓": "field jacket",
    "후드집업": "hooded zip-up jacket",
    "후드티": "hoodie",
    "스웻셔츠": "sweatshirt",
    "반소매 티셔츠": "t-shirt",
    "긴소매 티셔츠": "long sleeve t-shirt",
    "니트/스웨터": "sweater",
    "가디건": "cardigan",
    "상의": "upper clothing",
    "하의": "pants",
    "아우터": "outerwear jacket",
}

GROUNDING_DINO_FALLBACK_PROMPTS_BY_PRIMARY = {
    "leather jacket": (
        "leather jacket",
        "jacket. outerwear. coat. clothing",
        "black leather jacket. glossy jacket. folded jacket",
    ),
    "denim jeans": (
        "denim jeans",
        "jeans. pants. trousers. clothing",
        "black denim jeans. gray denim jeans. folded pants",
    ),
    "slacks trousers": (
        "slacks trousers",
        "pants. trousers. clothing",
    ),
    "cotton pants": (
        "cotton pants",
        "pants. trousers. clothing",
    ),
    "jogger pants": (
        "jogger pants",
        "pants. trousers. clothing",
    ),
    "shorts": (
        "shorts",
        "short pants. pants. clothing",
    ),
    "bomber jacket": (
        "bomber jacket",
        "jacket. outerwear. clothing",
    ),
    "field jacket": (
        "field jacket",
        "jacket. outerwear. coat. clothing",
    ),
    "hooded zip-up jacket": (
        "hooded zip-up jacket",
        "jacket. hoodie. outerwear. clothing",
    ),
    "outerwear jacket": (
        "outerwear jacket",
        "jacket. coat. clothing",
    ),
    "pants": (
        "pants",
        "trousers. bottoms. clothing",
    ),
    "upper clothing": (
        "upper clothing",
        "shirt. top. clothing",
    ),
}

COLOR_RGB_CENTROIDS = {
    "black": (25, 25, 25),
    "white": (235, 235, 225),
    "gray": (125, 125, 125),
    "blue": (40, 95, 180),
    "red": (175, 45, 45),
    "green": (55, 120, 70),
    "yellow": (220, 190, 65),
    "brown": (105, 70, 45),
    "pink": (215, 120, 155),
    "purple": (110, 70, 145),
    "orange": (210, 115, 45),
}

DENIM_COLOR_CENTROIDS = {
    "black": (32, 32, 35),
    "gray": (95, 95, 100),
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
DARK_DENIM_COLORS = {"black", "gray"}
DENIM_DARK_COLOR_GROUP = {"black", "gray"}
DENIM_BLUE_COLOR_GROUP = {"blue"}
DIFFERENT_COLOR_GROUP_LIMIT = 3
SAME_COLOR_MATCH_COUNT = 220
DESIGN_SIMILARITY_MATCH_COUNT = 140
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


def infer_excluded_color_from_text(text: Optional[str]) -> str:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    if not compact:
        return ""
    exclusion_tokens = {
        re.sub(r"[^a-z0-9가-힣]+", "", token.lower())
        for token in EXCLUDED_COLOR_PATTERNS
    }
    for canonical, alias_set in COLOR_ALIASES.items():
        for alias in alias_set | {canonical}:
            alias_compact = re.sub(r"[^a-z0-9가-힣]+", "", alias.lower())
            if not alias_compact:
                continue
            for exclusion in exclusion_tokens:
                if not exclusion:
                    continue
                if f"{alias_compact}{exclusion}" in compact or f"{exclusion}{alias_compact}" in compact:
                    return canonical
    return ""


def apply_excluded_color_phrase(query: str, intent: QueryIntent) -> QueryIntent:
    excluded_color = infer_excluded_color_from_text(query)
    intent_color = normalize_color(intent.color)
    if (
        excluded_color
        and intent_color == excluded_color
        and intent.color_mode in {"different", "target"}
    ):
        return QueryIntent(
            reasoning=f"{intent.reasoning} excluded_color={excluded_color}".strip(),
            color="",
            color_mode="different",
            design=intent.design,
        )
    return intent


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


def infer_color_from_text(text: Optional[str]) -> str:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    if not compact:
        return ""
    for canonical, alias_set in COLOR_ALIASES.items():
        blocked_terms = COLOR_BLOCKED_TERMS.get(canonical, set())
        for alias in alias_set | {canonical}:
            alias_text = alias.lower()
            alias_compact = re.sub(r"[^a-z0-9가-힣]+", "", alias_text)
            matched = alias_text in normalized or (len(alias_compact) >= 2 and alias_compact in compact)
            if not matched:
                continue
            blocked = False
            for term in blocked_terms:
                term_text = term.lower()
                term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
                if alias_text in term_text and term_text in normalized:
                    blocked = True
                    break
                if alias_compact and alias_compact in term_compact and term_compact in compact:
                    blocked = True
                    break
            if not blocked:
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
    if (
        item_color == target_color
        and not (
            (query_attrs or {}).get("is_denim_context")
            and normalize_color((query_attrs or {}).get("color")) == "gray"
            and item_color != "gray"
        )
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


KOREAN_DENIM_TONE_OVERRIDES = (
    ("black", ("흑청",)),
    ("dark_blue", ("진청",)),
    ("medium", ("중청",)),
    ("light", ("연청",)),
)

LIGHT_DENIM_TONE_TERMS = {
    "light blue",
    "l.blue",
    "l denim",
    "l.denim",
    "light denim",
    "light indigo",
    "light wash",
    "light washed",
    "light vintage",
    "salt washed",
    "vintage light",
    "washed light",
    "bleach",
    "bleached",
    "ice blue",
    "연청",
    "라이트",
    "라이트 블루",
    "라이트블루",
    "라이트 인디고",
    "라이트인디고",
    "라이트 워싱",
    "밝은 블루",
    "밝은 인디고",
    "밝은청",
    "연한 청",
    "연한청",
    "블리치",
}

MEDIUM_DENIM_TONE_TERMS = {
    "medium blue",
    "mediumblue",
    "medium denim",
    "medium indigo",
    "medium wash",
    "mid blue",
    "midblue",
    "mid denim",
    "mid wash",
    "m.blue",
    "중청",
    "미디엄 블루",
    "미디엄블루",
    "미디엄 인디고",
    "미디엄인디고",
    "중간청",
}

DARK_BLUE_DENIM_TONE_TERMS = {
    "dark blue",
    "darkblue",
    "d.blue",
    "dblue",
    "deep blue",
    "deepblue",
    "dark denim",
    "deep denim",
    "진청",
    "딥블루",
    "다크블루",
    "다크 블루",
    "어두운 청",
    "어두운청",
}

DARK_DENIM_TONE_TERMS = {
    "dark blue",
    "d.blue",
    "dark denim",
    "dark indigo",
    "raw denim",
    "one washed",
    "black",
    "oil black",
    "washed black",
    "deep blue",
    "진청",
    "흑청",
    "생지",
    "다크",
    "다크블루",
    "딥블루",
    "인디고 다크",
    "블랙",
    "오일 블랙",
}


BLACK_DENIM_TONE_TERMS = {
    "black denim",
    "blackdenim",
    "washed black",
    "oil black",
    "raw black",
    "흑청",
    "블랙 데님",
    "블랙데님",
    "오일 블랙",
}


INDIGO_DENIM_TONE_TERMS = {
    "raw denim",
    "rawdenim",
    "raw indigo",
    "rawindigo",
    "indigo",
    "dark indigo",
    "deep indigo",
    "one wash",
    "one washed",
    "생지",
    "인디고",
    "딥 인디고",
}


def denim_tone_from_reason(reason: str) -> str:
    reason = normalize_text(reason)
    if "denim_context_light" in reason:
        return "light"
    if "denim_context_dark" in reason:
        return "dark"
    if "denim_context_medium" in reason:
        return "medium"
    return ""


def infer_denim_tone_from_text(text: str) -> str:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    for tone, terms in KOREAN_DENIM_TONE_OVERRIDES:
        if any(term in normalized or term in compact for term in terms):
            return tone
    for term in BLACK_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "black"
    for term in INDIGO_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "indigo"
    for term in DARK_BLUE_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "dark_blue"
    for term in LIGHT_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "light"
    for term in MEDIUM_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "medium"
    for term in DARK_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "dark"
    return ""


NEW_DENIM_TONES = {
    "white",
    "light_blue",
    "mid_blue",
    "dark_blue",
    "black",
    "indigo",
    "gray",
    "brown",
}


def normalize_candidate_denim_tone(value: str) -> str:
    tone = normalize_text(value)
    aliases = {
        "lightblue": "light_blue",
        "light blue": "light_blue",
        "midblue": "mid_blue",
        "mediumblue": "mid_blue",
        "medium blue": "mid_blue",
        "darkblue": "dark_blue",
        "dark blue": "dark_blue",
    }
    tone = aliases.get(tone, tone)
    return tone if tone in NEW_DENIM_TONES or tone in {"light", "medium", "dark"} else ""


def compact_denim_tone_text(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    return normalized, compact


def has_text_or_compact_term(normalized: str, compact: str, term: str) -> bool:
    term_text = normalize_text(term)
    term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
    return term_text in normalized or (term_compact and term_compact in compact)


def candidate_indigo_name_modifier_tone(text: str) -> str:
    normalized, compact = compact_denim_tone_text(text)
    if not has_text_or_compact_term(normalized, compact, "indigo") and not has_text_or_compact_term(
        normalized,
        compact,
        "\uc778\ub514\uace0",
    ):
        return ""

    raw_terms = (
        "raw indigo",
        "raw denim",
        "one wash",
        "one washed",
        "deep indigo",
        "\uc0dd\uc9c0",
        "\ub525 \uc778\ub514\uace0",
        "\uc6d0\uc6cc\uc2dc",
    )
    if any(has_text_or_compact_term(normalized, compact, term) for term in raw_terms):
        return "indigo"

    medium_terms = (
        "medium indigo",
        "indigo medium",
        "mid indigo",
        "m tone indigo",
        "m-tone indigo",
        "mton indigo",
        "mton",
        "m tone",
        "m-tone",
        "m.blue",
        "medium wash",
        "\uc778\ub514\uace0 \ubbf8\ub4d0",
        "\ubbf8\ub4d0 \uc778\ub514\uace0",
        "\uc778\ub514\uace0 \ubbf8\ub514\uc5c4",
        "\ubbf8\ub514\uc5c4 \uc778\ub514\uace0",
        "m\ud1a4",
        "m\ud1a4 \uc778\ub514\uace0",
        "\uc911\uccad",
    )
    if any(has_text_or_compact_term(normalized, compact, term) for term in medium_terms):
        return "mid_blue"

    dark_terms = (
        "dark indigo",
        "indigo dark",
        "dark blue indigo",
        "\uc778\ub514\uace0 \ub2e4\ud06c",
        "\ub2e4\ud06c \uc778\ub514\uace0",
        "\uc9c4\uccad",
    )
    if any(has_text_or_compact_term(normalized, compact, term) for term in dark_terms):
        return "dark_blue"

    return ""


def get_candidate_denim_tone(item) -> str:
    stored_tone = normalize_candidate_denim_tone(item.get("denim_tone") or "")
    candidate_text = " ".join(
        str(item.get(field) or "")
        for field in ("name", "brand_name", "sub_category")
    )
    if stored_tone == "indigo":
        name_modifier_tone = candidate_indigo_name_modifier_tone(candidate_text)
        if name_modifier_tone:
            return name_modifier_tone
    if stored_tone:
        return stored_tone

    inferred_tone = infer_denim_tone_from_text(candidate_text)
    if inferred_tone == "medium":
        return "mid_blue"
    if inferred_tone == "light":
        return "light_blue"
    return inferred_tone



def denim_tone_adjustment(item, query_attrs, color_mode: str) -> tuple[float, str]:
    query_attrs = query_attrs or {}
    explicit_text_tone = normalize_text(query_attrs.get("denim_tone_source")) == "text"
    if not query_attrs.get("is_denim_context"):
        return 0.0, ""
    if color_mode != "same" and not (explicit_text_tone and color_mode in {"target", "ignore"}):
        return 0.0, ""
    target_tone = query_attrs.get("denim_tone") or ""
    if target_tone not in {"dark", "light", "medium", "dark_blue", "black", "indigo"}:
        return 0.0, ""

    tone_scale = 1.0 if color_mode == "same" else 0.85
    candidate_tone = get_candidate_denim_tone(item)
    if not candidate_tone:
        return 0.0, ""

    if target_tone == "dark":
        if normalize_color(query_attrs.get("color")) == "gray" and candidate_tone in {"black", "dark_blue", "indigo", "dark"}:
            return 0.0, candidate_tone
        if candidate_tone in {"black", "dark_blue", "indigo", "dark"}:
            return 0.18 * tone_scale, candidate_tone
        if candidate_tone == "gray":
            return 0.0, candidate_tone
        if candidate_tone == "mid_blue":
            return -0.06 * tone_scale, candidate_tone
        if candidate_tone in {"light_blue", "light", "white", "brown"}:
            return -0.15 * tone_scale, candidate_tone

    if target_tone == "black":
        if candidate_tone == "black":
            return 0.22 * tone_scale, candidate_tone
        if candidate_tone in {"dark_blue", "indigo", "dark"}:
            return -0.05 * tone_scale, candidate_tone
        if candidate_tone == "gray":
            return -0.06 * tone_scale, candidate_tone
        if candidate_tone in {"mid_blue", "medium", "light_blue", "light", "white", "brown"}:
            return -0.15 * tone_scale, candidate_tone

    if target_tone == "indigo":
        if candidate_tone == "indigo":
            return 0.18 * tone_scale, candidate_tone
        if candidate_tone in {"dark_blue", "dark"}:
            return 0.08 * tone_scale, candidate_tone
        if candidate_tone == "black":
            return -0.04 * tone_scale, candidate_tone
        if candidate_tone == "mid_blue":
            return -0.03 * tone_scale, candidate_tone
        if candidate_tone in {"light_blue", "light", "white", "gray", "brown"}:
            return -0.12 * tone_scale, candidate_tone

    if target_tone == "dark_blue":
        if candidate_tone in {"dark_blue", "dark"}:
            return 0.14 * tone_scale, candidate_tone
        if candidate_tone == "indigo":
            return 0.06 * tone_scale, candidate_tone
        if candidate_tone in {"mid_blue", "medium"}:
            return -0.03 * tone_scale, candidate_tone
        if candidate_tone == "black":
            return -0.10 * tone_scale, candidate_tone
        if candidate_tone in {"light_blue", "light", "white", "gray", "brown"}:
            return -0.16 * tone_scale, candidate_tone

    if target_tone == "light":
        if candidate_tone in {"light_blue", "light"}:
            return 0.12 * tone_scale, candidate_tone
        if candidate_tone == "white":
            return 0.04 * tone_scale, candidate_tone
        if candidate_tone == "mid_blue":
            return -0.03 * tone_scale, candidate_tone
        if candidate_tone in {"dark_blue", "black", "indigo", "dark"}:
            return -0.24 * tone_scale, candidate_tone
        if candidate_tone in {"gray", "brown"}:
            return -0.08 * tone_scale, candidate_tone

    if target_tone == "medium":
        if candidate_tone in {"mid_blue", "medium"}:
            return 0.08 * tone_scale, candidate_tone
        if candidate_tone == "indigo":
            return -0.12 * tone_scale, candidate_tone
        if candidate_tone in {"light", "light_blue"}:
            return -0.08 * tone_scale, candidate_tone
        if candidate_tone in {"dark", "dark_blue", "black"}:
            return -0.12 * tone_scale, candidate_tone

    if candidate_tone == target_tone:
        return 0.08 * tone_scale, candidate_tone
    return 0.0, candidate_tone


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
        normalized_candidate = {
            "color": color,
            "score": score,
            "source": normalize_text(candidate.get("source")) or "unknown",
            "confidence": normalize_text(candidate.get("confidence")) or "medium",
        }
        if candidate.get("rgb") is not None:
            normalized_candidate["rgb"] = candidate.get("rgb")
        named_color = normalize_text(candidate.get("named_color"))
        if named_color:
            normalized_candidate["named_color"] = named_color
        if candidate.get("named_rgb") is not None:
            normalized_candidate["named_rgb"] = candidate.get("named_rgb")
        normalized_candidates.append(normalized_candidate)
        seen_colors.add(color)

    return sorted(normalized_candidates, key=lambda candidate: candidate["score"], reverse=True)[:3]


def parse_rgb(value) -> tuple[int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(round(float(value[index]))))) for index in range(3))
        except (TypeError, ValueError):
            return None
    return None


GRAY_NAMED_COLORS = {
    "darkslategray",
    "darkslategrey",
    "slategray",
    "slategrey",
    "lightslategray",
    "lightslategrey",
    "dimgray",
    "dimgrey",
    "darkgray",
    "darkgrey",
    "gray",
    "grey",
}


DENIM_BLUE_NAMED_COLORS = {
    "lightblue",
    "steelblue",
    "royalblue",
    "darkblue",
    "navy",
    "midnightblue",
}


DENIM_DARK_NAMED_COLORS = {
    "black",
    "dimgray",
    "dimgrey",
    "darkgray",
    "darkgrey",
    "gray",
    "grey",
    "darkslategray",
    "darkslategrey",
    "midnightblue",
    "navy",
    "darkblue",
}


def denim_query_named_color(named_color: str, rgb, query_attrs) -> tuple[str, tuple[int, int, int] | None]:
    query_attrs = query_attrs or {}
    if (
        not query_attrs.get("is_denim_context")
        or normalize_text(named_color) not in GRAY_NAMED_COLORS
    ):
        return named_color, None

    raw_rgb = parse_rgb(rgb)
    if not raw_rgb:
        return named_color, None

    brightness = sum(raw_rgb) / 3
    blue_bias = raw_rgb[2] - max(raw_rgb[0], raw_rgb[1])
    tone = normalize_text(query_attrs.get("denim_tone"))
    query_color = normalize_color(query_attrs.get("color"))
    if query_color == "blue":
        if tone == "light":
            remapped = "lightblue"
        elif tone == "medium":
            remapped = "steelblue"
        elif tone == "dark":
            remapped = "midnightblue" if brightness < 78 and blue_bias < 12 else "navy"
        elif brightness >= 145:
            remapped = "lightblue"
        elif brightness >= 105:
            remapped = "steelblue"
        else:
            remapped = "midnightblue" if blue_bias < 12 else "navy"
    else:
        return named_color, None

    return remapped, parse_named_rgb_seed(remapped)


def query_named_color_info(query_attrs) -> dict:
    for candidate in (query_attrs or {}).get("color_candidates") or []:
        raw_rgb = parse_rgb(candidate.get("rgb"))
        named_rgb = parse_rgb(candidate.get("named_rgb"))
        rgb = named_rgb or raw_rgb
        named_color = normalize_text(candidate.get("named_color"))
        if rgb:
            if not named_color:
                named_color, _group, rgb = nearest_named_color(rgb)
            remapped_named_color, remapped_rgb = denim_query_named_color(named_color, raw_rgb or rgb, query_attrs)
            if remapped_rgb:
                named_color = remapped_named_color
                rgb = remapped_rgb
            return {"named_color": named_color, "rgb": rgb}
    return {}


def resolve_detailed_color(color_result: ColorExtractionResult | None) -> str:
    if not color_result:
        return ""
    for candidate in color_result.candidates or []:
        named_color = normalize_text(candidate.get("named_color"))
        if named_color:
            return named_color
        rgb = parse_rgb(candidate.get("named_rgb")) or parse_rgb(candidate.get("rgb"))
        if rgb:
            named_color, _group, _rgb = nearest_named_color(rgb)
            return normalize_text(named_color)
    return ""


def search_query_color_for_mode(
    query_image_color: str,
    intent: QueryIntent,
    color_result: ColorExtractionResult | None = None,
) -> str:
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    if color_mode != "same":
        return query_image_color
    return resolve_detailed_color(color_result) or query_image_color


def compact_prompt(*parts) -> str:
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def infer_named_color_from_text(text: str) -> tuple[str, tuple[int, int, int] | None]:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-z0-9가-힣]+", "", normalized)
    named_terms = (
        ("lightblue", ("light blue", "lightblue", "light indigo", "lightindigo", "light washed", "light vintage", "vintage light", "washed light", "라이트 블루", "라이트블루", "라이트 인디고", "라이트인디고", "라이트 워싱", "밝은 블루", "밝은 인디고", "밝은청", "연한 청", "연한청", "연청")),
        ("steelblue", ("steel blue", "스틸블루", "중청")),
        ("darkblue", ("dark blue", "dark denim", "dark indigo", "deep blue", "d.blue", "dblue", "다크블루", "진청", "딥블루")),
        ("navy", ("navy", "raw denim", "one washed", "one wash", "네이비", "남색", "인디고", "indigo", "생지")),
        ("midnightblue", ("midnight blue", "midnight", "흑청")),
        ("royalblue", ("royal blue",)),
        ("black", ("black", "블랙", "오일블랙", "oil black")),
        ("gray", ("gray", "grey", "그레이")),
    )
    for named_color, terms in named_terms:
        for term in terms:
            term_text = normalize_text(term)
            term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
            if term_text in normalized or (term_compact and term_compact in compact):
                _name, _group, rgb = nearest_named_color(parse_named_rgb_seed(named_color))
                return named_color, rgb
    return "", None


def parse_named_rgb_seed(named_color: str) -> tuple[int, int, int]:
    named_color = normalize_text(named_color)
    named_rgb_seeds = {
        "lightblue": (173, 216, 230),
        "steelblue": (70, 130, 180),
        "royalblue": (65, 105, 225),
        "darkblue": (0, 0, 139),
        "navy": (0, 0, 128),
        "midnightblue": (25, 25, 112),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }
    return named_rgb_seeds.get(named_color, (0, 0, 0))


def candidate_named_color_info(item, color_candidates) -> dict:
    for candidate in color_candidates or []:
        rgb = parse_rgb(candidate.get("named_rgb")) or parse_rgb(candidate.get("rgb"))
        named_color = normalize_text(candidate.get("named_color"))
        if rgb:
            if not named_color:
                named_color, _group, rgb = nearest_named_color(rgb)
            return {"named_color": named_color, "rgb": rgb}

    named_color, rgb = infer_named_color_from_text(
        " ".join(str(item.get(field) or "") for field in ("name", "brand_name"))
    )
    if rgb:
        return {"named_color": named_color, "rgb": rgb}
    return {}


def can_compare_named_color_groups(query_final_color, candidate_final_color, query_info, candidate_info, query_attrs) -> bool:
    query_group = color_group(query_final_color, query_attrs)
    candidate_group = color_group(candidate_final_color, query_attrs)
    if not query_group or not candidate_group or query_group == candidate_group:
        return True
    if not (query_attrs or {}).get("is_denim_context"):
        return False

    query_named = normalize_text(query_info.get("named_color"))
    candidate_named = normalize_text(candidate_info.get("named_color"))
    if query_named in DENIM_DARK_NAMED_COLORS and candidate_named in DENIM_DARK_NAMED_COLORS:
        return True
    if query_named in {"midnightblue", "navy", "darkblue"} and candidate_named in DENIM_BLUE_NAMED_COLORS:
        return True
    return False


def denim_named_tone(named_color: str) -> str:
    named_color = normalize_text(named_color)
    if named_color == "lightblue":
        return "light"
    if named_color in {"steelblue", "royalblue"}:
        return "medium"
    if named_color in DENIM_DARK_NAMED_COLORS:
        return "dark"
    return ""


def denim_named_tone_adjustment(query_named: str, candidate_named: str, adjustment: float, query_attrs) -> float:
    if not (query_attrs or {}).get("is_denim_context"):
        return adjustment

    query_tone = denim_named_tone(query_named)
    candidate_tone = denim_named_tone(candidate_named)
    if not query_tone or not candidate_tone or query_tone == candidate_tone:
        return adjustment

    if query_tone == "medium":
        if candidate_tone == "light":
            return min(adjustment, -0.08)
        if candidate_tone == "dark":
            return -0.05
    if {query_tone, candidate_tone} == {"light", "dark"}:
        return min(adjustment, -0.10)
    return adjustment


def dark_denim_query_active(query_attrs) -> bool:
    query_attrs = query_attrs or {}
    return (
        query_attrs.get("is_denim_context")
        and normalize_text(query_attrs.get("denim_tone")) == "dark"
        and normalize_color(query_attrs.get("color")) in {"gray", "black"}
    )


def gray_washed_dark_denim_query(query_attrs, query_named: str = "") -> bool:
    return (
        dark_denim_query_active(query_attrs)
        and normalize_color((query_attrs or {}).get("color")) == "gray"
        and normalize_text(query_named) in {
            "dimgray",
            "dimgrey",
            "gray",
            "grey",
            "darkgray",
            "darkgrey",
            "darkslategray",
            "darkslategrey",
        }
    )


GRAY_DENIM_NAMED_COLORS = {
    "dimgray",
    "dimgrey",
    "gray",
    "grey",
    "darkgray",
    "darkgrey",
    "darkslategray",
    "darkslategrey",
}


def strong_gray_candidate_evidence(item, color_candidates, candidate_named: str, candidate_tone: str) -> bool:
    if normalize_text(candidate_named) in GRAY_DENIM_NAMED_COLORS:
        return True
    if normalize_candidate_denim_tone(candidate_tone) == "gray":
        return True
    if normalize_color(item.get("dominant_color") or item.get("color")) == "gray":
        return True
    for candidate in color_candidates or []:
        if normalize_color(candidate.get("color")) != "gray":
            continue
        score = float(candidate.get("score", 0.0) or 0.0)
        source = normalize_text(candidate.get("source"))
        confidence = normalize_text(candidate.get("confidence"))
        if score >= 0.35 and confidence != "low" and source != "family":
            return True
    return False


def dark_indigo_candidate(candidate_named: str, candidate_tone: str) -> bool:
    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_candidate_denim_tone(candidate_tone)
    return candidate_named in {"midnightblue", "navy", "darkblue"} or candidate_tone in {"indigo", "dark_blue"}


def gray_same_dark_indigo_mismatch(item, color_candidates, candidate_named: str, candidate_tone: str, query_attrs, color_mode: str = "same") -> bool:
    return (
        color_mode == "same"
        and (query_attrs or {}).get("is_denim_context")
        and normalize_color((query_attrs or {}).get("color")) == "gray"
        and dark_indigo_candidate(candidate_named, candidate_tone)
        and not strong_gray_candidate_evidence(item, color_candidates, candidate_named, candidate_tone)
    )


def dark_denim_match_type(item, candidate_named: str, candidate_tone: str, query_attrs, query_named: str = "", color_candidates=None) -> str:
    if not dark_denim_query_active(query_attrs):
        return "none"

    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_text(candidate_tone)
    if candidate_named == "lightblue" or candidate_tone in {"light", "light_blue", "white"}:
        return "light_mismatch"

    item_text = normalize_text(
        " ".join(str(item.get(field) or "") for field in ("name", "brand_name", "sub_category"))
    )
    compact = re.sub(r"[^a-z0-9가-힣]+", "", item_text)
    washed_terms = {
        "gray", "grey", "그레이", "워싱", "washed", "washing",
        "fade", "faded", "vintage", "crack", "크랙", "slub", "슬럽",
    }
    black_terms = {"black", "블랙", "oil black", "raw black", "오일 블랙"}
    dark_indigo_terms = {
        "흑청", "midnight", "midnight blue", "navy", "네이비",
        "deep indigo", "dark indigo", "deep blue", "d.blue", "dblue",
        "darkblue", "dark blue", "딥 인디고", "딥블루", "다크블루", "생지",
    }

    def has_any(terms) -> bool:
        for term in terms:
            term_text = normalize_text(term)
            term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
            if term_text in item_text or (term_compact and term_compact in compact):
                return True
        return False

    has_washed = has_any(washed_terms)
    has_black = has_any(black_terms)
    has_dark_indigo = has_any(dark_indigo_terms)
    if gray_washed_dark_denim_query(query_attrs, query_named) and strong_gray_candidate_evidence(
        item,
        color_candidates,
        candidate_named,
        candidate_tone,
    ):
        return "washed_gray"
    if candidate_named in {"midnightblue", "navy", "darkblue"} or has_dark_indigo:
        return "dark_indigo"
    if has_black or candidate_named == "black":
        return "black_only"
    if dark_denim_candidate_match(candidate_named, candidate_tone):
        return "dark_indigo" if candidate_named in {"midnightblue", "navy", "darkblue"} else "none"
    return "none"


def dark_denim_candidate_match(candidate_named: str, candidate_tone: str) -> bool:
    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_text(candidate_tone)
    return candidate_named in DENIM_DARK_NAMED_COLORS or candidate_tone in {"dark", "black", "dark_blue", "indigo"}


def dark_denim_adjustment(match_type: str) -> tuple[float, bool]:
    if match_type == "washed_gray":
        return 0.03, True
    if match_type == "dark_indigo":
        return 0.02, True
    if match_type == "black_only":
        return -0.03, False
    if match_type == "light_mismatch":
        return -0.10, False
    return 0.0, False


def legacy_dark_denim_adjustment(query_attrs, candidate_named: str, candidate_tone: str) -> tuple[float, bool]:
    if not dark_denim_query_active(query_attrs):
        return 0.0, False

    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_text(candidate_tone)
    if candidate_named == "lightblue" or candidate_tone in {"light", "light_blue", "white"}:
        return -0.10, False
    if dark_denim_candidate_match(candidate_named, candidate_tone):
        return 0.03, True
    return 0.0, False


def fine_color_adjustment(item, color_candidates, query_attrs, color_mode: str) -> tuple[float, str, str, float | None, float, bool, str]:
    if color_mode != "same":
        return 0.0, "", "", None, 0.0, False, "none"
    query_info = query_named_color_info(query_attrs)
    candidate_info = candidate_named_color_info(item, color_candidates)
    candidate_tone = get_candidate_denim_tone(item)
    dark_match_type = dark_denim_match_type(
        item,
        candidate_info.get("named_color", ""),
        candidate_tone,
        query_attrs,
        query_info.get("named_color", ""),
        color_candidates,
    )
    dark_adjustment, dark_match = dark_denim_adjustment(dark_match_type)
    query_final_color = normalize_color((query_attrs or {}).get("color"))
    candidate_final_color = normalize_color(item.get("dominant_color") or item.get("color"))
    if not candidate_final_color and color_candidates:
        candidate_final_color = normalize_color(color_candidates[0].get("color"))
    if (
        query_final_color
        and candidate_final_color
        and not can_compare_named_color_groups(
            query_final_color,
            candidate_final_color,
            query_info,
            candidate_info,
            query_attrs,
        )
    ):
        return dark_adjustment, query_info.get("named_color", ""), candidate_info.get("named_color", ""), None, dark_adjustment, dark_match, dark_match_type
    query_rgb = query_info.get("rgb")
    candidate_rgb = candidate_info.get("rgb")
    if not query_rgb or not candidate_rgb:
        return dark_adjustment, query_info.get("named_color", ""), candidate_info.get("named_color", ""), None, dark_adjustment, dark_match, dark_match_type

    distance = named_lab_distance(query_rgb, candidate_rgb)
    if distance <= 18:
        adjustment = 0.06
    elif distance <= 32:
        adjustment = 0.03
    elif distance >= 58:
        adjustment = -0.10
    elif distance >= 42:
        adjustment = -0.05
    else:
        adjustment = 0.0
    query_named = normalize_text(query_info.get("named_color"))
    candidate_named = normalize_text(candidate_info.get("named_color"))
    if (
        (query_attrs or {}).get("is_denim_context")
        and query_named in DENIM_DARK_NAMED_COLORS
        and candidate_named in DENIM_DARK_NAMED_COLORS
        and adjustment < 0
        and not gray_same_dark_indigo_mismatch(
            item,
            color_candidates,
            candidate_named,
            candidate_tone,
            query_attrs,
        )
    ):
        adjustment = 0.0
    adjustment = denim_named_tone_adjustment(query_named, candidate_named, adjustment, query_attrs)
    if gray_same_dark_indigo_mismatch(
        item,
        color_candidates,
        candidate_named,
        candidate_tone,
        query_attrs,
    ):
        dark_adjustment = min(dark_adjustment, 0.0)
        dark_match = False
    if dark_adjustment:
        if dark_adjustment > 0:
            adjustment = max(adjustment, dark_adjustment)
        else:
            adjustment = min(adjustment, dark_adjustment)
    return adjustment, query_info.get("named_color", ""), candidate_info.get("named_color", ""), distance, dark_adjustment, dark_match, dark_match_type


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
            group_weight = 0.7
            if normalize_text(candidate.get("confidence")) == "low":
                group_weight *= 0.35
            if normalize_text(candidate.get("source")) == "family":
                group_weight *= 0.7
            best_score = max(best_score, candidate_score * group_weight)
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


def query_color_targets(intent: QueryIntent, query_attrs) -> dict[str, float]:
    query_attrs = query_attrs or {}
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    explicit_color = normalize_color(intent.color)
    if explicit_color:
        return {explicit_color: 1.0}

    targets = {}
    excluded_color = normalize_color(query_attrs.get("excluded_color"))
    if color_mode == "different" and excluded_color:
        return {excluded_color: 1.0}

    primary_color = normalize_color(query_attrs.get("color"))
    color_confidence = normalize_text(query_attrs.get("color_confidence"))
    primary_weight = image_color_confidence_weight(query_attrs)
    if primary_color:
        targets[primary_color] = primary_weight
    if color_mode == "different":
        return targets

    raw_weights = query_attrs.get("search_color_weights") or {}
    if isinstance(raw_weights, dict):
        for color, weight in raw_weights.items():
            normalized = normalize_color(color)
            if not normalized:
                continue
            try:
                score = float(weight or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            same_denim_family = (
                color_mode == "same"
                and primary_color
                and query_attrs.get("is_denim_context")
                and color_group(normalized, query_attrs) == color_group(primary_color, query_attrs)
            )
            if (
                color_mode == "same"
                and primary_color
                and normalized != primary_color
                and score < 0.5
                and not same_denim_family
            ):
                continue
            if score > 0:
                weighted_score = min(score, 1.0)
                if normalized == primary_color:
                    weighted_score = min(weighted_score, primary_weight)
                elif color_mode == "same" and primary_color:
                    weighted_score *= primary_weight
                targets[normalized] = max(targets.get(normalized, 0.0), weighted_score)

    for color in query_attrs.get("secondary_colors") or []:
        normalized = normalize_color(color)
        if normalized:
            if color_mode == "same" and primary_color and color_confidence in {"high", "medium"}:
                continue
            secondary_weight = 0.25 if color_confidence == "low" else 0.35
            targets[normalized] = max(targets.get(normalized, 0.0), secondary_weight)

    return targets


def best_color_match_score(item_color, color_candidates, target_weights, query_attrs=None):
    best_score = 0.0
    exact_match = False
    group_match = False
    matched_target = ""
    for target_color, target_weight in (target_weights or {}).items():
        if target_weight <= 0:
            continue
        candidate_score, candidate_exact, candidate_group = color_candidate_match_score(
            color_candidates,
            target_color,
            query_attrs,
        )
        dominant_score, dominant_exact, dominant_group = dominant_color_match_score(
            item_color,
            target_color,
            query_attrs,
        )
        score = max(candidate_score, dominant_score) * target_weight
        if score > best_score:
            best_score = score
            exact_match = candidate_exact or dominant_exact
            group_match = candidate_group or dominant_group
            matched_target = target_color
    return best_score, exact_match, group_match, matched_target


def image_color_confidence_weight(query_attrs) -> float:
    confidence = normalize_text((query_attrs or {}).get("color_confidence"))
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.85
    if confidence == "low":
        return 0.55
    return 1.0


def denim_color_adjustment_scale(candidate_tone: str, target_color: str, color_mode: str, query_attrs) -> float:
    query_attrs = query_attrs or {}
    normalized_candidate_tone = normalize_candidate_denim_tone(candidate_tone)
    query_denim_tone = normalize_text(query_attrs.get("denim_tone"))
    if (
        query_attrs.get("is_denim_context")
        and normalize_text(query_attrs.get("denim_tone_source")) == "text"
        and query_denim_tone == "light"
        and normalize_color(target_color) == "blue"
        and normalized_candidate_tone in {"indigo", "black", "dark_blue", "dark"}
    ):
        return 0.35
    if (
        query_attrs.get("is_denim_context")
        and normalize_text(query_attrs.get("denim_tone_source")) == "text"
        and query_denim_tone == "medium"
        and normalize_color(target_color) == "blue"
        and normalized_candidate_tone in {"indigo", "black", "dark_blue", "dark"}
    ):
        return 0.55
    if (
        color_mode == "same"
        and query_attrs.get("is_denim_context")
        and not query_attrs.get("design_similarity_mode")
        and query_denim_tone == "dark"
        and normalize_color(target_color) == "gray"
        and normalized_candidate_tone == "gray"
        and normalize_text(query_attrs.get("color_source")) != "explicit_text"
    ):
        return 0.25
    return 1.0


# ---------------------------------------------------------------------------
# LLM 카테고리 후처리
# ---------------------------------------------------------------------------

# CATEGORY_KEYWORDS 역방향 룩업: 값(main, sub) → 키 목록 (정규화용)
_MAIN_CAT_ALIASES: dict[str, list[str]] = {}
_SUB_CAT_ALIASES: dict[str, list[str]] = {}
for _kw, (_mc, _sc) in CATEGORY_KEYWORDS.items():
    if _mc:
        _MAIN_CAT_ALIASES.setdefault(_mc, []).append(_kw)
    if _sc:
        _SUB_CAT_ALIASES.setdefault(_sc, []).append(_kw)

# sub_category 표준 표기 → 별칭 집합 (LLM 출력 근사 매핑용)
_SUB_CAT_SURFACE_ALIASES: dict[str, set[str]] = {
    "후드티": {"후드티", "후디", "hoodie", "후드"},
    "스웻셔츠": {"스웻셔츠", "스웨트셔츠", "맨투맨", "sweatshirt"},
    "반소매 티셔츠": {"반소매 티셔츠", "반팔 티셔츠", "반팔티", "반팔", "short sleeve", "short sleeve t-shirt"},
    "긴소매 티셔츠": {"긴소매 티셔츠", "긴팔 티셔츠", "긴팔티", "롱 티셔츠", "long sleeve", "long sleeve t-shirt", "롱티"},
    "니트/스웨터": {"니트", "스웨터", "니트/스웨터", "knit", "sweater"},
    "가디건": {"가디건", "cardigan"},
    "레더자켓": {"레더자켓", "가죽자켓", "레더 자켓", "leather jacket"},
    "블루종/MA-1": {"블루종", "ma-1", "ma1", "bomber", "봄버", "항공점퍼"},
    "사파리/헌팅자켓": {"야상", "사파리", "헌팅", "필드자켓", "필드 자켓", "field jacket", "safari", "hunting jacket"},
    "데님팬츠": {"청바지", "데님팬츠", "데님 팬츠", "진", "jeans", "denim jeans"},
    "슬랙스/슈트 팬츠": {"슬랙스", "정장바지", "슈트바지", "slacks"},
    "트레이닝/조거 팬츠": {"조거", "조거팬츠", "트레이닝팬츠", "jogger", "jogger pants"},
    "코튼 팬츠": {"코튼팬츠", "면바지", "코튼 팬츠", "cotton pants"},
    "숏팬츠": {"반바지", "숏팬츠", "숏 팬츠", "shorts", "short pants"},
    "후드집업": {"후드집업", "집업", "zip-up", "hooded zip-up"},
}

# main_category 표준 표기 근사 매핑
_MAIN_CAT_SURFACE_ALIASES: dict[str, set[str]] = {
    "상의": {"상의", "top", "tops", "탑", "티", "셔츠류"},
    "하의": {"하의", "pants", "bottoms", "바텀", "바지류"},
    "아우터": {"아우터", "outerwear", "outer", "겉옷", "점퍼", "자켓류"},
}


def normalize_llm_category(
    llm_main: str,
    llm_sub: str,
) -> tuple[str, str]:
    """LLM이 자유 텍스트로 출력한 카테고리를 DB 정규화 값으로 변환한다.

    탐색 우선순위:
    1. 직접 일치 (_SUB_CAT_SURFACE_ALIASES / _MAIN_CAT_SURFACE_ALIASES)
    2. CATEGORY_KEYWORDS 역방향 탐색
    3. 빈 문자열 반환 → 호출 측에서 룰 기반 fallback 사용
    """
    raw_main = normalize_text(llm_main)
    raw_sub = normalize_text(llm_sub)

    # sub_category 정규화 -------------------------------------------------------
    normalized_sub = ""
    if raw_sub:
        # 1. 직접 일치 (표준 표기)
        for canonical, aliases in _SUB_CAT_SURFACE_ALIASES.items():
            if raw_sub in {normalize_text(a) for a in aliases} or raw_sub == normalize_text(canonical):
                normalized_sub = canonical
                break
        # 2. CATEGORY_KEYWORDS 역방향 탐색 (키워드가 LLM 출력에 포함되면)
        if not normalized_sub:
            for canonical, keywords in _SUB_CAT_ALIASES.items():
                if any(normalize_text(kw) in raw_sub or raw_sub in normalize_text(kw) for kw in keywords):
                    normalized_sub = canonical
                    break

    # main_category 정규화 ------------------------------------------------------
    normalized_main = ""
    if raw_main:
        # 1. 직접 일치
        for canonical, aliases in _MAIN_CAT_SURFACE_ALIASES.items():
            if raw_main in {normalize_text(a) for a in aliases} or raw_main == normalize_text(canonical):
                normalized_main = canonical
                break
        # 2. CATEGORY_KEYWORDS 역방향 탐색
        if not normalized_main:
            for canonical, keywords in _MAIN_CAT_ALIASES.items():
                if any(normalize_text(kw) in raw_main or raw_main in normalize_text(kw) for kw in keywords):
                    normalized_main = canonical
                    break

    # sub_category에서 main_category 유추 (main이 비었을 때)
    if normalized_sub and not normalized_main:
        for kw, (mc, sc) in CATEGORY_KEYWORDS.items():
            if sc == normalized_sub and mc:
                normalized_main = mc
                break

    return normalized_main, normalized_sub


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


def is_generic_pants_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    generic_terms = ("바지", "팬츠", "pants", "trousers")
    explicit_terms = (
        "청바지",
        "데님",
        "슬랙스",
        "정장바지",
        "슈트바지",
        "조거",
        "트레이닝",
        "코튼팬츠",
        "코튼 팬츠",
        "면바지",
        "반바지",
        "숏팬츠",
        "카고",
    )
    return any(term in normalized_query for term in generic_terms) and not any(
        term in normalized_query for term in explicit_terms
    )


def should_use_image_category_reconciliation(
    query: str,
    rule_main_categories,
    rule_sub_categories,
    analyzed_main_categories,
) -> bool:
    main_categories = list(analyzed_main_categories or rule_main_categories or [])
    return (
        bool(query)
        and is_generic_pants_query(query)
        and not rule_sub_categories
        and main_categories == ["하의"]
    )


def build_image_category_result(scores) -> ImageCategoryResult:
    ranked = sorted(
        (
            {
                "category": str(score.get("category") or ""),
                "score": float(score.get("score") or 0.0),
            }
            for score in scores or []
            if score.get("category")
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    if not ranked:
        return ImageCategoryResult("", "low", [])

    top = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = top["score"] - second_score
    if top["score"] >= 0.18 and margin >= 0.015:
        confidence = "high"
    elif top["score"] >= 0.16 and margin >= 0.008:
        confidence = "medium"
    else:
        confidence = "low"

    return ImageCategoryResult(top["category"], confidence, ranked[:5])


def category_filter_source(
    query: str,
    rule_sub_categories,
    analyzed_sub_categories,
    image_category_result: ImageCategoryResult | None = None,
) -> str:
    if rule_sub_categories:
        return "rule"
    image_category_result = image_category_result or ImageCategoryResult()
    if is_generic_pants_query(query):
        if image_category_result.confidence == "high" and image_category_result.category:
            return "image"
        if image_category_result.confidence in {"medium", "low"}:
            return "image_low_confidence"
        return "relaxed"
    if analyzed_sub_categories:
        return "llm"
    return "relaxed"


def main_category_for_sub_category(sub_category: str) -> str:
    if not sub_category:
        return ""
    for _keyword, (main_category, candidate_sub_category) in CATEGORY_KEYWORDS.items():
        if candidate_sub_category == sub_category and main_category:
            return main_category
    return ""


def reconcile_impossible_category_pair(main_categories, sub_categories):
    main_categories = list(main_categories or [])
    sub_categories = list(sub_categories or [])
    if not sub_categories:
        return main_categories, sub_categories

    required_main = main_category_for_sub_category(sub_categories[0])
    if required_main and main_categories != [required_main]:
        return [required_main], sub_categories
    return main_categories, sub_categories


def reconcile_category_filters(
    query: str,
    rule_main_categories,
    rule_sub_categories,
    analyzed_main_categories,
    analyzed_sub_categories,
    image_category_result: ImageCategoryResult | None = None,
):
    main_categories = list(analyzed_main_categories or rule_main_categories or [])
    sub_categories = list(analyzed_sub_categories or [])
    image_category_result = image_category_result or ImageCategoryResult()

    if (
        should_use_image_category_reconciliation(
            query,
            rule_main_categories,
            rule_sub_categories,
            main_categories,
        )
    ):
        if image_category_result.confidence == "high" and image_category_result.category:
            sub_categories = [image_category_result.category]
        else:
            sub_categories = []

    main_categories, sub_categories = reconcile_impossible_category_pair(
        main_categories,
        sub_categories,
    )

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
    generic_tokens.update(GENERIC_DESIGN_INSTRUCTION_TOKENS)

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
        return "blue"
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

    color_candidates = []
    search_color_weights = {}
    for candidate in valid_candidates[:5]:
        color = classify_color_by_lab(candidate["rgb"])
        ratio = float(candidate["ratio"])
        color_candidates.append({"color": color, "score": ratio, "rgb": candidate["rgb"]})
        if color and color != "multi_color":
            search_color_weights[color] = max(search_color_weights.get(color, 0.0), min(ratio, 1.0))

    top = valid_candidates[0]
    second_ratio = valid_candidates[1]["ratio"] if len(valid_candidates) > 1 else 0.0
    top_ratio = top["ratio"]

    if second_ratio >= 0.18 and abs(top_ratio - second_ratio) < 0.15:
        secondary_colors = []
        for candidate in color_candidates:
            color = candidate["color"]
            if color and color != "multi_color" and color not in secondary_colors:
                secondary_colors.append(color)
        return ColorExtractionResult(
            secondary_colors[0] if secondary_colors else "multi_color",
            "low",
            "mixed_color_clusters",
            top_ratio,
            second_ratio,
            candidates=color_candidates,
            secondary_colors=secondary_colors[1:],
            is_mixed_color=True,
            search_color_weights=search_color_weights,
        )

    color = classify_color_by_lab(top["rgb"])
    if top_ratio >= HIGH_COLOR_RATIO and top_ratio - second_ratio >= 0.18:
        confidence = "high"
    elif top_ratio >= MEDIUM_COLOR_RATIO:
        confidence = "medium"
    else:
        confidence = "low"

    secondary_colors = []
    for candidate in color_candidates[1:]:
        candidate_color = candidate["color"]
        if candidate_color and candidate_color != color and candidate_color not in secondary_colors:
            secondary_colors.append(candidate_color)

    return ColorExtractionResult(
        color,
        confidence,
        "",
        top_ratio,
        second_ratio,
        candidates=color_candidates,
        secondary_colors=secondary_colors,
        is_mixed_color=bool(secondary_colors and second_ratio >= 0.18),
        search_color_weights=search_color_weights,
    )


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


def extract_query_color_result(
    image_obj: Image.Image,
    denim_context: bool,
    pattern_context_text: str,
) -> ColorExtractionResult:
    result = extract_dominant_color_result_v2(
        image_obj,
        denim_context=denim_context,
        pattern_context_text=pattern_context_text,
    )
    if result.color or result.search_color_weights or result.candidates:
        return result

    fallback = extract_dominant_color_result(image_obj, denim_context=denim_context)
    fallback.reason = fallback.reason or "fallback_color_extraction"
    return fallback


async def analyze_query(
    user_query: str,
    rule_main_categories: list | None = None,
    rule_sub_categories: list | None = None,
) -> tuple[QueryIntent, list[str], list[str]]:
    """사용자 쿼리에서 검색 intent와 카테고리를 단일 Gemini 호출로 추론한다.

    반환값: (QueryIntent, main_categories, sub_categories)
    - LLM 성공 시: LLM 추론 카테고리 (normalize_llm_category 후처리)
    - LLM 실패 / gemini_client 없을 때: 룰 기반 fallback 카테고리
    """
    # ── 룰 기반 fallback 사전 계산 ────────────────────────────────────────────
    fallback_color = infer_color_from_text(user_query)
    fallback_color_mode = infer_color_mode(user_query)
    if fallback_color and fallback_color_mode == "ignore":
        fallback_color_mode = "target"
    fallback_intent = QueryIntent(
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
    fallback_intent = apply_excluded_color_phrase(user_query, fallback_intent)
    fallback_main = list(rule_main_categories or [])
    fallback_sub = list(rule_sub_categories or [])

    if not gemini_client or genai_types is None:
        return fallback_intent, fallback_main, fallback_sub

    # ── 허용 sub_category 목록 (프롬프트 힌트용) ─────────────────────────────
    _sub_cat_hint = ", ".join(sorted(_SUB_CAT_SURFACE_ALIASES.keys()))

    system_prompt = f"""\
You are a fashion search query analyzer for a Korean fashion app.
The user will provide a Korean (or mixed Korean/English) query about clothing.
Your job is to extract structured intent AND clothing category.

Fields to extract:
- reasoning: Brief explanation of your analysis (1~2 sentences in English).
- color: The target color explicitly mentioned (canonical English lowercase: black, white, gray, blue, red, green, yellow, brown, pink, purple, orange). Use "" if no color is mentioned.
- color_mode:
    - "target"    -> user wants items of a specific color (e.g. "검정 후드티 보여줘")
    - "same"      -> user wants the same color as a reference item (e.g. "같은 색으로 보여줘")
    - "different" -> user wants a different color (e.g. "색상이 다른 걸로 보여줘")
    - "ignore"    -> color is not relevant to the query
- design: Style, silhouette, length, fabric, or fit keywords in English, space-separated (e.g. "oversized wide-leg denim"). Use "" if absent.
- main_category: One of "상의", "하의", "아우터". Use "" if unclear.
- sub_category: The most specific clothing type from the hint list below, or your best inference even if not in the list. Use "" if truly unclear.
  Hint list: {_sub_cat_hint}

Rules:
- If a color word appears but the user is not requesting that color (e.g. describing a reference), set color_mode to "same" or "ignore" appropriately.
- Do NOT invent colors not mentioned in the query.
- For sub_category, prefer the closest match from the hint list; leave "" only when genuinely ambiguous.
- Return valid JSON only, no markdown.

Example:
{{"reasoning": "User wants a black oversized hoodie.", "color": "black", "color_mode": "target", "design": "oversized", "main_category": "상의", "sub_category": "후드티"}}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"User query: {user_query}",
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=QueryAnalysis,
            ),
        )
        if getattr(response, "parsed", None):
            parsed = response.parsed
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            data = json.loads(response.text)

        intent = QueryIntent(
            reasoning=data.get("reasoning", ""),
            color=normalize_color(data.get("color", "")),
            color_mode=data.get("color_mode", "ignore"),
            design=data.get("design", ""),
        )
        if intent.color_mode == "ignore":
            inferred_mode = infer_color_mode(user_query)
            if inferred_mode != "ignore":
                intent.color_mode = inferred_mode
        if normalize_color(intent.color) and intent.color_mode == "ignore":
            intent.color_mode = "target"
        intent = apply_excluded_color_phrase(user_query, intent)

        # ── 카테고리 후처리 ──────────────────────────────────────────────────
        llm_main_raw = data.get("main_category", "") or ""
        llm_sub_raw = data.get("sub_category", "") or ""
        norm_main, norm_sub = normalize_llm_category(llm_main_raw, llm_sub_raw)

        # LLM 추론 결과를 우선 사용하고, 비어 있는 필드만 룰 기반 결과로 보완한다.
        final_main = [norm_main] if norm_main else list(fallback_main)
        final_sub = [norm_sub] if norm_sub else list(fallback_sub)

        return intent, final_main, final_sub

    except Exception as exc:
        print(f"LLM analysis failed, fallback used: {exc}")
        return fallback_intent, fallback_main, fallback_sub



# analyze_query_intent는 하위 호환성을 위해 유지 (내부적으로 analyze_query 위임)
async def analyze_query_intent(user_query: str) -> QueryIntent:
    intent, _main, _sub = await analyze_query(user_query)
    return intent


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


def _project_clip_features(outputs, projection):
    """transformers 버전에 따라 get_image_features / get_text_features 가
    텐서를 돌려주기도 하고 BaseModelOutputWithPooling 객체를 돌려주기도 한다.
    어느 경우든 DB의 512차원 projected 임베딩과 같은 벡터를 반환하도록 보정한다."""
    if torch.is_tensor(outputs):
        return outputs
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    text_embeds = getattr(outputs, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = extract_feature_tensor(outputs)
    # projection이 아직 적용되지 않은 tower 출력이면(예: 이미지 768차원) projection을 적용한다.
    if projection is not None and hasattr(projection, "in_features") and pooled.shape[-1] == projection.in_features:
        return projection(pooled)
    return pooled


def encode_image_with_fashion_clip_api(image_obj: Image.Image):
    rgb_image = image_obj.convert("RGB")
    inputs = processor(images=rgb_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = _project_clip_features(outputs, getattr(model, "visual_projection", None))
    return F.normalize(features, p=2, dim=-1)


def encode_text_with_fashion_clip_api(text: str):
    return encode_texts_with_fashion_clip_api([text])


def encode_texts_with_fashion_clip_api(texts: list[str]):
    inputs = processor(
        text=list(texts),
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = _project_clip_features(outputs, getattr(model, "text_projection", None))
    return F.normalize(features, p=2, dim=-1)


def get_image_embedding(image_obj: Image.Image):
    return encode_image_with_fashion_clip_api(image_obj)


def get_text_embedding(text: str):
    return encode_text_with_fashion_clip_api(text)


DESIGN_PROMPT_EMBEDDING_CACHE = {}


def infer_image_category_from_features(image_features) -> ImageCategoryResult:
    prompt_items = []
    for category, prompts in IMAGE_CATEGORY_PROMPTS.items():
        for prompt in prompts:
            prompt_items.append((category, prompt))

    cache_key = "bottom_categories"
    text_features = IMAGE_CATEGORY_PROMPT_EMBEDDING_CACHE.get(cache_key)
    if text_features is None:
        text_features = encode_texts_with_fashion_clip_api([prompt for _category, prompt in prompt_items])
        IMAGE_CATEGORY_PROMPT_EMBEDDING_CACHE[cache_key] = text_features

    similarities = (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()
    category_scores = {}
    for (category, _prompt), score in zip(prompt_items, similarities):
        category_scores[category] = max(category_scores.get(category, -1.0), float(score))

    return build_image_category_result([
        {"category": category, "score": score}
        for category, score in category_scores.items()
    ])


def grounding_dino_prompt_for_label(clothing_label: str, main_categories=None, sub_categories=None) -> str:
    labels = [clothing_label, *(sub_categories or []), *(main_categories or [])]
    for label in labels:
        prompt = GROUNDING_DINO_PROMPTS_BY_LABEL.get(label)
        if prompt:
            return prompt
    return "clothing item"


def format_grounding_dino_prompt(prompt: str) -> str:
    labels = [label.strip() for label in re.split(r"[.;,]", str(prompt)) if label.strip()]
    if not labels:
        return "clothing item."
    return ". ".join(labels) + "."


def grounding_dino_prompts_for_label(clothing_label: str, main_categories=None, sub_categories=None) -> list[str]:
    primary = grounding_dino_prompt_for_label(clothing_label, main_categories, sub_categories)
    prompt_candidates = list(GROUNDING_DINO_FALLBACK_PROMPTS_BY_PRIMARY.get(primary, (primary,)))
    if "jacket" in primary and "jacket. outerwear. coat. clothing" not in prompt_candidates:
        prompt_candidates.append("jacket. outerwear. coat. clothing")
    if any(token in primary for token in ("pants", "trousers", "jeans")) and "pants. trousers. clothing" not in prompt_candidates:
        prompt_candidates.append("pants. trousers. clothing")
    prompt_candidates.append("clothing item")

    prompts = []
    seen = set()
    for candidate in prompt_candidates:
        formatted = format_grounding_dino_prompt(candidate)
        if formatted not in seen:
            seen.add(formatted)
            prompts.append(formatted)
    return prompts


def load_dino_sam_backend():
    global dino_sam_backend, dino_sam_failed
    if dino_sam_backend is not None:
        return dino_sam_backend
    if dino_sam_failed or not DINO_SAM_ENABLED:
        return None
    if not SAM_CHECKPOINT:
        dino_sam_failed = True
        print("GroundingDINO/SAM disabled: SAM_CHECKPOINT is not set.")
        return None
    if not os.path.exists(SAM_CHECKPOINT):
        dino_sam_failed = True
        print(f"GroundingDINO/SAM disabled: SAM checkpoint does not exist: {SAM_CHECKPOINT}")
        return None
    try:
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        dino_sam_failed = True
        print(f"GroundingDINO/SAM disabled: missing package: {exc}")
        return None
    try:
        selected_device = DINO_SAM_DEVICE or device
        processor_dino = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(selected_device)
        dino_model.eval()
        if SAM_MODEL_TYPE not in sam_model_registry:
            raise RuntimeError(f"Unsupported SAM model type: {SAM_MODEL_TYPE}")
        sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(selected_device)
        sam_predictor = SamPredictor(sam_model)
        dino_sam_backend = {
            "processor": processor_dino,
            "dino_model": dino_model,
            "sam_predictor": sam_predictor,
            "device": selected_device,
        }
        print(f"GroundingDINO/SAM loaded: dino={GROUNDING_DINO_MODEL_ID}, sam={SAM_MODEL_TYPE}, device={selected_device}")
        return dino_sam_backend
    except Exception as exc:
        dino_sam_failed = True
        print(f"GroundingDINO/SAM disabled: {exc}")
        return None


def post_process_grounding_dino_outputs(processor_dino, outputs, inputs, image_obj):
    target_sizes = [image_obj.size[::-1]]
    call_variants = (
        {
            "args": (outputs, inputs.input_ids),
            "kwargs": {
                "threshold": DINO_SAM_BOX_THRESHOLD,
                "text_threshold": DINO_SAM_TEXT_THRESHOLD,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs,),
            "kwargs": {
                "threshold": DINO_SAM_BOX_THRESHOLD,
                "text_threshold": DINO_SAM_TEXT_THRESHOLD,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs, inputs.input_ids),
            "kwargs": {
                "box_threshold": DINO_SAM_BOX_THRESHOLD,
                "text_threshold": DINO_SAM_TEXT_THRESHOLD,
                "target_sizes": target_sizes,
            },
        },
        {
            "args": (outputs,),
            "kwargs": {
                "box_threshold": DINO_SAM_BOX_THRESHOLD,
                "text_threshold": DINO_SAM_TEXT_THRESHOLD,
                "target_sizes": target_sizes,
            },
        },
    )
    last_error = None
    for variant in call_variants:
        try:
            return processor_dino.post_process_grounded_object_detection(
                *variant["args"],
                **variant["kwargs"],
            )[0]
        except TypeError as exc:
            last_error = exc
    raise last_error


def select_grounding_box(boxes, scores, image_size):
    if len(boxes) == 0:
        return None, 0.0
    width, height = image_size
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
    return tuple(int(round(value)) for value in boxes[best_index].tolist()), float(scores[best_index])


def detect_grounding_box(image_obj: Image.Image, prompt: str, backend):
    processor_dino = backend["processor"]
    dino_model = backend["dino_model"]
    inputs = processor_dino(images=image_obj, text=prompt, return_tensors="pt").to(backend["device"])
    with torch.no_grad():
        outputs = dino_model(**inputs)
    result = post_process_grounding_dino_outputs(processor_dino, outputs, inputs, image_obj)
    return select_grounding_box(result.get("boxes", []), result.get("scores", []), image_obj.size)


def bool_mask_bbox(mask) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(bbox, image_size, padding_ratio: float = 0.08):
    x0, y0, x1, y1 = bbox
    width, height = image_size
    pad = int(round(max(x1 - x0, y1 - y0) * padding_ratio))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def build_masked_crop(image_obj: Image.Image, mask, fallback_bbox) -> tuple[Image.Image, float, tuple[int, int, int, int]]:
    rgb_image = image_obj.convert("RGB")
    mask_array = np.asarray(mask).astype(bool)
    bbox = bool_mask_bbox(mask_array) or fallback_bbox
    bbox = pad_bbox(bbox, rgb_image.size)
    mask_image = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", rgb_image.size, "white")
    masked = Image.composite(rgb_image, white, mask_image)
    crop = masked.crop(bbox)
    mask_ratio = float(mask_array.sum()) / max(1, rgb_image.width * rgb_image.height)
    return crop, mask_ratio, bbox


def rank_query_sam_mask(mask, sam_score: float, image_size) -> float:
    width, height = image_size
    mask_array = np.asarray(mask).astype(bool)
    mask_ratio = float(mask_array.sum()) / max(1, width * height)
    rank = float(sam_score)
    if mask_ratio < 0.005:
        rank -= 0.35
    elif mask_ratio < 0.04:
        rank -= 0.18
    elif 0.08 <= mask_ratio <= 0.85:
        rank += 0.08
    elif mask_ratio > 0.95:
        rank -= 0.15
    return rank


def select_query_sam_mask(masks, scores, image_size) -> int:
    best_index = 0
    best_rank = None
    for index, (mask, score) in enumerate(zip(masks, scores)):
        rank = rank_query_sam_mask(mask, float(score), image_size)
        if best_rank is None or rank > best_rank:
            best_index = index
            best_rank = rank
    return best_index


def preprocess_query_image_with_dino_sam(
    image_obj: Image.Image,
    clothing_label: str,
    main_categories=None,
    sub_categories=None,
) -> QueryImagePreprocessResult:
    prompts = grounding_dino_prompts_for_label(clothing_label, main_categories, sub_categories)
    prompt = prompts[0] if prompts else "clothing item."
    backend = load_dino_sam_backend()
    if backend is None:
        return QueryImagePreprocessResult(image_obj, "original_fallback", prompt=prompt, error="backend_unavailable")
    try:
        rgb_image = image_obj.convert("RGB")
        bbox = None
        detection_score = 0.0
        used_prompt = prompt
        detection_errors = []
        for candidate_prompt in prompts:
            try:
                bbox, detection_score = detect_grounding_box(rgb_image, candidate_prompt, backend)
            except Exception as exc:
                detection_errors.append(f"{candidate_prompt}: {exc}")
                continue
            used_prompt = candidate_prompt
            if bbox is not None:
                break
        if bbox is None:
            error = "no_detection"
            if detection_errors:
                error = f"no_detection; {' | '.join(detection_errors[-2:])}"
            return QueryImagePreprocessResult(image_obj, "original_fallback", prompt=" | ".join(prompts), error=error)
        sam_predictor = backend["sam_predictor"]
        sam_predictor.set_image(np.array(rgb_image))
        masks, scores, _logits = sam_predictor.predict(box=np.array(bbox), multimask_output=True)
        if len(masks) == 0:
            crop = image_obj.convert("RGB").crop(pad_bbox(bbox, image_obj.size))
            return QueryImagePreprocessResult(crop, "groundingdino_box", used_prompt, bbox, detection_score, 0.0, 0.0)

        best_index = select_query_sam_mask(masks, scores, image_obj.size)
        crop, mask_ratio, mask_bbox = build_masked_crop(image_obj, masks[best_index], bbox)
        return QueryImagePreprocessResult(
            crop,
            "groundingdino_sam",
            used_prompt,
            mask_bbox,
            detection_score,
            float(scores[best_index]),
            mask_ratio,
        )
    except Exception as exc:
        print(f"GroundingDINO/SAM preprocess failed, original image used: {exc}")
        return QueryImagePreprocessResult(image_obj, "original_fallback", prompt=prompt, error=str(exc))


def infer_design_details_from_image_features(image_features, clothing_label: str) -> list[str]:
    prompt_items = []
    for detail, prompts in DESIGN_DETAIL_PROMPTS.items():
        for prompt in prompts:
            prompt_items.append((detail, f"a photo of {prompt}"))

    cache_key = "default"
    text_features = DESIGN_PROMPT_EMBEDDING_CACHE.get(cache_key)
    if text_features is None:
        text_features = encode_texts_with_fashion_clip_api([prompt for _detail, prompt in prompt_items])
        DESIGN_PROMPT_EMBEDDING_CACHE[cache_key] = text_features

    similarities = (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()
    detail_scores = {}
    for (detail, _prompt), score in zip(prompt_items, similarities):
        detail_scores[detail] = max(detail_scores.get(detail, -1.0), float(score))

    relevant_groups = DESIGN_CONFLICT_GROUPS
    if normalize_text(clothing_label) in {"바지", "팬츠", "데님팬츠", "하의", "pants", "denim jeans"}:
        relevant_groups = ({"wide", "straight", "curved", "cargo", "shorts", "cropped"},)

    label_text = normalize_text(clothing_label)
    if label_text in {"바지", "팬츠", "데님팬츠", "하의", "pants", "denim jeans"}:
        relevant_groups = ({"wide", "straight", "curved", "cargo", "shorts", "cropped"},)

    selected = []
    for group in relevant_groups:
        ranked = sorted(
            ((detail, detail_scores.get(detail, -1.0)) for detail in group),
            key=lambda row: row[1],
            reverse=True,
        )
        if not ranked:
            continue
        best_detail, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        if best_score >= 0.12 and best_score - second_score >= 0.003:
            selected.append(best_detail)

    return sorted(set(selected))


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
    color_reason: str = "",
    detected_color: str = "",
    secondary_colors=None,
    is_mixed_color: bool = False,
    pattern: str = "",
    search_color_weights=None,
    image_design_details=None,
    color_candidates=None,
    image_category_result: ImageCategoryResult | None = None,
    category_filter_source_value: str = "",
):
    text_for_attributes = f"{query or ''} {intent.design or ''}"
    material = infer_attribute_from_text(text_for_attributes, MATERIAL_ALIASES)
    is_denim_context = material == "denim" or is_denim_query_context(query, main_categories, sub_categories)
    query_denim_tone = infer_denim_tone_from_text(query or "") if is_denim_context else ""
    design_denim_tone = infer_denim_tone_from_text(intent.design or "") if is_denim_context else ""
    text_denim_tone = query_denim_tone or design_denim_tone
    image_denim_tone = denim_tone_from_reason(color_reason)
    denim_tone = text_denim_tone or image_denim_tone
    text_design_details = set(infer_design_details(f"{query or ''} {intent.design or ''}"))
    fit = infer_attribute_from_text(text_for_attributes, FIT_ALIASES)
    if "straight" in text_design_details:
        fit = "straight"
    elif not fit:
        for detail in ("wide", "cropped", "shorts", "curved"):
            if detail in text_design_details:
                fit = detail
                break
    combined_design_details = sorted(text_design_details | set(image_design_details or []))
    explicit_color = normalize_color(intent.color)
    color_source = "explicit_text" if explicit_color else ("image" if image_color else "none")
    excluded_color = infer_excluded_color_from_text(query)
    image_category_result = image_category_result or ImageCategoryResult()
    attrs = {
        "main_category": main_categories[0] if main_categories else "",
        "sub_category": sub_categories[0] if sub_categories else "",
        "color": image_color,
        "color_source": color_source,
        "excluded_color": excluded_color,
        "target_color_confidence": "high" if explicit_color else color_confidence,
        "detected_color": detected_color or image_color,
        "secondary_colors": secondary_colors or [],
        "color_candidates": color_candidates or [],
        "is_mixed_color": is_mixed_color,
        "pattern": pattern,
        "search_color_weights": search_color_weights or {},
        "color_confidence": color_confidence,
        "color_reason": color_reason,
        "color_uncertain": color_confidence == "low",
        "material": material,
        "fit": fit,
        "is_denim_context": is_denim_context,
        "denim_tone": denim_tone,
        "denim_tone_source": "text" if text_denim_tone else ("image" if image_denim_tone else ""),
        "design_similarity_mode": is_design_similarity_query(query),
        "design_details": combined_design_details,
        "text_design_details": sorted(text_design_details),
        "image_design_details": sorted(set(image_design_details or [])),
        "image_category": image_category_result.category,
        "image_category_confidence": image_category_result.confidence,
        "image_category_scores": image_category_result.scores,
        "category_filter_source": category_filter_source_value or "relaxed",
    }
    attrs["target_color_group"] = color_group(image_color, attrs)
    return attrs


def build_search_warnings(intent: QueryIntent, color_result: ColorExtractionResult, query_attrs) -> list[str]:
    warnings = []
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    if color_mode in {"same", "different"}:
        has_color_target = bool(query_color_targets(intent, query_attrs))
        if not has_color_target:
            warnings.append("uploaded_image_color_not_detected")
        elif color_result.confidence == "low":
            warnings.append("uploaded_image_color_low_confidence")
    if query_attrs.get("design_similarity_mode") and not query_attrs.get("design_details"):
        warnings.append("design_similarity_uses_image_embedding_only")
    return warnings


def build_color_extracted_metadata(color_result: ColorExtractionResult) -> dict:
    return {
        "color": color_result.color,
        "detailed_color": resolve_detailed_color(color_result),
        "confidence": color_result.confidence,
        "reason": color_result.reason,
        "pattern": color_result.pattern,
        "secondary_colors": color_result.secondary_colors,
        "dominant_ratio": color_result.dominant_ratio,
        "is_mixed_color": color_result.is_mixed_color,
    }


def retrieval_relaxation_metadata(reason: str) -> dict:
    reason = normalize_text(reason)
    return {
        "retrieval_relaxed": bool(reason),
        "retrieval_relax_reason": reason,
    }


def build_enhanced_query(
    en_clothing_label: str,
    query_image_color: str,
    intent: QueryIntent,
    design_similarity_mode: bool = False,
    color_result: ColorExtractionResult | None = None,
):
    has_color_request = bool(intent.color.strip()) if intent.color else False
    has_color_condition = intent.color_mode in {"target", "same", "different"}
    has_design_request = bool(intent.design.strip()) if intent.design else False
    is_specific_query = has_color_request or has_color_condition or has_design_request
    query_color_prompt = search_query_color_for_mode(query_image_color, intent, color_result)

    if design_similarity_mode and not has_color_condition:
        enhanced_query = f"a photo of {en_clothing_label}"
        text_weight = 0.05
        image_weight = 0.95
    elif not is_specific_query:
        enhanced_query = f"a photo of {query_color_prompt} {en_clothing_label}" if query_color_prompt else f"a photo of {en_clothing_label}"
        text_weight = 0.10
        image_weight = 0.90
    elif intent.color_mode == "target" and has_color_request and not has_design_request:
        enhanced_query = f"a photo of {intent.color} {en_clothing_label}"
        text_weight = 0.45
        image_weight = 0.55
    elif intent.color_mode == "same" and not has_design_request:
        enhanced_query = f"a photo of {query_color_prompt} {en_clothing_label}" if query_color_prompt else f"a photo of {en_clothing_label}"
        text_weight = 0.25 if query_color_prompt else 0.15
        image_weight = 0.75 if query_color_prompt else 0.85
    elif intent.color_mode == "different" and not has_design_request:
        enhanced_query = compact_prompt("a photo of", intent.color, en_clothing_label)
        text_weight = 0.60 if has_color_request else 0.55
        image_weight = 0.40 if has_color_request else 0.45
    elif has_design_request and not has_color_request:
        enhanced_query = (
            f"a photo of {query_color_prompt} {intent.design} {en_clothing_label}"
            if intent.color_mode == "same" and query_color_prompt
            else compact_prompt("a photo of", intent.color if intent.color_mode == "different" else "", intent.design, en_clothing_label)
        )
        if intent.color_mode == "different":
            text_weight = 0.60
            image_weight = 0.40
        elif intent.color_mode == "same" and query_color_prompt:
            text_weight = 0.35
            image_weight = 0.65
        else:
            text_weight = 0.25
            image_weight = 0.75
    else:
        color_prompt = intent.color if intent.color_mode in {"target", "different"} else ""
        enhanced_query = compact_prompt("a photo of", color_prompt, intent.design, en_clothing_label)
        if intent.color_mode == "different":
            text_weight = 0.60
            image_weight = 0.40
        else:
            text_weight = 0.35
            image_weight = 0.65

    return {
        "enhanced_query": enhanced_query,
        "text_weight": text_weight,
        "image_weight": image_weight,
        "is_specific_query": is_specific_query,
        "design_similarity_mode": design_similarity_mode,
        "query_detailed_color": resolve_detailed_color(color_result),
    }


def should_exclude_candidate(item, item_color: str, target_color: str, color_mode: str, query_attrs):
    return False


def strict_color_target_active(color_mode: str, query_attrs) -> bool:
    if color_mode not in {"target", "same", "different"}:
        return False
    query_attrs = query_attrs or {}
    color_source = normalize_text(query_attrs.get("color_source"))
    confidence = normalize_text(
        query_attrs.get("target_color_confidence")
        or query_attrs.get("color_confidence")
    )
    return color_source == "explicit_text" or confidence == "high"


def candidate_conflict_reason(
    item_color: str,
    color_candidates,
    target_color_weights,
    color_mode: str,
    query_attrs,
    avoid_color_weights=None,
) -> str:
    if not strict_color_target_active(color_mode, query_attrs):
        return ""
    effective_target_weights = avoid_color_weights if color_mode == "different" and avoid_color_weights else target_color_weights
    if not effective_target_weights:
        return ""

    has_color_signal = bool(item_color or color_candidates)
    if not has_color_signal:
        return ""

    match_score, exact_match, group_match, _matched_target = best_color_match_score(
        item_color,
        color_candidates,
        effective_target_weights,
        query_attrs,
    )
    excluded_color = normalize_color((query_attrs or {}).get("excluded_color"))
    if color_mode == "different" and excluded_color and avoid_color_weights:
        has_match = exact_match
    else:
        has_match = exact_match or group_match or match_score > 0
    if color_mode in {"target", "same"} and not has_match:
        return "same_color_conflict"
    if color_mode == "different" and has_match:
        return "different_color_conflict"
    return ""


def candidate_bucket_from_conflict(conflict_reason: str) -> str:
    if conflict_reason:
        return "fallback_strict_conflict"
    return "preferred"


def candidate_bucket_priority(bucket: str) -> int:
    priorities = {
        "preferred": 0,
        "fallback_unknown_color": 1,
        "fallback_soft_conflict": 2,
        "fallback_strict_conflict": 3,
    }
    return priorities.get(bucket, 9)


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
    target_color_weights = query_color_targets(intent, query_attrs)
    target_color = next(iter(target_color_weights), "")
    avoid_color = ""
    avoid_color_weights = {}
    excluded_color = normalize_color((query_attrs or {}).get("excluded_color"))
    if color_mode == "different" and excluded_color:
        avoid_color = excluded_color
        avoid_color_weights = {excluded_color: 1.0}
    elif color_mode == "different" and normalize_color(intent.color):
        avoid_color = normalize_color((query_attrs or {}).get("color"))
        if avoid_color and avoid_color != normalize_color(intent.color):
            avoid_color_weights = {avoid_color: image_color_confidence_weight(query_attrs)}
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
        (
            named_adjustment,
            query_named_color,
            candidate_named_color,
            named_color_distance,
            dark_denim_adjustment,
            denim_dark_match,
            dark_denim_match_type,
        ) = fine_color_adjustment(
            item,
            color_candidates,
            query_attrs,
            color_mode,
        )
        if named_color_distance is None:
            tone_adjustment, candidate_denim_tone = denim_tone_adjustment(item, query_attrs, color_mode)
        else:
            tone_adjustment, candidate_denim_tone = 0.0, get_candidate_denim_tone(item)

        color_adjustment = 0.0
        candidate_match_score, color_matched, color_group_matched, matched_target_color = best_color_match_score(
            item_color,
            color_candidates,
            target_color_weights,
            query_attrs,
        )
        avoid_match_score, avoid_color_matched, avoid_color_group_matched, matched_avoid_color = best_color_match_score(
            item_color,
            color_candidates,
            avoid_color_weights,
            query_attrs,
        )
        dominant_match_score, dominant_color_matched, dominant_color_group_matched = dominant_color_match_score(
            item_color,
            matched_target_color or target_color,
            query_attrs,
        )
        conflict_reason = candidate_conflict_reason(
            item_color,
            color_candidates,
            target_color_weights,
            color_mode,
            query_attrs,
            avoid_color_weights,
        )
        candidate_bucket = candidate_bucket_from_conflict(conflict_reason)
        effective_match_score = candidate_match_score
        effective_color_matched = color_matched
        effective_group_matched = color_group_matched
        if not normalize_color(intent.color) and color_mode != "target":
            effective_match_score *= image_color_confidence_weight(query_attrs)
        item_color_group = color_group(display_color, query_attrs)
        if gray_same_dark_indigo_mismatch(
            item,
            color_candidates,
            candidate_named_color,
            candidate_denim_tone,
            query_attrs,
            color_mode,
        ):
            effective_match_score *= 0.35
        if design_similarity_mode:
            if color_mode == "target" and target_color:
                color_adjustment = 0.16 * effective_match_score if effective_color_matched or effective_group_matched else (-0.18 if has_color_signal else 0.0)
            elif color_mode == "same" and target_color:
                color_adjustment = 0.26 * effective_match_score if effective_color_matched or effective_group_matched else (-0.20 if has_color_signal else 0.0)
            elif (
                target_color
                and normalize_text(query_attrs.get("color_confidence")) == "high"
                and color_mode != "different"
            ):
                color_adjustment = 0.06 * effective_match_score if effective_color_matched or effective_group_matched else 0.0
            else:
                color_adjustment = 0.0
        elif color_mode == "target" and target_color:
            color_adjustment = 0.20 * effective_match_score if effective_color_matched or effective_group_matched else (-0.12 if has_color_signal else 0.0)
        elif color_mode == "same" and target_color:
            color_adjustment = 0.26 * effective_match_score if effective_color_matched or effective_group_matched else (-0.20 if has_color_signal else 0.0)
        elif color_mode == "different" and target_color:
            excluded_match = bool(excluded_color and avoid_color_matched)
            avoid_match = (
                excluded_match
                if excluded_color
                else (avoid_color_matched or avoid_color_group_matched)
            )
            if avoid_color_weights and avoid_match:
                color_adjustment = -0.32 * max(avoid_match_score, 0.5)
            elif avoid_color_weights and (effective_color_matched or effective_group_matched):
                color_adjustment = 0.22 * effective_match_score
            elif effective_color_matched or effective_group_matched:
                color_adjustment = -0.32 * max(effective_match_score, 0.5)
            else:
                color_adjustment = 0.22 if has_color_signal else 0.0
        elif color_mode == "ignore" and target_color and effective_color_matched and not design_similarity_mode:
            color_adjustment = 0.04 * effective_match_score

        color_adjustment *= denim_color_adjustment_scale(
            candidate_denim_tone,
            matched_target_color or target_color,
            color_mode,
            query_attrs,
        )

        final_score = (
            base_similarity
            + category_bonus
            + sub_category_bonus
            + color_adjustment
            + design_adjustment
            + tone_adjustment
            + named_adjustment
        )
        item["_ranking"] = {
            "base_similarity": round(base_similarity, 4),
            "final_score": round(final_score, 4),
            "color_mode": color_mode,
            "target_color": target_color,
            "target_color_weights": target_color_weights,
            "avoid_color": avoid_color,
            "avoid_color_weights": avoid_color_weights,
            "matched_target_color": matched_target_color,
            "matched_avoid_color": matched_avoid_color,
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
            "tone_adjustment": round(tone_adjustment, 4),
            "candidate_denim_tone": candidate_denim_tone,
            "target_denim_tone": query_attrs.get("denim_tone") or "",
            "named_color_adjustment": round(named_adjustment, 4),
            "dark_denim_adjustment": round(dark_denim_adjustment, 4),
            "denim_dark_match": denim_dark_match,
            "dark_denim_match_type": dark_denim_match_type,
            "query_named_color": query_named_color,
            "candidate_named_color": candidate_named_color,
            "named_color_distance": round(named_color_distance, 2) if named_color_distance is not None else None,
            "design_adjustment": round(design_adjustment, 4),
            "design_matches": design_matches,
            "design_conflicts": design_conflicts,
            "design_similarity_mode": design_similarity_mode,
            "candidate_bucket": candidate_bucket,
            "exclude_reason": conflict_reason,
            "retrieval_relaxed": bool(query_attrs.get("retrieval_relaxed")),
            "retrieval_relax_reason": query_attrs.get("retrieval_relax_reason") or "",
        }
        reranked.append((candidate_bucket_priority(candidate_bucket), final_score, item))

    reranked.sort(key=lambda row: (row[0], -row[1]))
    if color_mode != "different" or not target_color:
        return [item for _, _score, item in reranked[:limit]]

    selected = []
    skipped = []
    group_counts = {}
    for _priority, _score, item in reranked:
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
    search_warnings=None,
    query_attrs=None,
):
    intent_payload = intent.model_dump() if hasattr(intent, "model_dump") else intent.dict()
    print("\n[FashionCLIP Search Debug]")
    print(f"original_query={query}")
    print(f"intent={json.dumps(intent_payload, ensure_ascii=False)}")
    print(f"main_categories={main_categories}")
    print(f"sub_categories={sub_categories}")
    print(f"query_image_color={query_image_color}")
    if query_attrs:
        print(f"query_image_color_group={query_attrs.get('target_color_group') or ''}")
        print(f"query_denim_tone={query_attrs.get('denim_tone') or ''}")
        print(f"image_category={query_attrs.get('image_category') or ''}")
        print(f"image_category_confidence={query_attrs.get('image_category_confidence') or ''}")
        print(f"category_filter_source={query_attrs.get('category_filter_source') or ''}")
        print(f"image_preprocess_source={query_attrs.get('image_preprocess_source') or ''}")
        print(f"image_preprocess_prompt={query_attrs.get('image_preprocess_prompt') or ''}")
        print(f"image_preprocess_bbox={query_attrs.get('image_preprocess_bbox') or ''}")
        print(f"image_preprocess_detection_score={query_attrs.get('image_preprocess_detection_score') or 0.0}")
        print(f"image_preprocess_sam_score={query_attrs.get('image_preprocess_sam_score') or 0.0}")
        print(f"image_preprocess_mask_ratio={query_attrs.get('image_preprocess_mask_ratio') or 0.0}")
        if query_attrs.get("image_preprocess_error"):
            print(f"image_preprocess_error={query_attrs.get('image_preprocess_error')}")
        color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
        if color_mode == "ignore":
            print("query_color_targets=inactive")
        else:
            print(f"query_color_targets={json.dumps(query_color_targets(intent, query_attrs), ensure_ascii=False)}")
    print(f"color_confidence={color_confidence}")
    print(f"enhanced_query={enhanced_query}")
    print(f"design_similarity_mode={design_similarity_mode}")
    print(f"image_weight={image_weight}, text_weight={text_weight}, threshold={threshold}")
    print(f"rpc_filters={json.dumps(rpc_filters, ensure_ascii=False)}")
    print(f"raw_result_count={raw_result_count}")
    print(f"search_warnings={search_warnings or []}")
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
            f"candidate_color_candidates={ranking.get('candidate_color_candidates')}, "
            f"candidate_color_group={ranking.get('candidate_color_group')}, "
            f"matched_target_color={ranking.get('matched_target_color')}, "
            f"color_adjustment={ranking.get('color_adjustment')}, "
            f"tone_adjustment={ranking.get('tone_adjustment')}, "
            f"candidate_denim_tone={ranking.get('candidate_denim_tone')}, "
            f"named_color_adjustment={ranking.get('named_color_adjustment')}, "
            f"dark_denim_adjustment={ranking.get('dark_denim_adjustment')}, "
            f"denim_dark_match={ranking.get('denim_dark_match')}, "
            f"dark_denim_match_type={ranking.get('dark_denim_match_type')}, "
            f"query_named_color={ranking.get('query_named_color')}, "
            f"candidate_named_color={ranking.get('candidate_named_color')}, "
            f"named_color_distance={ranking.get('named_color_distance')}, "
            f"effective_color_match_score={ranking.get('effective_color_match_score')}, "
            f"main_category={item.get('main_category')}, "
            f"sub_category={item.get('sub_category')}"
        )


@app.post("/search")
async def search_clothes(file: UploadFile = File(None), query: str = Form(None)):
    if not file:
        raise HTTPException(status_code=400, detail="image is required")

    try:
        query = (query or "").strip()
        image_only_search = not query
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

        original_image_features = get_image_embedding(image_obj)
        image_features = original_image_features
        image_category_result = ImageCategoryResult()
        category_filter_source_value = "relaxed"

        # ── 룰 기반 카테고리 (항상 먼저 계산, LLM fallback 용도) ─────────────
        rule_main_categories, rule_sub_categories = extract_category_from_query(query)

        if image_only_search:
            main_categories = rule_main_categories
            sub_categories = rule_sub_categories
            intent = QueryIntent(reasoning="image only search", color="", color_mode="ignore", design="")
        else:
            # LLM이 category + intent를 한 번에 추론; 실패 시 룰 기반 fallback 자동 적용
            intent, main_categories, sub_categories = await analyze_query(
                query,
                rule_main_categories=rule_main_categories,
                rule_sub_categories=rule_sub_categories,
            )
            if should_use_image_category_reconciliation(
                query,
                rule_main_categories,
                rule_sub_categories,
                main_categories,
            ):
                try:
                    image_category_result = infer_image_category_from_features(original_image_features)
                except Exception as exc:
                    print(f"Image category inference failed, fallback used: {exc}")
                    image_category_result = ImageCategoryResult("", "skipped", [])
            main_categories, sub_categories = reconcile_category_filters(
                query,
                rule_main_categories,
                rule_sub_categories,
                main_categories,
                sub_categories,
                image_category_result=image_category_result,
            )
            category_filter_source_value = category_filter_source(
                query,
                rule_sub_categories,
                sub_categories,
                image_category_result,
            )

        if sub_categories:
            clothing_label = sub_categories[0]
        elif main_categories:
            clothing_label = main_categories[0]
        else:
            clothing_label = "clothing"

        en_clothing_label = LABEL_TO_EN.get(clothing_label, "fashion item")
        intent.design = sanitize_design_terms(intent.design, clothing_label, main_categories, sub_categories)
        design_similarity_mode = image_only_search or is_design_similarity_query(query)
        color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"

        image_preprocess_result = preprocess_query_image_with_dino_sam(
            image_obj,
            clothing_label,
            main_categories,
            sub_categories,
        )
        search_image_obj = image_preprocess_result.image
        if image_preprocess_result.source in {"groundingdino_sam", "groundingdino_box"}:
            image_features = get_image_embedding(search_image_obj)

        query_image_color = ""
        color_result = ColorExtractionResult("", "skipped", "color_not_requested")
        pattern_context_text = f"{query or ''} {intent.design or ''}"
        should_extract_image_attributes = (
            color_mode in {"same", "different"}
            or design_similarity_mode
            or should_run_pattern_classifier(pattern_context_text)
        )
        if should_extract_image_attributes:
            denim_context = is_denim_query_context(query, main_categories, sub_categories)
            color_result = extract_query_color_result(search_image_obj, denim_context, pattern_context_text)
            if color_mode in {"same", "different"}:
                query_image_color = normalize_color(color_result.color)
            elif design_similarity_mode and color_result.color:
                query_image_color = normalize_color(color_result.color)

        if image_only_search:
            enhanced_query = "a photo of fashion item"
            text_weight = 0.0
            image_weight = 1.0
            is_specific_query = False
        else:
            query_build = build_enhanced_query(
                en_clothing_label,
                query_image_color if color_result.confidence in {"high", "medium"} else "",
                intent,
                design_similarity_mode,
                color_result if color_result.confidence in {"high", "medium"} else None,
            )
            enhanced_query = query_build["enhanced_query"]
            text_weight = query_build["text_weight"]
            image_weight = query_build["image_weight"]
            is_specific_query = query_build["is_specific_query"]

        image_design_details = (
            infer_design_details_from_image_features(image_features, clothing_label)
            if design_similarity_mode
            else []
        )
        if text_weight > 0:
            text_features = get_text_embedding(enhanced_query)
            query_embedding = F.normalize((image_features * image_weight) + (text_features * text_weight), p=2, dim=-1)
        else:
            query_embedding = F.normalize(image_features, p=2, dim=-1)
        query_embedding_list = query_embedding.squeeze().tolist()

        has_design_request = bool((intent.design or "").strip())
        threshold = 0.23 if image_only_search else (0.23 if design_similarity_mode else (0.22 if color_mode == "same" and has_design_request else (0.28 if color_mode == "same" else (0.30 if is_specific_query else 0.35))))
        if color_mode in {"same", "different"}:
            match_count = SAME_COLOR_MATCH_COUNT
        elif image_only_search or design_similarity_mode:
            match_count = DESIGN_SIMILARITY_MATCH_COUNT
        else:
            match_count = 100
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
        retrieval_relax_reason = ""
        if color_mode == "same" and len(response.data or []) < 20:
            threshold = 0.18
            retrieval_relax_reason = "same_color_low_count"
            response = supabase.rpc("match_clothes_fashion", {
                "query_embedding": query_embedding_list,
                "match_threshold": threshold,
                "match_count": match_count,
                **rpc_filters,
            }).execute()
        if design_similarity_mode and not response.data and sub_categories:
            threshold = 0.20
            retrieval_relax_reason = "design_no_subcategory_results"
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
            retrieval_relax_reason = "design_relaxed_subcategory_filter"
            response = supabase.rpc("match_clothes_fashion", {
                "query_embedding": query_embedding_list,
                "match_threshold": threshold,
                "match_count": match_count,
                **rpc_filters,
            }).execute()

        ranking_image_color = normalize_color(color_result.color) if color_result.color else query_image_color
        query_attrs = build_query_attrs(
            main_categories,
            sub_categories,
            ranking_image_color,
            query,
            intent,
            color_result.confidence,
            color_result.reason,
            color_result.color,
            color_result.secondary_colors,
            color_result.is_mixed_color,
            color_result.pattern,
            color_result.search_color_weights,
            image_design_details,
            color_result.candidates,
            image_category_result,
            category_filter_source_value,
        )
        query_attrs["image_only_search"] = image_only_search
        query_attrs["design_similarity_mode"] = design_similarity_mode
        query_attrs["image_preprocess_source"] = image_preprocess_result.source
        query_attrs["image_preprocess_prompt"] = image_preprocess_result.prompt
        query_attrs["image_preprocess_bbox"] = image_preprocess_result.bbox
        query_attrs["image_preprocess_detection_score"] = round(image_preprocess_result.detection_score, 4)
        query_attrs["image_preprocess_sam_score"] = round(image_preprocess_result.sam_score, 4)
        query_attrs["image_preprocess_mask_ratio"] = round(image_preprocess_result.mask_ratio, 4)
        query_attrs["image_preprocess_error"] = image_preprocess_result.error
        query_attrs.update(retrieval_relaxation_metadata(retrieval_relax_reason))
        search_warnings = build_search_warnings(intent, color_result, query_attrs)
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
            search_warnings=search_warnings,
            query_attrs=query_attrs,
        )

        return {
            "message": "Success",
            "model": FASHION_CLIP_MODEL_ID,
            "enhanced_query": enhanced_query,
            "color_extracted": build_color_extracted_metadata(color_result),
            "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.dict(),
            "query_image_attributes": query_attrs,
            "search_warnings": search_warnings,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("FashionCLIP search server error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
