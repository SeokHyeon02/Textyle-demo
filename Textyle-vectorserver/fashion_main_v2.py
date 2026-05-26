"""TexTyle FashionCLIP Text-only Search Server (v2).

Differences from fashion_main.py (v1):
  - No image embedding. Only text-vector search.
  - K-means color extraction (fashion_color_extraction.py) runs unconditionally
    and feeds the extracted color into Gemini.
  - Gemini receives the image, the user's Korean query, and the pre-extracted
    color info. It returns a long English design_description (40-80 words)
    covering silhouette/fit/material/details, plus an `enhanced_query` that
    combines target color + design.
  - is_fashion validation is performed by Gemini (the CLIP zero-shot validator
    used in v1 is removed).
  - The reranker is copied verbatim from v1 so color matching, denim tone
    matching, and design-detail scoring behave identically.

Endpoint port: 8002 (v1 keeps using 8001).

To run:
    python -m uvicorn fashion_main_v2:app --host 0.0.0.0 --port 8002 --reload
"""

import io
import importlib
import json
import os
import re
import traceback
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import numpy as np

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
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
FASHION_CLIP_MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
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

app = FastAPI(title="TexTyle FashionCLIP Text-only Search Server (v2)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading FashionCLIP (text-only mode)... model={FASHION_CLIP_MODEL_ID}, device={device}")
model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL_ID)
model.eval()
print("FashionCLIP loaded (text encoder will be used for search)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class QueryIntent(BaseModel):
    reasoning: str = Field(default="", description="brief analysis reasoning")
    is_fashion: bool = Field(default=True, description="true if image shows clothing/fashion")
    color: str = Field(default="", description="target color, empty if absent")
    color_mode: str = Field(default="ignore", description="target/same/different/ignore")
    design: str = Field(default="", description="short design keywords (rerank legacy)")
    design_description: str = Field(default="", description="long English description of design, silhouette, material, details — NO color words")
    enhanced_query: str = Field(default="", description="final search query in English combining color + design_description")


@dataclass
class ImageQualityResult:
    is_usable: bool
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


# ---------------------------------------------------------------------------
# Constants (copied verbatim from fashion_main.py — preserved for rerank parity)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

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
    for term in LIGHT_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "light"
    for term in DARK_DENIM_TONE_TERMS:
        term_text = normalize_text(term)
        term_compact = re.sub(r"[^a-z0-9가-힣]+", "", term_text)
        if term_text in normalized or (term_compact and term_compact in compact):
            return "dark"
    return ""


def denim_tone_adjustment(item, query_attrs, color_mode: str) -> tuple[float, str]:
    if color_mode != "same" or not query_attrs.get("is_denim_context"):
        return 0.0, ""
    target_tone = query_attrs.get("denim_tone") or ""
    if target_tone not in {"dark", "light", "medium"}:
        return 0.0, ""

    candidate_text = " ".join(
        str(item.get(field) or "")
        for field in ("name", "brand_name", "sub_category")
    )
    candidate_tone = infer_denim_tone_from_text(candidate_text)
    if not candidate_tone or candidate_tone == target_tone:
        return 0.0, candidate_tone
    if target_tone == "medium":
        if candidate_tone == "light":
            return -0.08, candidate_tone
        if candidate_tone == "dark":
            return -0.05, candidate_tone
    return -0.10, candidate_tone


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
            "rgb": candidate.get("rgb"),
            "named_color": normalize_text(candidate.get("named_color")),
            "named_rgb": candidate.get("named_rgb"),
        })
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


def dark_denim_match_type(item, candidate_named: str, candidate_tone: str, query_attrs, query_named: str = "") -> str:
    if not dark_denim_query_active(query_attrs):
        return "none"

    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_text(candidate_tone)
    if candidate_named == "lightblue" or candidate_tone == "light":
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
    if gray_washed_dark_denim_query(query_attrs, query_named) and (
        candidate_named in {"gray", "grey", "dimgray", "dimgrey", "darkgray", "darkgrey", "darkslategray", "darkslategrey"}
        or has_washed
    ):
        return "washed_gray"
    if candidate_named in {"midnightblue", "navy", "darkblue"} or has_dark_indigo:
        return "dark_indigo"
    if has_black or candidate_named == "black":
        return "black_only" if not has_washed and not has_dark_indigo else "washed_gray"
    if dark_denim_candidate_match(candidate_named, candidate_tone):
        return "dark_indigo" if candidate_named in {"midnightblue", "navy", "darkblue"} else "none"
    return "none"


def dark_denim_candidate_match(candidate_named: str, candidate_tone: str) -> bool:
    candidate_named = normalize_text(candidate_named)
    candidate_tone = normalize_text(candidate_tone)
    return candidate_named in DENIM_DARK_NAMED_COLORS or candidate_tone == "dark"


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
    if candidate_named == "lightblue" or candidate_tone == "light":
        return -0.10, False
    if dark_denim_candidate_match(candidate_named, candidate_tone):
        return 0.03, True
    return 0.0, False


def fine_color_adjustment(item, color_candidates, query_attrs, color_mode: str) -> tuple[float, str, str, float | None, float, bool, str]:
    if color_mode != "same":
        return 0.0, "", "", None, 0.0, False, "none"
    query_info = query_named_color_info(query_attrs)
    candidate_info = candidate_named_color_info(item, color_candidates)
    candidate_tone = infer_denim_tone_from_text(
        " ".join(str(item.get(field) or "") for field in ("name", "brand_name", "sub_category"))
    )
    dark_match_type = dark_denim_match_type(
        item,
        candidate_info.get("named_color", ""),
        candidate_tone,
        query_attrs,
        query_info.get("named_color", ""),
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
    ):
        adjustment = 0.0
    adjustment = denim_named_tone_adjustment(query_named, candidate_named, adjustment, query_attrs)
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


def query_color_targets(intent: QueryIntent, query_attrs) -> dict[str, float]:
    query_attrs = query_attrs or {}
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    explicit_color = normalize_color(intent.color)
    if explicit_color:
        return {explicit_color: 1.0}

    targets = {}
    primary_color = normalize_color(query_attrs.get("color"))
    color_confidence = normalize_text(query_attrs.get("color_confidence"))
    if primary_color:
        targets[primary_color] = 1.0

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
                targets[normalized] = max(targets.get(normalized, 0.0), min(score, 1.0))

    for color in query_attrs.get("secondary_colors") or []:
        normalized = normalize_color(color)
        if normalized:
            if color_mode == "same" and primary_color and color_confidence in {"high", "medium"}:
                continue
            targets[normalized] = max(targets.get(normalized, 0.0), 0.45)

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


# ---------------------------------------------------------------------------
# Color extraction pipeline (Lab / segmentation / K-means)
# ---------------------------------------------------------------------------

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


def extract_query_color_result(image_obj: Image.Image, denim_context: bool, pattern_context_text: str) -> ColorExtractionResult:
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


# ---------------------------------------------------------------------------
# Text encoder (CLIP) — text-only mode
# ---------------------------------------------------------------------------

def encode_text_with_fashion_clip_api(text: str):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        max_length=77,
        padding="max_length",
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
    return F.normalize(features, p=2, dim=-1)


def get_text_embedding(text: str):
    return encode_text_with_fashion_clip_api(text)


# ---------------------------------------------------------------------------
# Gemini multimodal intent analyzer (NEW — replaces v1's text-only analyzer)
# ---------------------------------------------------------------------------

GEMINI_V2_SYSTEM_PROMPT = """\
You are a fashion search query analyzer. You receive:
1. A user's Korean text query about clothing.
2. An image of a clothing item (or related to clothing).
3. Pre-extracted color information (from pixel-level K-means analysis — very accurate).

Your task:
- TRUST the pre-extracted color absolutely. DO NOT re-guess colors from the image.
- Describe the clothing's DESIGN, SILHOUETTE, FIT, MATERIAL, and DETAILS in English ONLY.
- DO NOT include any color hue word (no "red", "blue", "grey", "navy", "black", "white", etc.) inside `design_description`.
- Identify the user's intent: what target color do they want, and what is their color_mode?

Fields to return as JSON:
- reasoning: 1 short sentence summarizing your analysis (English).
- is_fashion: true if the image clearly shows a clothing/fashion item; false for non-fashion (food, landscape, faces only, etc.).
- color: target color the user wants (canonical lowercase English: black/white/gray/blue/red/green/yellow/brown/pink/purple/orange). "" if the user did not ask for a specific color.
- color_mode:
    * "target"    -> user wants a specific color (e.g. "검정 후드티").
    * "same"      -> user wants the same color as the image.
    * "different" -> user wants a color DIFFERENT from the image.
    * "ignore"    -> color is not relevant.
- design: short keyword summary (2-4 English words like "oversized hoodie"). For legacy rerank usage.
- design_description: LONG detailed English description (40-80 words) covering:
    * silhouette and fit (oversized / relaxed / regular / slim / cropped)
    * shoulders, sleeves, body shape
    * length (cropped / regular / long)
    * material/fabric (cotton fleece, denim, leather, knit, etc.)
    * surface details (kangaroo pocket, ribbed cuffs, drawstring, zipper, buttons, embroidery, prints, patches, distressed, hood, collar, panels, etc.)
    * pattern TYPE only if non-solid (striped / checked / floral) — but NEVER a color name.
- enhanced_query: ONE final English search query combining the target color + the design description. Format guideline: "a photo of <color> <garment-type> <key details>". 60-100 words OK.
    * If color_mode is "target": use the user's requested color word.
    * If color_mode is "same": use the pre-extracted dominant color word.
    * If color_mode is "different": include the user's requested color (if any). Do NOT include the image's original color.
    * If color_mode is "ignore": no color word.

Return JSON only — no markdown, no commentary.
"""


async def analyze_query_intent_v2(
    user_query: str,
    image_obj: Image.Image,
    color_result: ColorExtractionResult,
) -> QueryIntent:
    fallback_color = infer_color_from_text(user_query)
    fallback_color_mode = infer_color_mode(user_query)
    if fallback_color_mode == "ignore" and fallback_color:
        fallback_color_mode = "target"

    fallback_design = " ".join(
        value for value in [
            infer_attribute_from_text(user_query, FIT_ALIASES),
            infer_attribute_from_text(user_query, MATERIAL_ALIASES),
        ] if value
    )

    fallback_enhanced_color = (
        fallback_color
        if fallback_color_mode in {"target", "different"}
        else (color_result.color if fallback_color_mode == "same" else "")
    )
    fallback_enhanced = " ".join(
        word for word in [
            "a photo of",
            fallback_enhanced_color or color_result.color,
            fallback_design,
            "clothing",
        ] if word
    ).strip()

    fallback = QueryIntent(
        reasoning="rule based fallback",
        is_fashion=True,
        color=fallback_color,
        color_mode=fallback_color_mode,
        design=fallback_design,
        design_description=fallback_design or "clothing",
        enhanced_query=fallback_enhanced or "a photo of clothing",
    )

    if not gemini_client or genai_types is None:
        return fallback

    # Serialize image as JPEG bytes for Gemini Vision
    img_buffer = io.BytesIO()
    image_obj.convert("RGB").save(img_buffer, format="JPEG", quality=90)
    image_bytes = img_buffer.getvalue()

    color_summary = (
        "Pre-extracted color information (highly accurate, from pixel analysis):\n"
        f"- Dominant color: {color_result.color or 'unknown'} "
        f"(ratio {color_result.dominant_ratio:.0%}, confidence {color_result.confidence})\n"
        f"- Secondary colors: {', '.join(color_result.secondary_colors) or 'none'}\n"
        f"- Mixed/multi-color: {color_result.is_mixed_color}\n"
        f"- Pattern: {color_result.pattern or 'solid'}\n"
        f"- Color weights: {color_result.search_color_weights}"
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                f"User query: {user_query}\n\n{color_summary}",
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=GEMINI_V2_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=QueryIntent,
                temperature=0.1,
            ),
        )
        if getattr(response, "parsed", None):
            parsed = response.parsed
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            data = json.loads(response.text)

        intent = QueryIntent(
            reasoning=data.get("reasoning", ""),
            is_fashion=bool(data.get("is_fashion", True)),
            color=normalize_color(data.get("color", "")),
            color_mode=data.get("color_mode", "ignore"),
            design=data.get("design", ""),
            design_description=data.get("design_description", ""),
            enhanced_query=data.get("enhanced_query", ""),
        )
        if intent.color_mode not in {"target", "same", "different", "ignore"}:
            intent.color_mode = "ignore"
        if intent.color_mode == "ignore":
            inferred_mode = infer_color_mode(user_query)
            if inferred_mode != "ignore":
                intent.color_mode = inferred_mode
        if normalize_color(intent.color) and intent.color_mode == "ignore":
            intent.color_mode = "target"
        if not intent.enhanced_query:
            intent.enhanced_query = fallback.enhanced_query
        return intent
    except Exception as exc:
        print(f"Gemini v2 analysis failed, fallback used: {exc}")
        return fallback


# ---------------------------------------------------------------------------
# Rerank helpers (copied verbatim from fashion_main.py for parity)
# ---------------------------------------------------------------------------

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
):
    design_text = f"{intent.design or ''} {intent.design_description or ''}"
    text_for_attributes = f"{query or ''} {design_text}"
    material = infer_attribute_from_text(text_for_attributes, MATERIAL_ALIASES)
    is_denim_context = material == "denim" or is_denim_query_context(query, main_categories, sub_categories)
    text_design_details = set(infer_design_details(text_for_attributes))
    combined_design_details = sorted(text_design_details | set(image_design_details or []))
    attrs = {
        "main_category": main_categories[0] if main_categories else "",
        "sub_category": sub_categories[0] if sub_categories else "",
        "color": image_color,
        "detected_color": detected_color or image_color,
        "secondary_colors": secondary_colors or [],
        "color_candidates": color_candidates or [],
        "is_mixed_color": is_mixed_color,
        "pattern": pattern,
        "search_color_weights": search_color_weights or {},
        "color_confidence": color_confidence,
        "color_reason": color_reason,
        "color_uncertain": color_confidence == "low",
        "is_denim_context": is_denim_context,
        "denim_tone": denim_tone_from_reason(color_reason),
        "design_similarity_mode": is_design_similarity_query(query),
        "design_details": combined_design_details,
        "text_design_details": sorted(text_design_details),
        "image_design_details": sorted(set(image_design_details or [])),
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
    target_color_weights = query_color_targets(intent, query_attrs)
    target_color = next(iter(target_color_weights), "")
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
            dark_denim_adj,
            denim_dark_match,
            dark_denim_match_type_value,
        ) = fine_color_adjustment(
            item,
            color_candidates,
            query_attrs,
            color_mode,
        )
        if named_color_distance is None:
            tone_adjustment, candidate_denim_tone = denim_tone_adjustment(item, query_attrs, color_mode)
        else:
            tone_adjustment, candidate_denim_tone = 0.0, infer_denim_tone_from_text(
                " ".join(str(item.get(field) or "") for field in ("name", "brand_name", "sub_category"))
            )

        color_adjustment = 0.0
        candidate_match_score, color_matched, color_group_matched, matched_target_color = best_color_match_score(
            item_color,
            color_candidates,
            target_color_weights,
            query_attrs,
        )
        dominant_match_score, dominant_color_matched, dominant_color_group_matched = dominant_color_match_score(
            item_color,
            matched_target_color or target_color,
            query_attrs,
        )
        effective_match_score = candidate_match_score
        effective_color_matched = color_matched
        effective_group_matched = color_group_matched
        if not normalize_color(intent.color) and color_mode != "target":
            effective_match_score *= image_color_confidence_weight(query_attrs)
        item_color_group = color_group(display_color, query_attrs)
        if design_similarity_mode:
            color_adjustment = 0.0
        elif color_mode == "target" and target_color:
            color_adjustment = 0.20 * effective_match_score if effective_color_matched or effective_group_matched else (-0.12 if has_color_signal else 0.0)
        elif color_mode == "same" and target_color:
            color_adjustment = 0.26 * effective_match_score if effective_color_matched or effective_group_matched else (-0.20 if has_color_signal else 0.0)
        elif color_mode == "different" and target_color:
            if effective_color_matched or effective_group_matched:
                color_adjustment = -0.32 * max(effective_match_score, 0.5)
            else:
                color_adjustment = 0.22 if has_color_signal else 0.0
        elif color_mode == "ignore" and target_color and effective_color_matched and not design_similarity_mode:
            color_adjustment = 0.04 * effective_match_score

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
            "matched_target_color": matched_target_color,
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
            "dark_denim_adjustment": round(dark_denim_adj, 4),
            "denim_dark_match": denim_dark_match,
            "dark_denim_match_type": dark_denim_match_type_value,
            "query_named_color": query_named_color,
            "candidate_named_color": candidate_named_color,
            "named_color_distance": round(named_color_distance, 2) if named_color_distance is not None else None,
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
    design_similarity_mode: bool,
    threshold: float,
    rpc_filters,
    raw_result_count: int,
    results,
    search_warnings=None,
    query_attrs=None,
):
    intent_payload = intent.model_dump() if hasattr(intent, "model_dump") else intent.dict()
    print("\n[FashionCLIP v2 Search Debug]")
    print(f"original_query={query}")
    print(f"intent={json.dumps(intent_payload, ensure_ascii=False)}")
    print(f"main_categories={main_categories}")
    print(f"sub_categories={sub_categories}")
    print(f"query_image_color={query_image_color}")
    if query_attrs:
        print(f"query_image_color_group={query_attrs.get('target_color_group') or ''}")
        print(f"query_denim_tone={query_attrs.get('denim_tone') or ''}")
        color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
        if design_similarity_mode or color_mode == "ignore":
            print("query_color_targets=inactive")
        else:
            print(f"query_color_targets={json.dumps(query_color_targets(intent, query_attrs), ensure_ascii=False)}")
    print(f"color_confidence={color_confidence}")
    print(f"enhanced_query={enhanced_query}")
    print(f"design_similarity_mode={design_similarity_mode}")
    print(f"threshold={threshold}")
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
            f"candidate_color_group={ranking.get('candidate_color_group')}, "
            f"matched_target_color={ranking.get('matched_target_color')}, "
            f"color_adjustment={ranking.get('color_adjustment')}, "
            f"tone_adjustment={ranking.get('tone_adjustment')}, "
            f"named_color_adjustment={ranking.get('named_color_adjustment')}, "
            f"main_category={item.get('main_category')}, "
            f"sub_category={item.get('sub_category')}"
        )


# ---------------------------------------------------------------------------
# /search endpoint
# ---------------------------------------------------------------------------

def _run_rpc(query_embedding_list, threshold, match_count, rpc_filters):
    return supabase.rpc("match_clothes_fashion", {
        "query_embedding": query_embedding_list,
        "match_threshold": threshold,
        "match_count": match_count,
        **rpc_filters,
    }).execute()


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

        # ─── STAGE 0a: K-means color extraction (always run) ───
        main_categories, sub_categories = extract_category_from_query(query)
        denim_context_pre = is_denim_query_context(query, main_categories, sub_categories)
        pattern_context_text = f"{query or ''}"
        color_result = extract_query_color_result(image_obj, denim_context_pre, pattern_context_text)

        # ─── STAGE 0b: Gemini multimodal intent analysis ───
        if image_only_search:
            same_color_word = color_result.color or ""
            intent = QueryIntent(
                reasoning="image only search",
                is_fashion=True,
                color="",
                color_mode="same" if same_color_word else "ignore",
                design="",
                design_description="",
                enhanced_query=(
                    f"a photo of {same_color_word} clothing".strip()
                    if same_color_word
                    else "a photo of fashion item"
                ),
            )
        else:
            intent = await analyze_query_intent_v2(query, image_obj, color_result)

        # ─── Reject non-fashion images (Gemini judges) ───
        if not intent.is_fashion:
            raise HTTPException(
                status_code=400,
                detail="의류가 명확히 보이는 이미지를 업로드해주세요.",
            )

        # Sanitize design terms vs. category labels (legacy behavior)
        if sub_categories:
            clothing_label = sub_categories[0]
        elif main_categories:
            clothing_label = main_categories[0]
        else:
            clothing_label = "clothing"
        intent.design = sanitize_design_terms(intent.design, clothing_label, main_categories, sub_categories)
        design_similarity_mode = image_only_search or is_design_similarity_query(query)
        color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"

        query_image_color = ""
        if color_mode in {"same", "different"} or design_similarity_mode:
            query_image_color = normalize_color(color_result.color)

        # ─── STAGE 1: Text-only vector search ───
        enhanced_query = (
            intent.enhanced_query.strip()
            if intent.enhanced_query and intent.enhanced_query.strip()
            else "a photo of fashion item"
        )
        is_specific_query = bool(intent.color or intent.design or intent.design_description)

        query_embedding = get_text_embedding(enhanced_query)
        query_embedding_list = query_embedding.squeeze().tolist()

        has_design_request = bool((intent.design_description or "").strip()) or bool((intent.design or "").strip())
        threshold = (
            0.23 if image_only_search
            else (0.23 if design_similarity_mode
            else (0.22 if color_mode == "same" and has_design_request
            else (0.28 if color_mode == "same"
            else (0.30 if is_specific_query else 0.35))))
        )
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
        response = _run_rpc(query_embedding_list, threshold, match_count, rpc_filters)

        if color_mode == "same" and len(response.data or []) < 20:
            threshold = 0.18
            response = _run_rpc(query_embedding_list, threshold, match_count, rpc_filters)
        if design_similarity_mode and not response.data and sub_categories:
            threshold = 0.20
            response = _run_rpc(query_embedding_list, threshold, match_count, rpc_filters)
        if design_similarity_mode and not response.data and sub_categories and main_categories:
            rpc_filters = {
                "filter_main_categories": main_categories,
                "filter_sub_categories": None,
            }
            response = _run_rpc(query_embedding_list, threshold, match_count, rpc_filters)

        # ─── STAGE 2: Rerank ───
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
            [],  # v2: no image-derived design details
            color_result.candidates,
        )
        query_attrs["image_only_search"] = image_only_search
        query_attrs["design_similarity_mode"] = design_similarity_mode
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
            "version": "v2-textonly",
            "enhanced_query": enhanced_query,
            "design_description": intent.design_description,
            "color_extracted": {
                "color": color_result.color,
                "confidence": color_result.confidence,
                "pattern": color_result.pattern,
                "secondary_colors": color_result.secondary_colors,
                "dominant_ratio": color_result.dominant_ratio,
                "is_mixed_color": color_result.is_mixed_color,
            },
            "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.dict(),
            "query_image_attributes": query_attrs,
            "search_warnings": search_warnings,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("FashionCLIP v2 search server error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
