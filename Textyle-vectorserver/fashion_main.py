import io
import json
import os
import re
import traceback
from typing import Optional

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
from PIL import Image
from pydantic import BaseModel, Field
from supabase import Client, create_client
from transformers import CLIPModel, CLIPProcessor


"""
Required Supabase RPC:

create or replace function match_clothes_fashion(
  query_embedding vector(512),
  match_threshold float,
  match_count int,
  filter_main_categories text[] default null,
  filter_sub_categories text[] default null
)
returns table (
  brand text,
  name text,
  product_link text,
  image_url text,
  main_category text,
  sub_category text,
  dominant_color text,
  fit text,
  material text,
  similarity float
)
language sql stable
as $$
  select
    clothes.brand,
    clothes.name,
    clothes.product_link,
    clothes.image_url,
    clothes.main_category,
    clothes.sub_category,
    clothes.dominant_color,
    clothes.fit,
    clothes.material,
    1 - (clothes.fashion_embedding <=> query_embedding) as similarity
  from clothes
  where clothes.fashion_embedding is not null
    and (filter_main_categories is null or clothes.main_category = any(filter_main_categories))
    and (filter_sub_categories is null or clothes.sub_category = any(filter_sub_categories))
    and 1 - (clothes.fashion_embedding <=> query_embedding) > match_threshold
  order by clothes.fashion_embedding <=> query_embedding
  limit match_count;
$$;
"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
FASHION_CLIP_MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and genai else None

app = FastAPI(title="TexTyle FashionCLIP Search Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading FashionCLIP... model={FASHION_CLIP_MODEL_ID}, device={device}")
model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL_ID)
model.eval()
print("FashionCLIP loaded")


class QueryIntent(BaseModel):
    reasoning: str = Field(description="query analysis reasoning")
    color: str = Field(description="target color, empty string if absent")
    color_mode: str = Field(description="target, same, different, or ignore")
    design: str = Field(description="fit, material, length, or design phrase")


CATEGORY_KEYWORDS = {
    "후드티": ("상의", "후드티"),
    "후디": ("상의", "후드티"),
    "맨투맨": ("상의", "맨투맨"),
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
    "니트/스웨터": "knit sweater",
    "가디건": "cardigan",
    "레더자켓": "leather jacket",
    "블루종/MA-1": "blouson jacket",
    "데님팬츠": "denim jeans",
    "슬랙스/정장 팬츠": "slacks trousers",
    "트레이닝/조거 팬츠": "jogger pants",
    "카고팬츠": "cargo pants",
    "숏팬츠": "shorts",
}

COLOR_ALIASES = {
    "black": {"black", "블랙", "검정", "검정색", "까만색", "흑색", "흑청"},
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

DIFFERENT_COLOR_PATTERNS = ("색상이 다른", "색상 다른", "색이 다른", "다른 색", "색만 다른", "컬러가 다른", "컬러 다른")
SAME_COLOR_PATTERNS = ("같은 색", "색상은 그대로", "색은 그대로", "동일한 색", "같은 컬러", "컬러는 그대로")

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
    labels = " ".join([*(main_categories or []), *(sub_categories or []), query or ""]).lower()
    denim_terms = ("denim", "jean", "jeans", "데님", "청바지", "흑청", "진청", "중청", "연청")
    return any(term in labels for term in denim_terms)


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


def extract_dominant_color(image_obj: Image.Image, denim_context: bool = False) -> str:
    width, height = image_obj.size
    cropped = image_obj.crop((
        int(width * 0.2),
        int(height * 0.15),
        int(width * 0.8),
        int(height * 0.85),
    )).resize((64, 64))
    pixels = list(cropped.getdata())
    filtered_pixels = [
        (r, g, b)
        for r, g, b in pixels
        if not ((r > 242 and g > 242 and b > 242) or (r < 12 and g < 12 and b < 12))
    ] or pixels

    if denim_context:
        denim_color = classify_denim_color_from_pixels(filtered_pixels)
        if denim_color:
            return denim_color

    avg = tuple(sum(channel) / len(filtered_pixels) for channel in zip(*filtered_pixels))
    return min(
        COLOR_RGB_CENTROIDS,
        key=lambda color: sum((avg[idx] - COLOR_RGB_CENTROIDS[color][idx]) ** 2 for idx in range(3)),
    )


async def analyze_query_intent(user_query: str) -> QueryIntent:
    fallback = QueryIntent(
        reasoning="rule based fallback",
        color=infer_attribute_from_text(user_query, COLOR_ALIASES),
        color_mode=infer_color_mode(user_query),
        design=" ".join(
            value for value in [
                infer_attribute_from_text(user_query, FIT_ALIASES),
                infer_attribute_from_text(user_query, MATERIAL_ALIASES),
            ] if value
        ),
    )

    if not gemini_client or genai_types is None:
        return fallback

    system_prompt = """
You are a fashion search query analyzer.
Extract color_mode, color, and design from the Korean user query.
color_mode must be one of target, same, different, ignore.
Use target for an explicit requested color, same for same-color requests,
different for different-color requests, and ignore when color is irrelevant.
design should contain fit, material, length, or detail in English. Use "" if absent.
Return JSON only.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=f"{system_prompt}\nUser query: {user_query}",
            config=genai_types.GenerateContentConfig(
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


def get_image_embedding(image_obj: Image.Image):
    clip_image = crop_center_region(image_obj.convert("RGB"))
    inputs = processor(images=clip_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = extract_feature_tensor(model.get_image_features(**inputs))
        image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features


def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = extract_feature_tensor(model.get_text_features(**inputs))
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features


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


def build_query_attrs(main_categories, sub_categories, image_color: str, query: str, intent: QueryIntent):
    text_for_attributes = f"{query or ''} {intent.design or ''}"
    material = infer_attribute_from_text(text_for_attributes, MATERIAL_ALIASES)
    fit = infer_attribute_from_text(text_for_attributes, FIT_ALIASES)
    if not material and is_denim_query_context(query, main_categories, sub_categories):
        material = "denim"
    return {
        "main_category": main_categories[0] if main_categories else "",
        "sub_category": sub_categories[0] if sub_categories else "",
        "color": image_color,
        "material": material,
        "fit": fit,
    }


def rerank_results(results, intent: QueryIntent, query_attrs, limit: int = 10):
    color_mode = intent.color_mode if intent.color_mode in {"target", "same", "different", "ignore"} else "ignore"
    target_color = normalize_color(intent.color) or normalize_color(query_attrs.get("color"))
    reranked = []

    for item in results or []:
        base_similarity = float(item.get("similarity", item.get("score", 0.0)) or 0.0)
        item_color = normalize_color(item.get("dominant_color") or item.get("color"))
        item_material = normalize_attribute(item.get("material"), MATERIAL_ALIASES)
        item_fit = normalize_attribute(item.get("fit"), FIT_ALIASES)

        category_bonus = score_exact_or_unknown(item.get("main_category"), query_attrs.get("main_category"), 0.14, -0.18)
        sub_category_bonus = score_exact_or_unknown(item.get("sub_category"), query_attrs.get("sub_category"), 0.18, -0.10)
        material_bonus = score_exact_or_unknown(item_material, query_attrs.get("material"), 0.12, -0.04)
        fit_bonus = score_exact_or_unknown(item_fit, query_attrs.get("fit"), 0.08, -0.02)

        color_adjustment = 0.0
        if color_mode == "target" and target_color:
            color_adjustment = 0.20 if item_color == target_color else (-0.12 if item_color else 0.0)
        elif color_mode == "same" and target_color:
            color_adjustment = 0.16 if item_color == target_color else (-0.14 if item_color else 0.0)
        elif color_mode == "different" and target_color:
            color_adjustment = -0.22 if item_color == target_color else (0.10 if item_color else 0.0)
        elif color_mode == "ignore" and target_color and item_color == target_color:
            color_adjustment = 0.04

        final_score = base_similarity + category_bonus + sub_category_bonus + material_bonus + fit_bonus + color_adjustment
        item["_ranking"] = {
            "base_similarity": round(base_similarity, 4),
            "final_score": round(final_score, 4),
            "color_mode": color_mode,
            "target_color": target_color,
            "candidate_color": item_color,
            "query_material": query_attrs.get("material"),
            "candidate_material": item_material,
            "query_fit": query_attrs.get("fit"),
            "candidate_fit": item_fit,
            "category_bonus": round(category_bonus, 4),
            "sub_category_bonus": round(sub_category_bonus, 4),
            "material_bonus": round(material_bonus, 4),
            "fit_bonus": round(fit_bonus, 4),
            "color_adjustment": round(color_adjustment, 4),
        }
        reranked.append((final_score, item))

    reranked.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in reranked[:limit]]


@app.post("/search")
async def search_clothes(file: UploadFile = File(None), query: str = Form(None)):
    if not file or not query:
        raise HTTPException(status_code=400, detail="image and query are required")

    try:
        content = await file.read()
        image_obj = Image.open(io.BytesIO(content)).convert("RGB")

        main_categories, sub_categories = extract_category_from_query(query)
        denim_context = is_denim_query_context(query, main_categories, sub_categories)
        query_image_color = extract_dominant_color(image_obj, denim_context=denim_context)

        if sub_categories:
            clothing_label = sub_categories[0]
        elif main_categories:
            clothing_label = main_categories[0]
        else:
            clothing_label = "clothing"

        en_clothing_label = LABEL_TO_EN.get(clothing_label, "fashion item")
        intent = await analyze_query_intent(query)

        has_color_request = bool(intent.color.strip()) if intent.color else False
        has_color_condition = intent.color_mode in {"target", "same", "different"}
        has_design_request = bool(intent.design.strip()) if intent.design else False
        is_specific_query = has_color_condition or has_design_request

        if not is_specific_query:
            enhanced_query = f"a photo of {query_image_color} {en_clothing_label}" if denim_context and query_image_color else f"a photo of {en_clothing_label}"
            text_weight = 0.10
            image_weight = 0.90
        elif intent.color_mode == "target" and has_color_request and not has_design_request:
            enhanced_query = f"a photo of {intent.color} {en_clothing_label}"
            text_weight = 0.65
            image_weight = 0.35
        elif intent.color_mode in {"same", "different"} and not has_design_request:
            enhanced_query = f"a photo of {en_clothing_label}"
            text_weight = 0.20
            image_weight = 0.80
        elif has_design_request and not has_color_request:
            enhanced_query = f"a photo of {intent.design} {en_clothing_label}"
            text_weight = 0.40
            image_weight = 0.60
        else:
            color_prompt = intent.color if intent.color_mode == "target" else ""
            enhanced_query = f"a photo of {color_prompt} {intent.design} {en_clothing_label}".strip()
            text_weight = 0.50
            image_weight = 0.50

        image_features = get_image_embedding(image_obj)
        text_features = get_text_embedding(enhanced_query)
        query_embedding = F.normalize((image_features * image_weight) + (text_features * text_weight), p=2, dim=-1)
        query_embedding_list = query_embedding.squeeze().tolist()

        threshold = 0.45 if is_specific_query else 0.55
        response = supabase.rpc("match_clothes_fashion", {
            "query_embedding": query_embedding_list,
            "match_threshold": threshold,
            "match_count": 100,
            "filter_main_categories": main_categories if main_categories else None,
            "filter_sub_categories": sub_categories if sub_categories else None,
        }).execute()

        query_attrs = build_query_attrs(main_categories, sub_categories, query_image_color, query, intent)
        results = rerank_results(response.data, intent, query_attrs, limit=10)

        return {
            "message": "Success",
            "model": FASHION_CLIP_MODEL_ID,
            "enhanced_query": enhanced_query,
            "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.dict(),
            "query_image_attributes": query_attrs,
            "results": results,
        }

    except Exception as exc:
        print("FashionCLIP search server error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
