from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client, Client
from PIL import Image
import io
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import traceback
from deep_translator import GoogleTranslator
import re
import torch.nn.functional as F
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Optional
import json
# 1. 환경 변수 로드 및 Supabase 클라이언트 초기화
# -------------------------------------------------------------
# 1. 환경 변수 강제 로드 및 에러 체크
# -------------------------------------------------------------
# main.py 파일이 있는 폴더의 절대 경로를 찾아서 .env 파일을 강제로 지정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
# 강제로 해당 경로의 .env를 읽습니다.
load_dotenv(dotenv_path=env_path)

SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# -------------------------------------------------------------

# FastAPI 앱 생성
app = FastAPI(title="TexTyle Vector Search Server")

# 2. AI 모델 초기화 (서버 켜질 때 한 번만 로드)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
print(f"AI 모델 로딩 중... (Device: {device})")

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
print("AI 모델 로딩 완료!")

#  DB 카테고리 구조 매핑 (실제 sub_category 값 기준)
# ------------------------------------------------------------------ #

CATEGORY_KEYWORDS = {
        # ── 상의 ──────────────────────────────────────────────
        "후드티":       ("상의", "후드티"),
        "후디":         ("상의", "후드티"),
        "후드":         ("상의", "후드티"),
        "스웻셔츠":     ("상의", "스웻셔츠"),
        "맨투맨":       ("상의", "스웻셔츠"),
        "긴팔":         ("상의", "긴소매 티셔츠"),
        "긴소매":       ("상의", "긴소매 티셔츠"),
        "반팔":         ("상의", "반소매 티셔츠"),
        "티셔츠":       ("상의", "반소매 티셔츠"),
        "티":           ("상의", "반소매 티셔츠"),
        "니트":         ("상의", "니트/스웨터"),
        "스웨터":       ("상의", "니트/스웨터"),
        "셔츠":         ("상의", "셔츠"),
        "남방":         ("상의", "셔츠"),

        # ── 하의 ──────────────────────────────────────────────
        "슬랙스":       ("하의", "슬랙스/슈트 팬츠"),
        "슈트팬츠":     ("하의", "슬랙스/슈트 팬츠"),
        "정장바지":     ("하의", "슬랙스/슈트 팬츠"),
        "데님":         ("하의", "데님팬츠"),
        "청바지":       ("하의", "데님팬츠"),
        "진":           ("하의", "데님팬츠"),
        "반바지":       ("하의", "숏팬츠"),
        "숏팬츠":       ("하의", "숏팬츠"),
        "쇼츠":         ("하의", "숏팬츠"),
        "코튼팬츠":     ("하의", "코튼 팬츠"),
        "면바지":       ("하의", "코튼 팬츠"),
        "치노":         ("하의", "코튼 팬츠"),
        "트레이닝":     ("하의", "트레이닝/조거 팬츠"),
        "조거":         ("하의", "트레이닝/조거 팬츠"),
        "조깅":         ("하의", "트레이닝/조거 팬츠"),
        "운동복":       ("하의", "트레이닝/조거 팬츠"),
        "바지":         ("하의", None),   # 애매할 때 main만 필터

        # ── 아우터 ────────────────────────────────────────────
        "블루종":       ("아우터", "블루종/MA-1"),
        "MA1":          ("아우터", "블루종/MA-1"),
        "MA-1":         ("아우터", "블루종/MA-1"),
        "봄버":         ("아우터", "블루종/MA-1"),
        "슈트자켓":     ("아우터", "슈트/블레이저 자켓"),
        "블레이저":     ("아우터", "슈트/블레이저 자켓"),
        "정장자켓":     ("아우터", "슈트/블레이저 자켓"),
        "후드집업":     ("아우터", "후드집업"),
        "집업":         ("아우터", "후드집업"),
        "롱패딩":       ("아우터", "롱패딩"),
        "코치자켓":     ("아우터", "코치자켓"),
        "윈드브레이커": ("아우터", "코치자켓"),
        "경량패딩":     ("아우터", "경량패딩/패딩 베스트"),
        "패딩베스트":   ("아우터", "경량패딩/패딩 베스트"),
        "조끼패딩":     ("아우터", "경량패딩/패딩 베스트"),
        "숏패딩":       ("아우터", "숏패딩"),
        "패딩":         ("아우터", None),  # 종류 애매할 때 main만 필터
        "레더자켓":     ("아우터", "레더자켓"),
        "가죽자켓":     ("아우터", "레더자켓"),
        "싱글코트":     ("아우터", "겨울 싱글코트"),
        "코트":         ("아우터", "겨울 싱글코트"),
        "가디건":       ("아우터", "가디건"),
        "카디건":       ("아우터", "가디건"),
        "사파리":       ("아우터", "사파리/헌팅자켓"),
        "헌팅자켓":     ("아우터", "사파리/헌팅자켓"),
        "자켓":         ("아우터", None),  # 종류 애매할 때 main만 필터
    }

    # ------------------------------------------------------------------ #
    #  CLIP zero-shot 레이블 → (main_category, sub_category)
    # ------------------------------------------------------------------ #
CLIP_LABEL_TO_CATEGORY = {
        # 상의
        "hoodie":                       ("상의", "후드티"),
        "sweatshirt":                   ("상의", "스웻셔츠"),
        "long sleeve t-shirt":          ("상의", "긴소매 티셔츠"),
        "short sleeve t-shirt":         ("상의", "반소매 티셔츠"),
        "knit sweater":                  ("상의", "니트/스웨터"),
        "shirt":                        ("상의", "셔츠"),
        # 하의
        "dress pants slacks":           ("하의", "슬랙스/슈트 팬츠"),
        "denim jeans":                  ("하의", "데님팬츠"),
        "shorts":                       ("하의", "숏팬츠"),
        "cotton casual pants":          ("하의", "코튼 팬츠"),
        "jogger sweatpants":            ("하의", "트레이닝/조거 팬츠"),
        # 아우터
        "bomber jacket MA-1":           ("아우터", "블루종/MA-1"),
        "suit blazer jacket":           ("아우터", "슈트/블레이저 자켓"),
        "zip-up hoodie":                ("아우터", "후드집업"),
        "long padded puffer coat":      ("아우터", "롱패딩"),
        "coach jacket windbreaker":     ("아우터", "코치자켓"),
        "light padded vest":            ("아우터", "경량패딩/패딩 베스트"),
        "short padded jacket":          ("아우터", "숏패딩"),
        "leather jacket":               ("아우터", "레더자켓"),
        "single breasted winter coat":  ("아우터", "겨울 싱글코트"),
        "cardigan":                     ("아우터", "가디건"),
        "safari hunting jacket":        ("아우터", "사파리/헌팅자켓"),
    }

CLIP_LABELS = list(CLIP_LABEL_TO_CATEGORY.keys())

# ------------------------------------------------------------------ #
# 한국어 의류 라벨 -> CLIP 최적화 영문 라벨 매핑
# ------------------------------------------------------------------ #
LABEL_TO_EN = {
    # ── 상의 ──────────────────────────────────────────────
    "후드티": "hoodie",
    "스웻셔츠": "sweatshirt",
    "긴소매 티셔츠": "long sleeve t-shirt",
    "반소매 티셔츠": "short sleeve t-shirt",
    "니트/스웨터": "knit sweater",
    "셔츠": "shirt",
    "상의": "top",  # sub_category가 없을 때의 fallback

    # ── 하의 ──────────────────────────────────────────────
    "슬랙스/슈트 팬츠": "dress pants slacks",
    "데님팬츠": "denim jeans",
    "숏팬츠": "shorts",
    "코튼 팬츠": "cotton casual pants",
    "트레이닝/조거 팬츠": "jogger sweatpants",
    "하의": "pants",  # sub_category가 없을 때의 fallback

    # ── 아우터 ────────────────────────────────────────────
    "블루종/MA-1": "bomber jacket MA-1",
    "슈트/블레이저 자켓": "suit blazer jacket",
    "후드집업": "zip-up hoodie",
    "롱패딩": "long padded puffer coat",
    "코치자켓": "coach jacket windbreaker",
    "경량패딩/패딩 베스트": "light padded vest",
    "숏패딩": "short padded jacket",
    "레더자켓": "leather jacket",
    "겨울 싱글코트": "single breasted winter coat",
    "가디건": "cardigan",
    "사파리/헌팅자켓": "safari hunting jacket",
    "아우터": "outerwear"  # sub_category가 없을 때의 fallback
}

def classify_clothing_type(image_obj, processor, model, device, top_k=2):
    inputs = processor(
        text=CLIP_LABELS,
        images=image_obj,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    # 🚀 argmax 대신 topk를 사용하여 상위 K개 확률과 인덱스 추출
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
    
    main_cats = set() # 중복 방지를 위해 set 사용
    sub_cats = set()

    for i in range(top_k):
        prob = top_probs[0][i].item()
        idx = top_indices[0][i].item()
        
        # 확률이 너무 낮은(예: 10% 미만) 후보는 필터링
        if prob < 0.1:
            continue

        clip_label = CLIP_LABELS[idx]
        main_cat, sub_cat = CLIP_LABEL_TO_CATEGORY[clip_label]
        
        main_cats.add(main_cat)
        if sub_cat:
            sub_cats.add(sub_cat)

        print(f"🤖 CLIP 후보 {i+1}: {clip_label} (확률: {prob:.2f}) → main: {main_cat}, sub: {sub_cat}")

    # DB에 배열로 넘기기 위해 list로 변환하여 반환
    return list(main_cats), list(sub_cats)

def extract_category_from_query(query: str):
    main_cats = set()
    sub_cats = set()
    
    for keyword, (main_cat, sub_cat) in CATEGORY_KEYWORDS.items():
        if keyword in query:
            main_cats.add(main_cat)
            if sub_cat:
                sub_cats.add(sub_cat)
                
    # 결과가 없으면 빈 리스트 반환
    return list(main_cats), list(sub_cats)

class QueryIntent(BaseModel):
    reasoning: str = Field(
        description="사용자의 문장을 분석하여 색상 요청과 디자인/핏 요청이 각각 무엇인지, 혹은 없는지 단계별로 분석한 논리적 사고 과정 (한국어로 작성)"
    )
    color: str = Field(          # Optional 제거
        description="분석 결과를 바탕으로 추출한 색상/패턴 관련 영어 단어 명사구 (예: 'red', 'striped'). 없으면 빈 문자열"
    )
    design: str = Field(         # Optional 제거
        description="분석 결과를 바탕으로 추출한 핏/기장/소재/디자인 관련 영어 단어 명사구 (예: 'oversized', 'long length', 'v-neck'). 없으면 빈 문자열"
    )
async def analyze_query_intent(user_query: str) -> QueryIntent:
    system_prompt = """너는 매우 정교한 패션 검색 쿼리 분석가야.
        사용자의 입력을 분석해서 변경하고 싶은 'color'와 'design' 속성을 추출해.
        반드시 'reasoning' 필드에 분석 과정을 먼저 상세히 적은 후, 그 결론을 바탕으로 color와 design 값을 CLIP이 이해하기 쉬운 영어 명사로 도출해.
        **변경 요청이 없다면 해당 필드는 반드시 빈 문자열("")로 둬. null을 쓰지 마.**

        [예시]
        사용자: "이거 예쁜데, 핏은 좀 더 루즈하고 까만색인 거 없어?"
        reasoning: "사용자는 기존 옷을 기준으로 두 가지 속성 변경을 요청함. 1) 핏은 더 루즈하게('loose fit'), 2) 색상은 까만색('black'). 따라서 design과 color 모두 값이 존재함."
        color: "black"
        design: "loose fit"
        """
    try:
        prompt = f"{system_prompt}\n\n사용자 입력: {user_query}"
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=QueryIntent, 
                temperature=0.0, # 창의성 배제, 무조건 가장 확률이 높은 정확한 정답만 출력
            ),
        )
        
        parsed_data = json.loads(response.text)
        
        # 서버 로그에서 LLM이 어떻게 생각했는지 확인 가능
        print(f"🧠 [LLM 추론 과정]: {parsed_data.get('reasoning')}")
        
        return QueryIntent(reasoning=parsed_data.get("reasoning", ""),
        color=parsed_data.get("color", ""),
        design=parsed_data.get("design", ""),
    )
    except Exception as e:
        print(f"⚠️ 최고 정밀도 분석 실패: {e}")
        return QueryIntent(reasoning="에러 발생", color="", design="")

# 3. 검색 API 엔드포인트
@app.post("/search")
async def search_clothes(
    file: UploadFile = File(None), 
    query: str = Form(None)
):
    # 1. 입구컷: 둘 다 없으면 거절!
    if not file or not query:
        raise HTTPException(status_code=400, detail="이미지와 검색어를 모두 입력해야 검색이 가능합니다.")
    # ------------------------------------------------------------------ #

    try:
        # ✅ 1. 이미지 로드를 위로 올림 (CLIP 분류에 image_obj 필요)
        content = await file.read()
        image_obj = Image.open(io.BytesIO(content)).convert("RGB")

        # ✅ 2. 카테고리 결정
        main_categories, sub_categories = extract_category_from_query(query)
        if not main_categories:
            main_categories, sub_categories = classify_clothing_type(image_obj, processor, model, device)

        print(f"🏷️ 카테고리 배열: main={main_categories}, sub={sub_categories}")

        # 2. 텍스트 프롬프트용 라벨 생성 (가장 확실한 첫 번째 카테고리만 사용)
        if sub_categories:
            clothing_label = sub_categories[0]
        elif main_categories:
            clothing_label = main_categories[0]
        else:
            clothing_label = "clothing"
        
        en_clothing_label = LABEL_TO_EN.get(clothing_label, "fashion item")

        intent = await analyze_query_intent(query)
        has_color_request = bool(intent.color.strip()) if intent.color else False
        has_design_request = bool(intent.design.strip()) if intent.design else False
        is_specific_query = has_color_request or has_design_request

        if not is_specific_query:
            # 🅰️ 단순 검색 - 이미지만 사용
            enhanced_query = f"a photo of {en_clothing_label}"
            text_weight = 0.1
            image_weight = 0.9

        elif has_color_request and not has_design_request:
            # 🅱️ 색상만 변경 요청
            enhanced_query = f"a photo of {intent.color} {en_clothing_label}"
            text_weight = 0.7
            image_weight = 0.3

        elif has_design_request and not has_color_request:
            # 🅲 디자인/핏 등 색상 외 특징 요청
            enhanced_query = f"a photo of {intent.design} {en_clothing_label}"
            text_weight = 0.4
            image_weight = 0.6

        else:
            # 🅳 색상 + 디자인 모두 변경
            enhanced_query = f"a photo of {intent.color} {intent.design} {en_clothing_label}"
            text_weight = 0.5
            image_weight = 0.5

        print(f"✨ 최종 AI 입력 텍스트: '{enhanced_query}'")

        # 4. 모델 입력 및 임베딩 추출
        with torch.no_grad():
            # 텍스트 임베딩 (512차원)
            text_inputs = processor(text=[enhanced_query], return_tensors="pt", padding=True).to(device)
            text_outputs = model.get_text_features(**text_inputs)
            text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs

            # 이미지 임베딩 (512차원)
            image_inputs = processor(images=image_obj, return_tensors="pt").to(device)
            image_outputs = model.get_image_features(**image_inputs)
            image_features = image_outputs.pooler_output if hasattr(image_outputs, 'pooler_output') else image_outputs
            # 정규화
            text_features = F.normalize(text_features, p=2, dim=-1)
            image_features = F.normalize(image_features, p=2, dim=-1)

            embedding = (image_features * image_weight) + (text_features * text_weight)
            embedding = F.normalize(embedding, p=2, dim=-1)

            query_embedding_list = embedding.squeeze().tolist()

        threshold = 0.55 if is_specific_query else 0.70


        response = supabase.rpc("match_clothes", {
            "query_embedding":      query_embedding_list,
            "match_threshold":      threshold,
            "match_count":          10,
            "filter_main_categories": main_categories if main_categories else None,
            "filter_sub_categories": sub_categories if sub_categories else None
        }).execute()
        return {"message": "Success", "results": response.data}
        
    except Exception as e:
        print("\n❌ 서버 에러 상세:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))