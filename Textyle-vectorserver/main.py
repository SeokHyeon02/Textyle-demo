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
# -------------------------------------------------------------

# FastAPI 앱 생성
app = FastAPI(title="TexTyle Vector Search Server")

# 2. AI 모델 초기화 (서버 켜질 때 한 번만 로드)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
print(f"AI 모델 로딩 중... (Device: {device})")

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
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
#  ① 쿼리에서 카테고리 추출
# ------------------------------------------------------------------ #
def extract_category_from_query(query: str):
    for keyword, (main_cat, sub_cat) in CATEGORY_KEYWORDS.items():
        if keyword in query:
            return main_cat, sub_cat
    return None, None


# ------------------------------------------------------------------ #
#  ① CLIP zero-shot으로 이미지 카테고리 분류
# ------------------------------------------------------------------ #
def classify_clothing_type(image_obj, processor, model, device):
    inputs = processor(
        text=CLIP_LABELS,
        images=image_obj,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    best_idx = probs.argmax().item()
    clip_label = CLIP_LABELS[best_idx]
    main_cat, sub_cat = CLIP_LABEL_TO_CATEGORY[clip_label]

    print(f"🤖 CLIP 분류 결과: {clip_label} → main: {main_cat}, sub: {sub_cat}")
    return main_cat, sub_cat

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
        color_keywords = ["색상", "색", "컬러", "빨간", "파란", "검정", "흰색", "노란", "초록", "분홍", "베이지", "네이비"]
        design_keywords = ["패턴", "무늬", "로고", "재질", "소재", "핏", "사이즈", "디자인", "스타일", "소매", "넥", "카라", "기장"]
        stop_words = r"(이 사진과|사진이랑|이거랑|비슷한|찾아줘|보여줘|알려줘|의류|옷|있는|주세요|이미지|찾기|검색)"

        has_color_request = any(keyword in query for keyword in color_keywords)
        has_design_request = any(keyword in query for keyword in design_keywords)
        is_specific_query = has_color_request or has_design_request

        # ✅ 1. 이미지 로드를 위로 올림 (CLIP 분류에 image_obj 필요)
        content = await file.read()
        image_obj = Image.open(io.BytesIO(content)).convert("RGB")

        # ✅ 2. 카테고리 결정
        main_category, sub_category = extract_category_from_query(query)
        if main_category is None:
            main_category, sub_category = classify_clothing_type(image_obj, processor, model, device)

        print(f"🏷️ 카테고리: main={main_category}, sub={sub_category}")

        # ✅ 3. clothing_label 설정 (enhanced_query에 사용)
        clothing_label = sub_category if sub_category else main_category

        if not is_specific_query:
            # 🅰️ 단순 검색 - 이미지만 사용
            enhanced_query = f"a photo of {clothing_label}, fashion item"
            text_weight = 0.0
            image_weight = 1.0

        elif has_color_request and not has_design_request:
            # 🅱️ 색상만 변경 요청
            cleaned_query = re.sub(stop_words, "", query).strip()
            translated_query = GoogleTranslator(source='ko', target='en').translate(cleaned_query)
            enhanced_query = f"a photo of {clothing_label}, {translated_query} color, fashion item"
            text_weight = 0.85
            image_weight = 0.15

        elif has_color_request and has_design_request:
            # 🅲 디자인 유지 + 색상 변경
            cleaned_query = re.sub(stop_words, "", query).strip()
            translated_query = GoogleTranslator(source='ko', target='en').translate(cleaned_query)
            enhanced_query = f"a photo of {clothing_label}, {translated_query}, similar style different color, fashion item"
            text_weight = 0.7
            image_weight = 0.3

        else:
            # 🅳 디자인/핏 등 색상 외 특징 요청
            cleaned_query = re.sub(stop_words, "", query).strip()
            translated_query = GoogleTranslator(source='ko', target='en').translate(cleaned_query)
            enhanced_query = f"a photo of {clothing_label}, {translated_query}, fashion item"
            text_weight = 0.6
            image_weight = 0.4

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

            # 가중치 적용
            if text_weight == 0.0:
                embedding = image_features
            else:
                embedding = (image_features * image_weight) + (text_features * text_weight)
                embedding = F.normalize(embedding, p=2, dim=-1)  # ← .norm() 대신 F.normalize로 통일

            query_embedding_list = embedding.squeeze().tolist()

        # 5. DB 검색 - 요청 유형별 동적 임계값
        if text_weight == 0.0:
            threshold = 0.70
        elif has_color_request and has_design_request:
            threshold = 0.45
        elif has_color_request:
            threshold = 0.50
        else:
            threshold = 0.60


        response = supabase.rpc("match_clothes", {
            "query_embedding":      query_embedding_list,
            "match_threshold":      threshold,
            "match_count":          10,
            "filter_main_category": main_category,   # 예: "하의"
            "filter_sub_category":  sub_category     # 예: "반바지"
        }).execute()
        return {"message": "Success", "results": response.data}
        
    except Exception as e:
        print("\n❌ 서버 에러 상세:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))