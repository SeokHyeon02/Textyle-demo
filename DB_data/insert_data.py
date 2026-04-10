import os
import sys
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client
from dotenv import load_dotenv

# -------------------------------------------------------------
# 1. 환경 변수 및 Supabase 연결 설정
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ .env 파일에서 Supabase 정보를 불러오지 못했습니다.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------------------
# 2. AI 모델 로드 (최초 실행 시에만 로딩 시간 소요)
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

CATEGORY_MAP = {
    # 아우터 (002)
    "002020": "가디건", "002008": "환절기 코트", "002023": "플리스", 
    "002027": "경량패딩/패딩 베스트", "002007": "겨울 싱글코트", "002025": "무스탕", 
    "002009": "겨울 기타코트", "002013": "롱패딩", "002012": "숏패딩", 
    "002024": "겨울 더블코트", "002022": "후드집업", "002017": "트러커자켓", 
    "002001": "블루종/MA-1", "002006": "코치자켓", "002002": "레더자켓", 
    "002003": "슈트/블레이저 자켓", "002014": "사파리/헌팅자켓",
    
    # 상의 (001)
    "001010": "긴소매 티셔츠", "001005": "스웻셔츠", "001002": "셔츠", 
    "001001": "반소매 티셔츠", "001006": "니트/스웨터", "001003": "피케/카라 티셔츠", 
    "001004": "후드티",
    
    # 하의 (003)
    "003002": "데님팬츠", "003004": "트레이닝/조거 팬츠", "003007": "코튼 팬츠", 
    "003008": "슬랙스/슈트 팬츠", "003009": "숏팬츠"
}

def get_categories_from_code(category_code: str):
    """
    카테고리 코드를 입력받아 (메인_카테고리, 하위_카테고리) 튜플을 반환합니다.
    """
    sub_category = CATEGORY_MAP.get(category_code, "기타") # 맵에 없으면 '기타'로 처리
    
    prefix = category_code[:3]
    if prefix == "001":
        main_category = "상의"
    elif prefix == "002":
        main_category = "아우터"
    elif prefix == "003":
        main_category = "하의"
    else:
        main_category = "기타"
        
    return main_category, sub_category

def insert_clothes_data(name: str, image_url: str, shop_link: str, main_category: str, sub_category: str, price: int, brand_name: str):
    print(f"🔄 처리 중: [{brand_name}] [{main_category} > {sub_category}] {name} - {price}원")
    
    try:
        # 0. 중복 데이터 사전 검사 (AI 모델 실행 전 리소스 절약)
        existing_data = supabase.table("clothes").select("image_url").eq("image_url", image_url).execute()
        
        if len(existing_data.data) > 0:
            print(f"⏩ 이미 등록된 상품입니다. 처리를 건너뜁니다: {name}")
            return  # 함수 종료 (중복 시 아래 다운로드 및 모델 실행을 하지 않음)

        # 1. 이미지 URL에서 사진 다운로드
        print("  -> 새 상품 확인됨. 이미지 다운로드 및 벡터 변환 시작...")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 2. CLIP 모델로 벡터 변환
        inputs = processor(
            text=[""], 
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embedding_list = image_features.squeeze().tolist()

        # 3. Supabase 테이블에 데이터 삽입 (브랜드명 포함)
        data, count = supabase.table("clothes").upsert({
            "brand_name": brand_name,         # 브랜드명 저장 추가
            "name": name,
            "main_category": main_category,   
            "sub_category": sub_category,     
            "price": price,
            "image_url": image_url,
            "shop_link": shop_link,
            "embedding": embedding_list
        }).execute()

        print(f"✅ 성공적으로 DB에 저장되었습니다: [{brand_name}] {name}")

    except Exception as e:
        print(f"❌ 데이터 처리/삽입 실패: {e}")

# -------------------------------------------------------------
# 6. n8n 터미널 명령어 실행부
# -------------------------------------------------------------
if __name__ == "__main__":
    # 터미널에서 인자값을 받아 실행합니다.
    # 인자 순서: 스크립트명(0), 이름(1), URL(2), 링크(3), 카테고리(4), 가격(5), 브랜드명(6)
    clothes_name = sys.argv[1]
    clothes_url = sys.argv[2]
    clothes_link = sys.argv[3]
    
    # 카테고리 코드를 받아 메인, 하위 카테고리로 분리
    raw_category_code = sys.argv[4]
    main_cat, sub_cat = get_categories_from_code(raw_category_code)
    
    try:
        clothes_price = int(sys.argv[5])
    except ValueError:
        print("❌ 가격은 숫자로 입력해야 합니다.")
        sys.exit(1)
        
    # 새롭게 추가된 브랜드명 인자 받아오기 (인자가 누락될 경우를 대비해 예외 처리 추가)
    clothes_brand = sys.argv[6] if len(sys.argv) > 6 else ""

    insert_clothes_data(clothes_name, clothes_url, clothes_link, main_cat, sub_cat, clothes_price, clothes_brand)