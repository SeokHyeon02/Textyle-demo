import os
import sys
import torch
import requests
import numpy as np
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
model.eval()

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

MIN_IMAGE_SIDE = 160
MAX_ASPECT_RATIO = 4.0
MIN_PIXEL_STD = 8.0

VALIDATION_LABELS_BY_MAIN_CATEGORY = {
    "상의": [
        "a product photo of a shirt",
        "a product photo of a t-shirt",
        "a product photo of a hoodie",
        "a product photo of a sweatshirt",
        "a product photo of a knit sweater",
    ],
    "하의": [
        "a product photo of pants",
        "a product photo of jeans",
        "a product photo of shorts",
        "a product photo of slacks",
        "a product photo of jogger pants",
    ],
    "아우터": [
        "a product photo of a jacket",
        "a product photo of a coat",
        "a product photo of a cardigan",
        "a product photo of a blazer",
        "a product photo of a padded jacket",
    ],
}

NEGATIVE_VALIDATION_LABELS = [
    "a photo of food",
    "a photo of shoes",
    "a photo of a bag",
    "a photo of electronics",
    "a landscape photo",
    "a portrait photo of a face",
    "a screenshot of text",
    "a blank image",
]

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

def download_image(image_url: str):
    response = requests.get(image_url, timeout=15, headers=REQUEST_HEADERS)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if content_type and "image/" not in content_type:
        raise ValueError(f"이미지 URL이 아닙니다. content-type={content_type}")

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise ValueError("이미지 파일을 열 수 없습니다.") from exc

    return image

def validate_basic_image_quality(image: Image.Image):
    width, height = image.size
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        return False, f"이미지가 너무 작습니다. size={width}x{height}"

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"이미지 비율이 비정상적입니다. aspect_ratio={aspect_ratio:.2f}"

    resized = image.resize((96, 96))
    pixel_std = float(np.array(resized).std())
    if pixel_std < MIN_PIXEL_STD:
        return False, f"이미지가 거의 단색이거나 비어 있습니다. pixel_std={pixel_std:.2f}"

    return True, "ok"

def crop_center_region(image: Image.Image):
    width, height = image.size
    left = int(width * 0.08)
    top = int(height * 0.05)
    right = int(width * 0.92)
    bottom = int(height * 0.95)
    return image.crop((left, top, right, bottom))

def validate_clothing_image(image: Image.Image, main_category: str):
    expected_labels = VALIDATION_LABELS_BY_MAIN_CATEGORY.get(main_category, [])
    positive_labels = [label for labels in VALIDATION_LABELS_BY_MAIN_CATEGORY.values() for label in labels]
    prompts = expected_labels + [
        label for label in positive_labels
        if label not in expected_labels
    ] + NEGATIVE_VALIDATION_LABELS

    clip_image = crop_center_region(image)
    inputs = processor(
        text=prompts,
        images=clip_image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    expected_count = len(expected_labels)
    positive_count = len(positive_labels)
    negative_start = len(prompts) - len(NEGATIVE_VALIDATION_LABELS)

    positive_score = probs[:positive_count].sum().item()
    expected_score = probs[:expected_count].sum().item() if expected_count else positive_score
    negative_score = probs[negative_start:].sum().item()

    best_index = int(torch.argmax(probs).item())
    best_label = prompts[best_index]

    if positive_score < 0.45 or negative_score > positive_score:
        return False, (
            "의류 이미지로 보기 어렵습니다. "
            f"positive={positive_score:.2f}, negative={negative_score:.2f}, best='{best_label}'"
        )

    if expected_labels and expected_score < 0.20 and negative_score > 0.20:
        return False, (
            f"요청 카테고리({main_category})와 이미지가 맞지 않을 가능성이 높습니다. "
            f"expected={expected_score:.2f}, positive={positive_score:.2f}, best='{best_label}'"
        )

    return True, (
        f"positive={positive_score:.2f}, expected={expected_score:.2f}, "
        f"negative={negative_score:.2f}, best='{best_label}'"
    )

def validate_image_for_insert(image: Image.Image, main_category: str):
    ok, reason = validate_basic_image_quality(image)
    if not ok:
        return False, reason

    return validate_clothing_image(image, main_category)

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
        image = download_image(image_url)

        is_valid_image, validation_reason = validate_image_for_insert(image, main_category)
        if not is_valid_image:
            print(f"🚫 잘못된 이미지로 판단되어 DB 삽입을 거부합니다: {validation_reason}")
            return

        print(f"  -> 이미지 검증 통과: {validation_reason}")

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
