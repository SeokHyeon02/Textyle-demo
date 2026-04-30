import os
import re
from io import BytesIO
from urllib.parse import urljoin

import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from supabase import Client, create_client
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL 또는 SUPABASE_KEY가 .env에 없습니다.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
TARGET_COLUMN = "fashion_embedding"
PAGE_SIZE = 1000
FETCH_ONLY_MISSING = os.environ.get("FETCH_ONLY_MISSING_FASHION_EMBEDDING", "true").strip().lower() in {
    "1", "true", "yes", "y"
}
ORDER_COLUMN = os.environ.get("FASHION_EMBEDDING_ORDER_COLUMN", "image_url")

IMAGE_URL_PATTERN = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
DIRECT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


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
        if value is None:
            continue
        if attr_name == "last_hidden_state" and value.ndim == 3:
            return value[:, 0, :]
        return value

    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]

    raise TypeError(f"임베딩 텐서를 찾지 못했습니다: {type(model_output)}")


def load_fashion_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FashionCLIP 모델 로드 중... model={MODEL_ID}, device={device}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("FashionCLIP 모델 로드 완료")
    return model, processor, device


def get_fashion_image_embedding(image: Image.Image, model, processor, device):
    clip_image = crop_center_region(image.convert("RGB"))
    inputs = processor(images=clip_image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = extract_feature_tensor(model.get_image_features(**inputs))
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.squeeze().tolist()


def update_all_fashion_embeddings():
    model, processor, device = load_fashion_clip()

    print("Supabase에서 데이터를 가져옵니다 (1000개씩 분할 로드)...")
    print(f"비어 있는 fashion_embedding만 처리: {FETCH_ONLY_MISSING}")
    print(f"조회 정렬 기준: {ORDER_COLUMN}")

    all_items = []
    start = 0
    limit = PAGE_SIZE

    while True:
        query = (
            supabase
            .table("clothes")
            .select(f"name, image_url, {TARGET_COLUMN}")
            .order(ORDER_COLUMN, desc=False)
            .range(start, start + limit - 1)
        )

        if FETCH_ONLY_MISSING:
            query = query.is_(TARGET_COLUMN, "null")

        response = query.execute()

        data = response.data
        if not data:
            break

        all_items.extend(data)
        print(f"누적 {len(all_items)}개 데이터 로드 완료...")

        if len(data) < limit:
            break

        start += limit

    if not all_items:
        print("업데이트할 데이터가 없습니다.")
        return

    print(f"\n총 {len(all_items)}개의 fashion_embedding 업데이트를 시작합니다.\n")

    failed_items = []

    for index, item in enumerate(all_items, 1):
        name = item.get("name") or "이름 없음"
        image_url = item.get("image_url")

        if not image_url:
            failed_items.append((name, "image_url 없음"))
            print(f"[{index}/{len(all_items)}] 건너뜀: {name} - image_url 없음")
            continue

        print(f"[{index}/{len(all_items)}] 업데이트 중: {name}")

        try:
            image = download_product_image(image_url)
            embedding = get_fashion_image_embedding(image, model, processor, device)

            update_response = (
                supabase
                .table("clothes")
                .update({TARGET_COLUMN: embedding})
                .eq("image_url", image_url)
                .execute()
            )

            verify_response = (
                supabase
                .table("clothes")
                .select(TARGET_COLUMN)
                .eq("image_url", image_url)
                .limit(1)
                .execute()
            )

            verified_item = verify_response.data[0] if verify_response.data else {}
            saved_embedding = verified_item.get(TARGET_COLUMN)
            saved_dim = len(saved_embedding) if isinstance(saved_embedding, list) else None
            saved_preview = str(saved_embedding)[:24] if saved_embedding is not None else "None"

            print(
                "   -> "
                f"embedding_dim={len(embedding)}, "
                f"db_embedding_is_null={saved_embedding is None}, "
                f"db_embedding_dim={saved_dim}, "
                f"db_embedding_preview={saved_preview}, "
                f"returned_rows={len(update_response.data or [])}, "
                f"verified_rows={len(verify_response.data or [])}"
            )

        except Exception as exc:
            failed_items.append((name, str(exc)))
            print(f"[{index}/{len(all_items)}] 실패: {name} - {exc}")

    print("\nFashionCLIP 임베딩 업데이트 완료")
    print(f"실패 상품: {len(failed_items)}개")

    if failed_items:
        failed_log_path = os.path.join(BASE_DIR, "fashion_embedding_failed.log")
        with open(failed_log_path, "w", encoding="utf-8") as log_file:
            for name, reason in failed_items:
                log_file.write(f"{name}\t{reason}\n")
        print(f"실패 목록 저장: {failed_log_path}")


if __name__ == "__main__":
    update_all_fashion_embeddings()
