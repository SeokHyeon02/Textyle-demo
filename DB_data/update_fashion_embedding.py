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

from update import (
    COLOR_CONFIDENCE_DB_COLUMN,
    COLOR_DB_COLUMN,
    classify_color_from_name,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
TARGET_COLUMN = "fashion_embedding"
ID_COLUMN = os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id")
NAME_COLUMN = os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name")
IMAGE_COLUMN = os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", "image_url")
MAIN_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_MAIN_CATEGORY_COLUMN", "main_category")
SUB_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category")
ORDER_COLUMN = os.environ.get("FASHION_EMBEDDING_ORDER_COLUMN", ID_COLUMN)
PAGE_SIZE = int(os.environ.get("FASHION_EMBEDDING_PAGE_SIZE", "1000"))
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

IMAGE_URL_PATTERN = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
DIRECT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def build_image_request_headers(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if "msscdn.net" in (url or ""):
        headers["Referer"] = "https://www.musinsa.com/"

    return headers


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
        raise ValueError("image_url is empty")

    headers = build_image_request_headers(image_or_product_url)

    first_response = requests.get(image_or_product_url, timeout=15, headers=headers)
    first_response.raise_for_status()
    content_type = first_response.headers.get("content-type", "").lower()

    if "image/" in content_type or is_direct_image_url(image_or_product_url):
        return Image.open(BytesIO(first_response.content)).convert("RGB")

    image_url = extract_first_image_url_from_html(first_response.text, image_or_product_url)
    if not image_url:
        raise ValueError("Could not find an image URL from the product page")

    image_response = requests.get(image_url, timeout=15, headers=build_image_request_headers(image_url))
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

    raise TypeError(f"Cannot find feature tensor from {type(model_output)}")


def load_fashion_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FashionCLIP... model={MODEL_ID}, device={device}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("FashionCLIP loaded")
    return model, processor, device


def get_fashion_image_embedding(image: Image.Image, model, processor, device):
    clip_image = crop_center_region(image.convert("RGB"))
    inputs = processor(images=clip_image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = extract_feature_tensor(model.get_image_features(**inputs))
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.squeeze().tolist()


def fetch_all_items():
    all_items = []
    last_order_value = None
    select_columns = [
        ID_COLUMN,
        NAME_COLUMN,
        IMAGE_COLUMN,
        MAIN_CATEGORY_COLUMN,
        SUB_CATEGORY_COLUMN,
    ]
    if ORDER_COLUMN not in select_columns:
        select_columns.append(ORDER_COLUMN)

    while True:
        query = (
            supabase
            .table("clothes")
            .select(", ".join(select_columns))
            .order(ORDER_COLUMN, desc=False)
            .limit(PAGE_SIZE)
        )

        if last_order_value is not None:
            query = query.gt(ORDER_COLUMN, last_order_value)

        response = query.execute()
        data = response.data or []
        if not data:
            break

        all_items.extend(data)
        print(f"Loaded rows: {len(all_items)}")

        if len(data) < PAGE_SIZE:
            break

        next_order_value = data[-1].get(ORDER_COLUMN)
        if next_order_value is None or next_order_value == last_order_value:
            raise RuntimeError(f"Cannot continue pagination with ORDER_COLUMN={ORDER_COLUMN}")
        last_order_value = next_order_value

    return all_items


def extract_color_attributes(_image: Image.Image, item):
    item_name = item.get(NAME_COLUMN) or ""

    color_from_name = classify_color_from_name(item_name)
    if color_from_name:
        return {
            COLOR_DB_COLUMN: color_from_name,
            COLOR_CONFIDENCE_DB_COLUMN: "high",
            "color_reason": "product_name",
        }

    return {
        COLOR_DB_COLUMN: None,
        COLOR_CONFIDENCE_DB_COLUMN: None,
        "color_reason": "not_found_in_product_name",
    }


def build_update_payload(embedding, color_attributes):
    payload = {
        TARGET_COLUMN: embedding,
        COLOR_DB_COLUMN: color_attributes.get(COLOR_DB_COLUMN),
        COLOR_CONFIDENCE_DB_COLUMN: color_attributes.get(COLOR_CONFIDENCE_DB_COLUMN),
    }

    return payload


def update_all_fashion_embeddings():
    model, processor, device = load_fashion_clip()

    print("Loading all clothes rows from Supabase.")
    print("Every row will be regenerated, even if embedding or color columns are already filled.")
    print(f"Embedding column: {TARGET_COLUMN}")
    print(f"Color columns: {COLOR_DB_COLUMN}, {COLOR_CONFIDENCE_DB_COLUMN}")
    print(f"Order column: {ORDER_COLUMN}")
    print(f"Page size: {PAGE_SIZE}")
    print(f"Dry run: {DRY_RUN}")

    all_items = fetch_all_items()
    if not all_items:
        print("No rows to update.")
        return

    print(f"\nStarting fashion_embedding update for {len(all_items)} rows.\n")

    failed_items = []

    for index, item in enumerate(all_items, 1):
        row_id = item.get(ID_COLUMN)
        name = item.get(NAME_COLUMN) or "unnamed"
        image_url = item.get(IMAGE_COLUMN)

        if not image_url:
            failed_items.append((row_id, name, "image_url missing"))
            print(f"[{index}/{len(all_items)}] skipped: id={row_id} {name} - image_url missing")
            continue

        if row_id is None or row_id == "":
            failed_items.append((row_id, name, "id missing"))
            print(f"[{index}/{len(all_items)}] skipped: {name} - id missing")
            continue

        print(f"[{index}/{len(all_items)}] updating: id={row_id} {name}")

        try:
            image = download_product_image(image_url)
            embedding = get_fashion_image_embedding(image, model, processor, device)
            color_attributes = extract_color_attributes(image, item)
            payload = build_update_payload(embedding, color_attributes)

            update_response = None
            if not DRY_RUN:
                update_response = (
                    supabase
                    .table("clothes")
                    .update(payload)
                    .eq(ID_COLUMN, row_id)
                    .execute()
                )

            verify_response = (
                supabase
                .table("clothes")
                .select(f"{TARGET_COLUMN}, {COLOR_DB_COLUMN}, {COLOR_CONFIDENCE_DB_COLUMN}")
                .eq(ID_COLUMN, row_id)
                .limit(1)
                .execute()
            ) if not DRY_RUN else None

            verified_item = verify_response.data[0] if verify_response and verify_response.data else {}
            saved_embedding = verified_item.get(TARGET_COLUMN)
            saved_dim = len(saved_embedding) if isinstance(saved_embedding, list) else None
            saved_color = verified_item.get(COLOR_DB_COLUMN)
            saved_color_confidence = verified_item.get(COLOR_CONFIDENCE_DB_COLUMN)

            print(
                "   -> "
                f"embedding_dim={len(embedding)}, "
                f"color={payload[COLOR_DB_COLUMN]}, "
                f"color_confidence={payload[COLOR_CONFIDENCE_DB_COLUMN]}, "
                f"color_reason={color_attributes.get('color_reason')}, "
                f"db_embedding_is_null={saved_embedding is None}, "
                f"db_embedding_dim={saved_dim}, "
                f"db_color={saved_color}, "
                f"db_color_confidence={saved_color_confidence}, "
                f"returned_rows={len(update_response.data or []) if update_response else 0}, "
                f"verified_rows={len(verify_response.data or []) if verify_response else 0}"
            )

        except Exception as exc:
            failed_items.append((row_id, name, str(exc)))
            print(f"[{index}/{len(all_items)}] failed: id={row_id} {name} - {exc}")

    print("\nFashionCLIP embedding update complete")
    print(f"Failed items: {len(failed_items)}")

    if failed_items:
        failed_log_path = os.path.join(BASE_DIR, "fashion_embedding_failed.log")
        with open(failed_log_path, "w", encoding="utf-8") as log_file:
            for row_id, name, reason in failed_items:
                log_file.write(f"{row_id}\t{name}\t{reason}\n")
        print(f"Failed item log saved: {failed_log_path}")


if __name__ == "__main__":
    update_all_fashion_embeddings()
