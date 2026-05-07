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
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_ID = os.environ.get("FASHION_CLIP_MODEL_ID", "patrickjohncyh/fashion-clip")
TARGET_COLUMN = "fashion_embedding"
ORDER_COLUMN = "image_url"
PAGE_SIZE = int(os.environ.get("FASHION_EMBEDDING_PAGE_SIZE", "1000"))

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
        raise ValueError("image_url is empty")

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
        raise ValueError("Could not find an image URL from the product page")

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
    select_columns = ["name", "image_url"]
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


def update_all_fashion_embeddings():
    model, processor, device = load_fashion_clip()

    print("Loading all clothes rows from Supabase.")
    print("Every row will be regenerated, even if fashion_embedding is already filled.")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Order column: {ORDER_COLUMN}")
    print(f"Page size: {PAGE_SIZE}")

    all_items = fetch_all_items()
    if not all_items:
        print("No rows to update.")
        return

    print(f"\nStarting fashion_embedding update for {len(all_items)} rows.\n")

    failed_items = []

    for index, item in enumerate(all_items, 1):
        name = item.get("name") or "unnamed"
        image_url = item.get("image_url")

        if not image_url:
            failed_items.append((name, "image_url missing"))
            print(f"[{index}/{len(all_items)}] skipped: {name} - image_url missing")
            continue

        print(f"[{index}/{len(all_items)}] updating: {name}")

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

            print(
                "   -> "
                f"embedding_dim={len(embedding)}, "
                f"db_embedding_is_null={saved_embedding is None}, "
                f"db_embedding_dim={saved_dim}, "
                f"returned_rows={len(update_response.data or [])}, "
                f"verified_rows={len(verify_response.data or [])}"
            )

        except Exception as exc:
            failed_items.append((name, str(exc)))
            print(f"[{index}/{len(all_items)}] failed: {name} - {exc}")

    print("\nFashionCLIP embedding update complete")
    print(f"Failed items: {len(failed_items)}")

    if failed_items:
        failed_log_path = os.path.join(BASE_DIR, "fashion_embedding_failed.log")
        with open(failed_log_path, "w", encoding="utf-8") as log_file:
            for name, reason in failed_items:
                log_file.write(f"{name}\t{reason}\n")
        print(f"Failed item log saved: {failed_log_path}")


if __name__ == "__main__":
    update_all_fashion_embeddings()
