import os
import tempfile

import numpy as np
from PIL import Image

from fashion_color_utils import IMAGE_COLUMN, NAME_COLUMN, download_product_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
FASHION_CLIP_API_MODEL_ID = os.environ.get("FASHION_CLIP_API_MODEL_ID", "fashion-clip")
TARGET_COLUMN = "fashion_embedding"
ID_COLUMN = os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id")
MAIN_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_MAIN_CATEGORY_COLUMN", "main_category")
SUB_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category")
ORDER_COLUMN = os.environ.get("FASHION_EMBEDDING_ORDER_COLUMN", ID_COLUMN)
PAGE_SIZE = int(os.environ.get("FASHION_EMBEDDING_PAGE_SIZE", "1000"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("FASHION_EMBEDDING_BATCH_SIZE", "32"))
VERIFY_UPDATES = os.environ.get("VERIFY_FASHION_EMBEDDING_UPDATES", "false").strip().lower() in {
    "1", "true", "yes", "y"
}
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

supabase = None


def load_environment():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(DB_DATA_DIR, ".env"))


def get_supabase_client():
    global supabase
    if supabase is not None:
        return supabase

    load_environment()
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")

    from supabase import create_client

    supabase = create_client(supabase_url, supabase_key)
    return supabase


def load_fashion_clip():
    from fashion_clip.fashion_clip import FashionCLIP

    print(f"Loading FashionCLIP API... model={FASHION_CLIP_API_MODEL_ID}")
    fclip = FashionCLIP(FASHION_CLIP_API_MODEL_ID)
    print("FashionCLIP loaded")
    return fclip


def save_temp_rgb_image(image: Image.Image):
    rgb_image = image.convert("RGB")
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    rgb_image.save(temp_path, format="JPEG", quality=95)
    return temp_path


def encode_images_with_fashion_clip_api(image_paths, fclip):
    image_embeddings = fclip.encode_images(image_paths, batch_size=EMBEDDING_BATCH_SIZE)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    return image_embeddings


def fetch_all_items():
    client = get_supabase_client()
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
            client
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


def update_batch(batch_items, fclip, total_count, failed_items):
    client = get_supabase_client()
    temp_paths = []
    prepared_items = []

    for item_index, item in batch_items:
        row_id = item.get(ID_COLUMN)
        name = item.get(NAME_COLUMN) or "unnamed"
        image_url = item.get(IMAGE_COLUMN)
        try:
            image = download_product_image(image_url)
            temp_path = save_temp_rgb_image(image)
            temp_paths.append(temp_path)
            prepared_items.append((item_index, item, temp_path))
        except Exception as exc:
            failed_items.append((row_id, name, str(exc)))
            print(f"[{item_index}/{total_count}] failed: id={row_id} {name} - {exc}")

    if not prepared_items:
        return

    try:
        embeddings = encode_images_with_fashion_clip_api(temp_paths, fclip)
    except Exception as exc:
        for item_index, item, _temp_path in prepared_items:
            row_id = item.get(ID_COLUMN)
            name = item.get(NAME_COLUMN) or "unnamed"
            failed_items.append((row_id, name, str(exc)))
            print(f"[{item_index}/{total_count}] failed: id={row_id} {name} - {exc}")
        return
    finally:
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    for embedding_index, (item_index, item, _temp_path) in enumerate(prepared_items):
        row_id = item.get(ID_COLUMN)
        name = item.get(NAME_COLUMN) or "unnamed"
        try:
            embedding = embeddings[embedding_index].tolist()
            payload = {TARGET_COLUMN: embedding}

            update_response = None
            if not DRY_RUN:
                update_response = (
                    client
                    .table("clothes")
                    .update(payload)
                    .eq(ID_COLUMN, row_id)
                    .execute()
                )

            verify_response = None
            if VERIFY_UPDATES and not DRY_RUN:
                verify_response = (
                    client
                    .table("clothes")
                    .select(TARGET_COLUMN)
                    .eq(ID_COLUMN, row_id)
                    .limit(1)
                    .execute()
                )

            verified_item = verify_response.data[0] if verify_response and verify_response.data else {}
            saved_embedding = verified_item.get(TARGET_COLUMN)
            saved_dim = len(saved_embedding) if isinstance(saved_embedding, list) else None
            print(
                "   -> "
                f"[{item_index}/{total_count}] "
                f"id={row_id} {name}, "
                f"embedding_dim={len(embedding)}, "
                f"returned_rows={len(update_response.data or []) if update_response else 0}, "
                f"verified_rows={len(verify_response.data or []) if verify_response else 0}, "
                f"db_embedding_dim={saved_dim}"
            )
        except Exception as exc:
            failed_items.append((row_id, name, str(exc)))
            print(f"[{item_index}/{total_count}] failed: id={row_id} {name} - {exc}")


def update_all_fashion_embeddings():
    fclip = load_fashion_clip()
    print("Loading all clothes rows from Supabase.")
    print("Only fashion_embedding will be regenerated. Color columns will not be changed.")
    print(f"Embedding column: {TARGET_COLUMN}")
    print(f"Order column: {ORDER_COLUMN}")
    print(f"Page size: {PAGE_SIZE}")
    print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"Verify updates: {VERIFY_UPDATES}")
    print(f"Dry run: {DRY_RUN}")

    all_items = fetch_all_items()
    if not all_items:
        print("No rows to update.")
        return

    failed_items = []
    batch_items = []
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

        print(f"[{index}/{len(all_items)}] queued: id={row_id} {name}")
        batch_items.append((index, item))
        if len(batch_items) >= EMBEDDING_BATCH_SIZE:
            print(f"Encoding batch: {len(batch_items)} items")
            update_batch(batch_items, fclip, len(all_items), failed_items)
            batch_items = []

    if batch_items:
        print(f"Encoding final batch: {len(batch_items)} items")
        update_batch(batch_items, fclip, len(all_items), failed_items)

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
