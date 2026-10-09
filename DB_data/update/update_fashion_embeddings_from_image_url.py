import argparse
import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from fashion_color_utils import IMAGE_COLUMN, NAME_COLUMN, download_product_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
DEFAULT_IMAGE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "textyle_fashion_clip_images")
WORKFLOW_VERSION = "2026-05-20-sub-category-chunks-v1"


def load_environment(env_path):
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)


def get_supabase_client(env_path):
    load_environment(env_path)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing.")

    from supabase import create_client

    return create_client(supabase_url, supabase_key)


def load_fashion_clip(model_id):
    from fashion_clip.fashion_clip import FashionCLIP

    print(f"Loading FashionCLIP... model={model_id}")
    model = FashionCLIP(model_id)
    print("FashionCLIP loaded")
    return model


def fetch_product_rows(client, args):
    requested_ids = {str(value) for value in args.ids}
    select_columns = [args.id_column, args.name_column, args.image_url_column, args.sub_category_column]
    if args.order_column not in select_columns:
        select_columns.append(args.order_column)

    product_rows = []
    last_order_value = None
    while True:
        page_limit = args.page_size
        if args.limit:
            remaining = args.limit - len(product_rows)
            if remaining <= 0:
                break
            page_limit = min(page_limit, remaining)

        query = (
            client
            .table(args.table)
            .select(", ".join(select_columns))
            .order(args.order_column, desc=False)
            .limit(page_limit)
        )
        if last_order_value is not None:
            query = query.gt(args.order_column, last_order_value)
        if requested_ids:
            query = query.in_(args.id_column, list(requested_ids))
        elif args.start_id is not None:
            query = query.gte(args.id_column, args.start_id)
        if args.sub_category:
            query = query.in_(args.sub_category_column, args.sub_category)

        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        for row in rows:
            row_id = str(row.get(args.id_column) or "")
            if row_id:
                product_rows.append(row)

        if requested_ids or len(rows) < page_limit:
            break
        next_order_value = rows[-1].get(args.order_column)
        if next_order_value is None or next_order_value == last_order_value:
            raise RuntimeError(f"Cannot continue pagination with order column {args.order_column}")
        last_order_value = next_order_value

    return product_rows


def log_entry_for_row(row, args, reason):
    return {
        "id": row.get(args.id_column) or "",
        "name": row.get(args.name_column) or "",
        "image_url": row.get(args.image_url_column) or "",
        "reason": reason,
    }


def append_issue_log(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "image_url", "reason"], delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_image_for_row(row, args):
    product_id = str(row.get(args.id_column) or "")
    image_url = row.get(args.image_url_column) or ""
    if not image_url:
        raise ValueError("image_url missing")

    os.makedirs(args.image_cache_dir, exist_ok=True)
    image_path = os.path.join(args.image_cache_dir, f"{product_id}.jpg")
    if os.path.exists(image_path) and not args.refresh_downloaded_images:
        return image_path

    image = download_product_image(image_url)
    image.convert("RGB").save(image_path, format="JPEG", quality=95)
    return image_path


def predownload_images(product_rows, args):
    image_paths = {}
    failed = []
    failed_log = os.path.join(BASE_DIR, "fashion_embedding_update_failed.log")

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {
            executor.submit(download_image_for_row, row, args): (index, row)
            for index, row in enumerate(product_rows, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            index, row = futures[future]
            product_id = str(row.get(args.id_column) or "")
            completed += 1
            try:
                image_paths[product_id] = future.result()
            except Exception as exc:
                entry = log_entry_for_row(row, args, f"download_failed: {exc}")
                failed.append(entry)
                append_issue_log(failed_log, entry)
                print(f"[download {index}/{len(product_rows)}] failed id={product_id}: {exc}")
            if completed % 50 == 0 or completed == len(product_rows):
                print(f"Downloaded images: {completed}/{len(product_rows)}")
    return image_paths, failed


def encode_image_paths(fclip, image_paths, batch_size):
    embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
    embeddings = np.asarray(embeddings)
    norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def flush_update_batch(client, update_batch, args):
    if not update_batch:
        return 0
    updated_rows = 0
    for row, embedding in update_batch:
        response = (
            client
            .table(args.table)
            .update({args.embedding_column: embedding})
            .eq(args.id_column, row.get(args.id_column))
            .execute()
        )
        updated_rows += len(response.data or [])
    return updated_rows


def update_fashion_embeddings(args):
    args.image_cache_dir = os.path.abspath(args.image_cache_dir)
    args.download_workers = max(1, args.download_workers)
    args.embedding_batch_size = max(1, args.embedding_batch_size)
    args.update_batch_size = max(1, args.update_batch_size)

    client = get_supabase_client(args.env)
    product_rows = fetch_product_rows(client, args)
    if not product_rows:
        print("No DB rows matched.")
        return

    print(f"Workflow version: {WORKFLOW_VERSION}")
    print(f"Table: {args.table}")
    print(f"Image source column: {args.image_url_column}")
    print(f"Embedding column: {args.embedding_column}")
    print(f"Image cache dir: {args.image_cache_dir}")
    print(f"Rows selected: {len(product_rows)}")
    print(f"Start id: {args.start_id if args.start_id is not None else '-'}")
    print(f"Sub category filter: {', '.join(args.sub_category) if args.sub_category else '-'}")
    print(f"Apply: {args.apply}")
    print(f"Download workers: {args.download_workers}")
    print(f"Embedding batch size: {args.embedding_batch_size}")
    print(f"Update batch size: {args.update_batch_size}")

    print("Downloading images.")
    image_paths, download_failed = predownload_images(product_rows, args)
    failed_ids = {str(row["id"]) for row in download_failed}

    fclip = load_fashion_clip(args.fashion_clip_model_id)
    failed_log = os.path.join(BASE_DIR, "fashion_embedding_update_failed.log")
    prepared = []
    failed_count = len(download_failed)
    encoded_count = 0
    updated_count = 0
    update_batch = []

    for index, row in enumerate(product_rows, start=1):
        product_id = str(row.get(args.id_column) or "")
        if not product_id or product_id in failed_ids:
            continue
        image_path = image_paths.get(product_id)
        if not image_path:
            entry = log_entry_for_row(row, args, "downloaded image path missing")
            append_issue_log(failed_log, entry)
            failed_count += 1
            continue
        prepared.append((index, row, image_path))

        if len(prepared) >= args.embedding_batch_size:
            encoded_count, updated_count, failed_count, update_batch = process_embedding_batch(
                client, prepared, fclip, args, failed_log, encoded_count, updated_count, failed_count, update_batch
            )
            prepared = []

    if prepared:
        encoded_count, updated_count, failed_count, update_batch = process_embedding_batch(
            client, prepared, fclip, args, failed_log, encoded_count, updated_count, failed_count, update_batch
        )
    if args.apply and update_batch:
        try:
            returned_rows = flush_update_batch(client, update_batch, args)
            print(f"   final_batch_update_rows={returned_rows}")
        except Exception as exc:
            for failed_row, _embedding in update_batch:
                entry = log_entry_for_row(failed_row, args, f"final_batch_update_failed: {exc}")
                append_issue_log(failed_log, entry)
                failed_count += 1
            print(f"   final_batch_update_failed rows={len(update_batch)} error={exc}")
        update_batch = []

    print("\nFashionCLIP embedding update complete")
    print(f"Prepared embeddings: {encoded_count}")
    print(f"Prepared DB updates: {updated_count}")
    print(f"Failed: {failed_count}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to update Supabase.")
    if failed_count:
        print(f"Failed log: {failed_log}")


def process_embedding_batch(client, prepared, fclip, args, failed_log, encoded_count, updated_count, failed_count, update_batch):
    image_paths = [image_path for _index, _row, image_path in prepared]
    try:
        embeddings = encode_image_paths(fclip, image_paths, args.embedding_batch_size)
    except Exception as exc:
        for _index, row, _image_path in prepared:
            entry = log_entry_for_row(row, args, f"embedding_failed: {exc}")
            append_issue_log(failed_log, entry)
            failed_count += 1
        print(f"Embedding batch failed rows={len(prepared)} error={exc}")
        return encoded_count, updated_count, failed_count, update_batch

    for embedding_index, (item_index, row, _image_path) in enumerate(prepared):
        embedding = embeddings[embedding_index].tolist()
        product_id = row.get(args.id_column)
        name = row.get(args.name_column) or ""
        encoded_count += 1
        updated_count += 1
        print(f"[{item_index}] id={product_id} name={name} embedding_dim={len(embedding)}")
        if args.apply:
            update_batch.append((row, embedding))
            if len(update_batch) >= args.update_batch_size:
                try:
                    returned_rows = flush_update_batch(client, update_batch, args)
                    print(f"   batch_update_rows={returned_rows}")
                except Exception as exc:
                    for failed_row, _embedding in update_batch:
                        entry = log_entry_for_row(failed_row, args, f"batch_update_failed: {exc}")
                        append_issue_log(failed_log, entry)
                        failed_count += 1
                    print(f"   batch_update_failed rows={len(update_batch)} error={exc}")
                update_batch = []
    return encoded_count, updated_count, failed_count, update_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate fashion_embedding from Supabase image_url with FashionCLIP.")
    parser.add_argument("--env", default=os.path.join(DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", NAME_COLUMN))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", IMAGE_COLUMN))
    parser.add_argument("--sub-category-column", default=os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category"))
    parser.add_argument("--embedding-column", default="fashion_embedding")
    parser.add_argument("--order-column", default=os.environ.get("FASHION_EMBEDDING_ORDER_COLUMN", "id"))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--image-cache-dir", default=DEFAULT_IMAGE_CACHE_DIR)
    parser.add_argument("--refresh-downloaded-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--update-batch-size", type=int, default=50)
    parser.add_argument("--fashion-clip-model-id", default=os.environ.get("FASHION_CLIP_API_MODEL_ID", "fashion-clip"))
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--sub-category", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    update_fashion_embeddings(parse_args())
