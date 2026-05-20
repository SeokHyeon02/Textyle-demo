import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fashion_color_utils import IMAGE_COLUMN, download_product_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(DB_DATA_DIR)
TEST_DIR = os.path.join(DB_DATA_DIR, "test")
DEFAULT_IMAGE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "textyle_groundingdino_sam_images")
UPDATE_WORKFLOW_VERSION = "2026-05-19-parallel-download-batch-upsert-v1"

VERIFY_MODULE_PATH = os.path.join(TEST_DIR, "verify_groundingdino_sam_color_extraction.py")


def load_verify_module():
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(
        "verify_groundingdino_sam_color_extraction",
        VERIFY_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load verification module: {VERIFY_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_module = load_verify_module()

DEFAULT_DINO_MODEL_ID = verify_module.DEFAULT_DINO_MODEL_ID
DEFAULT_PROMPT = verify_module.DEFAULT_PROMPT
DEFAULT_SAM_CHECKPOINT = verify_module.DEFAULT_SAM_CHECKPOINT
DEFAULT_SAM_MODEL_TYPE = verify_module.DEFAULT_SAM_MODEL_TYPE
FINAL_COLOR_CATEGORIES = verify_module.FINAL_COLOR_CATEGORIES
load_models = verify_module.load_models
normalize_product_id = verify_module.normalize_product_id
verify_image = verify_module.verify_image


CONFIDENCE_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


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


def parse_json_array(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def confidence_is_allowed(confidence, minimum):
    if not minimum:
        return True
    return CONFIDENCE_ORDER.get(confidence or "", -1) >= CONFIDENCE_ORDER.get(minimum, 0)


def optional_column_payload(payload, column_name, value):
    if column_name:
        payload[column_name] = value


def result_search_candidates(result):
    search_candidates = parse_json_array(result.search_colors_json)
    if search_candidates:
        return search_candidates[:3]

    candidates = []
    for candidate in parse_json_array(result.candidates_json):
        color = candidate.get("color") or candidate.get("fashion_color")
        if not color:
            continue
        candidates.append(
            {
                "color": color,
                "score": float(candidate.get("ratio") or 0.0),
                "source": "image",
                "confidence": result.color_confidence,
                "base_color": color,
            }
        )
    return candidates[:3]


def build_payload(result, args):
    payload = {
        args.dominant_color_column: result.extracted_color,
        args.color_confidence_column: result.color_confidence,
        args.color_candidates_column: result_search_candidates(result),
    }
    optional_column_payload(payload, args.color_reason_column, result.color_reason)
    optional_column_payload(payload, args.named_color_column, result.extracted_named_color)
    optional_column_payload(payload, args.pre_hint_color_column, result.pre_hint_color)
    optional_column_payload(payload, args.pattern_hint_column, result.pattern_hint)
    optional_column_payload(payload, args.pattern_vit_column, result.should_run_pattern_vit)
    optional_column_payload(payload, args.color_source_column, "groundingdino_sam")
    optional_column_payload(payload, args.named_candidates_column, parse_json_array(result.named_candidates_json))
    optional_column_payload(payload, args.sam_candidates_column, parse_json_array(result.sam_candidates_json))
    return payload


def fetch_product_rows(client, args):
    requested_ids = {str(value) for value in args.ids}
    select_columns = [args.id_column, args.name_column, args.image_url_column]
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


def product_names_from_rows(rows, args):
    return {
        str(row.get(args.id_column)): row.get(args.name_column) or ""
        for row in rows
        if row.get(args.id_column) is not None
    }


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
    if args.download_workers <= 1:
        for index, row in enumerate(product_rows, start=1):
            product_id = str(row.get(args.id_column) or "")
            try:
                image_paths[product_id] = download_image_for_row(row, args)
            except Exception as exc:
                failed.append(log_entry_for_row(row, args, str(exc)))
                print(f"[download {index}/{len(product_rows)}] failed id={product_id}: {exc}")
        return image_paths, failed

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
                failed.append(log_entry_for_row(row, args, str(exc)))
                print(f"[download {index}/{len(product_rows)}] failed id={product_id}: {exc}")
            if completed % 50 == 0 or completed == len(product_rows):
                print(f"Downloaded images: {completed}/{len(product_rows)}")
    return image_paths, failed


def should_update_result(result, args):
    if result.status != "ok":
        return False, f"status={result.status}"
    if result.extracted_color not in FINAL_COLOR_CATEGORIES:
        return False, f"invalid_color={result.extracted_color}"
    if not confidence_is_allowed(result.color_confidence, args.min_confidence):
        return False, f"confidence={result.color_confidence}"
    return True, ""


def log_entry_for_row(row, args, reason):
    return {
        "id": row.get(args.id_column) or "",
        "name": row.get(args.name_column) or "",
        "image_url": row.get(args.image_url_column) or "",
        "reason": reason,
    }


def write_issue_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "image_url", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def flush_update_batch(client, update_batch, args, failed):
    if not update_batch:
        return

    payloads = []
    for row, payload in update_batch:
        row_id = row.get(args.id_column)
        batch_payload = {args.id_column: row_id}
        batch_payload.update(payload)
        payloads.append(batch_payload)

    try:
        response = (
            client
            .table(args.table)
            .upsert(payloads, on_conflict=args.id_column)
            .execute()
        )
        print(f"   batch_upsert_rows={len(response.data or [])}")
    except Exception as exc:
        for row, _payload in update_batch:
            failed.append(log_entry_for_row(row, args, f"batch_upsert_failed: {exc}"))
        print(f"   batch_upsert_failed rows={len(update_batch)} error={exc}")


def update_groundingdino_sam_colors(args):
    args.image_cache_dir = os.path.abspath(args.image_cache_dir)
    args.download_workers = max(1, args.download_workers)
    args.update_batch_size = max(1, args.update_batch_size)

    client = get_supabase_client(args.env)
    product_rows = fetch_product_rows(client, args)
    product_names = product_names_from_rows(product_rows, args)
    if not product_rows:
        print("No DB rows matched.")
        return

    print(f"Table: {args.table}")
    print(f"Image source column: {args.image_url_column}")
    print(f"Image cache dir: {args.image_cache_dir}")
    print(f"Rows selected: {len(product_rows)}")
    print(f"Start id: {args.start_id if args.start_id is not None else '-'}")
    print(f"Apply: {args.apply}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Download workers: {args.download_workers}")
    print(f"Update batch size: {args.update_batch_size}")
    print(f"Update workflow version: {UPDATE_WORKFLOW_VERSION}")
    print(f"Verify module: {VERIFY_MODULE_PATH}")
    print(f"Verify workflow version: {getattr(verify_module, 'VERIFY_WORKFLOW_VERSION', 'unknown')}")
    print("Downloading images.")
    image_paths, download_failed = predownload_images(product_rows, args)
    download_failed_ids = {str(row["id"]) for row in download_failed}
    print("Loading GroundingDINO + SAM models.")
    models = load_models(args.dino_model_id, args.sam_checkpoint, args.sam_model_type, args.device)

    updated = 0
    skipped = []
    failed = list(download_failed)
    update_batch = []

    for index, row in enumerate(product_rows, start=1):
        product_id = str(row.get(args.id_column) or "")
        if not product_id:
            skipped.append(log_entry_for_row(row, args, "id missing"))
            print(f"[{index}/{len(product_rows)}] skipped: id missing")
            continue
        if product_id in download_failed_ids:
            continue

        try:
            image_path = image_paths.get(product_id)
            if not image_path:
                raise ValueError("downloaded image path missing")
            result, _image, _mask = verify_image(image_path, models, args, product_names)
            allowed, reason = should_update_result(result, args)
            if not allowed:
                skipped.append(log_entry_for_row(row, args, reason))
                print(f"[{index}/{len(product_rows)}] skipped id={product_id}: {reason}")
                continue

            payload = build_payload(result, args)
            print(
                f"[{index}/{len(product_rows)}] id={product_id} "
                f"name={result.product_name} "
                f"color={result.extracted_color} "
                f"confidence={result.color_confidence} "
                f"reason={result.color_reason}"
            )

            if args.preview and updated < args.preview:
                print(f"   payload={payload}")

            if args.apply:
                update_batch.append((row, payload))
                if len(update_batch) >= args.update_batch_size:
                    flush_update_batch(client, update_batch, args, failed)
                    update_batch = []
            updated += 1
        except Exception as exc:
            failed.append(log_entry_for_row(row, args, str(exc)))
            print(f"[{index}/{len(product_rows)}] failed id={product_id}: {exc}")

    if args.apply and update_batch:
        flush_update_batch(client, update_batch, args, failed)

    print("\nGroundingDINO+SAM color update complete")
    print(f"Prepared updates: {updated}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to update Supabase.")

    if skipped:
        skipped_log = os.path.join(BASE_DIR, "groundingdino_sam_color_update_skipped.log")
        write_issue_log(skipped_log, skipped)
        print(f"Skipped log: {skipped_log}")
    if failed:
        failed_log = os.path.join(BASE_DIR, "groundingdino_sam_color_update_failed.log")
        write_issue_log(failed_log, failed)
        print(f"Failed log: {failed_log}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate DB color columns with GroundingDINO + SAM.")
    parser.add_argument("--env", default=os.path.join(DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", IMAGE_COLUMN))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--image-cache-dir", default=DEFAULT_IMAGE_CACHE_DIR)
    parser.add_argument("--refresh-downloaded-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--update-batch-size", type=int, default=50)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dino-model-id", default=DEFAULT_DINO_MODEL_ID)
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default=DEFAULT_SAM_MODEL_TYPE)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default=os.environ.get("DINO_SAM_DEVICE", "cpu"))
    parser.add_argument("--dominant-color-column", default="dominant_color")
    parser.add_argument("--color-confidence-column", default="color_confidence")
    parser.add_argument("--color-candidates-column", default="color_candidates")
    parser.add_argument("--color-reason-column", default="")
    parser.add_argument("--named-color-column", default="")
    parser.add_argument("--pre-hint-color-column", default="")
    parser.add_argument("--pattern-hint-column", default="")
    parser.add_argument("--pattern-vit-column", default="")
    parser.add_argument("--color-source-column", default="")
    parser.add_argument("--named-candidates-column", default="")
    parser.add_argument("--sam-candidates-column", default="")
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    update_groundingdino_sam_colors(parse_args())
