import argparse
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(DB_DATA_DIR)
TEST_DIR = os.path.join(DB_DATA_DIR, "test")
DEFAULT_IMAGE_DIR = os.path.join(DB_DATA_DIR, "image_jpg_700")

if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from verify_groundingdino_sam_color_extraction import (  # noqa: E402
    DEFAULT_DINO_MODEL_ID,
    DEFAULT_PROMPT,
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_SAM_MODEL_TYPE,
    FINAL_COLOR_CATEGORIES,
    load_models,
    normalize_product_id,
    verify_image,
)


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
        return search_candidates

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
    return candidates


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


def fetch_product_names(client, args):
    requested_ids = {str(value) for value in args.ids}
    select_columns = [args.id_column, args.name_column]
    if args.order_column not in select_columns:
        select_columns.append(args.order_column)

    product_names = {}
    last_order_value = None
    while True:
        query = (
            client
            .table(args.table)
            .select(", ".join(select_columns))
            .order(args.order_column, desc=False)
            .limit(args.page_size)
        )
        if last_order_value is not None:
            query = query.gt(args.order_column, last_order_value)
        if requested_ids:
            query = query.in_(args.id_column, list(requested_ids))

        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        for row in rows:
            row_id = str(row.get(args.id_column) or "")
            if row_id:
                product_names[row_id] = row.get(args.name_column) or ""

        if requested_ids or len(rows) < args.page_size:
            break
        next_order_value = rows[-1].get(args.order_column)
        if next_order_value is None or next_order_value == last_order_value:
            raise RuntimeError(f"Cannot continue pagination with order column {args.order_column}")
        last_order_value = next_order_value

    return product_names


def image_path_for_id(image_dir, product_id):
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(image_dir, f"{product_id}{extension}")
        if os.path.exists(path):
            return path
    return ""


def select_product_ids(product_names, args):
    product_ids = sorted(product_names.keys(), key=lambda value: int(value) if value.isdigit() else value)
    if args.ids:
        requested = {str(value) for value in args.ids}
        product_ids = [product_id for product_id in product_ids if product_id in requested]
    if args.limit:
        product_ids = product_ids[: args.limit]
    return product_ids


def should_update_result(result, args):
    if result.status != "ok":
        return False, f"status={result.status}"
    if result.extracted_color not in FINAL_COLOR_CATEGORIES:
        return False, f"invalid_color={result.extracted_color}"
    if not confidence_is_allowed(result.color_confidence, args.min_confidence):
        return False, f"confidence={result.color_confidence}"
    return True, ""


def update_groundingdino_sam_colors(args):
    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)

    client = get_supabase_client(args.env)
    product_names = fetch_product_names(client, args)
    product_ids = select_product_ids(product_names, args)
    if not product_ids:
        print("No DB rows matched.")
        return

    print(f"Table: {args.table}")
    print(f"Image dir: {image_dir}")
    print(f"Rows selected: {len(product_ids)}")
    print(f"Apply: {args.apply}")
    print(f"Min confidence: {args.min_confidence}")
    print("Loading GroundingDINO + SAM models.")
    models = load_models(args.dino_model_id, args.sam_checkpoint, args.sam_model_type, args.device)

    updated = 0
    skipped = []
    failed = []

    for index, product_id in enumerate(product_ids, start=1):
        image_path = image_path_for_id(image_dir, product_id)
        if not image_path:
            skipped.append((product_id, "local image missing"))
            print(f"[{index}/{len(product_ids)}] skipped id={product_id}: local image missing")
            continue

        try:
            result, _image, _mask = verify_image(image_path, models, args, product_names)
            allowed, reason = should_update_result(result, args)
            if not allowed:
                skipped.append((product_id, reason))
                print(f"[{index}/{len(product_ids)}] skipped id={product_id}: {reason}")
                continue

            payload = build_payload(result, args)
            print(
                f"[{index}/{len(product_ids)}] id={product_id} "
                f"name={result.product_name} "
                f"color={result.extracted_color} "
                f"confidence={result.color_confidence} "
                f"reason={result.color_reason}"
            )

            if args.preview and updated < args.preview:
                print(f"   payload={payload}")

            if args.apply:
                response = (
                    client
                    .table(args.table)
                    .update(payload)
                    .eq(args.id_column, product_id)
                    .execute()
                )
                print(f"   returned_rows={len(response.data or [])}")
            updated += 1
        except Exception as exc:
            failed.append((product_id, str(exc)))
            print(f"[{index}/{len(product_ids)}] failed id={product_id}: {exc}")

    print("\nGroundingDINO+SAM color update complete")
    print(f"Prepared updates: {updated}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to update Supabase.")

    if skipped:
        skipped_log = os.path.join(BASE_DIR, "groundingdino_sam_color_update_skipped.log")
        with open(skipped_log, "w", encoding="utf-8") as file:
            for product_id, reason in skipped:
                file.write(f"{product_id}\t{reason}\n")
        print(f"Skipped log: {skipped_log}")
    if failed:
        failed_log = os.path.join(BASE_DIR, "groundingdino_sam_color_update_failed.log")
        with open(failed_log, "w", encoding="utf-8") as file:
            for product_id, reason in failed:
                file.write(f"{product_id}\t{reason}\n")
        print(f"Failed log: {failed_log}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate DB color columns with GroundingDINO + SAM.")
    parser.add_argument("--env", default=os.path.join(DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    update_groundingdino_sam_colors(parse_args())
