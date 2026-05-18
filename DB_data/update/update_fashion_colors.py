import os

from fashion_color_utils import (
    COLOR_CANDIDATES_DB_COLUMN,
    COLOR_CONFIDENCE_DB_COLUMN,
    COLOR_DB_COLUMN,
    IMAGE_COLUMN,
    NAME_COLUMN,
    SUB_CATEGORY_COLUMN,
    download_product_image,
    extract_color_attributes,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
ID_COLUMN = os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id")
MAIN_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_MAIN_CATEGORY_COLUMN", "main_category")
ORDER_COLUMN = os.environ.get("FASHION_COLOR_ORDER_COLUMN", ID_COLUMN)
PAGE_SIZE = int(os.environ.get("FASHION_COLOR_PAGE_SIZE", "1000"))
VERIFY_UPDATES = os.environ.get("VERIFY_FASHION_COLOR_UPDATES", "false").strip().lower() in {
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


def build_color_payload(color_attributes):
    return {
        COLOR_DB_COLUMN: color_attributes.get(COLOR_DB_COLUMN),
        COLOR_CONFIDENCE_DB_COLUMN: color_attributes.get(COLOR_CONFIDENCE_DB_COLUMN),
        COLOR_CANDIDATES_DB_COLUMN: color_attributes.get(COLOR_CANDIDATES_DB_COLUMN) or [],
    }


def update_color_for_item(client, item, item_index, total_count):
    row_id = item.get(ID_COLUMN)
    name = item.get(NAME_COLUMN) or "unnamed"
    image_url = item.get(IMAGE_COLUMN)

    if not image_url:
        raise ValueError("image_url missing")
    image = download_product_image(image_url)
    color_attributes = extract_color_attributes(image, item)
    payload = build_color_payload(color_attributes)

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
            .select(f"{COLOR_DB_COLUMN}, {COLOR_CONFIDENCE_DB_COLUMN}, {COLOR_CANDIDATES_DB_COLUMN}")
            .eq(ID_COLUMN, row_id)
            .limit(1)
            .execute()
        )

    verified_item = verify_response.data[0] if verify_response and verify_response.data else {}
    print(
        "   -> "
        f"[{item_index}/{total_count}] "
        f"id={row_id} {name}, "
        f"color={payload[COLOR_DB_COLUMN]}, "
        f"color_confidence={payload[COLOR_CONFIDENCE_DB_COLUMN]}, "
        f"color_candidates={payload[COLOR_CANDIDATES_DB_COLUMN]}, "
        f"color_reason={color_attributes.get('color_reason')}, "
        f"returned_rows={len(update_response.data or []) if update_response else 0}, "
        f"verified_rows={len(verify_response.data or []) if verify_response else 0}, "
        f"db_color={verified_item.get(COLOR_DB_COLUMN)}, "
        f"db_color_confidence={verified_item.get(COLOR_CONFIDENCE_DB_COLUMN)}, "
        f"db_color_candidates={verified_item.get(COLOR_CANDIDATES_DB_COLUMN)}"
    )


def update_all_fashion_colors():
    client = get_supabase_client()
    print("Loading all clothes rows from Supabase.")
    print("Only color columns will be regenerated. fashion_embedding will not be changed.")
    print(f"Color columns: {COLOR_DB_COLUMN}, {COLOR_CONFIDENCE_DB_COLUMN}, {COLOR_CANDIDATES_DB_COLUMN}")
    print(f"Order column: {ORDER_COLUMN}")
    print(f"Page size: {PAGE_SIZE}")
    print(f"Verify updates: {VERIFY_UPDATES}")
    print(f"Dry run: {DRY_RUN}")

    all_items = fetch_all_items()
    if not all_items:
        print("No rows to update.")
        return

    failed_items = []
    for index, item in enumerate(all_items, 1):
        row_id = item.get(ID_COLUMN)
        name = item.get(NAME_COLUMN) or "unnamed"
        if row_id is None or row_id == "":
            failed_items.append((row_id, name, "id missing"))
            print(f"[{index}/{len(all_items)}] skipped: {name} - id missing")
            continue
        try:
            update_color_for_item(client, item, index, len(all_items))
        except Exception as exc:
            failed_items.append((row_id, name, str(exc)))
            print(f"[{index}/{len(all_items)}] failed: id={row_id} {name} - {exc}")

    print("\nFashion color update complete")
    print(f"Failed items: {len(failed_items)}")
    if failed_items:
        failed_log_path = os.path.join(BASE_DIR, "fashion_color_failed.log")
        with open(failed_log_path, "w", encoding="utf-8") as log_file:
            for row_id, name, reason in failed_items:
                log_file.write(f"{row_id}\t{name}\t{reason}\n")
        print(f"Failed item log saved: {failed_log_path}")


if __name__ == "__main__":
    update_all_fashion_colors()
