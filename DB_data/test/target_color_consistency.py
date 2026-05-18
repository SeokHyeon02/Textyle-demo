import argparse
import csv
import importlib.util
import os
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
UPDATE_DIR = os.path.join(BASE_DIR, "update")
if UPDATE_DIR not in sys.path:
    sys.path.insert(0, UPDATE_DIR)

DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "target_color_consistency_report.csv")
PAGE_SIZE = 1000
TRUSTED_EXTRACTION_CONFIDENCES = {"high", "medium"}
COLOR_FAMILIES = (
    {"blue", "navy", "indigo"},
)
COLOR_DB_COLUMN = "dominant_color"
COLOR_CONFIDENCE_DB_COLUMN = "color_confidence"
COLOR_CANDIDATES_DB_COLUMN = "color_candidates"
IMAGE_COLUMN = "image_url"
MAIN_CATEGORY_COLUMN = "main_category"
NAME_COLUMN = "name"
SUB_CATEGORY_COLUMN = "sub_category"

TARGET_PRESETS = {
    "top_sweater": {
        "category": "상의",
        "sub_category_contains": "스웨터",
    },
    "jeans": {
        "category": "하의",
        "sub_category_contains": "데님",
    },
    "leather_jacket": {
        "category": "아우터",
        "sub_category_contains": "레더",
    },
}


def load_fashion_color_utils():
    module_path = os.path.join(UPDATE_DIR, "fashion_color_utils.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find fashion_color_utils.py at {module_path}")

    module_name = "_target_color_fashion_color_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fashion_color_utils.py from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_color_value(value):
    return (value or "").strip().lower()


def stored_candidate_colors(stored_color, color_candidates=None):
    colors = []
    primary_color = normalize_color_value(stored_color)
    if primary_color:
        colors.append(primary_color)

    for candidate in color_candidates or []:
        if not isinstance(candidate, dict):
            continue
        candidate_color = normalize_color_value(candidate.get("color"))
        if candidate_color and candidate_color not in colors:
            colors.append(candidate_color)
    return colors


def compare_colors(stored_color, extracted_color, extracted_confidence="high", color_candidates=None):
    stored = normalize_color_value(stored_color)
    extracted = normalize_color_value(extracted_color)
    confidence = normalize_color_value(extracted_confidence)
    candidate_colors = stored_candidate_colors(stored, color_candidates)

    if confidence not in TRUSTED_EXTRACTION_CONFIDENCES:
        return "low_confidence"
    if not extracted:
        return "missing_extracted"
    if not candidate_colors:
        return "missing_stored"
    if stored == extracted:
        return "match"
    if extracted in candidate_colors:
        return "candidate_match"
    if any(candidate in family and extracted in family for candidate in candidate_colors for family in COLOR_FAMILIES):
        return "family_match"
    return "mismatch"


def build_report_row(item, extracted, image_error=""):
    extracted = extracted or {}
    status = "image_error" if image_error else compare_colors(
        item.get(COLOR_DB_COLUMN),
        extracted.get("color"),
        extracted.get("confidence"),
        item.get(COLOR_CANDIDATES_DB_COLUMN),
    )

    return {
        "id": item.get("id"),
        "product_name": item.get(NAME_COLUMN),
        "name": item.get(NAME_COLUMN),
        "main_category": item.get(MAIN_CATEGORY_COLUMN),
        "sub_category": item.get(SUB_CATEGORY_COLUMN),
        "stored_color": item.get(COLOR_DB_COLUMN) or "",
        "stored_color_confidence": item.get(COLOR_CONFIDENCE_DB_COLUMN) or "",
        "stored_color_candidates": item.get(COLOR_CANDIDATES_DB_COLUMN) or "",
        "extracted_color": extracted.get("color") or "",
        "extracted_confidence": extracted.get("confidence") or "",
        "extracted_reason": extracted.get("reason") or "",
        "dominant_ratio": extracted.get("dominant_ratio", 0.0),
        "second_ratio": extracted.get("second_ratio", 0.0),
        "status": status,
        "image_error": image_error,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare stored dominant_color values for a configurable product target."
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to inspect.")
    parser.add_argument("--offset", type=int, default=0, help="Initial row offset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="CSV report path.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include rows whose stored dominant_color is empty.",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Write only mismatch, missing, low-confidence, and image-error rows.",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_PRESETS),
        default="top_sweater",
        help="Reusable target preset. Override individual filters with category options.",
    )
    parser.add_argument(
        "--target-label",
        default="",
        help="Label printed in logs. Defaults to the selected target preset.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional exact main_category filter, e.g. 상의, 하의, 아우터.",
    )
    parser.add_argument(
        "--sub-category",
        default="",
        help="Optional exact sub_category filter.",
    )
    parser.add_argument(
        "--sub-category-contains",
        default=None,
        help="Optional substring filter for sub_category, e.g. 스웨터, 데님, 레더.",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Optional substring filter for product name.",
    )
    args = parser.parse_args(argv)
    preset = TARGET_PRESETS[args.target]
    if args.category is None:
        args.category = preset["category"]
    if args.sub_category_contains is None:
        args.sub_category_contains = preset["sub_category_contains"]
    if not args.target_label:
        args.target_label = args.target
    return args


def load_supabase_client():
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in DB_data/.env")
    return create_client(supabase_url, supabase_key)


def apply_text_filter(query, column, exact_value="", contains_value=""):
    exact_value = (exact_value or "").strip()
    contains_value = (contains_value or "").strip()
    if exact_value:
        return query.eq(column, exact_value)
    if contains_value:
        return query.ilike(column, f"%{contains_value}%")
    return query


def fetch_items(
    client,
    limit,
    offset,
    include_missing,
    category="",
    sub_category="",
    sub_category_contains="",
    name_contains="",
):
    selected_columns = [
        "id",
        NAME_COLUMN,
        IMAGE_COLUMN,
        MAIN_CATEGORY_COLUMN,
        SUB_CATEGORY_COLUMN,
        COLOR_DB_COLUMN,
        COLOR_CONFIDENCE_DB_COLUMN,
        COLOR_CANDIDATES_DB_COLUMN,
    ]
    query = (
        client
        .table("clothes")
        .select(", ".join(selected_columns))
        .order("id", desc=False)
        .range(offset, offset + limit - 1)
    )

    if not include_missing:
        query = query.not_.is_(COLOR_DB_COLUMN, "null")
    if category:
        query = query.eq(MAIN_CATEGORY_COLUMN, category.strip())
    query = apply_text_filter(
        query,
        SUB_CATEGORY_COLUMN,
        exact_value=sub_category,
        contains_value=sub_category_contains,
    )
    if name_contains:
        query = query.ilike(NAME_COLUMN, f"%{name_contains.strip()}%")

    response = query.execute()
    return response.data or []


def extract_image_color_for_item(item):
    color_utils = load_fashion_color_utils()

    image = color_utils.download_product_image(item.get(IMAGE_COLUMN))
    denim_context = color_utils.is_denim_context(
        item.get(NAME_COLUMN) or "",
        item.get(SUB_CATEGORY_COLUMN) or "",
    )
    return color_utils.extract_dominant_color_result(image, denim_context=denim_context)


def should_write_row(row, only_mismatches):
    if not only_mismatches:
        return True
    return row["status"] not in {"match", "candidate_match", "family_match"}


def write_report(rows, output_path):
    fieldnames = [
        "id",
        "product_name",
        "name",
        "main_category",
        "sub_category",
        "stored_color",
        "stored_color_confidence",
        "stored_color_candidates",
        "extracted_color",
        "extracted_confidence",
        "extracted_reason",
        "dominant_ratio",
        "second_ratio",
        "status",
        "image_error",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def main():
    args = parse_args()
    client = load_supabase_client()
    items = fetch_items(
        client,
        limit=args.limit,
        offset=args.offset,
        include_missing=args.include_missing,
        category=args.category,
        sub_category=args.sub_category,
        sub_category_contains=args.sub_category_contains,
        name_contains=args.name_contains,
    )

    report_rows = []
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started target color consistency check at {started_at}")
    print(f"Target: {args.target_label}")
    print(
        "Filters: "
        f"category={args.category or '-'}, "
        f"sub_category={args.sub_category or '-'}, "
        f"sub_category_contains={args.sub_category_contains or '-'}, "
        f"name_contains={args.name_contains or '-'}"
    )
    print(f"Rows loaded: {len(items)}")

    for index, item in enumerate(items, 1):
        image_error = ""
        extracted = {}
        try:
            extracted = extract_image_color_for_item(item)
        except Exception as exc:
            image_error = str(exc)

        row = build_report_row(item, extracted, image_error=image_error)
        if should_write_row(row, args.only_mismatches):
            report_rows.append(row)

        print(
            f"[{index}/{len(items)}] "
            f"id={row['id']} "
            f"name={row['product_name'] or '-'} "
            f"stored={row['stored_color'] or '-'} "
            f"candidates={row['stored_color_candidates'] or '-'} "
            f"extracted={row['extracted_color'] or '-'} "
            f"confidence={row['extracted_confidence'] or '-'} "
            f"status={row['status']}"
        )

    write_report(report_rows, args.output)
    print(f"\nReport saved: {args.output}")
    print(f"Rows written: {len(report_rows)}")
    print("Summary:")
    for status, count in sorted(summarize(report_rows).items()):
        print(f"- {status}: {count}")


if __name__ == "__main__":
    main()
