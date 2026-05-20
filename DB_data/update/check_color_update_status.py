import argparse
import json
import os
from collections import Counter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
FINAL_COLORS = {"white", "black", "red", "yellow", "green", "blue", "purple", "gray", "orange", "brown", "pink"}


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


def fetch_rows(client, args):
    select_columns = [
        args.id_column,
        args.name_column,
        args.dominant_color_column,
        args.color_confidence_column,
        args.color_candidates_column,
        args.color_reason_column,
        args.named_color_column,
        args.pre_hint_color_column,
        args.pattern_hint_column,
        args.pattern_vit_column,
    ]
    select_columns = [column for column in select_columns if column]

    rows = []
    last_order_value = None
    while len(rows) < args.limit:
        page_limit = min(args.page_size, args.limit - len(rows))
        query = (
            client
            .table(args.table)
            .select(", ".join(select_columns))
            .order(args.order_column, desc=False)
            .limit(page_limit)
        )
        if args.start_id is not None and last_order_value is None:
            query = query.gte(args.id_column, args.start_id)
        if last_order_value is not None:
            query = query.gt(args.order_column, last_order_value)

        response = query.execute()
        page_rows = response.data or []
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(page_rows) < page_limit:
            break
        next_order_value = page_rows[-1].get(args.order_column)
        if next_order_value is None or next_order_value == last_order_value:
            raise RuntimeError(f"Cannot continue pagination with order column {args.order_column}")
        last_order_value = next_order_value
    return rows


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def analyze_rows(rows, args):
    color_counts = Counter()
    confidence_counts = Counter()
    reason_counts = Counter()
    problems = []
    pattern_vit_count = 0

    for row in rows:
        color = row.get(args.dominant_color_column) or ""
        confidence = row.get(args.color_confidence_column) or ""
        candidates = as_list(row.get(args.color_candidates_column))
        reason = row.get(args.color_reason_column) or ""
        if row.get(args.pattern_vit_column) is True:
            pattern_vit_count += 1

        color_counts[color or "(empty)"] += 1
        confidence_counts[confidence or "(empty)"] += 1
        reason_counts[reason or "(empty)"] += 1

        row_problems = []
        if not color:
            row_problems.append("dominant_color missing")
        elif color not in FINAL_COLORS:
            row_problems.append(f"invalid dominant_color={color}")
        if not confidence:
            row_problems.append("color_confidence missing")
        if not candidates:
            row_problems.append("color_candidates missing")
        elif len(candidates) > 3:
            row_problems.append(f"too many color_candidates={len(candidates)}")
        if not reason:
            row_problems.append("color_reason missing")

        if row_problems:
            problems.append({
                "id": row.get(args.id_column),
                "name": row.get(args.name_column),
                "dominant_color": color,
                "color_confidence": confidence,
                "color_reason": reason,
                "candidate_count": len(candidates),
                "problems": row_problems,
            })

    return {
        "checked_rows": len(rows),
        "problem_rows": len(problems),
        "pattern_vit_rows": pattern_vit_count,
        "color_distribution": dict(color_counts.most_common()),
        "confidence_distribution": dict(confidence_counts.most_common()),
        "reason_distribution": dict(reason_counts.most_common()),
        "sample_problems": problems[: args.problem_preview],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Check Supabase color update status without modifying DB.")
    parser.add_argument("--env", default=os.path.join(DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--dominant-color-column", default="dominant_color")
    parser.add_argument("--color-confidence-column", default="color_confidence")
    parser.add_argument("--color-candidates-column", default="color_candidates")
    parser.add_argument("--color-reason-column", default="color_reason")
    parser.add_argument("--named-color-column", default="extracted_named_color")
    parser.add_argument("--pre-hint-color-column", default="pre_hint_color")
    parser.add_argument("--pattern-hint-column", default="pattern_hint")
    parser.add_argument("--pattern-vit-column", default="should_run_pattern_vit")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--problem-preview", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    client = get_supabase_client(args.env)
    rows = fetch_rows(client, args)
    result = analyze_rows(rows, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
