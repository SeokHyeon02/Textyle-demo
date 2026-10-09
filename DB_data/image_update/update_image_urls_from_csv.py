import argparse
import csv
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
TABLE_NAME = os.environ.get("IMAGE_URL_UPDATE_TABLE", "clothes")
CSV_PATH = os.environ.get(
    "IMAGE_URL_UPDATE_CSV_PATH",
    os.path.join(BASE_DIR, "selected_shop_links.csv"),
)

SKIP_WORDS = ("수정", "크롭", "캡쳐", "캡처")
DELETE_WORD = "삭제"


supabase = None


def log(message: str):
    print(message, flush=True)


def get_supabase_client():
    global supabase

    if supabase is not None:
        return supabase

    log("[INFO] Supabase 클라이언트를 준비하는 중입니다.")
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(dotenv_path=os.path.join(DB_DATA_DIR, ".env"))

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        log(".env 파일에서 SUPABASE_URL 또는 SUPABASE_KEY를 찾을 수 없습니다.")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)
    log("[INFO] Supabase 클라이언트 준비 완료")
    return supabase


def is_url(value: str):
    return value.startswith("http://") or value.startswith("https://")


def should_skip_by_note(value: str):
    return any(word in value for word in SKIP_WORDS)


def update_image_url_by_id(row_id: str, new_image_url: str, dry_run: bool):
    if dry_run:
        return 0

    client = get_supabase_client()
    existing_rows = (
        client.table(TABLE_NAME)
        .select("id")
        .eq("id", row_id)
        .execute()
    )
    matched_count = len(existing_rows.data or [])
    if matched_count == 0:
        return 0

    response = (
        client.table(TABLE_NAME)
        .update({"image_url": new_image_url})
        .eq("id", row_id)
        .execute()
    )

    return len(response.data or []) or matched_count


def delete_rows_by_id(row_id: str, dry_run: bool):
    if dry_run:
        return 0

    client = get_supabase_client()
    existing_rows = (
        client.table(TABLE_NAME)
        .select("id")
        .eq("id", row_id)
        .execute()
    )
    matched_count = len(existing_rows.data or [])
    if matched_count == 0:
        return 0

    client.table(TABLE_NAME).delete().eq("id", row_id).execute()
    return matched_count


def process_csv(csv_path: str, dry_run: bool):
    log(f"[INFO] CSV 읽기 시작: {csv_path}")
    log(f"[INFO] 실행 모드: {'dry-run' if dry_run else 'DB update'}")

    counts = {
        "total": 0,
        "updated": 0,
        "deleted": 0,
        "dry_run_targets": 0,
        "dry_run_delete_targets": 0,
        "skipped_header": 0,
        "skipped_short_row": 0,
        "skipped_note": 0,
        "skipped_empty_action": 0,
        "skipped_not_url": 0,
        "not_found": 0,
    }

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        for line_number, row in enumerate(reader, start=1):
            counts["total"] += 1

            if line_number == 1 and row and row[0].strip().lower() in ("saved_at", "date", "날짜"):
                counts["skipped_header"] += 1
                continue

            if len(row) < 4:
                counts["skipped_short_row"] += 1
                log(f"[SKIP] line {line_number}: CSV는 날짜,id,url,바꿀 url/캡쳐/삭제 4칸이어야 합니다.")
                continue

            row_id = row[1].strip()
            current_image_url = row[2].strip()
            action_value = row[3].strip()

            if not row_id:
                counts["skipped_short_row"] += 1
                log(f"[SKIP] line {line_number}: id 값이 없습니다.")
                continue

            if not action_value:
                counts["skipped_empty_action"] += 1
                log(f"[SKIP] line {line_number}: 바꿀 url/캡쳐/삭제 값이 비어 있습니다. id={row_id}")
                continue

            if DELETE_WORD in action_value:
                log(f"[ROW DELETE TARGET] line {line_number}: id={row_id}")
                deleted_count = delete_rows_by_id(row_id, dry_run)
                if dry_run:
                    counts["dry_run_delete_targets"] += 1
                    log(f"[DRY RUN ROW DELETE] line {line_number}: id={row_id}")
                elif deleted_count == 0:
                    counts["not_found"] += 1
                    log(f"[MISS] line {line_number}: DB에서 삭제할 row를 찾지 못했습니다. id={row_id}")
                else:
                    counts["deleted"] += deleted_count
                    log(f"[ROW DELETE] line {line_number}: DB row {deleted_count}개 삭제")
                continue

            if should_skip_by_note(action_value):
                counts["skipped_note"] += 1
                log(f"[SKIP] line {line_number}: '{action_value}' 메모가 있습니다. id={row_id}")
                continue

            if not is_url(action_value):
                counts["skipped_not_url"] += 1
                log(f"[SKIP] line {line_number}: 바꿀 값이 URL이 아닙니다. id={row_id}, value={action_value}")
                continue

            log(f"[TARGET] line {line_number}: id={row_id}")
            matched_count = update_image_url_by_id(row_id, action_value, dry_run)
            if dry_run:
                counts["dry_run_targets"] += 1
                log(f"[DRY RUN] line {line_number}: id={row_id}, {current_image_url} -> {action_value}")
            elif matched_count == 0:
                counts["not_found"] += 1
                log(f"[MISS] line {line_number}: DB에서 id를 찾지 못했습니다. id={row_id}")
            else:
                counts["updated"] += matched_count
                log(f"[UPDATE] line {line_number}: DB row {matched_count}개 업데이트")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="CSV의 날짜,id,url,바꿀 url/캡쳐/삭제 형식에 따라 clothes.image_url을 업데이트하거나 row를 삭제합니다."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="읽을 CSV 파일 경로")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB를 수정하지 않고 업데이트/삭제 대상만 출력",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        log(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
        sys.exit(1)

    counts = process_csv(args.csv, args.dry_run)

    log("\n완료")
    for key, value in counts.items():
        log(f"- {key}: {value}")


if __name__ == "__main__":
    main()
