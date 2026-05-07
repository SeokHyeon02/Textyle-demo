import csv
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "selected_shop_links.csv"

load_dotenv(dotenv_path=BASE_DIR / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
STORAGE_BASE_URL = os.environ.get("STORAGE_BASE_URL")
if not STORAGE_BASE_URL and SUPABASE_URL:
    STORAGE_BASE_URL = (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/image/image_capture"
    )


def load_csv_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.reader(csv_file))


def is_target_row(row):
    return len(row) == 3


def append_storage_urls(csv_rows):
    updated_count = 0
    for row in csv_rows:
        if not is_target_row(row):
            continue
        row_id = row[1].strip()
        if not row_id:
            continue
        row.append(f"{STORAGE_BASE_URL}/{row_id}.jpg")
        updated_count += 1

    return updated_count


def save_csv_rows(csv_rows):
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)


def main():
    if not STORAGE_BASE_URL:
        raise ValueError("STORAGE_BASE_URL or SUPABASE_URL is missing in .env")

    csv_rows = load_csv_rows()
    updated_count = append_storage_urls(csv_rows)
    save_csv_rows(csv_rows)
    print(f"Updated {updated_count} CSV rows with storage URLs")


if __name__ == "__main__":
    main()
