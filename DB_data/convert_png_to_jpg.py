import csv
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "selected_shop_links.csv"
INPUT_DIR = BASE_DIR / "image"
OUTPUT_DIR = BASE_DIR / "image_jpg_700"
MAX_SIZE = 700
QUALITY = 90


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_csv_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.reader(csv_file))


def is_target_row(row):
    return len(row) == 3


def collect_target_ids(csv_rows):
    target_ids = []
    for row in csv_rows:
        if not is_target_row(row):
            continue
        row_id = row[1].strip()
        if row_id:
            target_ids.append(row_id)

    return target_ids


def convert_image_to_jpg(source_path: Path, target_stem: str):
    output_path = OUTPUT_DIR / f"{target_stem}.jpg"

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        image.save(output_path, "JPEG", quality=QUALITY, optimize=True)

    return output_path


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    csv_rows = load_csv_rows()
    target_ids = collect_target_ids(csv_rows)
    if not target_ids:
        print(f"No 3-column rows found in {CSV_PATH}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    image_files = sorted(
        path for path in INPUT_DIR.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No PNG/JPG files found in {INPUT_DIR}")
        return

    if len(image_files) > len(target_ids):
        raise ValueError(
            f"Not enough IDs in CSV: images={len(image_files)}, ids={len(target_ids)}"
        )

    for source_path, target_id in zip(image_files, target_ids):
        output_path = convert_image_to_jpg(source_path, target_id)
        print(f"{source_path.name} -> {output_path.name}")

    print(f"Done. Converted {len(image_files)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
