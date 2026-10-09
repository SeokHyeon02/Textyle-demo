import argparse
import os
from types import SimpleNamespace

import update_groundingdino_sam_colors as color_update


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_CACHE_DIR = os.path.join(
    color_update.DEFAULT_IMAGE_CACHE_DIR,
    "indigo_denim",
)

DEFAULT_INDIGO_KEYWORDS = (
    "indigo",
    "raw denim",
    "raw indigo",
    "one wash",
    "one washed",
    "onewash",
    "deep indigo",
    "dark indigo",
    "인디고",
    "생지",
    "딥인디고",
    "딥 인디고",
    "다크인디고",
    "다크 인디고",
)


def normalize_match_text(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def row_matches_indigo_keywords(row, name_column, keywords):
    name = normalize_match_text(row.get(name_column) or "")
    return bool(name) and any(normalize_match_text(keyword) in name for keyword in keywords)


def filter_indigo_rows(rows, name_column, keywords):
    return [
        row
        for row in rows
        if row_matches_indigo_keywords(row, name_column, keywords)
    ]


def build_base_args(args):
    return SimpleNamespace(
        env=args.env,
        table=args.table,
        id_column=args.id_column,
        name_column=args.name_column,
        image_url_column=args.image_url_column,
        sub_category_column=args.sub_category_column,
        order_column=args.order_column,
        page_size=args.page_size,
        image_cache_dir=args.image_cache_dir,
        refresh_downloaded_images=args.refresh_downloaded_images,
        download_workers=args.download_workers,
        update_batch_size=args.update_batch_size,
        prompt=args.prompt,
        dino_model_id=args.dino_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        dominant_color_column=args.dominant_color_column,
        color_confidence_column=args.color_confidence_column,
        color_candidates_column=args.color_candidates_column,
        color_reason_column=args.color_reason_column,
        named_color_column=args.named_color_column,
        pre_hint_color_column=args.pre_hint_color_column,
        pattern_hint_column=args.pattern_hint_column,
        pattern_vit_column=args.pattern_vit_column,
        color_source_column=args.color_source_column,
        named_candidates_column=args.named_candidates_column,
        sam_candidates_column=args.sam_candidates_column,
        denim_tone_column=args.denim_tone_column,
        candidate_mode=args.candidate_mode,
        allow_name_fallback=args.allow_name_fallback,
        only_missing_denim_tone=False,
        min_confidence=args.min_confidence,
        ids=args.ids,
        start_id=args.start_id,
        sub_category=args.sub_category,
        limit=args.scan_limit,
        preview=args.preview,
        apply=args.apply,
    )


def update_indigo_denim_colors(args):
    base_args = build_base_args(args)
    original_fetch_product_rows = color_update.fetch_product_rows
    keywords = tuple(args.keyword)

    def fetch_indigo_rows(client, update_args):
        rows = original_fetch_product_rows(client, update_args)
        filtered_rows = filter_indigo_rows(rows, update_args.name_column, keywords)
        if args.limit:
            filtered_rows = filtered_rows[:args.limit]
        print(f"Indigo keyword scan rows: {len(rows)}")
        print(f"Indigo keyword matched rows: {len(filtered_rows)}")
        print(f"Indigo keywords: {', '.join(keywords)}")
        return filtered_rows

    color_update.fetch_product_rows = fetch_indigo_rows
    try:
        color_update.update_groundingdino_sam_colors(base_args)
    finally:
        color_update.fetch_product_rows = original_fetch_product_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate GroundingDINO/SAM color columns only for denim rows whose product names look indigo-related."
    )
    parser.add_argument("--env", default=os.path.join(color_update.DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", color_update.IMAGE_COLUMN))
    parser.add_argument("--sub-category-column", default=os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category"))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--scan-limit", type=int, default=5000, help="Maximum denim rows to scan before keyword filtering.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum matched indigo rows to process after filtering. 0 means all matched rows.")
    parser.add_argument("--image-cache-dir", default=DEFAULT_IMAGE_CACHE_DIR)
    parser.add_argument("--refresh-downloaded-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--update-batch-size", type=int, default=50)
    parser.add_argument("--prompt", default=color_update.DEFAULT_PROMPT)
    parser.add_argument("--dino-model-id", default=color_update.DEFAULT_DINO_MODEL_ID)
    parser.add_argument("--sam-checkpoint", default=color_update.DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default=color_update.DEFAULT_SAM_MODEL_TYPE)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default=os.environ.get("DINO_SAM_DEVICE", "cpu"))
    parser.add_argument("--dominant-color-column", default="dominant_color")
    parser.add_argument("--color-confidence-column", default="color_confidence")
    parser.add_argument("--color-candidates-column", default="color_candidates")
    parser.add_argument("--color-reason-column", default="color_reason")
    parser.add_argument("--named-color-column", default="extracted_named_color")
    parser.add_argument("--pre-hint-color-column", default="pre_hint_color")
    parser.add_argument("--pattern-hint-column", default="pattern_hint")
    parser.add_argument("--pattern-vit-column", default="should_run_pattern_vit")
    parser.add_argument("--color-source-column", default="")
    parser.add_argument("--named-candidates-column", default="")
    parser.add_argument("--sam-candidates-column", default="")
    parser.add_argument("--denim-tone-column", default="denim_tone")
    parser.add_argument(
        "--candidate-mode",
        choices=[color_update.IMAGE_SEARCH_CANDIDATE_MODE, color_update.FULL_DEBUG_CANDIDATE_MODE],
        default=color_update.IMAGE_SEARCH_CANDIDATE_MODE,
    )
    parser.add_argument("--allow-name-fallback", action="store_true")
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--sub-category", nargs="*", default=["데님팬츠"])
    parser.add_argument("--keyword", nargs="*", default=list(DEFAULT_INDIGO_KEYWORDS))
    parser.add_argument("--preview", type=int, default=10)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    update_indigo_denim_colors(parse_args())
