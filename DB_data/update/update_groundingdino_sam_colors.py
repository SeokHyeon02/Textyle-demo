import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fashion_color_utils import IMAGE_COLUMN, download_product_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(DB_DATA_DIR)
TEST_DIR = os.path.join(DB_DATA_DIR, "test")
DEFAULT_IMAGE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "textyle_groundingdino_sam_images")
UPDATE_WORKFLOW_VERSION = "2026-05-20-sub-category-chunks-v1"

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

IMAGE_SEARCH_CANDIDATE_MODE = "image_search"
FULL_DEBUG_CANDIDATE_MODE = "full_debug"

DENIM_TONE_SUB_CATEGORY = "데님팬츠"
DENIM_TONE_EMPTY = ""
DENIM_NAME_TONE_KEYWORDS = {
    "white": {"white", "화이트", "아이보리", "ivory", "에크루", "ecru", "크림", "cream", "베이지", "beige"},
    "light_blue": {
        "lightblue",
        "lightindigo",
        "indigolight",
        "lightwash",
        "라이트블루",
        "라이트인디고",
        "인디고라이트",
        "연청",
        "연한청",
    },
    "mid_blue": {"midblue", "mediumblue", "mediumindigo", "중청"},
    "dark_blue": {"darkblue", "deepblue", "darkwash", "진청", "딥블루", "다크블루"},
    "black": {"black", "블랙", "흑청", "blackdenim"},
    "indigo": {"indigo", "rawdenim", "rawindigo", "인디고", "생지"},
    "gray": {"gray", "grey", "그레이", "차콜", "charcoal"},
    "brown": {"brown", "브라운", "카키", "khaki", "모카", "mocha", "초코", "choco", "카멜", "camel"},
}

PRODUCT_NAME_COLOR_KEYWORDS = [
    ("white", {"white", "화이트", "아이보리", "ivory", "에크루", "ecru", "크림", "cream", "베이지", "beige"}),
    ("black", {"black", "블랙", "흑청", "blackdenim"}),
    ("gray", {"gray", "grey", "그레이", "차콜", "charcoal"}),
    ("blue", {"blue", "블루", "중청", "진청", "연청", "인디고", "indigo", "lightblue", "darkblue"}),
    ("brown", {"brown", "브라운", "카키", "khaki", "모카", "mocha", "초코", "choco", "카멜", "camel"}),
    ("red", {"red", "레드", "버건디", "burgundy", "와인", "wine"}),
    ("green", {"green", "그린", "올리브", "olive"}),
    ("yellow", {"yellow", "옐로우", "노랑", "머스타드", "mustard"}),
    ("purple", {"purple", "퍼플", "보라", "바이올렛", "violet"}),
    ("pink", {"pink", "핑크"}),
    ("orange", {"orange", "오렌지"}),
]


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


def compact_text(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def product_name_tokens(value):
    return {
        token
        for token in re.split(r"[^a-z0-9가-힣]+", str(value or "").lower())
        if token
    }


def should_match_color_keyword(keyword, compact_name, tokens):
    keyword_text = str(keyword or "").lower()
    if not keyword_text:
        return False
    compact_keyword = compact_text(keyword_text)
    if not compact_keyword:
        return False

    is_ascii = all(ord(ch) < 128 for ch in keyword_text)
    if is_ascii and len(compact_keyword) <= 5:
        return keyword_text in tokens or compact_keyword in tokens
    return compact_keyword in compact_name


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def confidence_from_score(score):
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def image_color_candidates(result):
    candidates = []
    for candidate in parse_json_array(result.candidates_json):
        color = candidate.get("color") or candidate.get("fashion_color")
        if not color:
            continue
        score = safe_float(candidate.get("ratio"))
        candidates.append(
            {
                "color": color,
                "score": score,
                "source": "image",
                "confidence": confidence_from_score(score),
                "base_color": color,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def legacy_search_candidates(result):
    search_candidates = parse_json_array(result.search_colors_json)
    if search_candidates:
        return search_candidates[:3]

    return image_color_candidates(result)[:3]


def infer_product_name_color(product_name):
    compact_name = compact_text(product_name)
    tokens = product_name_tokens(product_name)
    for color, keywords in PRODUCT_NAME_COLOR_KEYWORDS:
        if color not in FINAL_COLOR_CATEGORIES:
            continue
        if any(should_match_color_keyword(keyword, compact_name, tokens) for keyword in keywords):
            return color
    return ""


def product_name_color_hint(result):
    inferred_color = infer_product_name_color(result.product_name)
    if inferred_color:
        return inferred_color
    if result.color_hint in FINAL_COLOR_CATEGORIES:
        return result.color_hint
    return ""


def name_fallback_candidate(result):
    color_hint = product_name_color_hint(result)
    if not color_hint:
        return None
    return {
        "color": color_hint,
        "score": 1.0,
        "source": "product_name",
        "confidence": "high",
        "base_color": color_hint,
    }


def result_search_candidates(result, args):
    if args.candidate_mode == FULL_DEBUG_CANDIDATE_MODE:
        return legacy_search_candidates(result)

    image_candidates = image_color_candidates(result)
    name_candidate = name_fallback_candidate(result)
    if name_candidate:
        candidates = [name_candidate]
        candidates.extend(candidate for candidate in image_candidates if candidate["color"] != name_candidate["color"])
        return candidates[:3]

    if image_candidates and confidence_is_allowed(image_candidates[0].get("confidence"), args.min_confidence):
        return image_candidates[:3]

    return image_candidates[:3]


def payload_color_reason(result, candidates, args):
    top = candidates[0] if candidates else None
    hint = product_name_color_hint(result)
    image_candidates = image_color_candidates(result)
    image_top = image_candidates[0] if image_candidates else None

    if args.candidate_mode == FULL_DEBUG_CANDIDATE_MODE:
        return result.color_reason
    if top and top.get("source") == "product_name":
        if image_top and image_top["color"] == top["color"]:
            return "product_name_image_agree"
        if image_top:
            return "product_name_priority"
        return "product_name_only"
    if top and hint and top["color"] == hint:
        return "image_name_agree"
    if top and hint and top["color"] != hint:
        return "image_name_conflict"
    if top:
        return "image_only"
    if args.allow_name_fallback and hint:
        return "low_image_confidence_name_fallback"
    return result.color_reason or "no_image_color_candidate"


def payload_dominant_color(result, candidates, args):
    if args.candidate_mode == FULL_DEBUG_CANDIDATE_MODE:
        return result.extracted_color
    if candidates:
        return candidates[0]["color"]
    color_hint = product_name_color_hint(result)
    if color_hint:
        return color_hint
    return result.extracted_color


def payload_color_confidence(result, candidates, args):
    if args.candidate_mode == FULL_DEBUG_CANDIDATE_MODE:
        return result.color_confidence
    if candidates:
        return candidates[0].get("confidence") or result.color_confidence
    if product_name_color_hint(result):
        return "high"
    return result.color_confidence


def candidate_rgb(result):
    for candidate in parse_json_array(result.candidates_json):
        rgb = candidate.get("rgb")
        if isinstance(rgb, list) and len(rgb) >= 3:
            return tuple(safe_float(value) for value in rgb[:3])
    if result.dominant_rgb:
        return result.dominant_rgb
    return None


def rgb_brightness(rgb):
    if not rgb:
        return None
    r, g, b = rgb[:3]
    return (0.299 * r) + (0.587 * g) + (0.114 * b)


def denim_tone_from_name(product_name):
    compact_name = compact_text(product_name)
    medium_indigo_terms = {
        "mediumindigo",
        "midindigo",
        "mtonindigo",
        "m톤인디고",
        "인디고미듐",
        "미듐인디고",
        "인디고미디엄",
        "미디엄인디고",
    }
    dark_blue_indigo_terms = {
        "darkindigo",
        "indigodark",
        "인디고다크",
        "다크인디고",
    }
    raw_indigo_terms = {
        "deepindigo",
        "rawindigo",
        "rawdenim",
        "onewash",
        "onewashed",
        "딥인디고",
        "생지",
    }
    if any(term in compact_name for term in medium_indigo_terms):
        return "mid_blue"
    if any(term in compact_name for term in dark_blue_indigo_terms):
        return "dark_blue"
    if any(term in compact_name for term in raw_indigo_terms):
        return "indigo"
    for tone, keywords in DENIM_NAME_TONE_KEYWORDS.items():
        if any(keyword in compact_name for keyword in keywords):
            return tone
    return DENIM_TONE_EMPTY


def denim_tone_from_image(color, rgb):
    brightness = rgb_brightness(rgb)
    if color == "white":
        return "white"
    if color == "black":
        return "black"
    if color == "gray":
        return "gray"
    if color == "brown":
        return "brown"
    if color != "blue":
        return DENIM_TONE_EMPTY
    if brightness is None:
        return "mid_blue"
    if brightness >= 170:
        return "light_blue"
    if brightness >= 95:
        return "mid_blue"
    return "dark_blue"


def denim_tone_for_result(result, row, args, candidates):
    if row.get(args.sub_category_column) != DENIM_TONE_SUB_CATEGORY:
        return DENIM_TONE_EMPTY

    image_tone = denim_tone_from_image(
        candidates[0]["color"] if candidates else result.extracted_color,
        candidate_rgb(result),
    )
    name_tone = denim_tone_from_name(result.product_name)
    if name_tone:
        return name_tone
    blue_tones = {"light_blue", "mid_blue", "dark_blue", "indigo"}
    if name_tone == "white" and image_tone in {"white", "gray"}:
        return "white"
    if image_tone in {"light_blue", "mid_blue", "dark_blue"} and name_tone in blue_tones:
        return name_tone
    if image_tone:
        if name_tone == "indigo" and image_tone in {"dark_blue", "mid_blue"}:
            return "indigo"
        return image_tone
    return name_tone


def payload_debug_info(result, row, args, candidates):
    return {
        "candidate_mode": args.candidate_mode,
        "stored_candidates": candidates,
        "legacy_candidates": legacy_search_candidates(result)[:3],
        "image_candidates": image_color_candidates(result)[:3],
        "product_name_color_hint": product_name_color_hint(result),
        "denim_tone": denim_tone_for_result(result, row, args, candidates),
    }


def build_payload(result, args, row):
    candidates = result_search_candidates(result, args)
    payload = {
        args.dominant_color_column: payload_dominant_color(result, candidates, args),
        args.color_confidence_column: payload_color_confidence(result, candidates, args),
        args.color_candidates_column: candidates,
    }
    optional_column_payload(payload, args.color_reason_column, payload_color_reason(result, candidates, args))
    optional_column_payload(payload, args.named_color_column, result.extracted_named_color)
    optional_column_payload(payload, args.pre_hint_color_column, result.pre_hint_color)
    optional_column_payload(payload, args.pattern_hint_column, result.pattern_hint)
    optional_column_payload(payload, args.pattern_vit_column, result.should_run_pattern_vit)
    optional_column_payload(payload, args.color_source_column, "groundingdino_sam")
    optional_column_payload(payload, args.named_candidates_column, parse_json_array(result.named_candidates_json))
    optional_column_payload(payload, args.sam_candidates_column, parse_json_array(result.sam_candidates_json))
    optional_column_payload(payload, args.denim_tone_column, denim_tone_for_result(result, row, args, candidates))
    return payload


def should_include_row_by_denim_tone(row, args):
    if not args.only_missing_denim_tone or not args.denim_tone_column:
        return True
    value = row.get(args.denim_tone_column)
    return value is None or str(value).strip() == ""


def fetch_product_rows(client, args):
    requested_ids = {str(value) for value in args.ids}
    select_columns = [args.id_column, args.name_column, args.image_url_column, args.sub_category_column]
    if args.order_column not in select_columns:
        select_columns.append(args.order_column)
    if args.only_missing_denim_tone and args.denim_tone_column and args.denim_tone_column not in select_columns:
        select_columns.append(args.denim_tone_column)

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
        if args.only_missing_denim_tone and args.denim_tone_column:
            query = query.or_(f"{args.denim_tone_column}.is.null,{args.denim_tone_column}.eq.")

        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        for row in rows:
            row_id = str(row.get(args.id_column) or "")
            if row_id and should_include_row_by_denim_tone(row, args):
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
                entry = log_entry_for_row(row, args, str(exc))
                failed.append(entry)
                append_issue_log(args.failed_log_path, entry)
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
                entry = log_entry_for_row(row, args, str(exc))
                failed.append(entry)
                append_issue_log(args.failed_log_path, entry)
                print(f"[download {index}/{len(product_rows)}] failed id={product_id}: {exc}")
            if completed % 50 == 0 or completed == len(product_rows):
                print(f"Downloaded images: {completed}/{len(product_rows)}")
    return image_paths, failed


def should_update_result(result, args, row):
    if result.status != "ok":
        return False, f"status={result.status}"

    candidates = result_search_candidates(result, args)
    dominant_color = payload_dominant_color(result, candidates, args)
    confidence = payload_color_confidence(result, candidates, args)
    if dominant_color not in FINAL_COLOR_CATEGORIES:
        return False, f"invalid_color={dominant_color}"
    if not confidence_is_allowed(confidence, args.min_confidence):
        if args.allow_name_fallback and name_fallback_candidate(result):
            return True, ""
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


def issue_log_path(kind):
    return os.path.join(BASE_DIR, f"groundingdino_sam_color_update_{kind}.log")


def initialize_issue_log(path):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "image_url", "reason"], delimiter="\t")
        writer.writeheader()


def append_issue_log(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "image_url", "reason"], delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def flush_update_batch(client, update_batch, args, failed):
    if not update_batch:
        return

    for row, payload in update_batch:
        try:
            response = (
                client
                .table(args.table)
                .update(payload)
                .eq(args.id_column, row.get(args.id_column))
                .execute()
            )
            print(f"   batch_update_rows={len(response.data or [])}")
        except Exception as exc:
            entry = log_entry_for_row(row, args, f"batch_update_failed: {exc}")
            failed.append(entry)
            append_issue_log(args.failed_log_path, entry)
            print(f"   batch_update_failed id={row.get(args.id_column)} error={exc}")


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
    print(f"Sub category filter: {', '.join(args.sub_category) if args.sub_category else '-'}")
    print(f"Apply: {args.apply}")
    print(f"Candidate mode: {args.candidate_mode}")
    print(f"Allow name fallback: {args.allow_name_fallback}")
    print(f"Only missing denim tone: {args.only_missing_denim_tone}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Download workers: {args.download_workers}")
    print(f"Update batch size: {args.update_batch_size}")
    print(f"Update workflow version: {UPDATE_WORKFLOW_VERSION}")
    print(f"Verify module: {VERIFY_MODULE_PATH}")
    print(f"Verify workflow version: {getattr(verify_module, 'VERIFY_WORKFLOW_VERSION', 'unknown')}")
    args.skipped_log_path = issue_log_path("skipped")
    args.failed_log_path = issue_log_path("failed")
    initialize_issue_log(args.skipped_log_path)
    initialize_issue_log(args.failed_log_path)
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
            entry = log_entry_for_row(row, args, "id missing")
            skipped.append(entry)
            append_issue_log(args.skipped_log_path, entry)
            print(f"[{index}/{len(product_rows)}] skipped: id missing")
            continue
        if product_id in download_failed_ids:
            continue

        try:
            image_path = image_paths.get(product_id)
            if not image_path:
                raise ValueError("downloaded image path missing")
            result, _image, _mask = verify_image(image_path, models, args, product_names)
            allowed, reason = should_update_result(result, args, row)
            if not allowed:
                entry = log_entry_for_row(row, args, reason)
                skipped.append(entry)
                append_issue_log(args.skipped_log_path, entry)
                print(f"[{index}/{len(product_rows)}] skipped id={product_id}: {reason}")
                continue

            payload = build_payload(result, args, row)
            debug_info = payload_debug_info(result, row, args, payload[args.color_candidates_column])
            print(
                f"[{index}/{len(product_rows)}] id={product_id} "
                f"name={result.product_name} "
                f"color={payload[args.dominant_color_column]} "
                f"confidence={payload[args.color_confidence_column]} "
                f"reason={payload.get(args.color_reason_column) or result.color_reason} "
                f"denim_tone={debug_info['denim_tone'] or '-'}"
            )

            if args.preview and updated < args.preview:
                print(f"   payload={payload}")
                print(f"   color_debug={debug_info}")

            if args.apply:
                update_batch.append((row, payload))
                if len(update_batch) >= args.update_batch_size:
                    flush_update_batch(client, update_batch, args, failed)
                    update_batch = []
            updated += 1
        except Exception as exc:
            entry = log_entry_for_row(row, args, str(exc))
            failed.append(entry)
            append_issue_log(args.failed_log_path, entry)
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
        write_issue_log(args.skipped_log_path, skipped)
        print(f"Skipped log: {args.skipped_log_path}")
    if failed:
        write_issue_log(args.failed_log_path, failed)
        print(f"Failed log: {args.failed_log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate DB color columns with GroundingDINO + SAM.")
    parser.add_argument("--env", default=os.path.join(DB_DATA_DIR, ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", IMAGE_COLUMN))
    parser.add_argument("--sub-category-column", default=os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category"))
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
        choices=[IMAGE_SEARCH_CANDIDATE_MODE, FULL_DEBUG_CANDIDATE_MODE],
        default=IMAGE_SEARCH_CANDIDATE_MODE,
    )
    parser.add_argument("--allow-name-fallback", action="store_true")
    parser.add_argument("--only-missing-denim-tone", action="store_true")
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--sub-category", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    update_groundingdino_sam_colors(parse_args())
