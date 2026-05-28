"""
Colab runner for updating indigo-related denim color columns.

Expected location:
    DB_data/update/colab_update_indigo_denim_colors.py

Typical Colab usage:
    %cd /content/Textyle-demo/DB_data/update
    !python colab_update_indigo_denim_colors.py --install-deps --download-sam --limit 20
    !python colab_update_indigo_denim_colors.py --limit 20 --apply

Required secrets/environment:
    SUPABASE_URL
    SUPABASE_KEY

The script is dry-run by default. Supabase is updated only when --apply is passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_DATA_DIR = BASE_DIR.parent
ROOT_DIR = DB_DATA_DIR.parent
DEFAULT_SAM_CHECKPOINT = Path("/content/sam_vit_b_01ec64.pth")
SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.check_call(command)


def install_dependencies() -> None:
    packages = [
        "python-dotenv",
        "supabase",
        "pillow",
        "requests",
        "numpy",
        "transformers",
        "segment-anything",
    ]
    run_command([sys.executable, "-m", "pip", "install", "-q", *packages])


def download_sam_checkpoint(path: Path) -> None:
    if path.exists():
        print(f"SAM checkpoint already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run_command(["wget", "-O", str(path), SAM_VIT_B_URL])


def load_colab_secrets() -> None:
    try:
        from google.colab import userdata
    except Exception:
        return

    for key in ("SUPABASE_URL", "SUPABASE_KEY"):
        if os.environ.get(key):
            continue
        try:
            value = userdata.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = value


def require_project_files() -> None:
    required = [
        BASE_DIR / "update_indigo_denim_colors.py",
        BASE_DIR / "update_groundingdino_sam_colors.py",
        BASE_DIR / "fashion_color_utils.py",
        DB_DATA_DIR / "test" / "verify_groundingdino_sam_color_extraction.py",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required project files:\n{joined}")


def require_environment() -> None:
    missing = [key for key in ("SUPABASE_URL", "SUPABASE_KEY") if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Set them in Colab secrets or os.environ before running."
        )


def build_update_argv(args: argparse.Namespace) -> list[str]:
    update_args = [
        "--env",
        str(args.env),
        "--table",
        args.table,
        "--id-column",
        args.id_column,
        "--name-column",
        args.name_column,
        "--image-url-column",
        args.image_url_column,
        "--sub-category-column",
        args.sub_category_column,
        "--order-column",
        args.order_column,
        "--page-size",
        str(args.page_size),
        "--scan-limit",
        str(args.scan_limit),
        "--limit",
        str(args.limit),
        "--image-cache-dir",
        str(args.image_cache_dir),
        "--download-workers",
        str(args.download_workers),
        "--update-batch-size",
        str(args.update_batch_size),
        "--sam-checkpoint",
        str(args.sam_checkpoint),
        "--sam-model-type",
        args.sam_model_type,
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--device",
        args.device,
        "--min-confidence",
        args.min_confidence,
        "--preview",
        str(args.preview),
        "--sub-category",
        *args.sub_category,
    ]
    if args.ids:
        update_args.extend(["--ids", *args.ids])
    if args.start_id is not None:
        update_args.extend(["--start-id", str(args.start_id)])
    if args.refresh_downloaded_images:
        update_args.append("--refresh-downloaded-images")
    if args.allow_name_fallback:
        update_args.append("--allow-name-fallback")
    if args.apply:
        update_args.append("--apply")
    return update_args


def run_update(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import update_indigo_denim_colors

    update_argv = build_update_argv(args)
    print("Running indigo denim update")
    print(f"Project root: {ROOT_DIR}")
    print(f"Apply: {args.apply}")
    print(f"Limit: {args.limit}")
    print(f"SAM checkpoint: {args.sam_checkpoint}")
    update_indigo_denim_colors.update_indigo_denim_colors(
        update_indigo_denim_colors.parse_args(update_argv)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab-safe runner for indigo denim GroundingDINO/SAM DB updates."
    )
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--download-sam", action="store_true")
    parser.add_argument("--env", default=str(DB_DATA_DIR / ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", "image_url"))
    parser.add_argument("--sub-category-column", default=os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category"))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--sub-category", nargs="*", default=["\ub370\ub2d8\ud32c\uce20"])
    parser.add_argument("--image-cache-dir", default="/content/textyle_indigo_denim_images")
    parser.add_argument("--refresh-downloaded-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--update-batch-size", type=int, default=25)
    parser.add_argument("--sam-checkpoint", default=os.environ.get("SAM_CHECKPOINT", str(DEFAULT_SAM_CHECKPOINT)))
    parser.add_argument("--sam-model-type", default=os.environ.get("SAM_MODEL_TYPE", "vit_b"))
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default=os.environ.get("DINO_SAM_DEVICE", "cuda"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--preview", type=int, default=10)
    parser.add_argument("--allow-name-fallback", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_project_files()
    if args.install_deps:
        install_dependencies()
    if args.download_sam:
        download_sam_checkpoint(Path(args.sam_checkpoint))
    load_colab_secrets()
    require_environment()
    run_update(args)


if __name__ == "__main__":
    main()
