import argparse
import csv
import os
import shlex
import subprocess
import sys


def read_failed_ids(log_path, reason_contains=""):
    ids = []
    seen = set()
    with open(log_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames and "id" in reader.fieldnames:
            for row in reader:
                if reason_contains and reason_contains not in str(row.get("reason") or ""):
                    continue
                row_id = str(row.get("id") or "").strip()
                if row_id and row_id not in seen:
                    seen.add(row_id)
                    ids.append(row_id)
            return ids

    with open(log_path, encoding="utf-8-sig") as file:
        for line in file:
            row_id = line.split("\t", 1)[0].strip()
            if row_id and row_id.lower() != "id" and row_id not in seen:
                seen.add(row_id)
                ids.append(row_id)
    return ids


def chunked(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def write_ids(ids, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(" ".join(ids))
        file.write("\n")


def build_command(ids, args):
    if args.kind == "color":
        script = "update/update_groundingdino_sam_colors.py"
        extra_args = [
            "--download-workers", str(args.download_workers),
            "--update-batch-size", str(args.update_batch_size),
            "--device", args.device,
            "--sam-checkpoint", args.sam_checkpoint,
            "--sam-model-type", args.sam_model_type,
            "--color-reason-column", "color_reason",
            "--named-color-column", "extracted_named_color",
            "--pre-hint-color-column", "pre_hint_color",
            "--pattern-hint-column", "pattern_hint",
            "--pattern-vit-column", "should_run_pattern_vit",
        ]
    else:
        script = "update/update_fashion_embeddings_from_image_url.py"
        extra_args = [
            "--download-workers", str(args.download_workers),
            "--embedding-batch-size", str(args.embedding_batch_size),
            "--update-batch-size", str(args.update_batch_size),
        ]

    parts = [args.python, script, "--ids", *ids, *extra_args]
    if args.apply:
        parts.append("--apply")
    return parts


def print_command(parts):
    print(" ".join(shlex.quote(part) for part in parts))


def run_command(parts):
    print_command(parts)
    subprocess.run(parts, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Retry update jobs for ids listed in a failed TSV log.")
    parser.add_argument("log_path")
    parser.add_argument("--kind", choices=["color", "embedding"], required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--execute", action="store_true", help="Run retry commands. Without this, commands are printed only.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-ids", type=int, default=0)
    parser.add_argument("--reason-contains", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--update-batch-size", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.log_path):
        raise FileNotFoundError(args.log_path)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")

    ids = read_failed_ids(args.log_path, args.reason_contains)
    if args.max_ids:
        ids = ids[:args.max_ids]
    print(f"Found failed ids: {len(ids)}", file=sys.stderr)
    if args.output:
        write_ids(ids, args.output)
        print(f"Wrote ids to {args.output}", file=sys.stderr)

    for chunk_index, id_chunk in enumerate(chunked(ids, args.chunk_size), start=1):
        print(f"# chunk {chunk_index}: {len(id_chunk)} ids", file=sys.stderr)
        command = build_command(id_chunk, args)
        if args.execute:
            run_command(command)
        else:
            print_command(command)


if __name__ == "__main__":
    main()
