import argparse
import csv
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError


BASE_DIR = Path(__file__).resolve().parent
DB_DATA_DIR = BASE_DIR.parent
PROJECT_ROOT = DB_DATA_DIR.parent
VECTOR_SERVER_DIR = PROJECT_ROOT / "Textyle-vectorserver"
DEFAULT_CSV_PATH = BASE_DIR / "manual_color_labels.csv"
DENIM_SUB_CATEGORY = "\ub370\ub2d8\ud32c\uce20"

FINAL_COLORS = {
    "white",
    "black",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "gray",
    "orange",
    "brown",
    "pink",
}

DENIM_TONES = {
    "light",
    "medium",
    "dark",
    "black",
    "gray",
    "raw",
}

CSV_COLUMNS = [
    "saved_at",
    "id",
    "manual_detail_color",
    "manual_final_color",
    "denim_tone",
    "sub_category",
    "shop_link",
    "name",
]


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual Color Review</title>
  <style>
    :root { font-family: Arial, sans-serif; color: #1f2428; background: #f5f5f2; }
    body { margin: 0; padding: 20px; }
    header { max-width: 1280px; margin: 0 auto 16px; display: flex; gap: 12px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 22px; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .controls label { display: grid; gap: 4px; font-size: 12px; font-weight: 700; color: #59636e; }
    select, input, textarea { box-sizing: border-box; max-width: 100%; border: 1px solid #b9b7ad; border-radius: 6px; padding: 8px; background: #fff; color: #1f2428; font: inherit; }
    #subCategory { width: min(360px, 76vw); }
    button { box-sizing: border-box; border: 1px solid #b9b7ad; border-radius: 6px; padding: 8px 10px; background: #fff; color: #1f2428; font-weight: 700; cursor: pointer; white-space: nowrap; }
    button:hover { background: #eeeee8; }
    .status { max-width: 1280px; margin: 0 auto 12px; color: #59636e; font-size: 13px; }
    .grid { max-width: 1280px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
    .card { background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
    .image-wrap { position: relative; width: 100%; aspect-ratio: 4 / 5; background: #ecebe6; }
    .image-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .badge { position: absolute; top: 8px; left: 8px; padding: 4px 7px; border-radius: 999px; background: rgba(0,0,0,.72); color: #fff; font-size: 12px; font-weight: 700; }
    .name { padding: 9px 10px 4px; font-size: 13px; font-weight: 700; line-height: 1.35; }
    .meta { padding: 0 10px 8px; color: #59636e; font-size: 12px; line-height: 1.4; }
    .form { padding: 0 10px 12px; display: grid; gap: 8px; min-width: 0; }
    .colors { display: grid; grid-template-columns: repeat(auto-fit, minmax(72px, 1fr)); gap: 6px; }
    .color-btn { min-width: 0; padding: 8px 6px; font-size: 13px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; }
    .color-btn.active { outline: 3px solid #2f6feb; }
    .row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(120px, .45fr); gap: 8px; min-width: 0; }
    .row.actions { grid-template-columns: 1fr 1fr; }
    .denim-row[hidden] { display: none; }
    .pager { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .pager input { width: 74px; }
    .pager button:disabled { opacity: .45; cursor: not-allowed; }
    .save { background: #1f883d; border-color: #1f883d; color: white; }
    .open { box-sizing: border-box; text-align: center; text-decoration: none; border: 1px solid #b9b7ad; border-radius: 6px; padding: 8px 10px; color: #1f2428; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .saved { opacity: .42; }
    @media (max-width: 520px) {
      body { padding: 12px; }
      .grid { grid-template-columns: 1fr; }
      .row, .row.actions { grid-template-columns: 1fr; }
      .colors { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Manual Color Review</h1>
    <div class="controls">
      <label>Show only sub_category <select id="subCategory"></select></label>
      <label>Start ID <input id="startId" type="number" min="0" style="width:90px" /></label>
      <label>Limit <input id="limit" type="number" min="1" value="50" style="width:70px" /></label>
      <button id="loadBtn">Load</button>
      <div class="pager">
        <button id="prevPage" type="button">Prev</button>
        <span id="pageInfo">Page 1 / 1</span>
        <button id="nextPage" type="button">Next</button>
      </div>
    </div>
  </header>
  <div class="status" id="status">Loading...</div>
  <main class="grid" id="grid"></main>
  <script>
    const COLORS = ["black","white","gray","blue","red","yellow","green","purple","orange","brown","pink"];
    const DENIM_SUB_CATEGORY = "\ub370\ub2d8\ud32c\uce20";
    const DENIM_TONES = ["light","medium","dark","black","gray","raw"];
    const MUSINSA_COLOR_LABELS = [
      "블랙","라이트 그레이","그레이","다크 그레이","네이비","아이보리","화이트","오트밀","블루","다크네이비",
      "베이지","레드","카키","다크 그린","버건디","브라운","다크 브라운","그린","퍼플","스카이 블루","다크 블루",
      "올리브","오렌지","민트","핑크","피치","라이트 핑크","샌드","라이트 그린","옐로우","라벤더","다크핑크",
      "머스타드","다크 베이지","딥레드","라이트 옐로우","라이트 브라운","페일 핑크","다크 오렌지","브릭",
      "라임","카멜","실버","라이트 오렌지","골드","인디고","연청","흑청","중청","진청"
    ];
    const INITIAL_SUB_CATEGORY = "__INITIAL_SUB_CATEGORY__";
    const grid = document.getElementById("grid");
    const statusEl = document.getElementById("status");
    const subSelect = document.getElementById("subCategory");
    const startId = document.getElementById("startId");
    const limit = document.getElementById("limit");
    const pageInfo = document.getElementById("pageInfo");
    const prevPage = document.getElementById("prevPage");
    const nextPage = document.getElementById("nextPage");
    let currentPage = 1;
    let totalPages = 1;

    function qs(params) {
      return Object.entries(params)
        .filter(([, value]) => value !== undefined && value !== null && String(value) !== "")
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join("&");
    }

    async function loadCategories() {
      const res = await fetch("/api/sub-categories");
      const data = await res.json();
      subSelect.innerHTML = `<option value="">All</option>` + data.sub_categories.map(
        value => `<option value="${String(value).replaceAll('"', '&quot;')}">${value}</option>`
      ).join("");
      if (INITIAL_SUB_CATEGORY) {
        subSelect.value = INITIAL_SUB_CATEGORY;
      }
    }

    function card(row) {
      const el = document.createElement("section");
      el.className = "card";
      el.dataset.id = row.id;
      let selectedFinalColor = row.manual_final_color || "";
      const isDenim = row.sub_category === DENIM_SUB_CATEGORY;
      const currentDetailColor = row.manual_detail_color || row.extracted_named_color || row.musinsa_color_label || "";
      const detailColorOptions = (currentDetailColor && !MUSINSA_COLOR_LABELS.includes(currentDetailColor))
        ? [currentDetailColor, ...MUSINSA_COLOR_LABELS]
        : MUSINSA_COLOR_LABELS;
      el.innerHTML = `
        <div class="image-wrap">
          <span class="badge">#${row.id}</span>
          <img src="${row.image_url || ""}" alt="" loading="lazy" />
        </div>
        <div class="name">${row.name || "unnamed"}</div>
        <div class="meta">
          sub_category: ${row.sub_category || ""}<br />
          current: ${row.dominant_color || ""} / ${row.extracted_named_color || ""}${row.manual_detail_color ? `<br />manual detail: ${row.manual_detail_color}` : ""}${isDenim ? `<br />denim_tone: ${row.denim_tone || ""}` : ""}
        </div>
        <div class="form">
          <div class="colors"></div>
          <div class="row">
            <select class="detail-color">
              <option value="">detail color</option>
              ${detailColorOptions.map(label => `<option value="${label}" ${currentDetailColor === label ? "selected" : ""}>${label}</option>`).join("")}
            </select>
          </div>
          <div class="denim-row" ${isDenim ? "" : "hidden"}>
            <select class="denim-tone">
              <option value="">denim_tone</option>
              ${DENIM_TONES.map(tone => `<option value="${tone}" ${row.denim_tone === tone ? "selected" : ""}>${tone}</option>`).join("")}
            </select>
          </div>
          <div class="row actions">
            <a class="open" href="${row.shop_link || "#"}" target="_blank">Open product</a>
            <button class="save">Save CSV</button>
          </div>
        </div>
      `;
      const colorWrap = el.querySelector(".colors");
      function setSelectedColor(color) {
        selectedFinalColor = color;
        colorWrap.querySelectorAll(".color-btn").forEach(item => {
          item.classList.toggle("active", item.dataset.color === color);
        });
      }
      COLORS.forEach(color => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "color-btn";
        btn.dataset.color = color;
        btn.textContent = color;
        btn.addEventListener("click", () => setSelectedColor(color));
        colorWrap.appendChild(btn);
      });
      if (selectedFinalColor) {
        setSelectedColor(selectedFinalColor);
      }
      el.querySelector(".save").addEventListener("click", async () => {
        if (!selectedFinalColor) {
          alert("Select manual_final_color first.");
          return;
        }
        const detailColor = el.querySelector(".detail-color").value.trim();
        if (!detailColor) {
          alert("Select manual_detail_color first.");
          return;
        }
        const payload = {
          id: row.id,
          manual_detail_color: detailColor,
          manual_final_color: selectedFinalColor,
          denim_tone: isDenim ? el.querySelector(".denim-tone").value : "",
          sub_category: row.sub_category || "",
          shop_link: row.shop_link || "",
          name: row.name || "",
        };
        const res = await fetch("/api/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) {
          alert(data.error || "save failed");
          return;
        }
        el.classList.add("saved");
        statusEl.textContent = `Saved id=${row.id} to ${data.csv_path}`;
      });
      return el;
    }

    async function loadRows(page = 1) {
      currentPage = Math.max(1, page);
      grid.innerHTML = "";
      statusEl.textContent = "Loading rows...";
      const selectedCategory = subSelect.value;
      const params = qs({sub_category: selectedCategory, start_id: startId.value, limit: limit.value, page: currentPage});
      const res = await fetch(`/api/rows?${params}`);
      const data = await res.json();
      if (!res.ok) {
        statusEl.textContent = data.error || "load failed";
        return;
      }
      data.rows.forEach(row => grid.appendChild(card(row)));
      const filterText = selectedCategory ? `sub_category="${selectedCategory}"` : "all sub_categories";
      totalPages = Math.max(1, data.total_pages || 1);
      currentPage = Math.min(data.page || currentPage, totalPages);
      pageInfo.textContent = `Page ${currentPage} / ${totalPages}`;
      prevPage.disabled = currentPage <= 1;
      nextPage.disabled = currentPage >= totalPages;
      statusEl.textContent = `Loaded ${data.rows.length} rows for ${filterText}. Total ${data.total_count || 0}. CSV: ${data.csv_path}`;
    }

    document.getElementById("loadBtn").addEventListener("click", () => loadRows(1));
    subSelect.addEventListener("change", () => loadRows(1));
    limit.addEventListener("change", () => loadRows(1));
    startId.addEventListener("change", () => loadRows(1));
    prevPage.addEventListener("click", () => loadRows(currentPage - 1));
    nextPage.addEventListener("click", () => loadRows(currentPage + 1));
    loadCategories().then(loadRows).catch(error => { statusEl.textContent = String(error); });
  </script>
</body>
</html>
"""


def load_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def require_config(args):
    load_env(Path(args.env))
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing.")
    return url.rstrip("/"), key


def rest_request_raw(args, method: str, path: str, payload=None, extra_headers: dict | None = None):
    url, key = require_config(args)
    data = None
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"
    req = Request(f"{url}/rest/v1/{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            response_headers = response.headers
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase HTTP {exc.code} for {method} {path}: {error_body}") from exc
    return json.loads(body) if body else [], response_headers


def rest_request(args, method: str, path: str, payload=None):
    body, _headers = rest_request_raw(args, method, path, payload=payload)
    return body


def validate_color(value: str, field_name: str, required: bool = False) -> str:
    color = (value or "").strip().lower()
    if not color:
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    if color not in FINAL_COLORS:
        raise ValueError(f"{field_name} must be one of {sorted(FINAL_COLORS)}: {color}")
    return color


def validate_denim_tone(value: str, required: bool = False) -> str:
    tone = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not tone:
        if required:
            raise ValueError("denim_tone is required")
        return ""
    aliases = {
        "lightblue": "light",
        "light_blue": "light",
        "mediumblue": "medium",
        "medium_blue": "medium",
        "midblue": "medium",
        "mid_blue": "medium",
        "darkblue": "dark",
        "dark_blue": "dark",
        "indigo": "raw",
        "raw_denim": "raw",
        "one_wash": "raw",
        "bluegray": "gray",
        "blue_gray": "gray",
        "grayblue": "gray",
        "gray_blue": "gray",
        "washedgray": "gray",
        "washed_gray": "gray",
        "washedblack": "black",
        "washed_black": "black",
    }
    tone = aliases.get(tone, tone)
    if tone not in DENIM_TONES:
        raise ValueError(f"denim_tone must be one of {sorted(DENIM_TONES)}: {tone}")
    return tone


def csv_path(args) -> Path:
    return Path(args.csv).resolve()


def append_csv_row(args, row: dict):
    path = csv_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_row = dict(row)
    if not normalized_row.get("manual_final_color"):
        normalized_row["manual_final_color"] = normalized_row.get("manual_color", "")
    if not normalized_row.get("manual_detail_color"):
        normalized_row["manual_detail_color"] = (
            normalized_row.get("manual_named_color")
            or normalized_row.get("musinsa_color_label")
            or ""
        )
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != CSV_COLUMNS:
                existing_rows = list(reader)
                with path.open("w", encoding="utf-8-sig", newline="") as rewrite_file:
                    writer = csv.DictWriter(rewrite_file, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        if not existing_row.get("manual_final_color"):
                            existing_row["manual_final_color"] = existing_row.get("manual_color", "")
                        if not existing_row.get("manual_detail_color"):
                            existing_row["manual_detail_color"] = (
                                existing_row.get("manual_named_color")
                                or existing_row.get("musinsa_color_label")
                                or ""
                            )
                        writer.writerow({column: existing_row.get(column, "") for column in CSV_COLUMNS})
    file_exists = path.exists() and path.stat().st_size > 0
    output = {column: normalized_row.get(column, "") for column in CSV_COLUMNS}
    output["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(output)
    return path


def count_rows(args, sub_category: str = "", start_id: str = "") -> int:
    params = {
        "select": args.id_column,
    }
    if start_id:
        params[f"{args.id_column}"] = f"gte.{start_id}"
    if sub_category:
        params[f"{args.sub_category_column}"] = f"eq.{sub_category}"
    _body, headers = rest_request_raw(
        args,
        "GET",
        f"{quote(args.table)}?{urlencode(params)}",
        extra_headers={
            "Prefer": "count=exact",
            "Range-Unit": "items",
            "Range": "0-0",
        },
    )
    content_range = headers.get("Content-Range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    return 0


def fetch_rows(args, sub_category: str = "", start_id: str = "", limit: int | None = None, page: int = 1):
    page_size = max(1, limit or args.limit)
    current_page = max(1, page)
    columns = [
        args.id_column,
        args.name_column,
        args.image_url_column,
        args.shop_link_column,
        args.main_category_column,
        args.sub_category_column,
        args.dominant_color_column,
        args.color_confidence_column,
        args.named_color_column,
        args.denim_tone_column,
    ]
    if args.musinsa_color_label_column:
        columns.append(args.musinsa_color_label_column)
    if args.manual_detail_color_column:
        columns.append(args.manual_detail_color_column)
    if args.manual_final_color_column:
        columns.append(args.manual_final_color_column)
    params = {
        "select": ",".join(dict.fromkeys(columns)),
        "order": f"{args.order_column}.asc",
        "limit": str(page_size),
        "offset": str((current_page - 1) * page_size),
    }
    if start_id:
        params[f"{args.id_column}"] = f"gte.{start_id}"
    if sub_category:
        params[f"{args.sub_category_column}"] = f"eq.{sub_category}"
    path = f"{quote(args.table)}?{urlencode(params)}"
    rows = rest_request(args, "GET", path)
    if args.musinsa_color_label_column:
        for row in rows:
            row["musinsa_color_label"] = row.get(args.musinsa_color_label_column, "")
    if args.manual_detail_color_column:
        for row in rows:
            row["manual_detail_color"] = row.get(args.manual_detail_color_column, "")
    if args.manual_final_color_column:
        for row in rows:
            row["manual_final_color"] = row.get(args.manual_final_color_column, "")
    return rows


def fetch_sub_categories(args):
    values = set()
    scanned = 0
    last_order_value = None
    page_size = min(args.page_size, args.category_scan_limit)
    select_columns = ",".join(dict.fromkeys([args.order_column, args.sub_category_column]))
    while scanned < args.category_scan_limit:
        limit = min(page_size, args.category_scan_limit - scanned)
        params = {
            "select": select_columns,
            "order": f"{args.order_column}.asc",
            "limit": str(limit),
        }
        if last_order_value is not None:
            params[args.order_column] = f"gt.{last_order_value}"
        rows = rest_request(args, "GET", f"{quote(args.table)}?{urlencode(params)}")
        if not rows:
            break
        scanned += len(rows)
        values.update(row.get(args.sub_category_column) for row in rows if row.get(args.sub_category_column))
        if len(rows) < limit:
            break
        next_order_value = rows[-1].get(args.order_column)
        if next_order_value is None or next_order_value == last_order_value:
            break
        last_order_value = next_order_value
    return sorted(values)


def fetch_row_by_id(args, row_id: str):
    columns = [
        args.id_column,
        args.name_column,
        args.sub_category_column,
        args.dominant_color_column,
        args.color_confidence_column,
        args.color_candidates_column,
        args.color_reason_column,
        args.named_color_column,
        args.denim_tone_column,
    ]
    if args.musinsa_color_label_column:
        columns.append(args.musinsa_color_label_column)
    if args.manual_detail_color_column:
        columns.append(args.manual_detail_color_column)
    if args.manual_final_color_column:
        columns.append(args.manual_final_color_column)
    params = {
        "select": ",".join(dict.fromkeys(columns)),
        args.id_column: f"eq.{row_id}",
        "limit": "1",
    }
    rows = rest_request(args, "GET", f"{quote(args.table)}?{urlencode(params)}")
    return rows[0] if rows else None


def build_payload(csv_row: dict, args, db_sub_category: str = ""):
    manual_final_color = validate_color(
        csv_row.get("manual_final_color") or csv_row.get("manual_color", ""),
        "manual_final_color",
        required=True,
    )
    manual_detail_color = (
        csv_row.get("manual_detail_color")
        or csv_row.get("manual_named_color")
        or csv_row.get("musinsa_color_label")
        or ""
    ).strip()
    csv_sub_category = csv_row.get("sub_category") or ""
    is_denim_row = db_sub_category == DENIM_SUB_CATEGORY or csv_sub_category == DENIM_SUB_CATEGORY
    denim_tone = validate_denim_tone(
        csv_row.get("denim_tone", ""),
    )
    candidates = [{
        "color": manual_final_color,
        "score": 1.0,
        "source": "manual",
        "confidence": "high",
    }]
    if manual_detail_color:
        candidates[0]["named_color"] = manual_detail_color
    payload = {
        args.dominant_color_column: manual_final_color,
        args.color_confidence_column: "high",
        args.color_candidates_column: candidates,
        args.color_reason_column: "manual_review",
    }
    if manual_detail_color:
        payload[args.named_color_column] = manual_detail_color
    if denim_tone and is_denim_row:
        payload[args.denim_tone_column] = denim_tone
    if getattr(args, "manual_detail_color_column", ""):
        payload[args.manual_detail_color_column] = manual_detail_color
    if getattr(args, "manual_final_color_column", ""):
        payload[args.manual_final_color_column] = manual_final_color
    return payload


def read_latest_csv_rows(args):
    path = csv_path(args)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    latest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for line_no, row in enumerate(reader, 2):
            row_id = (row.get("id") or "").strip()
            if not row_id:
                print(f"[skip] line={line_no} id missing")
                continue
            latest[row_id] = row
    return latest


def update_from_csv(args):
    rows_by_id = read_latest_csv_rows(args)
    allowed_sub_categories = set(args.sub_category or [])
    print(f"CSV rows after de-duplication: {len(rows_by_id)}")
    print(f"Apply updates: {args.apply}")
    if allowed_sub_categories:
        print(f"Sub-category filter: {sorted(allowed_sub_categories)}")

    updated = 0
    skipped = 0
    failed = 0
    for row_id, csv_row in rows_by_id.items():
        try:
            db_row = fetch_row_by_id(args, row_id)
            if not db_row:
                skipped += 1
                print(f"[skip] id={row_id} not found")
                continue
            db_sub = db_row.get(args.sub_category_column) or ""
            csv_sub = csv_row.get("sub_category") or ""
            if allowed_sub_categories and db_sub not in allowed_sub_categories and csv_sub not in allowed_sub_categories:
                skipped += 1
                print(f"[skip] id={row_id} sub_category={db_sub or csv_sub}")
                continue
            payload = build_payload(csv_row, args, db_sub_category=db_sub)
            print(
                f"[{'apply' if args.apply else 'dry-run'}] "
                f"id={row_id} sub_category={db_sub} payload={json.dumps(payload, ensure_ascii=False)}"
            )
            if args.apply:
                params = urlencode({args.id_column: f"eq.{row_id}"})
                rest_request(args, "PATCH", f"{quote(args.table)}?{params}", payload=payload)
            updated += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] id={row_id} {exc}")
    print(f"Done. updated_or_planned={updated}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


class ReviewHandler(BaseHTTPRequestHandler):
    args = None

    def log_message(self, fmt, *values):
        print(f"[server] {self.address_string()} {fmt % values}")

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                initial_sub_category = (self.args.sub_category or [""])[0] if self.args.sub_category else ""
                page = HTML.replace("__INITIAL_SUB_CATEGORY__", initial_sub_category.replace("\\", "\\\\").replace('"', '\\"'))
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/sub-categories":
                self.send_json(200, {"sub_categories": fetch_sub_categories(self.args)})
                return
            if parsed.path == "/api/rows":
                qs = parse_qs(parsed.query)
                sub_category = (qs.get("sub_category") or [""])[0]
                start_id = (qs.get("start_id") or [""])[0]
                limit = max(1, int((qs.get("limit") or [self.args.limit])[0]))
                page_number = max(1, int((qs.get("page") or ["1"])[0]))
                total_count = count_rows(self.args, sub_category=sub_category, start_id=start_id)
                total_pages = max(1, (total_count + limit - 1) // limit)
                if page_number > total_pages:
                    page_number = total_pages
                rows = fetch_rows(
                    self.args,
                    sub_category=sub_category,
                    start_id=start_id,
                    limit=limit,
                    page=page_number,
                )
                self.send_json(
                    200,
                    {
                        "rows": rows,
                        "csv_path": str(csv_path(self.args)),
                        "total_count": total_count,
                        "total_pages": total_pages,
                        "page": page_number,
                        "limit": limit,
                    },
                )
                return
            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            payload["manual_final_color"] = validate_color(
                payload.get("manual_final_color", ""),
                "manual_final_color",
                required=True,
            )
            payload["manual_detail_color"] = (payload.get("manual_detail_color") or "").strip()
            if not payload["manual_detail_color"]:
                raise ValueError("manual_detail_color is required")
            if payload.get("sub_category") == DENIM_SUB_CATEGORY:
                payload["denim_tone"] = validate_denim_tone(payload.get("denim_tone", ""))
            else:
                payload["denim_tone"] = ""
            if not str(payload.get("id") or "").strip():
                raise ValueError("id is required")
            path = append_csv_row(self.args, payload)
            self.send_json(200, {"saved": True, "csv_path": str(path)})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def serve(args):
    require_config(args)
    ReviewHandler.args = args
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Manual color review server: {url}")
    print(f"CSV path: {csv_path(args)}")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review product colors manually, save CSV labels, and update Supabase color columns."
    )
    parser.add_argument("--env", default=str(DB_DATA_DIR / ".env"))
    parser.add_argument("--table", default=os.environ.get("IMAGE_VIEWER_TABLE", "clothes"))
    parser.add_argument("--id-column", default=os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id"))
    parser.add_argument("--name-column", default=os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name"))
    parser.add_argument("--image-url-column", default=os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", "image_url"))
    parser.add_argument("--shop-link-column", default=os.environ.get("IMAGE_VIEWER_SHOP_LINK_COLUMN", "shop_link"))
    parser.add_argument("--main-category-column", default=os.environ.get("IMAGE_VIEWER_MAIN_CATEGORY_COLUMN", "main_category"))
    parser.add_argument("--sub-category-column", default=os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category"))
    parser.add_argument("--order-column", default=os.environ.get("FASHION_COLOR_ORDER_COLUMN", "id"))
    parser.add_argument("--dominant-color-column", default="dominant_color")
    parser.add_argument("--color-confidence-column", default="color_confidence")
    parser.add_argument("--color-candidates-column", default="color_candidates")
    parser.add_argument("--color-reason-column", default="color_reason")
    parser.add_argument("--named-color-column", default="extracted_named_color")
    parser.add_argument("--denim-tone-column", default="denim_tone")
    parser.add_argument("--musinsa-color-label-column", default="")
    parser.add_argument("--manual-detail-color-column", default="")
    parser.add_argument("--manual-final-color-column", default="")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--category-scan-limit", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--secondary-score", type=float, default=0.55)
    parser.add_argument("--sub-category", nargs="*", default=[])
    parser.add_argument("--update-from-csv", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_from_csv:
        update_from_csv(args)
    else:
        serve(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
