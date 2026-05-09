import csv
import json
import os
import sys
import threading
import time
import webbrowser
from math import ceil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from supabase import Client, create_client


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

TABLE_NAME = os.environ.get("IMAGE_VIEWER_TABLE", "clothes")
ID_COLUMN = os.environ.get("IMAGE_VIEWER_ID_COLUMN", "id")
NAME_COLUMN = os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name")
IMAGE_COLUMN = os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", "image_url")
SHOP_LINK_COLUMN = os.environ.get("IMAGE_VIEWER_SHOP_LINK_COLUMN", "shop_link")
PAGE_SIZE = int(os.environ.get("IMAGE_VIEWER_PAGE_SIZE", "50"))
HOST = os.environ.get("IMAGE_VIEWER_HOST", "127.0.0.1")
PORT = int(os.environ.get("IMAGE_VIEWER_PORT", "8765"))
CSV_PATH = os.environ.get(
    "IMAGE_VIEWER_CSV_PATH",
    os.path.join(BASE_DIR, "selected_shop_links.csv"),
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(".env 파일에서 SUPABASE_URL 또는 SUPABASE_KEY를 찾을 수 없습니다.")
    sys.exit(1)


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DB Image URL Viewer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, "Malgun Gothic", sans-serif;
      background: #f5f5f2;
      color: #1f2428;
    }
    body {
      margin: 0;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin: 0 auto 18px;
      max-width: 1200px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .id-search {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .id-search input {
      width: 120px;
      padding: 8px 10px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      background: #fff;
      color: #1f2428;
      font-size: 13px;
    }
    .status {
      color: #59636e;
      font-size: 14px;
      text-align: right;
      min-width: 260px;
    }
    .page-controls {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .page-indicator {
      display: flex;
      align-items: center;
      gap: 4px;
      min-width: 84px;
      color: #1f2428;
      font-size: 13px;
      font-weight: 700;
      text-align: center;
    }
    .page-indicator input {
      width: 42px;
      padding: 7px 6px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      background: #fff;
      color: #1f2428;
      font-size: 13px;
      font-weight: 700;
      text-align: center;
    }
    .page-button {
      padding: 8px 12px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      background: #fff;
      color: #1f2428;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    .id-search button {
      padding: 8px 12px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      background: #fff;
      color: #1f2428;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    .id-search button:hover {
      background: #eeeee8;
    }
    .page-button:hover {
      background: #eeeee8;
    }
    .page-button:disabled {
      color: #8c959f;
      cursor: not-allowed;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 14px;
      max-width: 1200px;
      margin: 0 auto;
    }
    .card {
      position: relative;
      display: flex;
      flex-direction: column;
      min-height: 230px;
      overflow: hidden;
      border: 1px solid #d8d7d0;
      border-radius: 8px;
      background: #fff;
      cursor: pointer;
    }
    .card:focus {
      outline: 3px solid #2f6feb;
      outline-offset: 2px;
    }
    .image-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 5;
      background: #ecebe6;
    }
    .image-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .badge {
      position: absolute;
      top: 8px;
      left: 8px;
      padding: 4px 7px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      z-index: 2;
    }
    .product-name {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 8px 9px;
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
      text-align: left;
      overflow-wrap: anywhere;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .meta {
      padding: 9px 10px 10px;
      font-size: 12px;
      color: #4d5560;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }
    .button-row {
      display: flex;
      gap: 8px;
      margin: 0 10px 10px;
    }
    .save-button,
    .delete-button {
      flex: 1 1 0;
      padding: 8px 10px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    .save-button {
      background: #f7f7f4;
      color: #1f2428;
    }
    .delete-button {
      background: #fff1f1;
      border-color: #d9a8a8;
      color: #8f1d1d;
    }
    .save-button:hover {
      background: #eeeee8;
    }
    .delete-button:hover {
      background: #fde5e5;
    }
    .save-button:focus,
    .delete-button:focus {
      outline: 3px solid #2f6feb;
      outline-offset: 2px;
    }
    .save-button:disabled,
    .delete-button:disabled {
      color: #8c959f;
      cursor: not-allowed;
    }
    .empty {
      display: none;
      max-width: 1200px;
      margin: 36px auto 0;
      color: #59636e;
      text-align: center;
      font-size: 15px;
    }
  </style>
</head>
<body>
  <header>
    <h1>DB Image URL Viewer</h1>
    <div class="header-actions">
      <form class="id-search" id="idSearchForm">
        <input id="idSearchInput" type="text" inputmode="numeric" placeholder="id 입력" aria-label="id 입력" />
        <button type="submit">이미지 열기</button>
      </form>
      <div class="page-controls">
        <form class="page-indicator" id="pageJumpForm">
          <input id="pageInput" type="number" min="1" value="1" aria-label="이동할 페이지" />
          <span>/</span>
          <span id="totalPagesText">-</span>
        </form>
        <button class="page-button" id="skipButton" type="button">50개 건너뛰기</button>
      </div>
      <div class="status" id="status">불러오는 중...</div>
    </div>
  </header>
  <main class="grid" id="grid"></main>
  <p class="empty" id="empty">더 이상 표시할 image_url이 없습니다.</p>

  <script>
    const grid = document.getElementById("grid");
    const statusEl = document.getElementById("status");
    const emptyEl = document.getElementById("empty");
    const skipButton = document.getElementById("skipButton");
    const pageJumpForm = document.getElementById("pageJumpForm");
    const pageInput = document.getElementById("pageInput");
    const totalPagesText = document.getElementById("totalPagesText");
    const idSearchForm = document.getElementById("idSearchForm");
    const idSearchInput = document.getElementById("idSearchInput");

    let currentPage = 0;
    let totalPages = null;
    let pageSize = 50;
    let loading = false;
    let currentCount = 0;

    skipButton.addEventListener("click", () => {
      if (loading || totalPages === null || currentPage + 1 >= totalPages) return;
      loadPage(currentPage + 1);
    });

    pageJumpForm.addEventListener("submit", (event) => {
      event.preventDefault();
      if (loading || totalPages === null || totalPages === 0) return;

      const requestedPage = Number.parseInt(pageInput.value, 10);
      if (!Number.isFinite(requestedPage)) {
        pageInput.value = String(currentPage + 1);
        return;
      }

      const page = Math.min(Math.max(requestedPage, 1), totalPages);
      loadPage(page - 1);
    });

    idSearchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const id = idSearchInput.value.trim();
      if (!id) return;
      window.open(`/image?id=${encodeURIComponent(id)}`, "_blank", "width=520,height=760");
    });

    function updateStatus(message) {
      if (totalPages === null) {
        pageInput.value = "1";
        totalPagesText.textContent = "-";
      } else if (totalPages === 0) {
        pageInput.value = "0";
        totalPagesText.textContent = "0";
      } else {
        pageInput.value = String(currentPage + 1);
        totalPagesText.textContent = String(totalPages);
      }

      statusEl.textContent = `${message} / 현재 ${currentCount}개`;
      updatePageButtons();
    }

    function updatePageButtons() {
      skipButton.disabled =
        loading || totalPages === null || totalPages === 0 || currentPage + 1 >= totalPages;
      pageInput.disabled = loading || totalPages === null || totalPages === 0;
    }

    async function loadPage(page) {
      if (loading) return;
      loading = true;
      currentPage = Math.max(page, 0);
      grid.innerHTML = "";
      currentCount = 0;
      updateStatus("불러오는 중...");

      try {
        const offset = currentPage * pageSize;
        const response = await fetch(`/api/images?offset=${offset}`);
        if (!response.ok) {
          throw new Error(await response.text());
        }

        const payload = await response.json();
        currentPage = payload.page_index;
        totalPages = payload.total_pages;
        pageSize = payload.page_size;
        currentCount = payload.images.length;

        if (payload.images.length === 0) {
          currentCount = 0;
          updateStatus("표시할 이미지 없음");
          emptyEl.style.display = "block";
          return;
        }

        renderImages(payload.images);
        updateStatus("묶음 표시");
      } catch (error) {
        statusEl.textContent = "이미지를 불러오지 못했습니다.";
        console.error(error);
      } finally {
        loading = false;
        updatePageButtons();
      }
    }

    function renderImages(images) {
      emptyEl.style.display = "none";
      grid.innerHTML = "";

      images.forEach((item, index) => {
        const card = document.createElement("button");
        card.className = "card";
        card.type = "button";
        card.title = "클릭하면 화면에서 제거됩니다.";

        const imageWrap = document.createElement("div");
        imageWrap.className = "image-wrap";

        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = String(index + 1);

        const img = document.createElement("img");
        img.src = item.image_url;
        img.alt = item.name || `DB image ${item.absolute_index}`;
        img.loading = "eager";
        img.referrerPolicy = "no-referrer";

        const productName = document.createElement("div");
        productName.className = "product-name";
        productName.textContent = item.name || "제품명 없음";

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `id=${item.id} / ${item.image_url}`;

        const buttonRow = document.createElement("div");
        buttonRow.className = "button-row";

        const saveButton = document.createElement("button");
        saveButton.className = "save-button";
        saveButton.type = "button";
        saveButton.textContent = item.shop_link ? "shop_link 저장" : "shop_link 없음";
        saveButton.disabled = !item.shop_link;

        const deleteButton = document.createElement("button");
        deleteButton.className = "delete-button";
        deleteButton.type = "button";
        deleteButton.textContent = "삭제";

        saveButton.addEventListener("click", async (event) => {
          event.stopPropagation();
          saveButton.disabled = true;
          deleteButton.disabled = true;
          saveButton.textContent = "저장 중";

          try {
            await saveCsvValue(item.id, item.shop_link, "");
            saveButton.textContent = "저장됨";
            card.remove();
            loadNextBatchIfCurrentBatchCleared();
          } catch (error) {
            console.error(error);
            saveButton.disabled = false;
            deleteButton.disabled = false;
            saveButton.textContent = "저장 실패";
            setTimeout(() => {
              saveButton.textContent = "shop_link 저장";
            }, 1200);
          }
        });

        deleteButton.addEventListener("click", async (event) => {
          event.stopPropagation();
          saveButton.disabled = true;
          deleteButton.disabled = true;
          deleteButton.textContent = "저장 중";

          try {
            await saveCsvValue(item.id, item.shop_link || "", "삭제");
            deleteButton.textContent = "저장됨";
            card.remove();
            loadNextBatchIfCurrentBatchCleared();
          } catch (error) {
            console.error(error);
            saveButton.disabled = !item.shop_link;
            deleteButton.disabled = false;
            deleteButton.textContent = "저장 실패";
            setTimeout(() => {
              deleteButton.textContent = "삭제";
            }, 1200);
          }
        });

        imageWrap.append(badge, img, productName);
        buttonRow.append(saveButton, deleteButton);
        card.append(imageWrap, meta, buttonRow);
        card.addEventListener("click", () => {
          card.remove();
          loadNextBatchIfCurrentBatchCleared();
        });
        grid.appendChild(card);
      });
    }

    function loadNextBatchIfCurrentBatchCleared() {
      currentCount = grid.children.length;
      if (currentCount === 0) {
        updateStatus("현재 묶음 완료");
        if (totalPages !== null && currentPage + 1 < totalPages) {
          loadPage(currentPage + 1);
        }
      } else {
        updateStatus("묶음 표시");
      }
    }

    async function saveCsvValue(id, shopLink, action) {
      const response = await fetch("/api/save-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ id, shop_link: shopLink, action }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }
    }

    loadPage(0);
  </script>
</body>
</html>
"""


IMAGE_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DB Image Viewer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, "Malgun Gothic", sans-serif;
      background: #f5f5f2;
      color: #1f2428;
    }
    body {
      margin: 0;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      max-width: 440px;
      margin: 0 auto 18px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    .status {
      color: #59636e;
      font-size: 14px;
      text-align: right;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      max-width: 440px;
      margin: 0 auto;
    }
    .card {
      position: relative;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border: 1px solid #d8d7d0;
      border-radius: 8px;
      background: #fff;
    }
    .image-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 5;
      background: #ecebe6;
    }
    .image-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .badge {
      position: absolute;
      top: 8px;
      left: 8px;
      padding: 4px 7px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      z-index: 2;
    }
    .product-name {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 8px 9px;
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
      text-align: left;
      overflow-wrap: anywhere;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .meta {
      padding: 9px 10px 10px;
      font-size: 12px;
      color: #4d5560;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }
    .button-row {
      display: flex;
      gap: 8px;
      margin: 0 10px 10px;
    }
    .save-button,
    .delete-button {
      flex: 1 1 0;
      padding: 8px 10px;
      border: 1px solid #b9b7ad;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    .save-button {
      background: #f7f7f4;
      color: #1f2428;
    }
    .delete-button {
      background: #fff1f1;
      border-color: #d9a8a8;
      color: #8f1d1d;
    }
    .save-button:hover {
      background: #eeeee8;
    }
    .delete-button:hover {
      background: #fde5e5;
    }
    .save-button:disabled,
    .delete-button:disabled {
      color: #8c959f;
      cursor: not-allowed;
    }
    .empty {
      max-width: 440px;
      margin: 36px auto 0;
      color: #59636e;
      text-align: center;
      font-size: 15px;
    }
  </style>
</head>
<body>
  <header>
    <h1>DB Image Viewer</h1>
    <div class="status" id="status">불러오는 중...</div>
  </header>
  <main class="grid" id="grid"></main>
  <p class="empty" id="empty" style="display: none;">해당 id의 image_url이 없습니다.</p>

  <script>
    const grid = document.getElementById("grid");
    const statusEl = document.getElementById("status");
    const emptyEl = document.getElementById("empty");
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id") || "";

    async function loadImage() {
      if (!id) {
        showEmpty("id가 입력되지 않았습니다.");
        return;
      }

      try {
        const response = await fetch(`/api/image?id=${encodeURIComponent(id)}`);
        if (!response.ok) {
          throw new Error(await response.text());
        }

        const payload = await response.json();
        if (!payload.image) {
          showEmpty("해당 id의 image_url이 없습니다.");
          return;
        }

        renderImage(payload.image);
        statusEl.textContent = `id=${payload.image.id}`;
      } catch (error) {
        console.error(error);
        showEmpty("이미지를 불러오지 못했습니다.");
      }
    }

    function showEmpty(message) {
      statusEl.textContent = "확인 필요";
      emptyEl.textContent = message;
      emptyEl.style.display = "block";
    }

    function renderImage(item) {
      emptyEl.style.display = "none";
      grid.innerHTML = "";

      const card = document.createElement("div");
      card.className = "card";

      const imageWrap = document.createElement("div");
      imageWrap.className = "image-wrap";

      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "id";

      const img = document.createElement("img");
      img.src = item.image_url;
      img.alt = item.name || `DB image ${item.id}`;
      img.loading = "eager";
      img.referrerPolicy = "no-referrer";

      const productName = document.createElement("div");
      productName.className = "product-name";
      productName.textContent = item.name || "제품명 없음";

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `id=${item.id} / ${item.image_url}`;

      const buttonRow = document.createElement("div");
      buttonRow.className = "button-row";

      const saveButton = document.createElement("button");
      saveButton.className = "save-button";
      saveButton.type = "button";
      saveButton.textContent = item.shop_link ? "shop_link 저장" : "shop_link 없음";
      saveButton.disabled = !item.shop_link;

      const deleteButton = document.createElement("button");
      deleteButton.className = "delete-button";
      deleteButton.type = "button";
      deleteButton.textContent = "삭제";

      saveButton.addEventListener("click", async () => {
        saveButton.disabled = true;
        deleteButton.disabled = true;
        saveButton.textContent = "저장 중";

        try {
          await saveCsvValue(item.id, item.shop_link, "");
          saveButton.textContent = "저장됨";
          statusEl.textContent = "저장됨";
        } catch (error) {
          console.error(error);
          saveButton.disabled = false;
          deleteButton.disabled = false;
          saveButton.textContent = "저장 실패";
          setTimeout(() => {
            saveButton.textContent = "shop_link 저장";
          }, 1200);
        }
      });

      deleteButton.addEventListener("click", async () => {
        saveButton.disabled = true;
        deleteButton.disabled = true;
        deleteButton.textContent = "저장 중";

        try {
          await saveCsvValue(item.id, item.shop_link || "", "삭제");
          deleteButton.textContent = "저장됨";
          statusEl.textContent = "삭제 저장됨";
        } catch (error) {
          console.error(error);
          saveButton.disabled = !item.shop_link;
          deleteButton.disabled = false;
          deleteButton.textContent = "저장 실패";
          setTimeout(() => {
            deleteButton.textContent = "삭제";
          }, 1200);
        }
      });

      imageWrap.append(badge, img, productName);
      buttonRow.append(saveButton, deleteButton);
      card.append(imageWrap, meta, buttonRow);
      grid.appendChild(card);
    }

    async function saveCsvValue(id, shopLink, action) {
      const response = await fetch("/api/save-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ id, shop_link: shopLink, action }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }
    }

    loadImage();
  </script>
</body>
</html>
"""


def fetch_image_urls(offset: int, limit: int):
    start = max(offset, 0)
    end = start + limit - 1

    response = (
        supabase.table(TABLE_NAME)
        .select(f"{ID_COLUMN}, {NAME_COLUMN}, {IMAGE_COLUMN}, {SHOP_LINK_COLUMN}")
        .neq(IMAGE_COLUMN, "")
        .order(ID_COLUMN)
        .range(start, end)
        .execute()
    )

    rows = response.data or []
    images = []
    for row_index, row in enumerate(rows):
        image_url = row.get(IMAGE_COLUMN)
        if not image_url:
            continue
        shop_link = row.get(SHOP_LINK_COLUMN) or ""
        images.append(
            {
                "id": row.get(ID_COLUMN),
                "name": row.get(NAME_COLUMN) or "",
                "image_url": image_url,
                "shop_link": shop_link,
                "absolute_index": start + row_index + 1,
            }
        )

    return images


def fetch_image_by_id(row_id):
    response = (
        supabase.table(TABLE_NAME)
        .select(f"{ID_COLUMN}, {NAME_COLUMN}, {IMAGE_COLUMN}, {SHOP_LINK_COLUMN}")
        .eq(ID_COLUMN, row_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        return None

    row = rows[0]
    image_url = row.get(IMAGE_COLUMN)
    if not image_url:
        return None

    return {
        "id": row.get(ID_COLUMN),
        "name": row.get(NAME_COLUMN) or "",
        "image_url": image_url,
        "shop_link": row.get(SHOP_LINK_COLUMN) or "",
    }


def fetch_total_image_count():
    response = (
        supabase.table(TABLE_NAME)
        .select(ID_COLUMN, count="exact")
        .neq(IMAGE_COLUMN, "")
        .range(0, 0)
        .execute()
    )
    return response.count or 0


def save_csv_value(row_id, shop_link: str, action: str = ""):
    if not row_id:
        raise ValueError("row_id is empty")

    file_exists = os.path.exists(CSV_PATH)
    needs_header = not file_exists or os.path.getsize(CSV_PATH) == 0

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(["saved_at", "id", "shop_link"])
        if action:
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), row_id, shop_link, action])
        else:
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), row_id, shop_link])


class ImageViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/":
            self.send_html(HTML)
            return

        if parsed_url.path == "/image":
            self.send_html(IMAGE_HTML)
            return

        if parsed_url.path == "/api/images":
            self.send_images(parsed_url.query)
            return

        if parsed_url.path == "/api/image":
            self.send_image(parsed_url.query)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/api/save-url":
            self.save_url()
            return

        self.send_error(404, "Not Found")

    def log_message(self, fmt, *args):
        return

    def send_html(self, content: str):
        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, status_code: int, payload):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_images(self, query: str):
        params = parse_qs(query)

        try:
            total_count = fetch_total_image_count()
            total_pages = ceil(total_count / PAGE_SIZE) if total_count else 0
            requested_offset = int(params.get("offset", ["0"])[0])
            if total_pages == 0:
                offset = 0
            else:
                max_offset = (total_pages - 1) * PAGE_SIZE
                offset = min(max(requested_offset, 0), max_offset)
            page_index = offset // PAGE_SIZE if PAGE_SIZE else 0
            images = fetch_image_urls(offset, PAGE_SIZE)
            next_offset = offset + PAGE_SIZE
            remaining_count = max(total_count - next_offset, 0)
            self.send_json(
                200,
                {
                    "images": images,
                    "next_offset": next_offset,
                    "page_index": page_index,
                    "page_size": PAGE_SIZE,
                    "total_pages": total_pages,
                    "total_loaded": offset + len(images),
                    "total_count": total_count,
                    "remaining_count": remaining_count,
                },
            )
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def send_image(self, query: str):
        params = parse_qs(query)
        row_id = params.get("id", [""])[0].strip()

        if not row_id:
            self.send_json(400, {"error": "id is required"})
            return

        try:
            self.send_json(200, {"image": fetch_image_by_id(row_id)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def save_url(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
            row_id = payload.get("id", "")
            shop_link = payload.get("shop_link", "")
            action = payload.get("action", "")
            save_csv_value(row_id, shop_link, action)
            self.send_json(200, {"saved": True, "csv_path": CSV_PATH})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def open_browser(url: str):
    time.sleep(0.5)
    webbrowser.open(url)


def main():
    server = ThreadingHTTPServer((HOST, PORT), ImageViewerHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"DB image_url viewer started: {url}")
    print("이미지를 클릭하면 화면에서 제거됩니다. 50개가 모두 사라지면 다음 50개를 불러옵니다.")
    print(f"CSV 저장 위치: {CSV_PATH}")

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


