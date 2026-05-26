"""
v2 만들기 위한 참고용 코드

채널별 cosine 유사도 분포 측정 스크립트 (사전 실험).

목적
  DB(clothes 테이블) 의 embedding_512 (Marqo-FashionCLIP 으로 추출, L2-normalized 가정) 를
  corpus 로 사용하고, 각 옷의 원본 이미지를 Gemini Vision 으로 fit/color/detail 3채널
  텍스트로 분해한 뒤, 같은 Marqo-FashionCLIP 의 텍스트 인코더로 임베딩하여
  "채널별 텍스트 → 이미지 corpus" cosine 분포를 측정한다.
  채널 간 정답쌍 mean 격차로 추후 rank fusion 전략 (RRF vs CombSUM) 을 결정.

원칙
  - DB 에 저장된 벡터를 그대로 사용 (재인코딩 X, 검증용 1~2개 제외)
  - DB 벡터와 텍스트 인코더는 반드시 동일한 Marqo-FashionCLIP 임베딩 공간이어야 함
  - 키/시크릿 노출 금지

출력
  experiments/gemini_texts.json  (id, fit, color, detail, image_ref)
  experiments/results.json       (채널별 통계 + 채널 간 비교)
  experiments/results.csv        (표 형태 요약)

환경
  .env (DB_data 또는 Textyle-vectorserver) 에 SUPABASE_URL, SUPABASE_KEY,
  GEMINI_API_KEY, GEMINI_MODEL_NAME 가 있다는 전제.

실행
  python experiments/channel_cosine_analysis.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from supabase import Client, create_client

try:
    import open_clip
except ImportError:
    sys.exit("❌ open_clip_torch 미설치. `pip install open_clip_torch` 후 재실행.")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("❌ google-genai 미설치. `pip install google-genai` 후 재실행.")

try:
    from pydantic import BaseModel, Field
except ImportError:
    sys.exit("❌ pydantic 미설치. `pip install pydantic` 후 재실행.")


# ─────────────────────────────────────────────────────────────────────
# 0. 환경 변수 로드
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_CANDIDATES = [
    PROJECT_ROOT / "DB_data" / ".env",
    PROJECT_ROOT / "Textyle-vectorserver" / ".env",
    SCRIPT_DIR / ".env",
    PROJECT_ROOT / ".env",
]
_loaded = False
for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"✅ .env 로드: {env_path}")
        _loaded = True
        break
if not _loaded:
    print("⚠️ .env 파일 없음 — 시스템 환경변수에 직접 설정되어 있어야 합니다.")

SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_KEY       = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME  = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

for k, v in [("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_KEY", SUPABASE_KEY),
             ("GEMINI_API_KEY", GEMINI_API_KEY)]:
    if not v:
        sys.exit(f"❌ {k} 미설정")

print(f"   GEMINI_MODEL_NAME = {GEMINI_MODEL_NAME}")

SAMPLE_SIZE        = int(os.environ.get("SAMPLE_SIZE", "30"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
GEMINI_DEFAULT_RETRY_DELAY = float(os.environ.get("GEMINI_DEFAULT_RETRY_DELAY", "30"))

OUTPUT_DIR         = SCRIPT_DIR
GEMINI_TEXTS_PATH  = OUTPUT_DIR / "gemini_texts.json"
RESULTS_JSON_PATH  = OUTPUT_DIR / "results.json"
RESULTS_CSV_PATH   = OUTPUT_DIR / "results.csv"


# ─────────────────────────────────────────────────────────────────────
# 1. 프롬프트 + 구조화 응답 스키마
#    한 번의 Gemini 호출로 3채널을 모두 받기 위해 단일 prompt + Pydantic schema 사용.
#    각 채널의 작성 규칙은 분리해서 전달 (서로 영향 안 가도록).
# ─────────────────────────────────────────────────────────────────────
CHANNEL_NAMES: list[str] = ["fit", "color", "detail"]


class ChannelDescriptions(BaseModel):
    """Gemini 구조화 응답: 한 옷의 fit / color / detail 채널 텍스트."""
    fit: str = Field(
        ...,
        description=(
            "Fit & silhouette ONLY. Ignore color/pattern/material/logo. "
            "Cover overall fit (oversized/relaxed/regular/slim/tight), shoulders "
            "(dropped/structured), sleeves (wide/loose/fitted/tapered), body "
            "(boxy wide/fitted/cropped), length, defining fit detail. "
            "English noun phrase. Max 50 words. No color, no fabric, no pattern words."
        ),
    )
    color: str = Field(
        ...,
        description=(
            "All colors part-by-part. Ignore fit/silhouette/shape. "
            "Order: dominant color FIRST with specific shade (e.g. royal blue, deep crimson); "
            "then secondary colors with location ('<part> is <color>' for logo/collar/cuff/trim/pocket); "
            "then overall tone (warm/cool, bright/muted/pastel/dark). "
            "English. Dominant first, minor accents last. Max 40 words."
        ),
    )
    detail: str = Field(
        ...,
        description=(
            "All surface patterns and design/construction details. Ignore fit/silhouette and "
            "ignore color hue names (pattern types OK). "
            "Order: MAIN pattern or most prominent detail FIRST (e.g. large graphic print on chest, "
            "all-over floral, solid plain surface); then every other detail with location "
            "('<detail> at/on <location>'): logo placement, prints, embroidery, distressed/rips, "
            "patches, pockets, zippers, buttons, drawstrings, ribbed cuffs/hem, hood, "
            "collar/neckline, seams, panels, ruffles, lettering/text. "
            "English. Most prominent first. Max 50 words. NOT fit, NOT color names."
        ),
    )


COMBINED_PROMPT = """You are a fashion analyst. Look at the garment in the image and produce \
THREE independent descriptions of it: fit, color, and detail.

Each description must focus on its own aspect ONLY and ignore the other two aspects entirely \
(e.g. the fit description must contain no color words; the color description must contain no \
fit/silhouette words; the detail description must name pattern types but no color hues).

Follow the per-field rules described in the response schema exactly (length caps, ordering, \
English only, noun phrase style, no preamble).

Return JSON with exactly these keys: "fit", "color", "detail". \
Each value is a single phrase (no preamble, no commentary outside the JSON)."""


# ─────────────────────────────────────────────────────────────────────
# 2. 클라이언트 / 디바이스
# ─────────────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = pick_device()
print(f"🖥️  device = {device}")


# ─────────────────────────────────────────────────────────────────────
# 3. DB 스키마 탐색 (한 행 가져와 컬럼/벡터 형식 확인)
# ─────────────────────────────────────────────────────────────────────
def _to_float_list(emb) -> list[float]:
    """DB 의 vector 컬럼이 list 또는 JSON 문자열로 올 수 있으므로 통일."""
    if emb is None:
        return []
    if isinstance(emb, str):
        emb = json.loads(emb)
    return list(map(float, emb))


def explore_schema() -> None:
    print("\n" + "=" * 60)
    print("DB 스키마 탐색")
    print("=" * 60)
    resp = (
        supabase.table("clothes")
        .select("*")
        .not_.is_("embedding_512", "null")
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        sys.exit("❌ embedding_512 가 채워진 행이 0개. 먼저 index_clip.py 를 실행하세요.")
    row = rows[0]
    print("\n샘플 행 (한 개) 의 컬럼:")
    for k, v in row.items():
        if isinstance(v, list):
            print(f"  {k:20s}: list (len={len(v)}, sample={v[:3]}...)")
        elif isinstance(v, dict):
            print(f"  {k:20s}: dict (keys={list(v.keys())[:5]})")
        elif isinstance(v, str) and len(v) > 60:
            print(f"  {k:20s}: str(len={len(v)}, {v[:50]}...)")
        else:
            print(f"  {k:20s}: {type(v).__name__} = {v}")

    emb = _to_float_list(row.get("embedding_512"))
    arr = np.asarray(emb, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    print(f"\n📐 embedding_512:")
    print(f"   dim   = {arr.shape[0]}")
    print(f"   norm  = {norm:.6f}  "
          f"({'L2-정규화됨' if 0.99 < norm < 1.01 else 'raw — 코사인 전에 정규화 적용함'})")
    if arr.shape[0] != 512:
        print(f"⚠️ 차원 불일치 ({arr.shape[0]} ≠ 512). Marqo-FashionCLIP 와 호환 안 될 수 있음.")


# ─────────────────────────────────────────────────────────────────────
# 4. 샘플링 (sub_category 별 라운드로빈으로 다양성 확보)
# ─────────────────────────────────────────────────────────────────────
def sample_rows(n: int) -> list[dict]:
    resp = (
        supabase.table("clothes")
        .select("id, name, sub_category, main_category, image_url, image_url_hq, embedding_512")
        .not_.is_("embedding_512", "null")
        .limit(max(n * 10, 300))
        .execute()
    )
    candidates = resp.data or []
    if not candidates:
        return []
    if len(candidates) <= n:
        return candidates

    random.shuffle(candidates)

    by_cat: dict[str, list[dict]] = {}
    for row in candidates:
        cat = row.get("sub_category") or "_unknown"
        by_cat.setdefault(cat, []).append(row)

    selected: list[dict] = []
    while len(selected) < n:
        added = False
        for cat in list(by_cat.keys()):
            if not by_cat[cat]:
                continue
            selected.append(by_cat[cat].pop(0))
            added = True
            if len(selected) >= n:
                break
        if not added:
            break
    return selected[:n]


# ─────────────────────────────────────────────────────────────────────
# 5. 이미지 다운로드
# ─────────────────────────────────────────────────────────────────────
def download_image(url: str, timeout: int = 10) -> Optional[bytes]:
    if not url:
        return None
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        if len(r.content) < 1000:
            return None
        return r.content
    except Exception as exc:
        print(f"   [download] 실패 {url[:60]}: {exc}")
        return None


def _guess_mime(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            fmt = (im.format or "").lower()
        if fmt == "png":  return "image/png"
        if fmt == "webp": return "image/webp"
        if fmt == "gif":  return "image/gif"
        return "image/jpeg"
    except Exception:
        return "image/jpeg"


# ─────────────────────────────────────────────────────────────────────
# 6. Gemini Vision 호출 (429 시 retryDelay 만큼 대기 후 재시도)
# ─────────────────────────────────────────────────────────────────────
def _is_quota_error(exc: Exception) -> bool:
    m = str(exc)
    return ("429" in m) or ("RESOURCE_EXHAUSTED" in m) or ("quota" in m.lower())


def _extract_retry_delay(exc: Exception) -> float:
    m = str(exc)
    hit = re.search(r"['\"]retryDelay['\"]\s*[:=]\s*['\"]\s*(\d+(?:\.\d+)?)s", m)
    if hit:
        return float(hit.group(1))
    hit = re.search(r"retry in (\d+(?:\.\d+)?)s", m, re.IGNORECASE)
    if hit:
        return float(hit.group(1))
    return GEMINI_DEFAULT_RETRY_DELAY


def gemini_describe_channels(image_bytes: bytes) -> Optional[dict[str, str]]:
    """한 번의 Gemini 호출로 fit/color/detail 3채널을 모두 받는다.
    구조화 응답(JSON + Pydantic schema) 사용 → API 호출 횟수 1/3."""
    mime = _guess_mime(image_bytes)
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    COMBINED_PROMPT,
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ChannelDescriptions,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                print(f"   [Gemini] 빈 응답, 재시도 ({attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(2)
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"   [Gemini] JSON 파싱 실패 ({exc}), 재시도 ({attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(2)
                continue
            out = {ch: (parsed.get(ch) or "").strip() for ch in CHANNEL_NAMES}
            if not all(out.values()):
                missing = [ch for ch, v in out.items() if not v]
                print(f"   [Gemini] 누락 채널 {missing}, 재시도 ({attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(2)
                continue
            return out
        except Exception as exc:
            if _is_quota_error(exc) and attempt < GEMINI_MAX_RETRIES:
                delay = _extract_retry_delay(exc) + 1.0
                print(f"   ⏳ [Gemini] quota, {delay:.1f}s 대기 "
                      f"({attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(delay)
                continue
            print(f"   [Gemini] 실패: {exc}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────────
# 7. Marqo-FashionCLIP 로드 (open_clip)
# ─────────────────────────────────────────────────────────────────────
MODEL_ID = "hf-hub:Marqo/marqo-fashionCLIP"
print(f"\n⏳ {MODEL_ID} 로딩 중...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(MODEL_ID)
clip_tokenizer = open_clip.get_tokenizer(MODEL_ID)
clip_model = clip_model.to(device).eval()
print("✅ 모델 로딩 완료")


# pad token id 감지 — CLIP=0, SigLIP=1. 잘못 잡으면 패딩이 전부 "사용된 토큰" 으로
# 잡혀 토큰 길이가 항상 context_length 로 보이는 버그 발생.
def _detect_pad_id() -> int:
    for obj in (clip_tokenizer, getattr(clip_tokenizer, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("pad_token_id", "pad_id"):
            v = getattr(obj, attr, None)
            if v is not None:
                return int(v)
    # Fallback: 빈 문자열 토큰화 → 마지막 위치 = padding 으로 간주
    empty_ids = clip_tokenizer([""])[0]
    return int(empty_ids[-1].item())


PAD_ID = _detect_pad_id()
print(f"   pad_id = {PAD_ID}")


@torch.no_grad()
def encode_texts(texts: list[str]) -> torch.Tensor:
    """텍스트 N개 → L2-normalized (N, 512)."""
    tokens = clip_tokenizer(texts).to(device)
    feats = clip_model.encode_text(tokens, normalize=True)
    return feats


@torch.no_grad()
def encode_image_pil(pil: Image.Image) -> torch.Tensor:
    t = clip_preprocess(pil).unsqueeze(0).to(device)
    feats = clip_model.encode_image(t, normalize=True)
    return feats


def count_used_tokens(text: str) -> int:
    """open_clip context_length(77) 중 실제 사용된 토큰 수 (pad 제외)."""
    ids = clip_tokenizer([text])[0]
    return int((ids != PAD_ID).sum().item())


# ─────────────────────────────────────────────────────────────────────
# 8. 정합성 검증 — 1-2개 이미지 재인코딩 vs DB 벡터
# ─────────────────────────────────────────────────────────────────────
def validate_corpus_alignment(rows: list[dict], n_check: int = 2) -> None:
    print("\n" + "=" * 60)
    print(f"정합성 검증 ({n_check}개 이미지 재인코딩 vs DB 벡터)")
    print("=" * 60)
    checked = 0
    for row in rows:
        if checked >= n_check:
            break
        url = row.get("image_url_hq") or row.get("image_url") or ""
        img_bytes = download_image(url)
        if img_bytes is None:
            continue
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            print(f"  id={row['id']}: 이미지 디코딩 실패: {exc}")
            continue
        emb_db = torch.tensor(_to_float_list(row.get("embedding_512")),
                              dtype=torch.float32, device=device)
        emb_db = torch.nn.functional.normalize(emb_db, p=2, dim=-1)
        emb_re = encode_image_pil(pil).squeeze()
        cos = float((emb_db * emb_re).sum().item())
        status = "✅ 일치" if cos > 0.99 else "⚠️ 불일치 — 모델/전처리 버전 차이 가능"
        print(f"  id={row['id']}: cos(re-encoded, DB) = {cos:.4f}  {status}")
        checked += 1
    if checked == 0:
        print("  (검증 가능한 이미지 0개)")


# ─────────────────────────────────────────────────────────────────────
# 9. Gemini 텍스트 생성 (캐싱)
# ─────────────────────────────────────────────────────────────────────
def generate_channel_texts(rows: list[dict]) -> list[dict]:
    """각 행에 대해 3채널 텍스트 생성. GEMINI_TEXTS_PATH 에 캐싱."""
    cache: dict[str, dict] = {}
    if GEMINI_TEXTS_PATH.exists():
        try:
            data = json.loads(GEMINI_TEXTS_PATH.read_text(encoding="utf-8"))
            for entry in data:
                cache[str(entry["id"])] = entry
            print(f"\n📂 기존 캐시 로드: {len(cache)}개 ({GEMINI_TEXTS_PATH.name})")
        except Exception as exc:
            print(f"⚠️ 캐시 파싱 실패 (무시): {exc}")

    results: list[dict] = []
    print(f"\n🤖 Gemini Vision 3채널 텍스트 생성 (model={GEMINI_MODEL_NAME}, temp=0.1)\n")

    def _save_cache(items: list[dict]):
        GEMINI_TEXTS_PATH.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for idx, row in enumerate(rows, 1):
        rid = str(row["id"])
        url = row.get("image_url_hq") or row.get("image_url") or ""

        entry = dict(cache.get(rid, {}))
        entry["id"] = rid
        entry["image_ref"] = url
        entry["sub_category"] = row.get("sub_category") or ""

        # 이미 3채널 모두 있으면 스킵
        if all(entry.get(ch) for ch in CHANNEL_NAMES):
            print(f"[{idx}/{len(rows)}] id={rid}  (cache hit)")
            results.append(entry)
            continue

        print(f"[{idx}/{len(rows)}] id={rid}  {row.get('sub_category', '-')}")
        img_bytes = download_image(url)
        if img_bytes is None:
            print("   skip (이미지 다운로드 실패)")
            results.append(entry)
            continue

        # 단일 호출로 3채널 모두 받기 (구조화 응답)
        channels = gemini_describe_channels(img_bytes)
        if channels is None:
            print("   skip (Gemini 호출 실패)")
            results.append(entry)
            continue
        for ch in CHANNEL_NAMES:
            entry[ch] = channels[ch]
            print(f"   {ch}: {channels[ch][:80]}")

        results.append(entry)
        # 매 5개마다 캐시 flush (중단 대비)
        if idx % 5 == 0:
            _save_cache(results)

    _save_cache(results)
    print(f"\n💾 Gemini 텍스트 저장: {GEMINI_TEXTS_PATH}")
    return results


# ─────────────────────────────────────────────────────────────────────
# 10. 채널별 cosine 통계
# ─────────────────────────────────────────────────────────────────────
def _tensor_stats(t: torch.Tensor) -> dict:
    return {
        "mean": float(t.mean()),
        "std":  float(t.std()),
        "min":  float(t.min()),
        "max":  float(t.max()),
    }


def channel_cosine_stats(
    corpus: torch.Tensor,      # (N, 512), L2-normalized
    texts: list[str],
    label: str,
) -> dict:
    n = corpus.size(0)
    assert len(texts) == n

    txt = encode_texts(texts)                  # (N, 512), normalized
    sim = txt @ corpus.T                        # (N, N)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    diag = sim[eye]
    off  = sim[~eye]

    # 토큰 길이 체크
    token_lens = [count_used_tokens(t) for t in texts]
    over_limit = sum(1 for length in token_lens if length >= 75)
    if over_limit:
        print(f"   ⚠️ {over_limit}/{n} 텍스트가 75 토큰 이상 — 잘림 위험")

    def recall_at_k(k: int) -> float:
        topk = sim.topk(k, dim=-1).indices
        labels = torch.arange(n, device=sim.device)
        return float((topk == labels.unsqueeze(1)).any(dim=-1).float().mean())

    pos = _tensor_stats(diag)
    neg = _tensor_stats(off)
    all_ = _tensor_stats(sim.flatten())

    return {
        "channel": label,
        "n": n,
        "all": all_,
        "positive": pos,
        "negative": neg,
        "separation": pos["mean"] - neg["mean"],
        "range_pos": pos["max"] - pos["min"],
        "range_neg": neg["max"] - neg["min"],
        "recall@1": recall_at_k(1),
        "recall@5": recall_at_k(min(5, n)),
        "token_len_mean": float(np.mean(token_lens)),
        "token_len_max":  int(max(token_lens)),
        "token_over_75":  over_limit,
    }


def print_channel_report(r: dict) -> None:
    print(f"\n■ [{r['channel'].upper()}] n={r['n']}")
    print(f"  전체:    mean={r['all']['mean']:+.4f}  std={r['all']['std']:.4f}  "
          f"range=[{r['all']['min']:+.4f}, {r['all']['max']:+.4f}]")
    print(f"  정답쌍:  mean={r['positive']['mean']:+.4f}  std={r['positive']['std']:.4f}  "
          f"range=[{r['positive']['min']:+.4f}, {r['positive']['max']:+.4f}]")
    print(f"  오답쌍:  mean={r['negative']['mean']:+.4f}  std={r['negative']['std']:.4f}  "
          f"range=[{r['negative']['min']:+.4f}, {r['negative']['max']:+.4f}]")
    print(f"  분리도:  {r['separation']:+.4f}  (정답mean − 오답mean)")
    print(f"  Recall@1 = {r['recall@1']:.4f}   Recall@5 = {r['recall@5']:.4f}")
    print(f"  토큰: mean={r['token_len_mean']:.1f}, max={r['token_len_max']}, "
          f"≥75: {r['token_over_75']}/{r['n']}")


def channel_comparison(results_per_ch: dict[str, dict]) -> dict:
    pos_means = {ch: r["positive"]["mean"] for ch, r in results_per_ch.items()}
    gap = max(pos_means.values()) - min(pos_means.values())
    if gap > 0.10:
        rec = "RRF 권장 — 채널 간 절대 cosine 격차가 큼 (rank 기반 정규화가 효과적)"
    elif gap > 0.05:
        rec = "RRF 우선 + CombSUM 비교 — 격차 중간"
    else:
        rec = "CombSUM 도 후보 — 격차가 작아 절대값 합산도 유효"

    print("\n" + "=" * 60)
    print("채널 간 비교")
    print("=" * 60)
    header = f"{'channel':<10}{'pos_mean':>10}{'pos_std':>10}{'neg_mean':>10}{'sep':>10}{'R@1':>8}{'R@5':>8}"
    print(header)
    print("-" * len(header))
    for ch, r in results_per_ch.items():
        print(f"{ch:<10}"
              f"{r['positive']['mean']:>10.4f}"
              f"{r['positive']['std']:>10.4f}"
              f"{r['negative']['mean']:>10.4f}"
              f"{r['separation']:>10.4f}"
              f"{r['recall@1']:>8.3f}"
              f"{r['recall@5']:>8.3f}")
    print(f"\n정답 mean 격차 (max−min) = {gap:.4f}")
    print(f"📌 판정: {rec}")

    return {
        "pos_means": pos_means,
        "gap_max_min": gap,
        "recommendation": rec,
    }


# ─────────────────────────────────────────────────────────────────────
# 11. 결과 저장
# ─────────────────────────────────────────────────────────────────────
def save_results(results_per_ch: dict[str, dict], comparison: dict, n_pairs: int) -> None:
    out = {
        "n_pairs": n_pairs,
        "model":   MODEL_ID,
        "gemini_model": GEMINI_MODEL_NAME,
        "channels": results_per_ch,
        "comparison": comparison,
    }
    RESULTS_JSON_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n💾 결과 JSON: {RESULTS_JSON_PATH}")

    lines = [
        "channel,n,pos_mean,pos_std,pos_min,pos_max,"
        "neg_mean,neg_std,separation,recall@1,recall@5,"
        "tok_mean,tok_max,tok_over75"
    ]
    for ch, r in results_per_ch.items():
        lines.append(
            f"{ch},{r['n']},"
            f"{r['positive']['mean']:.6f},{r['positive']['std']:.6f},"
            f"{r['positive']['min']:.6f},{r['positive']['max']:.6f},"
            f"{r['negative']['mean']:.6f},{r['negative']['std']:.6f},"
            f"{r['separation']:.6f},"
            f"{r['recall@1']:.6f},{r['recall@5']:.6f},"
            f"{r['token_len_mean']:.2f},{r['token_len_max']},{r['token_over_75']}"
        )
    lines.append("")
    lines.append(f"# gap_max_min,{comparison['gap_max_min']:.6f}")
    lines.append(f"# recommendation,{comparison['recommendation']}")
    RESULTS_CSV_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"💾 결과 CSV : {RESULTS_CSV_PATH}")


# ─────────────────────────────────────────────────────────────────────
# 12. 메인
# ─────────────────────────────────────────────────────────────────────
def main():
    explore_schema()

    print("\n" + "=" * 60)
    print(f"샘플링: {SAMPLE_SIZE}개 (sub_category 별 라운드로빈)")
    print("=" * 60)
    rows = sample_rows(SAMPLE_SIZE)
    if not rows:
        sys.exit("❌ 샘플링 실패")
    print(f"✅ {len(rows)}개 가져옴")

    cat_dist: dict[str, int] = {}
    for r in rows:
        c = r.get("sub_category") or "_"
        cat_dist[c] = cat_dist.get(c, 0) + 1
    print("카테고리 분포:")
    for c, n in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {c:25s}: {n}")

    # 정합성 검증
    try:
        validate_corpus_alignment(rows, n_check=2)
    except Exception as exc:
        print(f"⚠️ 정합성 검증 실패 (스킵): {exc}")

    # Gemini 채널 텍스트
    text_entries = generate_channel_texts(rows)

    # 3채널 모두 있는 행만 사용
    id_to_row = {str(r["id"]): r for r in rows}
    valid: list[tuple[dict, dict]] = []
    for entry in text_entries:
        rid = entry.get("id")
        if rid not in id_to_row:
            continue
        if not all(entry.get(ch) for ch in CHANNEL_NAMES):
            continue
        valid.append((id_to_row[rid], entry))

    print(f"\n유효 페어 (3채널 모두 채워진 행): {len(valid)}/{len(rows)}")
    if len(valid) < 3:
        sys.exit("❌ 유효 페어가 너무 적어 통계 의미 없음.")

    # corpus 행렬 구성 (DB 벡터, 정규화 강제)
    corpus_list = [_to_float_list(row["embedding_512"]) for row, _ in valid]
    corpus = torch.tensor(corpus_list, dtype=torch.float32, device=device)
    corpus = torch.nn.functional.normalize(corpus, p=2, dim=-1)
    print(f"\ncorpus shape = {tuple(corpus.shape)}  (L2-normalized 적용 후)")

    # 채널별 분석
    results_per_ch: dict[str, dict] = {}
    for ch in CHANNEL_NAMES:
        texts = [entry[ch] for _, entry in valid]
        r = channel_cosine_stats(corpus, texts, ch)
        print_channel_report(r)
        results_per_ch[ch] = r

    # 채널 간 비교
    comparison = channel_comparison(results_per_ch)

    # 저장
    save_results(results_per_ch, comparison, n_pairs=len(valid))

    print("\n✅ 완료")


if __name__ == "__main__":
    main()