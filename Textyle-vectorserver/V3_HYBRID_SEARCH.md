# 🧪 fashion_main_v3.py — 하이브리드 검색 서버 (포트 8003)

텍스트 1차 검색 + 리랭킹/필터링 + **이미지 벡터 2차 점수 결합**을 한 파이프라인으로 묶은 실험용 검색 서버입니다.
v1(8001)·v2(8002)는 그대로 두고 A/B 비교용으로 추가했습니다.

---

## 1. 한 줄 요약

> 이미지·텍스트 벡터를 **합치지 않고 단계로 분리**한다.
> 색·디자인 의미는 **텍스트(Gemini description) 벡터**로 1차 검색 + 리랭킹해서 정합성을 보장하고,
> 그 다음 **걸러진 후보 안에서만** 원본 이미지 벡터의 시각 유사도를 가점으로 더해 최종 정렬한다.

---

## 2. 실행 방법

### 서버 기동
v1과 동시에 띄울 수 있습니다(포트만 다름).

```powershell
cd Textyle-vectorserver
$env:PYTHONIOENCODING="utf-8"
uvicorn fashion_main_v3:app --host 0.0.0.0 --port 8003 --reload
```

- `.env`는 v1과 동일한 파일을 그대로 사용합니다 (`SUPABASE_URL`, `SUPABASE_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`).
- Supabase RPC(`match_clothes_fashion`)·DB 스키마 변경 없음. `fashion_embedding` 컬럼(512차원 L2 정규화)만 추가로 읽습니다.
- `Application startup complete` 출력 후 `http://localhost:8003/docs`로 확인 가능.

### 앱에서 호출
앱은 `EXPO_PUBLIC_FASHION_API_URL`(예: `http://192.168.0.6:8001`)의 **포트만 8001 → 8003**으로 치환해 v3를 호출합니다. 별도 env 설정 불필요.

- 검색 화면에 **🧪 v3로 검색 (하이브리드)** 버튼과 테스트 노브 입력칸이 있습니다.
- 노브를 비우면 서버 기본값을 사용합니다.

---

## 3. 기존(v1/v2)과의 핵심 차이점

| 구분 | **v1** (8001) | **v2** (8002) | **v3** (8003, 본 문서) |
|---|---|---|---|
| 검색 벡터 | `이미지*0.55~0.65 + 텍스트*0.35~0.45` 가중합 | Gemini description의 텍스트 단일 벡터 | Gemini description의 텍스트 단일 벡터 |
| 이미지 벡터 사용 | 검색 벡터에 **혼합** | 사용 안 함 | **2차 점수(가점)로 분리 사용** |
| Gemini 입력 | 텍스트만 | 이미지+텍스트 → description | 이미지+텍스트 → description + 구조화 intent |
| 색상 정보원 | K-means(픽셀) | (실험에 따라 다름) | K-means(픽셀) — 리랭킹 색 매칭용, v1과 동일 |
| 리랭킹 | 있음 | 없음(유사도순) | **있음 + 색 페널티 강화 + 하드컷** |
| 색 교체 쿼리 약점 | 이미지 색이 1차 검색을 오염 | 해소(텍스트만) | 해소(텍스트만) + change-gate로 2차도 차단 |

**핵심 아이디어**: v1은 색을 바꾸려 해도 이미지 벡터에 “바꿔야 할 색”이 섞여 들어가 오염됐습니다. v3는 그 색 의미를 텍스트 단계로 옮겨 정합성을 확보하고, 이미지 벡터는 **색이 이미 정리된 뒤** 실루엣·디테일 보정용으로만 씁니다.

---

## 4. 검색 실행 시 로직

입력 예시: **회색 후드집업 사진 + "사진과 비슷한데 빨간색으로"**

```
[입력] 이미지 + 텍스트
   │
   ▼ 의류 검증 (validate_fashion_image, CLIP zero-shot)
   │  └ 비패션(음식/풍경 등)이면 400 거부
   │
   ▼ STAGE 0: Gemini 멀티모달 분석 (analyze_query_v3)
   │  입력: 이미지 + "사진과 비슷한데 빨간색으로"
   │  출력:
   │   - description = "a photo of red oversized hoodie zip-up, kangaroo pocket,
   │                    ribbed cuffs, drop shoulder, cotton fleece ..."   ← 검색 벡터로 사용
   │   - color=red, color_mode=target, design="oversized hoodie zip-up"   ← 리랭킹용
   │   - main/sub_category, is_fashion
   │  (+ 조건부 K-means 색 추출 → 리랭킹 query_attrs의 색 매칭 기준. same/different/
   │     디자인유사/패턴 쿼리일 때만 실행)
   │
   ▼ STAGE 1: 텍스트 단일 벡터 검색
   │  query_emb = FashionCLIP_text(description)
   │  supabase.rpc("match_clothes_fashion", ...) → 후보 100~220개
   │  (similarity = description 텍스트 ↔ 후보 이미지 임베딩의 코사인)
   │  (v1과 동일한 threshold/match_count + 폴백 재시도 로직 유지)
   │
   ▼ STAGE 2: 리랭킹 + 텍스트 점수 하드컷
   │  rerank_results(color_penalty_scale, return_all=True)
   │   → 각 후보 text_final_score = base_similarity + 카테고리/세부카테고리/색/디자인/톤 보정
   │   → 색 불일치(음수 색 보정)에는 color_penalty_scale(기본 2.0) 곱해 페널티 강화
   │  text_final_score < score_threshold 인 후보 **탈락**
   │  (전부 잘리면 빈 결과 대신 텍스트 점수 상위 10개로 폴백)
   │
   ▼ STAGE 3: 이미지 벡터 2차 가점 + 최종 정렬
   │  w2_eff = 0  (change-gate ON 이고 color_mode ∈ {target, different})  ← 색 교체라 이미지 차단
   │         = image_weight(w2)  (그 외: same / ignore / 이미지-only)
   │  if w2_eff > 0:
   │     살아남은 후보 id로 fashion_embedding 일괄 fetch
   │     image_sim = cosine(쿼리 이미지 벡터, 후보 이미지 벡터)   (둘 다 정규화 → dot)
   │     image_sim_norm = min-max 정규화 (살아남은 풀 안에서)
   │     final = text_final_score + w2_eff * image_sim_norm
   │  else:
   │     final = text_final_score
   │  sort by final 내림차순 → Top 10
   │
   ▼ [응답] Top 10 + 점수 분해
```

### 단계별 포인트
- **색 정합성은 텍스트 단계(STAGE 1~2)에서 확보** → 이미지 가점이 색을 흔들지 못함.
- **하드컷이 이미지 가점보다 먼저** → 색·카테고리가 틀린 후보는 이미지가 닮았다고 살아나지 못함.
- **change-gate**: "빨간색으로"처럼 색을 바꾸는 쿼리(target/different)에서는 원본(회색) 이미지 벡터가 다시 끼어들면 오염되므로 2차 가점을 끔.
- **preserve 쿼리**("같은 색으로", "비슷한 디자인", 이미지-only)에서는 2차 가점이 켜져 텍스트로 못 적는 실루엣·질감 뉘앙스를 보정.

---

## 5. 테스트 노브 (앱에서 조절, 폼 파라미터)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `image_weight` (w2) | `0.5` | 이미지 유사도가 줄 수 있는 **최대 가점** (min-max 정규화 후 곱) |
| `score_threshold` | `0.15` | 이 **텍스트 리랭킹 점수** 미만 후보는 탈락. 보수적 기본값(잘 안 잘림) |
| `color_penalty_scale` | `2.0` | 색 불일치 시 음수 색보정 페널티 배율 (v1 대비) |
| `change_gate` | `true` | target/different 쿼리에서 w2를 0으로(색 교체 시 이미지 차단) |

> 모두 서버에서 0~상한으로 clamp됩니다. 비우면 기본값 사용.

### 튜닝 가이드
- **회색 잔류가 거슬리면**: `score_threshold`를 올리거나(예: 0.30~0.40) `color_penalty_scale`을 키웁니다(예: 3~4).
- **결과가 너무 적게 나오면**: `score_threshold`를 낮춥니다(0이면 컷 사실상 해제).
- **이미지 디테일 보정 효과 보기**: preserve 쿼리에서 `image_weight`를 0 ↔ 0.5 ↔ 1.0으로 바꿔 비교.
- **색 교체에도 이미지 보정 실험**: `change_gate`를 꺼서 target 쿼리에도 w2를 적용(색 오염 위험 감수).

---

## 6. 응답 형식 (주요 필드)

```jsonc
{
  "version": "v3-hybrid",
  "enhanced_query": "a photo of red oversized hoodie ...",  // = Gemini description
  "knobs": { "image_weight": 0.5, "w2_effective": 0.0,
             "score_threshold": 0.15, "color_penalty_scale": 2.0, "change_gate": true },
  "color_extracted": { ... },   // K-means 결과(실행됐을 때)
  "intent": { "color": "red", "color_mode": "target", "design": "...", "description": "..." },
  "search_warnings": [ ... ],
  "results": [
    {
      "name": "...", "main_category": "...", "sub_category": "...", "image_url": "...",
      "_ranking": {
        "final_score": 0.71,            // 텍스트 리랭킹 점수
        "image_sim": 0.62,              // 쿼리↔후보 이미지 코사인 (가점 적용 시)
        "image_sim_norm": 0.84,         // 풀 내 min-max 정규화값
        "w2_effective": 0.5,            // 실제 적용된 w2
        "final_score_combined": 1.13    // 최종 정렬 점수
      }
    }
  ]
}
```

---

## 7. 디버그 로그 읽는 법

서버 콘솔에 매 검색마다 출력됩니다.

```
[FashionCLIP v3 Hybrid Debug]
original_query=사진과 비슷한데 빨간색으로
enhanced_query=a photo of red oversized hoodie zip-up ...
color_mode=target, change_gate=True, w2=0.5, w2_eff=0.0, score_threshold=0.15, color_penalty_scale=2.0
main_categories=['아우터'], sub_categories=['후드집업']
raw_result_count=100, survivors_after_cut=63, hard_cut_fallback=False, image_stage_applied=False
  [1] OOO 후드집업 | text=0.78 img_sim=None img_norm=None combined=0.78 | color=red cat=아우터>후드집업
  [2] ...
```

- `w2_eff=0.0` + `image_stage_applied=False` → change-gate가 작동해 2차 이미지 가점을 끈 상태(색 교체 쿼리).
- `survivors_after_cut` → 하드컷 후 남은 후보 수. 너무 적으면 `score_threshold`를 낮춰야 함.
- `hard_cut_fallback=True` → threshold가 전부 잘라내 상위 10개로 폴백한 상태(threshold가 과함).
- 결과 줄의 `text / img_sim / img_norm / combined` 분해로 어느 단계가 순위를 결정했는지 확인.

---

## 8. 주의 / 한계

- **2차 fetch 비용**: 살아남은 후보(~수십~100개)의 `fashion_embedding`을 한 번 더 가져옵니다. 테스트 용도라 수용 가능한 수준.
- **join 키**: 후보를 `id`로 매칭하며, 없으면 `image_url`로 폴백합니다.
- **change-gate를 끄면** 색 교체 쿼리에서 원본 색이 2차 정렬에 잔류 편향으로 끼어들 수 있습니다(의도된 트레이드오프).
- **코드 중복**: v1을 복사해 만든 실험 서버라 공통 로직이 중복됩니다. 방향이 확정되면 공통 모듈로 분리 가능.
- `build_enhanced_query` / `log_search_debug`는 v1에서 복사됐지만 v3에서는 사용하지 않습니다(향후 정리 대상).
