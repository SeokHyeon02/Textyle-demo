# 의류 색상 추출 실험 기록

날짜: 2026-05-18

이 문서는 Textyle 의류 이미지 색상 추출 실험 과정과 현재 채택한 방향을 정리한다. 현재 결과는 DB 전체 업데이트 전 검증 단계이며, 최종 운영 적용 전에는 더 많은 샘플과 실제 검색 결과로 추가 확인이 필요하다.

## 목표

이미지에서 배경, 인물, 로고, 그림자 영향을 줄이고 의류 본체 색상을 추출한다. 추출 결과는 이미지와 텍스트를 함께 사용하는 패션 검색에서 색상 필터링과 reranking 보조 신호로 사용한다.

## 초기 접근과 실패

처음에는 비교적 가벼운 이미지 처리 방식부터 실험했다.

- 중앙 crop 기반 대표색 추출
- 이미지 가장자리 픽셀 기반 배경색 제거
- `rembg` + `u2net_cloth_seg` 의류 segmentation
- mask 내부 k-means 색상 후보 추출
- 중앙, 하단, 목, 소매 영역 평균 기반 대표색 보정
- 제품명 색상 힌트 기반 보정

이 방식들은 일부 샘플에서는 동작했지만 안정적이지 않았다.

주요 실패 원인은 다음과 같다.

- 중앙 crop은 배경, 모델 피부, 로고, 흰 영역이 섞여 대표색이 흔들렸다.
- `rembg/u2net_cloth_seg`는 의류 전체가 아니라 상단, 하단, 허벅지 띠, 소매 일부처럼 특정 부분만 잡는 경우가 많았다.
- mask 시각화는 정상처럼 보여도 실제 `masked` 픽셀은 일부 영역만 포함되는 경우가 있었다.
- 체크, 스트라이프, 카모 같은 패턴 의류는 단일 대표색으로 강제하면 오분류가 늘었다.
- k-means `k=20`과 논문식 중앙/하단/목/소매 평균 방식은 복잡도가 늘었지만, 검증 샘플에서는 기존 방식보다 안정적이지 않았다.

따라서 `rembg/u2net_cloth_seg` 기반 색상 추출은 검색 품질에 바로 반영하기 어렵다고 판단했다.

## GroundingDINO + SAM 검증

다음 단계로 GroundingDINO와 SAM을 이용했다.

1. GroundingDINO로 의류 객체 bbox를 찾는다.
2. SAM에 bbox prompt를 전달해 의류 mask 후보를 생성한다.
3. mask 내부 픽셀만 수집한다.
4. k-means 색상 후보를 만든다.
5. 148개 named color 기준으로 세부색을 분류한다.
6. 최종 검색용 색상은 제한된 카테고리로 묶는다.
7. 제품명에 색상이 있으면 최종 검색색 결정에 우선 반영한다.

검증 스크립트:

```text
DB_data/test/verify_groundingdino_sam_color_extraction.py
```

최신 검증 샘플:

```text
C:\Users\leoho\Downloads\groundingdino_sam_color_report (1).csv
C:\Users\leoho\Downloads\groundingdino_sam_debug_sheet (1).jpg
```

최신 샘플 100건은 모두 `status=ok`로 처리됐다.

## 최종 색상 카테고리

검색용 최종 색상은 다음 11개로 정리했다.

```text
white, black, red, yellow, green, blue, purple, gray, orange, brown, pink
```

초기에는 `black`을 최종 색상에서 제외하는 방안도 검토했지만, 실제 DB 상품에는 블랙 상품이 많고 `6260`, `7195`처럼 제품명과 실제 색상이 명확히 블랙인 케이스가 있어 최종 카테고리에 유지했다.

## 148개 named color 사용

사용자가 제공한 MATLAB/Matplotlib named color 표를 기준으로 148개 세부 색상을 활용했다. 최종 검색색은 11개 카테고리로 유지하되, 이미지에서 가까운 named color를 별도 기록한다.

예:

- `extracted_color`: 검색용 최종색
- `extracted_named_color`: 148개 named color 중 대표 세부색
- `named_candidates_json`: named color 후보 목록
- `candidates_json`: 최종색 기준 후보 목록
- `search_colors_json`: 검색 확장용 후보와 가중치

이렇게 분리한 이유는 검색 필터는 단순해야 하지만, 디버깅과 세부 보정에는 더 촘촘한 색상 정보가 필요하기 때문이다.

## 주요 오류와 해결

### 1. 어두운 유채색이 black/gray로 쏠림

딥카키, 다크브라운, 네이비 계열이 `black` 또는 `gray`로 자주 분류됐다.

해결:

- 제품명 색상 힌트가 `blue`, `green`, `brown`인 경우 무채색 표준편차 판정보다 hue 보정을 먼저 적용했다.
- `hint_blue_neutral`, `hint_green_neutral`, `hint_brown_neutral` 이유를 후보에 남겼다.

개선 샘플:

- `3894.jpg`: 딥카키 -> `green`
- `7048.jpg`: DARK BROWN -> `brown`
- `1210.jpg`: 블루 스트라이프 -> `blue`
- `1561.jpg`: 빈티지네이비 -> `blue`

### 2. black 상품이 white로 추출됨

`6260`, `7195`는 제품명과 실제 상품이 블랙인데, 초기 결과에서는 흰 배경이 mask로 잡혀 `white`가 1순위가 됐다.

원인:

- GroundingDINO bbox 안에서 SAM 후보 중 배경 mask가 높은 score를 받았다.
- 기존 코드는 SAM 후보 중 `sam_score`가 가장 높은 mask만 선택했다.

해결:

- SAM 후보 mask를 모두 평가하도록 수정했다.
- 후보 mask별로 색상 후보를 임시 추출하고, 제품명 색상 힌트와 맞는 mask에 가점을 줬다.
- 제품명 색상이 white가 아닌데 mask가 white 위주이면 감점했다.

개선 결과:

- `6260.jpg`: `pre_hint_color=black`, `dominant_rgb=[22, 23, 26]`
- `7195.jpg`: `pre_hint_color=black`, `dominant_rgb=[28, 28, 31]`

### 3. 흰 옷이 흰 배경으로 잡힘

`7058`은 흰 셔츠인데 초기 수정 후 완전한 흰 배경이 선택되어 `dominant_rgb=[255,255,255]`, `dominant_ratio=1.0`이 됐다.

해결:

- 제품명 힌트가 white여도 `top_color=white`, `top_ratio>0.92`, `brightness>=248`, `mask_bbox_ratio>0.85`이면 순수 흰 배경으로 보고 감점했다.

개선 결과:

- `7058.jpg`: `dominant_rgb=[244,245,244]`, `dominant_ratio=0.868`

### 4. 후드집업 일부만 mask로 선택됨

`4397`은 투톤 후드집업인데 후드 일부만 선택되어 gray 단일 색상처럼 나왔다.

원인:

- 작은 부분 mask가 SAM 후보 중 선택됐다.
- 제품명은 Navy지만 선택 mask는 gray 단일 영역이었다.

해결:

- `mask_ratio`가 너무 낮은 mask를 감점했다.
- 작은 mask인데 제품명 색상과 다르고 한 색상 비율이 90% 이상이면 추가 감점했다.

개선 결과:

- `4397.jpg`: 최종 `blue`, 이미지 후보는 `black/gray` 혼합으로 바뀌었고, 이전처럼 gray 100% 단일 후보는 해소됐다.

## 최신 100건 검증 요약

최신 CSV 기준:

```text
status: ok 100

extracted_color:
black 39
gray 18
blue 12
brown 11
green 10
white 7
red 2
purple 1

color_confidence:
high 85
medium 10
low 5

color_reason:
dominant_image_color 66
product_name_priority 19
pattern_hint_needs_vit 8
moderate_image_color 5
ambiguous_second_color 2
```

핵심 샘플:

| 파일 | 제품명 요약 | 결과 | 비고 |
| --- | --- | --- | --- |
| `6260.jpg` | 블랙 코트 | `black` | 배경 mask 문제 해결 |
| `7195.jpg` | 블랙 티셔츠 | `black` | 배경 mask 문제 해결 |
| `7058.jpg` | 화이트 셔츠 | `white` | 순수 흰 배경 감점 적용 |
| `4397.jpg` | Navy 후드집업 | `blue` | 작은 부분 mask 문제 완화 |
| `3894.jpg` | 딥카키 셔츠 | `green` | hue 힌트 보정 |
| `7048.jpg` | 다크브라운 자켓 | `brown` | hue 힌트 보정 |
| `1210.jpg` | 블루 스트라이프 팬츠 | `blue` | 패턴 케이스, ViT 대상 |

## 패턴 처리 방향

체크, 스트라이프, 카모, 헤링본, 체커보드, 그래픽 패턴은 색상 추출과 분리한다.

현재 스크립트는 제품명에 패턴 키워드가 있으면 다음 값을 기록한다.

```text
pattern_hint
should_run_pattern_vit
```

실제 패턴 분류는 ViT 모델을 별도로 붙이는 방향이 적합하다. 색상 추출 로직에서 패턴까지 억지로 판정하면 체크/스트라이프 의류가 `multi_color`, `gray`, `white`로 흔들리는 문제가 커진다.

## DB 반영 스크립트

DB에 색상 결과를 반영할 때는 Supabase 상품을 다시 조회하고 로컬 이미지를 현재 GroundingDINO + SAM 방식으로 새로 분석해서 업데이트한다.

```text
DB_data/update/update_groundingdino_sam_colors.py
```

이 스크립트는 Supabase에서 `id`, `name`을 조회하고, `DB_data/image_jpg_700/<id>.jpg` 이미지를 현재 GroundingDINO + SAM 방식으로 새로 분석한 뒤 DB에 업데이트한다. 기본 실행은 dry-run이며 실제 DB를 수정하지 않는다.

```powershell
cd E:\캡스톤\DB_data
python .\update\update_groundingdino_sam_colors.py --ids 6260 7195 7058 4397
```

실제 업데이트는 명시적으로 `--apply`를 붙여야 한다.

```powershell
python .\update\update_groundingdino_sam_colors.py --ids 6260 7195 7058 4397 --apply
```

기본 업데이트 컬럼:

```text
dominant_color
color_confidence
color_candidates
```

추가 컬럼이 DB에 있을 경우 옵션으로 지정할 수 있다.

```powershell
python .\update\update_groundingdino_sam_colors.py `
  --color-reason-column color_reason `
  --named-color-column extracted_named_color `
  --pattern-hint-column pattern_hint `
  --pattern-vit-column should_run_pattern_vit `
  --apply
```

전체 DB를 처리하려면 `--ids`를 생략한다. 처음에는 반드시 `--limit` 또는 `--ids`로 작은 범위만 dry-run 한 뒤 적용한다.

```powershell
python .\update\update_groundingdino_sam_colors.py --limit 20
```

## 현재 판단

현재 GroundingDINO + SAM 방식은 기존 중앙 crop/rembg 방식보다 안정적이다. 특히 제품명 색상 힌트, 148개 named color, SAM 후보 mask ranking을 함께 사용하면서 주요 실패 케이스가 개선됐다.

다만 최종 운영 반영 전에는 다음 확인이 필요하다.

- 100건보다 큰 샘플에서 색상 분포가 과도하게 black/gray로 쏠리지 않는지 확인
- 패턴 의류는 ViT 분류 결과와 함께 저장
- DB 전체 업데이트 전 `--ids` 옵션으로 일부 상품만 먼저 업데이트
- 검색 API에서 `dominant_color`, `color_candidates`가 의도대로 reranking에 반영되는지 확인
