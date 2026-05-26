# 👕 Textyle

## 🔎 이미지와 자연어를 함께 이해하는 패션 검색 서비스

Textyle은 사용자가 업로드한 의류 이미지와 한국어 검색 문장을 함께 분석해, Supabase에 저장된 의류 데이터 중 조건에 맞는 상품을 찾아주는 모바일 기반 패션 검색 프로젝트입니다. Expo 앱에서 이미지를 선택하고 검색어를 입력하면, FastAPI 벡터 서버가 FashionCLIP 임베딩과 쿼리 해석 결과를 이용해 유사 상품을 반환합니다.

## ✨ 특징

- 🖼️ 이미지 기반 유사도 검색: 업로드한 의류 이미지를 FashionCLIP 이미지 임베딩으로 변환해 시각적으로 비슷한 상품을 검색합니다.
- 💬 자연어 조건 검색: "검은색 바지", "비슷한 디자인의 반팔"처럼 한국어로 입력된 조건을 색상, 카테고리, 디자인 의도로 분리해 검색에 반영합니다.
- 🎯 색상/카테고리 기반 후처리: 명시된 색상이나 반팔/긴팔 같은 세부 조건과 맞지 않는 후보를 최종 결과에서 제외해 검색 정확도를 높입니다.
- 🗄️ Supabase 연동: 의류 상품 데이터, 이미지 URL, 카테고리, 색상, 소재, 벡터 임베딩을 Supabase 테이블에 저장하고 RPC 검색 함수로 조회합니다.
- 🔄 데이터 갱신 스크립트: `DB_data`의 스크립트를 통해 FashionCLIP 임베딩과 제품명 기반 색상/소재 정보를 DB에 갱신할 수 있습니다.
- 📱 모바일 중심 UX: Expo Router 기반 React Native 앱에서 이미지 선택, 검색어 입력, 결과 확인, 상품 링크 이동 흐름을 제공합니다.

## 🧩 시스템 아키텍처

```text
                               👤 User
                                  |
                                  v
+------------------------------------------------------------------+
| 📱 Textyle-app                                                    |
| Expo / React Native / Expo Router                                |
| 이미지 선택 · 한국어 검색어 입력 · 결과 리스트 표시                 |
+------------------------------------------------------------------+
                                  |
                                  | multipart/form-data
                                  v
+------------------------------------------------------------------+
| 🧠 Textyle-vectorserver                                           |
| FastAPI / FashionCLIP Search API                                  |
| /search 요청 처리 · 이미지/텍스트 임베딩 생성 · 후보 재정렬          |
+------------------------------------------------------------------+
          |                         |                         |
          v                         v                         v
+-------------------+    +-------------------+    +-------------------+
| 💬 Query Analyzer |    | 🖼️ FashionCLIP    |    | 🎯 Reranker       |
| Gemini / fallback |    | image/text embed  |    | color/category    |
| intent 추출        |    | similarity vector |    | strict filtering  |
+-------------------+    +-------------------+    +-------------------+
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                                    v
+------------------------------------------------------------------+
| 🗄️ Supabase                                                       |
| clothes 테이블 / fashion_embedding / match_clothes_fashion RPC     |
| 상품 데이터 · 이미지 URL · 색상 · 소재 · 카테고리 · 벡터 검색         |
+------------------------------------------------------------------+
                                    ^
                                    |
+------------------------------------------------------------------+
| 🔄 DB_data Scripts                                                |
| update/insert_data.py / update/update_fashion_colors.py / update/update_fashion_embeddings_only.py |
| 상품 등록 · 색상 후보 갱신 · FashionCLIP 임베딩 갱신                 |
+------------------------------------------------------------------+
```

1. 👤 사용자가 모바일 앱에서 이미지를 업로드하고 검색 문장을 입력합니다.
2. 📡 앱은 `multipart/form-data` 요청으로 FastAPI 서버의 `/search` 엔드포인트를 호출합니다.
3. 🧠 서버는 Gemini 또는 rule-based fallback으로 검색 의도를 분석하고, FashionCLIP으로 이미지/텍스트 임베딩을 생성합니다.
4. 🎯 서버는 Supabase RPC 함수로 벡터 유사 후보를 가져온 뒤 색상, 카테고리, 세부 카테고리 조건을 기준으로 재정렬합니다.
5. 📱 앱은 반환된 상품 이미지, 이름, 가격, 링크를 결과 화면에 표시합니다.
6. 🔄 `DB_data` 스크립트는 별도 실행으로 상품 임베딩과 제품명 기반 속성 정보를 Supabase에 갱신합니다.

## 🛠️ 기술 스택

| 구분 | 기술 스택 | 역할 |
| --- | --- | --- |
| 📱 모바일 앱 | Expo, React Native, Expo Router, TypeScript | 이미지 업로드, 검색어 입력, 검색 결과 UI |
| 🔐 인증/데이터 | Supabase Auth, Supabase Database, Supabase RPC | 사용자 인증, 상품 데이터 저장, 벡터 검색 함수 실행 |
| 🚀 벡터 검색 서버 | Python, FastAPI, Uvicorn | `/search` API 제공, 검색 요청 처리 |
| 🧠 AI/임베딩 | FashionCLIP, OpenAI CLIP, PyTorch, Transformers | 이미지/텍스트 임베딩 생성 및 유사도 검색 |
| 💬 쿼리 분석 | Gemini API, rule-based fallback | 한국어 검색어 의도, 색상, 카테고리 조건 추출 |
| 🖼️ 이미지 처리 | Pillow | 업로드 이미지 로딩, 전처리, 중심 영역 crop |
| 🔄 데이터 갱신 | Python scripts, Supabase Python Client | 상품 등록, 속성 갱신, FashionCLIP 임베딩 갱신 |
| ✅ 테스트 | Python `unittest` | 쿼리 분석, 필터링, rerank 순수 로직 검증 |

## 🎨 의류 색상 추출 실험

이미지 기반 의류 색상 추출은 `GroundingDINO + SAM` 방식으로 검증 중입니다. 기존 `rembg/u2net_cloth_seg` 방식은 실패로 폐기했고, 현재는 별도 검증 스크립트에서 의류 mask와 색상 후보를 생성한 뒤 CSV 결과를 확인하는 흐름을 사용합니다.

폐기한 방식은 다음과 같습니다.

- 중앙 crop 기반 색상 추출
- 가장자리 픽셀 기반 배경색 제거
- `rembg` + `u2net_cloth_seg` 의류 mask
- mask 내부 k-means 대표색 추출
- 중앙/하단/목/소매 평균 기반 대표색 보정

실패 원인과 개선 과정은 `COLOR_EXTRACTION_EXPERIMENT_NOTES.md`에 정리되어 있습니다. 핵심은 `rembg/u2net_cloth_seg`가 의류 전체가 아니라 상단, 하단, 허벅지 띠 같은 일부 영역만 segment하는 경우가 많았다는 점입니다.

현재 검증 방향은 다음과 같습니다.

- GroundingDINO로 의류 객체 bbox를 찾습니다.
- SAM에 bbox prompt를 전달해 여러 mask 후보를 생성합니다.
- SAM 후보 mask를 색상 힌트, mask 크기, 배경 가능성 기준으로 ranking합니다.
- mask 내부 픽셀에서 k-means 색상 후보를 만들고, 148개 named color 기준 세부색도 함께 저장합니다.
- 최종 검색 색상은 `white, black, red, yellow, green, blue, purple, gray, orange, brown, pink` 11개로 정리합니다.
- 제품명 색상 힌트가 있으면 최종 검색색 결정에 우선 반영합니다.
- 체크, 스트라이프, 카모 등 패턴은 색상 추출과 분리하고 ViT 패턴 분류 대상으로 표시합니다.

검증 스크립트는 다음 위치에 있습니다.

```text
DB_data/test/verify_groundingdino_sam_color_extraction.py
```

실행 전 `SAM_CHECKPOINT`에 로컬 SAM checkpoint 경로를 지정해야 합니다.

```powershell
cd DB_data/test
$env:SAM_CHECKPOINT="E:\models\sam_vit_b_01ec64.pth"
python verify_groundingdino_sam_color_extraction.py --limit 20
```

검증 결과는 `groundingdino_sam_color_report.csv`와 `groundingdino_sam_debug_sheet.jpg`로 생성됩니다.

Supabase 상품을 현재 방식으로 새로 분석해 색상 컬럼에 반영하는 스크립트도 추가되어 있습니다. 기본 실행은 dry-run입니다.

```powershell
cd DB_data
python .\update\update_groundingdino_sam_colors.py --ids 6260 7195 7058 4397
```

이 스크립트는 Supabase에서 제품명을 읽고 `DB_data/image_jpg_700/<id>.jpg` 이미지를 GroundingDINO + SAM 방식으로 분석합니다. 실제 DB 업데이트는 명시적으로 `--apply`를 붙여 실행합니다.

```powershell
python .\update\update_groundingdino_sam_colors.py --ids 6260 7195 7058 4397 --apply
```

## 👖 FashionCLIP 데님 검색 실험

데님팬츠 검색은 FashionCLIP 이미지 임베딩만으로는 연청, 중청, 진청, 흑청처럼 색상 톤이 가까운 후보를 안정적으로 구분하기 어려워 별도 실험 흐름으로 관리합니다. 현재 서버는 데님 문맥에서 이미지 색상 후보, named color, 제품명 기반 톤 힌트, Supabase 후보 재정렬 점수를 함께 사용합니다.

상세 실험 내용과 폐기한 보조 스크립트는 `FASHIONCLIP_DENIM_SEARCH_EXPERIMENTS.md`에 정리되어 있습니다. 현재 기준으로 `DB_data/test/check_product_name_color_hint.py`와 `DB_data/update/export_denim_pants_rows.py`는 임시 검증용 스크립트였으므로 저장소에서 제거했습니다.

## 🚀 Textyle 프로젝트 실행 방법

### 1. ⚙️ 사전 준비

- 🔑 `Textyle-vectorserver` 폴더 안에 `.env` 파일을 생성하고 Supabase, Gemini 등 서버에서 사용하는 API 키를 입력합니다.
- 📱 `Textyle-app` 폴더 안에 환경 변수 파일이 필요한 경우 생성하고 앱에서 사용하는 API 키를 입력합니다.
- 🌐 `Textyle-app` 실행 환경에 `EXPO_PUBLIC_FASHION_API_URL`을 설정합니다. 예: `http://192.168.0.6:8001`
- 🔌 API URL은 FashionCLIP 서버 포트 `8001`을 바라보도록 맞춥니다.

### 2. 📱 모바일 앱 실행

프로젝트 루트 폴더에서 새 터미널을 열고 아래 명령어를 실행합니다.

```powershell
cd Textyle-app
npx expo start
```

### 3. 👕 FashionCLIP 벡터 서버 실행

FashionCLIP 기반 검색 서버를 사용할 경우 프로젝트 루트 폴더에서 새 터미널을 열고 아래 명령어를 실행합니다.

```powershell
cd Textyle-vectorserver
$env:PYTHONIOENCODING="utf-8"
uvicorn fashion_main:app --host 0.0.0.0 --port 8001 --reload
```

앱의 요청 주소도 `8001` 포트를 바라보도록 맞춥니다.

```powershell
$env:EXPO_PUBLIC_FASHION_API_URL="http://192.168.0.6:8001"
npx expo start
```

### 4. ⚠️ 주의 사항

- 🗄️ `fashion_main.py`는 Supabase RPC 함수 `match_clothes_fashion`과 `fashion_embedding` 컬럼을 사용합니다.
- 🧾 앱 화면이 계속 로딩 중이면 서버 터미널에서 `/search` 요청 로그가 어디까지 출력되는지 확인합니다.


### 추가된 기능

1️⃣ v2 URL 자동 생성
index.tsx:9:


const FASHION_API_V2_URL = FASHION_API_URL?.replace(/:8001(\/|$)/, ':8002$1');
기존 EXPO_PUBLIC_FASHION_API_URL에서 포트만 8001 → 8002로 치환. 별도 환경변수 설정 불필요.

2️⃣ 버전을 받는 검색 함수
searchClothes(version: 'v1' | 'v2') 형태로 변경. 두 버전 모두 같은 함수 재사용.

3️⃣ 두 개의 검색 버튼 (검색 화면)
보라색 버튼: v1 검색 (이미지+텍스트) 🔍 → 포트 8001
초록색 버튼: v2 검색 (텍스트 하나로 융합 후 검색) 🧪 → 포트 8002
각자 검색 중일 때만 해당 버튼이 ActivityIndicator 표시.

4️⃣ 결과 화면에 메타 정보 표시
우측 상단 버전 배지: v1 · 이미지+텍스트 (보라) / v2 · 텍스트 전용 (초록)
결과 위 메타 박스: 검색 쿼리, 추출된 색상 (v2), 디자인 설명 (v2)
v1은 enhanced_query/color_extracted/design_description 필드가 없으므로 자동으로 안 나옴(옵셔널 체이닝). v2 응답에서는 모두 표시됨.

5️⃣ 다른 옷 검색하기 버튼
이제 searchResults + resultVersion + resultMeta 모두 초기화.

사용 방법
1. 두 서버 동시 기동 (별도 터미널)

# 터미널 1: v1
cd Textyle-vectorserver
python -m uvicorn fashion_main:app --host 0.0.0.0 --port 8001 --reload

# 터미널 2: v2
cd Textyle-vectorserver
python -m uvicorn fashion_main_v2:app --host 0.0.0.0 --port 8002 --reload
2. Expo 앱 환경변수 확인
Textyle-app/.env 또는 EXPO_PUBLIC_FASHION_API_URL에 반드시 :8001을 포함해야 v2 URL이 자동 도출됩니다:


EXPO_PUBLIC_FASHION_API_URL=http://{IP주소}:8001
3. 앱 실행 후
레퍼런스 사진 첨부
"변경 요구사항 텍스트" 입력
v1 버튼 눌러서 결과 확인 → 다른 옷 검색하기
v2 버튼 눌러서 같은 입력으로 결과 비교

---

6️⃣ v2 검색 서버 (`fashion_main_v2.py`, 포트 8002)

v1(이미지+텍스트 벡터 가중합) 대비 핵심 변경점:

- **이미지 임베딩 제거**: CLIP 텍스트 인코더만 사용. 이미지의 시각 벡터가 검색 벡터에 섞이지 않아 색상 교체 쿼리의 정확도가 높아짐.
- **K-means 색상 추출 → Gemini 입력**: 이미지 픽셀에서 K-means로 색상을 먼저 추출해 Gemini에 전달. Gemini는 색을 재추측하지 않고 디자인/실루엣/소재만 영어로 서술(`design_description`).
- **세분화된 색상(detailed color) 활용**: K-means가 추출한 정밀 색명(navy, burgundy, olive, charcoal 등)을 `enhanced_query`에 사용. 리랭킹용 11가지 broad 카테고리(blue, red 등)로 단순화하기 전 값이므로 색조 수준의 검색 정밀도 향상. `color_mode=same`(같은 색 유지) 쿼리에만 적용.
- **Gemini `is_fashion` 검증**: v1의 CLIP zero-shot 패션 판별 대신 Gemini 응답으로 비패션 이미지 거부.
- **응답 필드 추가**: `design_description`, `color_extracted.detailed_color` 포함.
