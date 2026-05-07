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
| insert_data.py / update.py / update_fashion_embedding.py           |
| 상품 등록 · 제품명 기반 색상/소재 갱신 · FashionCLIP 임베딩 갱신     |
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

## 🚀 Textyle 프로젝트 실행 방법

### 1. ⚙️ 사전 준비

- 🔑 `Textyle-vectorserver` 폴더 안에 `.env` 파일을 생성하고 Supabase, Gemini 등 서버에서 사용하는 API 키를 입력합니다.
- 📱 `Textyle-app` 폴더 안에 환경 변수 파일이 필요한 경우 생성하고 앱에서 사용하는 API 키를 입력합니다.
- 🌐 `Textyle-app/app/(tabs)/index.tsx` 파일의 `SERVER_IP` 값을 현재 서버를 실행하는 컴퓨터의 IP 주소로 수정합니다.
- 🔌 같은 파일에서 요청 포트도 사용하는 CLIP 모델에 맞게 수정합니다.
  - 🧠 기본 CLIP 서버(`main.py`) 사용: `8000`
  - 👕 FashionCLIP 서버(`fashion_main.py`) 사용: `8001`

### 2. 📱 모바일 앱 실행

프로젝트 루트 폴더에서 새 터미널을 열고 아래 명령어를 실행합니다.

```powershell
cd Textyle-app
npx expo start
```

### 3. 🧠 기본 CLIP 벡터 서버 실행

기존 `main.py` 서버를 사용할 경우 프로젝트 루트 폴더에서 새 터미널을 열고 아래 명령어를 실행합니다.

```powershell
cd Textyle-vectorserver
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

앱의 요청 주소도 `8000` 포트를 바라보도록 맞춥니다.

```tsx
const response = await fetch(`http://${SERVER_IP}:8000/search`, {
  method: 'POST',
  body: formData,
});
```

### 4. 👕 FashionCLIP 벡터 서버 실행

FashionCLIP 기반 검색 서버를 사용할 경우 프로젝트 루트 폴더에서 새 터미널을 열고 아래 명령어를 실행합니다.

```powershell
cd Textyle-vectorserver
$env:PYTHONIOENCODING="utf-8"
uvicorn fashion_main:app --host 0.0.0.0 --port 8001 --reload
```

앱의 요청 주소도 `8001` 포트를 바라보도록 맞춥니다.

```tsx
const response = await fetch(`http://${SERVER_IP}:8001/search`, {
  method: 'POST',
  body: formData,
});
```

### 5. ⚠️ 주의 사항

- 🗄️ `fashion_main.py`는 Supabase RPC 함수 `match_clothes_fashion`과 `fashion_embedding` 컬럼을 사용합니다.
- 🧾 앱 화면이 계속 로딩 중이면 서버 터미널에서 `/search` 요청 로그가 어디까지 출력되는지 확인합니다.
