# TEXTYLE AI 검색 서버

사진 한 장과 한국어 검색어로 비슷한 옷을 찾는 패션 검색 서비스 **TEXTYLE**의 AI 검색 서버 저장소입니다.
FastAPI 서버(`Textyle-vectorserver`)가 Gemini로 검색어를 분석하고, FashionCLIP 이미지·텍스트 벡터를 의도에 맞는 비중으로 합쳐 Supabase에서 상품을 찾은 뒤 조건에 맞게 다시 정렬합니다.

## 특징

- **이미지 기반 유사도 검색**: 업로드한 옷 사진을 FashionCLIP 이미지 벡터로 바꿔 시각적으로 비슷한 상품을 찾습니다. 검색어 없이 사진만으로도 검색할 수 있습니다.
- **자연어 조건 검색**: "검정색으로", "같은 색인데 슬림한 핏"처럼 한국어로 입력한 조건을 Gemini가 색, 옷 종류, 디자인, 색 조건(같은 색 / 다른 색 / 지정 색)으로 나눕니다. 질의 의도에 따라 이미지·텍스트 벡터의 비중을 조절해 검색에 반영합니다.
- **옷 영역 분리 (정밀 분석)**: 요청하면 GroundingDINO와 SAM으로 사진에서 옷만 분리해 배경의 영향을 줄입니다.
- **조건 기반 재정렬**: 카테고리, 세부 카테고리(반소매·긴소매 등), 색, 데님 톤 점수로 후보를 다시 정렬합니다. 조건과 맞지 않는 후보는 제외하지 않고 순위를 뒤로 보냅니다.
- **Supabase 벡터 검색**: 상품 정보, 이미지 URL, 카테고리, 색, FashionCLIP 벡터를 `clothes` 테이블에 저장하고 pgvector 기반 RPC(`match_clothes_fashion`)로 조회합니다.
- **데이터 갱신 스크립트**: `DB_data`의 스크립트로 상품 등록, FashionCLIP 벡터 갱신, 사진 기반 색 갱신을 할 수 있습니다.
- **속도와 안정성**: 서버 시작 때 모델과 DB 연결을 미리 준비하고, 동시 검색 수를 제한합니다. Gemini 응답이 늦으면 규칙 기반 분석으로 전환하고, 단계별 처리 시간을 응답과 로그에 남깁니다.
- **웹 서비스 연동**: Spring API 서버를 거쳐 웹 서비스(tex-tyle.app)의 검색 기능으로 제공됩니다.

## 기술 스택

| 파트 | 기술 |
|---|---|
| BackEnd | Python, FastAPI, Uvicorn |
| AI | FashionCLIP, PyTorch, Transformers, Gemini API, GroundingDINO, SAM |
| DataBase | Supabase(PostgreSQL, pgvector) |
| Infra | AWS EC2, GitHub Actions, AWS SSM |
| etc. | Notion |

## 전체 구성

```
웹 (Textyle-web, Vercel)                       tex-tyle.app
  └─ Spring Boot API 서버 (Textile-Service)    api.tex-tyle.app
       └─ AI 검색 서버 (이 저장소, FastAPI :8001)  외부 비공개, Spring 서버만 접근
            └─ Supabase                         clothes 테이블 · fashion_embedding · match_clothes_fashion RPC
```

- 브라우저는 AI 서버를 직접 부르지 않습니다. Spring 서버가 검색 요청을 받아 이 서버의 `/search`로 중계합니다.
- 응답 형식은 Spring 서버의 `SearchResultDto`와 맞춰져 있습니다. 응답 필드를 바꿀 때는 Spring 쪽도 함께 확인해야 합니다.

## 저장소 구조

| 경로 | 내용 |
|---|---|
| `Textyle-vectorserver/fashion_main.py` | AI 검색 서버 본체 (API, 질의 분석, 임베딩, 재정렬) |
| `Textyle-vectorserver/fashion_color_extraction.py` | 사진 색 추출 |
| `Textyle-vectorserver/requirements.txt` | 서버 패키지 목록 |
| `DB_data/` | 상품 등록, 임베딩·색 갱신, 검증 스크립트 (사용법: `DB_data/scripts_guide.md`) |
| `.github/workflows/deploy-ai.yml` | AI 서버 자동 배포 |
| `COLOR_EXTRACTION_EXPERIMENT_NOTES.md` | 색 추출 실험 기록 |
| `FASHIONCLIP_DENIM_SEARCH_EXPERIMENTS.md` | 데님 검색 실험 기록 |
| `Textyle-app/` | 초기 Expo 앱 (현재 웹 프론트는 Textyle-web 저장소에서 관리) |

## 검색 흐름 (`POST /search`)

1. **입력 검증**: 검색어 최대 100자, 프롬프트 조작·SQL·스크립트 패턴 차단, 사진 최대 20MB
2. **사진 준비**: 휴대폰 사진의 회전 정보(EXIF)를 반영하고 긴 변 1280px로 축소
3. **질의 분석**: Gemini가 검색어에서 색, 옷 종류, 디자인, 색 조건(`color_mode`)을 추출합니다. 응답이 늦거나 실패하면 규칙 기반 분석으로 바로 전환합니다. 옷 종류가 비어 있으면 FashionCLIP으로 사진의 상위 카테고리를 채웁니다.
4. **옷 영역 분리 (선택)**: `use_grounding_dino=true`(웹의 "정밀 분석")이면 GroundingDINO로 옷을 찾고 SAM으로 배경을 지운 사진을 씁니다. 모델을 쓸 수 없으면 원본 사진으로 진행합니다.
5. **색 추출**: 같은 색 / 다른 색 조건이나 디자인 유사 검색일 때 사진에서 옷 색을 추출합니다.
6. **벡터 결합**: FashionCLIP 이미지 벡터와 텍스트 벡터를 아래 비중으로 합칩니다.
   `검색 벡터 = normalize(이미지 비중 × 이미지 벡터 + 텍스트 비중 × 텍스트 벡터)`
7. **벡터 검색**: Supabase RPC `match_clothes_fashion`으로 후보를 가져옵니다 (카테고리 필터, 후보 100~220개). 같은 색 검색에서 후보가 20개 미만이면 기준을 낮춰 한 번 더 찾습니다.
8. **재정렬**: 카테고리·색·디자인·데님 톤 점수를 더해 상위 10개를 반환합니다. 조건과 맞지 않는 후보는 제외하지 않고 순위를 뒤로 보냅니다.


## 로컬 실행


```bash
cd Textyle-vectorserver
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- torch는 `requirements.txt`의 안내대로 설치합니다. CPU 전용 주소(`--index-url .../whl/cpu`)로 설치하지 마세요.

`Textyle-vectorserver/.env`를 만듭니다. 실제 값은 팀 DM으로 공유됩니다.

```
SUPABASE_URL=
SUPABASE_KEY=
GEMINI_API_KEY=
GEMINI_MODEL_NAME=
SAM_CHECKPOINT=      # SAM 체크포인트 파일 경로 (정밀 분석용, 예: sam_vit_b_01ec64.pth)
```

서버 실행

```bash
uvicorn fashion_main:app --host 0.0.0.0 --port 8001
```

- Windows PowerShell에서 한글 로그가 깨지면 먼저 `$env:PYTHONIOENCODING="utf-8"`를 설정합니다.
- 처음 실행할 때 FashionCLIP과 GroundingDINO 모델을 내려받느라 시간이 걸립니다.
- `curl http://localhost:8001/health`로 확인합니다.
- Spring 서버를 로컬에서 함께 띄우면 기본 설정(`ai.server.url=http://localhost:8001`)으로 바로 연결됩니다.

### 선택 환경변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `TEXTYLE_ENABLE_DINO_SAM` | `1` | `0`이면 GroundingDINO·SAM 모델을 올리지 않습니다 (정밀 분석 요청도 원본 사진으로 처리) |
| `SAM_MODEL_TYPE` | `vit_b` | SAM 모델 종류 |
| `GROUNDING_DINO_MODEL_ID` | `IDEA-Research/grounding-dino-tiny` | GroundingDINO 모델 |
| `DINO_SAM_BOX_THRESHOLD` / `DINO_SAM_TEXT_THRESHOLD` | `0.28` / `0.25` | 옷 영역 검출 기준 |
| `DINO_SAM_DEVICE` | 자동 | DINO·SAM 실행 장치 |
| `FASHION_CLIP_MODEL_ID` | `patrickjohncyh/fashion-clip` | 임베딩 모델 |
| `GEMINI_TIMEOUT_MS` | `12000` | Gemini 응답 대기 상한 (Gemini API는 10초 미만을 거절) |
| `MAX_CONCURRENT_SEARCHES` | `2` | 동시에 실행하는 검색 수. 나머지는 순서대로 대기 |
| `MAX_IMAGE_SIDE` | `1280` | 업로드 사진 긴 변 최대 크기(px) |
| `MAX_UPLOAD_MB` | `20` | 업로드 사진 최대 용량 (Spring 서버와 같은 값) |
| `TEXTYLE_WARMUP` | `1` | 서버 시작 때 모델 로딩·프롬프트 임베딩·DB 연결을 미리 준비 |
| `SUPABASE_KEEPALIVE_INTERVAL` | `45` | 검색이 없을 때 Supabase 연결 유지용 1건 조회 간격(초). `0`이면 끔 |
| `SUPABASE_KEEPALIVE_EXPIRY` | `300` | 쉬고 있는 Supabase 연결을 유지하는 시간(초) |

## 운영 서버와 배포

- `main`에 `Textyle-vectorserver/` 변경이 푸시되면 `deploy-ai.yml`이 실행됩니다.
  - GitHub Actions가 AWS 임시 권한(OIDC)을 받아 SSM으로 서버에 명령을 보냅니다. GitHub에 AWS 키를 저장하지 않고, SSH 접속도 쓰지 않습니다.
  - 서버에서 `git pull` → (`requirements.txt`가 바뀌었으면) 패키지 설치 → 재시작 → `/health` 응답 확인(최대 10분)
- 직접 배포하려면 Actions → **Deploy AI server** → **Run workflow**를 누릅니다. 서버 인스턴스가 중지되어 있으면 실패하므로, 켠 뒤 다시 실행합니다.

서버에서 자주 쓰는 명령

```bash
curl -s http://127.0.0.1:8001/health      # 상태 확인
sudo journalctl -u textyle-ai -f          # 실시간 로그 ([Timing] 줄에 단계별 처리 시간)
sudo systemctl restart textyle-ai         # 재시작
```

## 상품 데이터 (Supabase)

- `clothes` 테이블에 상품 정보, 이미지 URL, 카테고리, 색, `fashion_embedding`(상품 사진의 FashionCLIP 벡터)을 저장합니다.
- 검색 색은 `white, black, red, yellow, green, blue, purple, gray, orange, brown, pink` 11가지로 정리합니다.
- 데님 바지는 별도로 `denim_tone`(`light, medium, dark, black, gray, raw`)을 씁니다. `manual_final_color`, `manual_detail_color`는 사람이 검수한 색입니다.
- 재정렬에서 색을 비교하려면 RPC 결과에 `dominant_color`, `color_candidates`, `manual_final_color`, `manual_detail_color`, `denim_tone` 컬럼이 포함되어야 합니다.
- 상품 등록과 갱신 스크립트 사용법은 `DB_data/scripts_guide.md`를 참고하세요. 일부 스크립트(예: `update_groundingdino_sam_colors.py`)는 기본이 미리보기(dry-run)라 `--apply`를 붙여야 반영되지만, 바로 DB에 쓰는 스크립트도 있으니 실행 전에 확인하세요.

## 실험 기록

- **색 추출**: 중앙 영역 자르기, `rembg/u2net_cloth_seg` 방식은 옷의 일부만 잡는 문제로 폐기하고 GroundingDINO + SAM으로 바꿨습니다. → `COLOR_EXTRACTION_EXPERIMENT_NOTES.md`
- **데님 검색**: 연청·중청·진청처럼 가까운 톤을 구분하려고 `denim_tone`과 수동 검수 색을 재정렬에 반영했습니다. → `FASHIONCLIP_DENIM_SEARCH_EXPERIMENTS.md`

## 관련 저장소

- 웹 프론트: `givenhhy/Textyle-web`
- Spring Boot API 서버: `sunghoonhan-hh/Textile-Service`
