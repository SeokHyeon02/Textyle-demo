# DB_data 스크립트 가이드

이 문서는 아래 3개 스크립트의 역할과 실행 방식을 정리한다.

- 🖼️ `view_db_images.py`
- 🔄 `convert_png_to_jpg.py`
- 🔗 `append_storage_urls_to_csv.py`

세 스크립트는 서로 연결되어 있지만 역할은 분리되어 있다.

1. `view_db_images.py`로 DB 이미지를 검토하고 CSV에 `id`, `shop_link`를 저장한다.
2. `convert_png_to_jpg.py`로 로컬 이미지 파일을 `id.jpg` 형식으로 변환한다.
3. `append_storage_urls_to_csv.py`로 CSV 마지막 컬럼에 storage URL을 추가한다.


## 1. 🖼️ `view_db_images.py`

📁 파일 위치:
- [view_db_images.py](./image_update/view_db_images.py)

### 목적

Supabase `clothes` 테이블에 저장된 이미지들을 브라우저에서 검토하기 위한 로컬 뷰어다.

이 스크립트는 다음 작업을 위해 사용한다.

- DB에 있는 `image_url`을 화면에 표시
- 각 이미지의 `id`, 제품명(`name`), `shop_link` 확인
- 버튼 클릭 시 `shop_link`를 CSV에 저장
- 이미지를 클릭하거나 버튼을 눌렀을 때 화면에서 제거
- 현재 50개 배치를 모두 처리하면 다음 50개 자동 로드
- 한 번에 50개를 건너뛰는 기능 제공

### 동작 방식

로컬 HTTP 서버를 띄우고 기본 브라우저를 연다.

- 기본 주소: `http://127.0.0.1:8765`
- 기본 배치 크기: `50`

브라우저 화면에는 다음 정보가 보인다.

- 이미지
- 좌상단 배치 번호
- 이미지 하단 제품명
- `id`와 `image_url`
- `shop_link 저장` 버튼
- 우측 상단 현재 개수 / 남은 개수

### CSV 저장 형식

버튼을 누르면 CSV 파일에 아래 형식으로 한 줄이 추가된다.

```csv
saved_at,id,shop_link
2026-05-07 20:05:33,2697,https://www.musinsa.com/products/4714964
```

기본 CSV 경로:

- `./image_update/selected_shop_links.csv`

### 주요 환경변수

`.env`에서 아래 값을 읽는다.

```env
SUPABASE_URL=...
SUPABASE_KEY=...
IMAGE_VIEWER_TABLE=clothes
IMAGE_VIEWER_ID_COLUMN=id
IMAGE_VIEWER_NAME_COLUMN=name
IMAGE_VIEWER_IMAGE_COLUMN=image_url
IMAGE_VIEWER_SHOP_LINK_COLUMN=shop_link
IMAGE_VIEWER_PAGE_SIZE=50
IMAGE_VIEWER_HOST=127.0.0.1
IMAGE_VIEWER_PORT=8765
IMAGE_VIEWER_CSV_PATH=./image_update/selected_shop_links.csv
```

값을 따로 지정하지 않으면 코드의 기본값을 사용한다.

### 실행 방법

```powershell
python ./image_update/view_db_images.py
```

### 주의점

- `image_url`이 비어 있지 않은 row만 조회한다.
- 정렬 기준은 `id` 오름차순이다.
- `shop_link`가 없는 row는 저장 버튼이 비활성화된다.
- CSV는 append 방식이라 같은 `id`가 여러 번 저장될 수 있다.


## 2. 🔄 `convert_png_to_jpg.py`

📁 파일 위치:
- [convert_png_to_jpg.py](./image_update/convert_png_to_jpg.py)

### 목적

`DB_data/image` 폴더 안의 이미지 파일들을 JPG로 변환하고, 파일명을 CSV에 기록된 `id`로 맞춘다.

현재 설정:

- 최대 픽셀: `700`
- JPEG 품질: `90`
- 지원 확장자: `.png`, `.jpg`, `.jpeg`

### 입력 데이터

1. 입력 이미지 폴더
   - `./image`
2. CSV 파일
   - `./image_update/selected_shop_links.csv`

이 스크립트는 CSV 전체를 읽지만, 실제 대상은 `쉼표가 2개인 줄`만 사용한다.

즉 CSV row 기준으로는 `컬럼 3개`인 줄만 처리 대상이다.

예:

```csv
2026-05-07 20:05:33,2697,https://www.musinsa.com/products/4714964
```

위 줄에서는 두 번째 값인 `2697`을 파일명으로 사용한다.

### 처리 규칙

- `len(row) == 3`인 줄만 대상
- 대상 줄의 두 번째 값(`id`)을 순서대로 수집
- 입력 이미지 파일도 파일명 기준 정렬 후 순서대로 처리
- 첫 번째 이미지 -> 첫 번째 `id`
- 두 번째 이미지 -> 두 번째 `id`

변환 결과 파일명 예:

```text
2697.jpg
```

### 출력 폴더

- `./image_jpg_700`

### 실행 방법

```powershell
python ./image_update/convert_png_to_jpg.py
```

### 주의점

- 이 스크립트는 CSV를 수정하지 않는다.
- CSV의 `id` 순서와 이미지 파일 순서가 정확히 맞아야 한다.
- `쉼표가 3개인 줄`, 즉 이미 추가 컬럼이 있는 row는 대상에서 제외된다.
- 이미지 개수가 대상 `id` 개수보다 많으면 에러를 낸다.

### 적합한 사용 시점

다음과 같은 경우에 사용한다.

- 사람이 직접 캡처한 이미지를 storage 업로드 전 정리할 때
- PNG/JPG 혼합 파일을 모두 700px JPG로 통일할 때
- DB의 `id` 기준으로 파일명을 통일할 때


## 3. 🔗 `append_storage_urls_to_csv.py`

📁 파일 위치:
- [append_storage_urls_to_csv.py](./image_update/append_storage_urls_to_csv.py)

### 목적

CSV에서 아직 URL이 없는 줄에 대해, 마지막 컬럼에 Supabase Storage 공개 URL을 자동으로 추가한다.

이 스크립트는 이미지 파일을 변환하지 않는다. CSV만 수정한다.

### URL 생성 규칙

기본 URL 형식:

```text
https://<supabase-project>.supabase.co/storage/v1/object/public/image/image_capture/<id>.jpg
```

예:

```text
https://luokxiiyouqoybyljooa.supabase.co/storage/v1/object/public/image/image_capture/2697.jpg
```

### 환경변수

`.env`에서 아래 값을 읽는다.

```env
SUPABASE_URL=https://luokxiiyouqoybyljooa.supabase.co
STORAGE_BASE_URL=https://luokxiiyouqoybyljooa.supabase.co/storage/v1/object/public/image/image_capture
```

우선순위:

1. `STORAGE_BASE_URL`가 있으면 그대로 사용
2. 없으면 `SUPABASE_URL`로 아래 경로를 자동 조합

```text
/storage/v1/object/public/image/image_capture
```

### 처리 대상

이 스크립트는 `쉼표가 2개인 줄`, 즉 `len(row) == 3`인 row만 처리한다.

예를 들어 아래 row:

```csv
2026-05-07 20:05:33,2697,https://www.musinsa.com/products/4714964
```

실행 후:

```csv
2026-05-07 20:05:33,2697,https://www.musinsa.com/products/4714964,https://luokxiiyouqoybyljooa.supabase.co/storage/v1/object/public/image/image_capture/2697.jpg
```

반대로 이미 `쉼표가 3개`인 줄은 건너뛴다.

### 실행 방법

```powershell
python ./image_update/append_storage_urls_to_csv.py
```

### 주의점

- 이 스크립트는 CSV 파일을 직접 덮어쓴다.
- 이미 4컬럼인 줄은 수정하지 않는다.
- `id`가 비어 있는 줄은 건너뛴다.
- URL은 항상 `<id>.jpg` 형식으로 붙는다.


## ✅ 권장 작업 순서

캡처 이미지를 새로 DB에 반영하는 흐름은 아래 순서를 권장한다.

1. `view_db_images.py`로 대상 row의 `id`, `shop_link`를 CSV에 모은다.
2. 필요한 이미지를 `./image` 폴더에 넣는다.
3. `convert_png_to_jpg.py`를 실행해 `id.jpg` 파일로 변환한다.
4. 변환된 JPG를 Supabase Storage의 `image/image_capture` 경로에 업로드한다.
5. `append_storage_urls_to_csv.py`를 실행해 CSV 마지막 컬럼에 storage URL을 붙인다.
6. 이후 `update_image_urls_from_csv.py`로 DB의 `image_url`을 교체한다.


## 🧾 CSV 상태별 의미

### 3컬럼 row

```csv
saved_at,id,shop_link
```

의미:

- `view_db_images.py`에서 저장만 된 상태
- 아직 storage URL이 붙지 않은 상태
- `convert_png_to_jpg.py`, `append_storage_urls_to_csv.py`의 처리 대상

### 4컬럼 row

```csv
saved_at,id,shop_link,storage_url
```

의미:

- storage URL까지 붙은 상태
- `append_storage_urls_to_csv.py`는 이 row를 건너뜀
- 이후 DB update 스크립트 입력으로 사용 가능


## ▶️ 실행 예시

```powershell
python ./image_update/view_db_images.py
python ./image_update/convert_png_to_jpg.py
python ./image_update/append_storage_urls_to_csv.py
```
