# FashionCLIP Denim Search Experiments

## 목적

데님팬츠 검색에서 FashionCLIP 임베딩 결과가 시각적으로는 가까워도 사용자가 기대하는 데님 톤과 어긋나는 문제가 있었다. 특히 연청, 중청, 진청, 흑청은 모두 데님/팬츠 문맥으로 묶이기 때문에 단순 벡터 유사도만으로 최종 순위를 정하면 색상 의도가 약하게 반영된다.

이 문서는 현재 데님 검색 보정 방향과 실험에서 폐기한 보조 스크립트를 정리한다.

## 관찰한 문제

- `청바지`, `데님`, `데님팬츠` 쿼리는 카테고리 일치는 잘 되지만 톤 차이가 검색 순위에 충분히 반영되지 않았다.
- 이미지에서 추출한 회색 계열이 실제로는 워싱된 블루 데님인 경우가 많았다.
- `black`, `gray`, `navy`, `indigo`, `darkblue`는 일반 색상 검색에서는 다른 색상처럼 보이지만 데님 문맥에서는 모두 진청/흑청 후보군으로 비교해야 하는 경우가 있었다.
- 제품명에 `연청`, `진청`, `흑청`, `raw denim`, `one wash`, `light washed` 같은 힌트가 있으면 이미지 픽셀보다 더 신뢰할 수 있는 경우가 있었다.

## 현재 적용 방향

### 1. 데님 문맥 감지

`Textyle-vectorserver/fashion_color_extraction.py`와 `Textyle-vectorserver/fashion_main.py`는 다음 표현을 데님 문맥으로 취급한다.

- `denim`, `jean`, `jeans`
- `데님`, `청바지`
- `흑청`, `진청`, `중청`, `연청`
- `raw denim`, `dark denim`

데님 문맥이 켜지면 일반 색상 분류보다 데님 전용 톤 보정이 우선된다.

### 2. 이미지 색상 후보 확장

색상 추출 결과는 단일 `color`만 쓰지 않고 후보 리스트와 검색 가중치를 함께 전달한다.

- `color_candidates`: 후보 색상, 점수, source, confidence, RGB, named color
- `search_color_weights`: 대표색과 인접 색상군의 검색 가중치
- `color_reason`: `denim_context_light`, `denim_context_medium`, `denim_context_dark` 같은 판단 근거

이 정보는 `/search` 응답의 `query_image_attributes`와 ranking debug 로그에 노출된다.

### 3. named color 기반 미세 보정

데님 검색에서는 최종 11개 색상 그룹만으로 비교하지 않고 PIL named color와 CIELab 거리를 함께 사용한다.

- `lightblue`는 연청 후보로 본다.
- `steelblue`, `royalblue`는 중청 후보로 본다.
- `navy`, `darkblue`, `midnightblue`, `black`, `gray` 계열은 진청/흑청 비교 후보로 본다.
- 회색으로 잡힌 업로드 이미지라도 데님 문맥과 blue bias가 있으면 `lightblue`, `steelblue`, `navy`, `midnightblue`로 재해석한다.

### 4. 제품명 기반 톤 보정

후보 상품의 `name`, `brand_name`, `sub_category`에서 톤 힌트를 찾는다.

- light: `light blue`, `light denim`, `light washed`, `연청`, `라이트 블루`, `밝은청`
- dark: `dark blue`, `dark denim`, `raw denim`, `one washed`, `진청`, `흑청`, `생지`, `오일 블랙`
- medium: named color가 `steelblue` 또는 `royalblue`에 가까운 경우

쿼리 이미지가 같은 색상 요청(`same`)이고 데님 톤이 다르면 순위 점수를 감점한다.

### 5. dark denim 예외 처리

진청/흑청 요청은 일반 색상 그룹 비교로는 너무 쉽게 누락된다. 현재 로직은 다음 타입을 구분한다.

- `washed_gray`: 그레이 워싱 또는 회색 계열 흑청
- `dark_indigo`: navy, darkblue, midnightblue, dark indigo 계열
- `black_only`: 블랙 데님이지만 워싱/인디고 힌트가 약한 후보
- `light_mismatch`: 연청 후보라서 진청/흑청 요청과 충돌

`washed_gray`와 `dark_indigo`는 소폭 가산하고, `light_mismatch`는 감점한다.

## 검색 서버 변경점 요약

- 빈 검색어도 이미지 단독 검색으로 허용한다.
- 앱은 하드코딩된 `SERVER_IP` 대신 `EXPO_PUBLIC_FASHION_API_URL`을 사용한다.
- FashionCLIP 검색 서버는 같은 색상 검색에서 후보 수를 늘리고, 결과가 부족하면 threshold를 한 번 낮춘다.
- 디자인 유사도 모드에서는 색상 보정보다 이미지 임베딩과 디자인 디테일을 우선한다.
- debug 로그에 `query_denim_tone`, `target_color_weights`, `candidate_named_color`, `dark_denim_match_type` 등을 출력한다.

## 폐기한 보조 스크립트

다음 파일은 임시 실험 또는 CSV 추출용으로만 사용했고 현재 검색/업데이트 경로에는 필요하지 않아 제거했다.

- `DB_data/test/check_product_name_color_hint.py`
- `DB_data/update/export_denim_pants_rows.py`

제품명 색상 힌트 검증은 `verify_groundingdino_sam_color_extraction.py`의 `infer_color_hint` 흐름과 서버 debug 로그로 확인한다. 데님팬츠 데이터 추출은 운영 Supabase 테이블에 직접 접근하는 임시 스크립트 대신, 필요한 컬럼과 대상 조건을 먼저 확인한 뒤 별도 일회성 쿼리로 수행한다.

## 남은 확인 항목

- 실제 Supabase 결과에서 `color_candidates`가 JSON 배열로 안정적으로 저장되는지 확인한다.
- 연청 이미지가 회색이나 흰색으로 과도하게 이동하지 않는지 샘플별 ranking 로그를 비교한다.
- 흑청 요청에서 순수 블랙 팬츠가 데님 후보보다 앞서는 케이스가 있는지 확인한다.
- 패턴 데님은 색상 추출과 분리해 ViT 패턴 분류 대상으로 계속 검증한다.
