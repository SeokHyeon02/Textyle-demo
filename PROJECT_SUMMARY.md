# Textyle 프로젝트 진행 요약

## 1. 프로젝트 개요

Textyle은 사용자가 의류 이미지를 업로드하고 한국어 자연어 검색어를 입력하면, 이미지와 문장을 함께 분석해 유사하거나 조건에 맞는 패션 상품을 찾아주는 모바일 패션 검색 서비스입니다.

예시 검색:

- `같은 색 와이드 청바지`
- `색상이 다른 검정 와이드 청바지`
- `검정 말고 다른 색 청바지`
- `비슷한 디자인의 레더자켓`

핵심 목표는 단순 텍스트 검색이 아니라, **이미지의 디자인/색상 정보와 사용자의 자연어 조건을 함께 반영하는 패션 검색 품질 개선**입니다.

## 2. 전체 구조

프로젝트는 크게 4개 영역으로 구성되어 있습니다.

| 영역 | 경로 | 역할 |
| --- | --- | --- |
| 모바일 앱 | `Textyle-app` | Expo/React Native 기반 이미지 업로드, 검색어 입력, 결과 표시 |
| 벡터 검색 서버 | `Textyle-vectorserver` | FastAPI, FashionCLIP, Gemini, 색상 추출, Supabase 검색 |
| DB/이미지 스크립트 | `DB_data` | 상품 이미지 관리, 색상 추출, DB 컬럼 업데이트 보조 |
| 인증 서버 | `Textyle-serviceserver` | Supabase Auth 기반 사용자 인증 보조 |

검색 흐름:

1. 사용자가 앱에서 이미지와 검색어를 입력합니다.
2. 앱이 FastAPI 서버 `/search`로 multipart 요청을 보냅니다.
3. 서버가 Gemini/rule 기반으로 검색 의도를 분석합니다.
4. FashionCLIP으로 이미지/텍스트 임베딩을 생성합니다.
5. Supabase RPC `match_clothes_fashion`으로 후보 상품을 가져옵니다.
6. 색상, 카테고리, 소재, 핏, 데님 톤 기준으로 재정렬합니다.
7. 앱에 결과와 검색 메타데이터를 반환합니다.

## 3. 주요 구현 내용

### 3.1 모바일 앱

- Expo Router 기반 탭 화면 구성
- 이미지 선택 및 업로드 기능 구현
- 한국어 검색어 입력 UI 구현
- 검색 결과 리스트 표시
- Supabase 연동 기반 북마크/로그인 흐름 구성
- 서버 응답의 검색 메타데이터 표시 구조 검토
  - `enhanced_query`
  - `color_extracted`
  - `design_description`
  - 검색 버전 표시 구조

### 3.2 벡터 검색 서버

활성 서버는 `Textyle-vectorserver/fashion_main.py`이며 기본 포트는 `8001`입니다.

구현된 주요 기능:

- FashionCLIP 이미지 임베딩 생성
- FashionCLIP 텍스트 임베딩 생성
- 이미지+텍스트 임베딩 가중합 검색
- Gemini 기반 자연어 의도 분석
- Gemini 실패 시 rule-based fallback
- 한국어 색상/소재/카테고리/핏 alias 처리
- Supabase RPC 검색 연동
- 검색 결과 reranking
- 비패션 이미지 거부 로직
- 검색 디버그/메타데이터 응답

## 4. 검색 품질 개선 과정

### 4.1 v1 방식

초기 방식은 이미지 임베딩과 텍스트 임베딩을 가중합해 검색했습니다.

장점:

- 이미지의 디자인, 실루엣, 형태 유사도를 잘 반영
- “비슷한 디자인” 검색에 유리

문제:

- 색상을 바꾸는 쿼리에서 이미지 색상이 검색 벡터에 계속 섞임
- 예: 파란 청바지 이미지를 올리고 `색상이 다른 검정 청바지`를 검색해도 파란 계열이 남는 문제

### 4.2 조원이 만든 v2 방식 검토

조원이 만든 v2 방식은 이미지 임베딩을 제거하고 CLIP 텍스트 인코더 중심으로 검색하는 구조였습니다.

핵심 아이디어:

- 이미지 임베딩 제거
- K-means로 색상 추출
- Gemini에는 색상을 다시 추측시키지 않고 디자인/실루엣/소재만 설명하게 함
- 상세 색상(`navy`, `burgundy`, `olive`, `charcoal` 등)을 `enhanced_query`에 사용
- Gemini 응답으로 비패션 이미지 검증

검토 결과:

- 색상 교체 검색에는 장점이 있음
- 하지만 이미지 디자인 유사도가 약해질 수 있음
- 따라서 v1을 완전히 버리기보다, **이미지 임베딩은 유지하되 색상 교체 상황에서 영향도를 줄이는 hybrid 방식**이 적합하다고 판단했습니다.

## 5. 최종 적용한 검색 전략

현재 적용 방향은 v1과 v2의 장점을 합친 hybrid 방식입니다.

### 5.1 같은 색 검색

예: `같은 색 와이드 청바지`

처리 방식:

- 업로드 이미지에서 색상을 추출
- broad color와 detailed color를 함께 사용
- `same` 검색에서는 detailed color를 `enhanced_query`에 반영
- 이미지 임베딩은 유지하여 디자인 유사도도 반영

예시:

```text
enhanced_query = "a photo of gainsboro wide-leg denim jeans"
```

### 5.2 다른 색 검색

예: `색상이 다른 와이드 청바지`

처리 방식:

- 업로드 이미지 색상은 피해야 할 색으로 사용
- 이미지 임베딩의 색상 영향도를 줄이기 위해 image weight 감소
- 디자인/카테고리는 유지

적용 가중치:

| 상황 | text_weight | image_weight |
| --- | ---: | ---: |
| 다른 색, 디자인 조건 없음 | 0.55 | 0.45 |
| 다른 색, 디자인 조건 있음 | 0.60 | 0.40 |

### 5.3 다른 색 + 명시 색상 검색

예: `색상이 다른 검정 와이드 청바지`

기존 문제:

- `검정`을 찾을 색이 아니라 피해야 할 색으로 잘못 처리할 수 있었음

수정 후:

- 업로드 이미지 색상은 avoid color
- 사용자가 말한 `검정`은 target color
- 검정 청바지를 상위로 올림

예시:

```text
enhanced_query = "a photo of black wide denim jeans"
```

### 5.4 제외형 문장 처리

예:

- `검정 말고 다른 색 청바지`
- `블랙 제외 와이드 청바지`
- `검정 빼고 청바지`

수정 내용:

- `말고`, `빼고`, `제외`, `except`, `without` 같은 제외 표현 감지
- 이 경우 명시 색상을 target이 아니라 `excluded_color`로 처리
- 제외색은 정확히 해당 색만 피하도록 조정

결과:

- `색상이 다른 검정 청바지`와 `검정 말고 다른 색 청바지`를 서로 다르게 처리할 수 있게 됨

### 5.5 카테고리 불가능 조합 보정

Gemini가 자연어를 해석하는 과정에서 서로 맞지 않는 카테고리 조합을 만들 수 있었습니다.

실제 발생 예:

```text
main_category = 아우터
sub_category = 데님팬츠
```

문제:

- `데님팬츠`는 DB 기준 `하의`에 속함
- 그런데 main category가 `아우터`로 들어가면 검색 필터가 아우터로 제한됨
- 결과적으로 청바지를 찾아야 하는 쿼리에서 윈드브레이커, 가디건, 후드집업 등이 반환됨

수정 내용:

- sub category의 소속 main category를 기준으로 불가능한 조합을 보정
- `데님팬츠`가 있으면 main category를 `하의`로 강제 보정
- 향후 다른 sub category에도 같은 방식으로 적용 가능

예시:

```text
아우터 + 데님팬츠 -> 하의 + 데님팬츠
```

### 5.6 디자인+색상 동시 유사 검색 보정

실제 발생 예:

```text
이 디자인과 색상이 비슷한 청바지 찾기
```

문제:

- `이 디자인` 표현 때문에 design similarity mode가 켜짐
- 이 모드에서는 같은 색상 보정이 일반 `same` 검색보다 약하게 적용됨
- `gray` 데님 후보의 색상 점수가 높아도 보정값이 작아져 blue/indigo/black 계열이 위로 올라옴
- `similar` 같은 지시어가 디자인 설명으로 남아 `enhanced_query`에 섞임
- 디버그 로그에서 실제 색상 타겟이 있는데도 `query_color_targets=inactive`로 표시됨

수정:

- `same + design similarity` 검색에서는 색상을 약한 참고값이 아니라 주요 조건으로 반영
- 디자인 유사 모드에서는 `gray` 데님 후보 색상 보너스를 0.25배로 줄이지 않도록 조정
- `similar` 지시어를 디자인 토큰에서 제거
- 디버그 로그가 실제 `query_color_targets`를 출력하도록 수정

검증:

- `test_design_and_same_color_gray_denim_prefers_gray_candidate` 추가
- `test_sanitize_design_terms_removes_similarity_instruction` 추가
- 관련 테스트 78개 통과

## 6. 색상 추출 및 DB 색상 개선

### 6.1 색상 구조

검색 품질을 위해 색상을 두 단계로 관리합니다.

| 구분 | 예시 | 용도 |
| --- | --- | --- |
| broad color | `blue`, `black`, `gray`, `white` | 필터링, reranking, 다른 색 판별 |
| detailed color | `navy`, `charcoal`, `gainsboro`, `darkslategray` | 같은 색 검색 정밀도 향상 |

### 6.2 청바지 색상 처리

청바지는 일반 색상보다 톤 구분이 중요해 별도 `denim_tone`을 사용합니다.

예시:

- `light_blue`
- `mid_blue`
- `dark_blue`
- `indigo`
- `black`
- `gray`
- `white`

검색에서 활용:

- `연청`, `중청`, `진청`, `흑청`, `생지`, `인디고` 검색 구분
- 검정/인디고/진청이 서로 잘못 섞이는 문제 완화

추가 보정:

- `연청`, `중청`, `흑청`은 원문 쿼리에서 최우선 하드코딩 처리
- Gemini가 `중청`을 `light blue`처럼 잘못 해석해도 원문 기준을 우선함
- `gray` 청바지 검색에서 `black` 데님 톤 보너스를 제거해 검정 계열 과다 노출을 완화
- `연청` 검색에서는 `indigo`, `black`, `dark_blue` 톤을 강하게 감점하고 broad `blue` 보너스를 축소해 딥 인디고 상위 노출을 완화
- `중청` 검색에서는 `indigo`, `black`, `dark_blue` 톤을 감점하고 broad `blue` 보너스를 축소해 딥 인디고 상위 노출을 완화
- `진청`은 `dark_blue` 톤으로 별도 인식하며, 중청보다 어둡고 흑청보다 밝은 데님 톤으로 점수 구간을 분리
- DB 재검수는 전체 청바지 재추출 대신 `인디고/생지/raw/one wash` 포함 데님팬츠만 선별하는 `DB_data/update/update_indigo_denim_colors.py` dry-run 스크립트를 추가
- 제품명 톤 파싱에서 `인디고 미듐`은 `mid_blue`, `인디고 다크`는 `dark_blue`, `딥 인디고/raw/생지`는 `indigo`로 분리
- 회색 워싱 검색에서는 `washed/slub` 단어만으로 `washed_gray` 보너스를 주지 않고, 실제 gray 후보색 또는 충분한 gray 이미지 비율이 있을 때만 인정
- 회색 same-color 데님 검색에서는 gray 근거가 약한 `indigo/navy/darkblue` 후보만 별도로 분리해 dark-indigo 보너스를 제거하고 색상 group match 점수를 축소

### 6.3 DB 색상 업데이트 검수

청바지 DB 색상은 이전에 업데이트되었고, 이번 작업에서는 검수용 TSV를 생성해 확인했습니다.

검수 파일:

```text
DB_data/update/color_review_sample_denim.tsv
```

30개 샘플 결과:

| 항목 | 결과 |
| --- | --- |
| 전체 행 수 | 30 |
| 필수 색상 필드 누락 | 없음 |
| `color_confidence` | 전부 `high` |
| `dominant_color` 분포 | blue 15, black 10, white 4, gray 1 |
| `denim_tone` 분포 | black 10, indigo 7, light_blue 5, white 4, mid_blue 3, gray 1 |

주의 행:

- 베이지/에크루/크림 계열은 현재 broad color 체계에서 `white`로 묶임
- 나중에 `beige`를 독립 색상으로 다룰지 결정 필요

## 7. API 평가 결과

청바지 전용 평가셋을 추가했습니다.

파일:

```text
Textyle-vectorserver/evaluation/denim_query_quality_cases.json
```

평가 케이스:

- 같은 색 청바지
- 다른 색 청바지
- 다른 색 + 검정 target
- 검정 제외
- 블랙 제외 + 와이드 청바지

실행 결과:

| 항목 | 결과 |
| --- | ---: |
| 평가 케이스 | 5 |
| intent accuracy | 1.00 |
| attribute match rate | 0.88 |
| error rate | 0.00 |

결과 파일:

```text
Textyle-vectorserver/evaluation/query_quality_results_denim_checks_after_exclusion.csv
Textyle-vectorserver/evaluation/query_quality_summary_denim_checks_after_exclusion.md
```

## 8. 테스트 현황

주요 테스트:

- 검색 reranking 단위 테스트
- 색상 추출 metadata 테스트
- query quality 평가 로더/메트릭 테스트
- DB 색상 검수 TSV export 테스트
- 실제 `/search` API 평가

통과한 테스트:

```text
test_search_ranking
test_fashion_color_extraction
test_query_quality_eval
test_export_color_review_sample
```

검증 내용:

- `same` 검색에서 detailed color 사용
- `different` 검색에서 이미지 색상 영향 감소
- `different + explicit target` 처리
- `excluded color` 문장 처리
- 청바지 tone reranking
- `연청`, `중청`, `흑청` 원문 우선 처리
- `아우터 + 데님팬츠` 같은 불가능한 카테고리 조합 보정
- 색상 metadata 응답
- 검수 TSV 생성 로직

## 9. 현재까지의 핵심 성과

- 이미지+텍스트 기반 패션 검색 서버 구현
- 한국어 자연어 검색 의도 분석 구현
- Supabase 벡터 검색 연동
- FashionCLIP 기반 이미지/텍스트 임베딩 검색 구현
- 검색 결과 reranking 구조 구축
- 색상/소재/카테고리/핏 alias 정리
- 청바지 색상 tone 검색 개선
- 잘못된 LLM 카테고리 조합 보정
- `같은 색`, `다른 색`, `검정 제외` 같은 실제 사용자 표현 처리
- v2 방식 검토 후 hybrid 방식으로 개선
- 청바지 DB 색상 검수 파일 생성 및 샘플 검증
- API 평가셋과 자동 평가 스크립트 구축

## 10. 남은 작업

우선순위 높은 작업:

1. 청바지 색상 TSV를 100개 이상으로 확대 검수
2. 베이지/크림/아이보리 계열을 `white`로 유지할지 별도 색상으로 분리할지 결정
3. 청바지 외 카테고리로 색상 업데이트 확대
   - 우선 후보: 슬랙스, 코튼 팬츠, 후드티, 맨투맨, 니트
   - 보류 후보: 패턴/프린트/복합 색상 상품
4. 앱 결과 화면에 검색 메타데이터 표시 개선
5. 서버 응답 속도 개선
   - 현재 일부 API 평가에서 응답 시간이 길게 측정됨
   - 모델 로딩, Gemini 호출, 이미지 전처리 구간별 시간 측정 필요

## 11. 현재 결론

현재 Textyle은 단순 이미지 검색이 아니라, **이미지의 디자인 정보와 한국어 자연어 조건을 함께 해석하는 패션 검색 시스템**으로 동작합니다.

특히 이번 개선으로 청바지 검색에서 다음 케이스를 구분할 수 있게 되었습니다.

- 같은 색 청바지 찾기
- 원본 이미지와 다른 색 청바지 찾기
- 원본과 다른 색이면서 검정 청바지 찾기
- 검정은 제외하고 다른 색 청바지 찾기

이는 실제 쇼핑 검색에서 자주 발생하는 색상 조건을 더 정확하게 처리하기 위한 핵심 개선입니다.

## 12. 최근 청바지 색상 보정 추가

- `M톤 인디고`, `인디고 미듐`, `medium indigo`처럼 제품명에 인디고가 포함되어도 실제 의미가 중청/워싱청인 경우 `indigo`가 아니라 `mid_blue`로 검색 보정합니다.
- `인디고 다크`, `dark indigo`는 생지 인디고가 아니라 진청 계열 `dark_blue`로 보정합니다.
- `raw denim`, `raw indigo`, `생지`, `딥 인디고`는 기존처럼 생지 인디고 `indigo`로 유지합니다.
- DB에 과거 `denim_tone=indigo`로 저장된 상품도 서버에서 제품명 modifier를 다시 확인해 잘못된 인디고 보너스를 막습니다.
- 검증: 벡터서버 검색 랭킹 테스트 87개, DB 인디고 재추출/색상 일관성 테스트 8개 통과.
