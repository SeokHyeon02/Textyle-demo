# Textyle-demo-edit 코드 수정 내역 총정리

본 문서는 통합 테스트 및 엣지 케이스(예외 상황) 해결을 위해 `Textyle-demo-edit` (프론트엔드 및 파이썬 백엔드) 폴더 내에 적용된 모든 수정 사항을 기록합니다. 향후 발표 및 디펜스 자료로 활용할 수 있습니다.

---

## 📱 1. 프론트엔드 (`Textyle-app`) 수정 사항

### 1.1 이미지 압축 및 크롭 기능 활성화 (UX 및 성능 개선)
*   **파일:** `app/(tabs)/index.tsx`
*   **변경 내용:** `Expo ImagePicker` 설정에 `allowsEditing: true`와 `quality: 0.6` 옵션을 추가했습니다.
*   **도입 목적:** 
    *   사용자가 옷 이미지를 업로드할 때 불필요한 배경을 잘라낼 수 있게 하여(크롭), AI(GroundingDINO)가 타겟 의상을 더 정확하게 인식할 수 있도록 돕습니다.
    *   최신 스마트폰의 고화질(10MB 이상) 사진을 60% 품질로 압축하여 백엔드로 전송함으로써, 네트워크 지연(Latency)을 줄이고 서버 메모리 과부하를 방지합니다.

### 1.2 서버 에러 핸들링 강화 (앱 크래시 방지)
*   **파일:** `app/(tabs)/index.tsx`
*   **변경 내용:** 서버 응답을 파싱하는 `JSON.parse(response.body)` 코드를 `try-catch` 블록으로 감쌌습니다.
*   **도입 목적:** 백엔드(스프링 부트 또는 파이썬)에서 500 에러 등으로 인해 JSON이 아닌 일반 HTML 에러 페이지를 반환할 경우, 프론트엔드 앱이 파싱 에러로 강제 종료(Crash)되는 현상을 방어합니다.

### 1.3 통신 엔드포인트(API Gateway) 변경
*   **파일:** `.env`
*   **변경 내용:** `EXPO_PUBLIC_FASHION_API_URL` 값을 파이썬 서버(`http://localhost:8001`)에서 스프링 부트 라우터(`http://localhost:8080/api`)로 변경했습니다.
*   **도입 목적:** MSA(Microservices Architecture) 구조로 전환함에 따라, 모바일 앱이 백엔드의 단일 진입점(API Gateway)인 스프링 부트만을 바라보도록 설정했습니다.

---

## 🐍 2. AI 백엔드 (`Textyle-vectorserver`) 수정 사항

### 2.1 프롬프트 인젝션 방어 로직 완화 (자연어 검색 개선)
*   **파일:** `fashion_main.py`
*   **변경 내용:** `prompt_injection_pattern` 정규식에서 "무시하고", "이전 명령" 등 지나치게 엄격했던 한국어 금지 키워드를 삭제했습니다.
*   **도입 목적:** 사용자가 "꽃무늬는 무시하고 단색으로 찾아줘"와 같은 정상적인 패션 검색 조건을 입력했을 때, 이를 악의적인 공격으로 오인하여 검색을 차단하는 부작용을 해결했습니다.

### 2.2 제로샷 이미지 유효성 검증(CLIP) 엣지 케이스 해결
*   **파일:** `fashion_main.py`
*   **변경 내용:**
    1.  `FASHION_IMAGE_LABELS` 배열에 `shorts`, `leggings`, `underwear` 등 하의 카테고리를 대폭 추가했습니다.
    2.  `NON_FASHION_IMAGE_LABELS`의 항목을 단순한 `"face"`에서 `"a close-up photo of a human face"`로 구체화했습니다.
    3.  `MIN_FACE_REJECT_SCORE` (얼굴 인식 거절 임계값)를 `0.18`에서 `0.25`로 상향 조정했습니다.
*   **도입 목적:** 살구색이나 베이지색 반바지를 입력했을 때 AI가 이를 사람의 피부나 얼굴로 오인하여 검색을 강제 종료하던 치명적인 버그를 완벽하게 수정했습니다. 이제 살구색 반바지 이미지도 정상적인 패션 아이템으로 인식되어 성공적으로 검색됩니다.


---

## 🔐 3. 로그인 및 북마크 시스템 전환 (Supabase -> Spring Boot)

### 3.1 회원가입 및 로그인 통신 경로 스프링 부트로 우회
*   **파일:** `app/signup.tsx`, `app/(tabs)/login.tsx`
*   **변경 내용:** 기존 프론트엔드에서 Supabase로 직접 요청하던 회원가입/로그인 코드를 전면 삭제하고, 스프링 부트 라우터(`/api/auth/join`, `/api/auth/login`)로 `fetch` 요청을 보내도록 변경했습니다. 구글 소셜 로그인 기능은 복잡도 최소화를 위해 삭제했습니다.
*   **도입 목적:** 캡스톤 프로젝트에서 백엔드 팀원의 역할과 기여도를 강조하기 위해, 인증 및 인가 책임을 프론트엔드가 아닌 스프링 부트 서버로 완전히 이관했습니다.

### 3.2 로컬 토큰 스토리지(AsyncStorage) 도입
*   **파일:** `app/(tabs)/login.tsx`
*   **변경 내용:** 스프링 부트 로그인 성공 시 반환받은 JWT `accessToken`, `userEmail`, `userNickname`을 React Native의 `@react-native-async-storage/async-storage`를 이용해 기기에 안전하게 저장하도록 변경했습니다. 

### 3.3 북마크(찜) 기능 스프링 부트 API와 통합
*   **파일:** `lib/bookmarks.ts`
*   **변경 내용:** 직접 Supabase DB(`user_bookmarks` 테이블)를 쿼리하던 코드를 완전히 삭제하고, 저장된 JWT 토큰을 Header에 담아 스프링 부트의 `/api/bookmarks` 엔드포인트로 통신하도록 모든 API 함수를 재작성했습니다.
*   **도입 목적:** 북마크 기능 역시 스프링 부트의 커스텀 비즈니스 로직(BookmarkController)을 거치도록 설계하여 완벽한 백엔드 통합을 이뤄냈습니다.

### 3.4 비로그인(게스트) UI 및 토큰 갱신 버그 해결
*   **파일:** `app/(tabs)/bookmarks.tsx`, `app/(tabs)/index.tsx`
*   **변경 내용:** 
    *   기존 Supabase의 세션 검사(`supabase.auth.getSession`)를 완전히 제거하고, `AsyncStorage`를 검사하는 커스텀 로직으로 대체했습니다.
    *   로그아웃 시 발생하던 캐시 고착화 버그를 해결하기 위해, 화면에 진입할 때마다 토큰 유무를 실시간으로 재검사하는 `useFocusEffect`를 적용했습니다.
*   **도입 목적:** 토큰 만료 또는 로그아웃 시 401 에러(Failed to fetch)가 발생하는 대신, 이미 구축되어 있던 훌륭한 "게스트용 로그인 유도 UI"가 즉시 사용자에게 올바르게 노출되도록 화면 생명주기(Lifecycle) 논리 회로를 수정했습니다.
