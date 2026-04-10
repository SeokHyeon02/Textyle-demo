## 🚀 프로젝트 실행 방법

### 1. ⚙️ 준비 사항 (Setup)
- **환경 변수 설정:** `Textyle-vectorserver` 폴더 안에 `.env` 파일을 새로 생성한 후, 공유받은 API 키/코드를 붙여넣습니다.
- `Textyle-app` 폴더 안에 `env` 파일을 새로 생성한 후, 공유받은 API 키/코드를 붙여넣습니다.
- **IP 주소 변경:** `Textyle-app/app/(tabs)/index.tsx` 파일을 열고, 코드 내의 IP 주소를 본인 컴퓨터의 현재 IP로 변경합니다.

### 2. ▶️ 프로젝트 실행 (Run)
> **💡 팁:** 앱과 서버가 동시에 돌아가야 하므로, **터미널(명령 프롬프트)을 2개** 열어서 각각 실행해 주세요.

**[터미널 1] 모바일 앱 실행**
- `Textyle-app` 폴더 경로에서 아래 명령어를 실행합니다.  
- npx expo start  
**[터미널 2] AI/벡터 서버 실행**
- `Textyle-vectorserver` 폴더 경로에서 아래 명령어를 실행합니다.
- uvicorn main:app --host 0.0.0.0 --port 8000 --reload
