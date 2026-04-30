## 🚀 Textyle 프로젝트 실행 방법

### 1. ⚙️ 사전 준비

- `Textyle-vectorserver` 폴더 안에 `.env` 파일을 생성하고 Supabase, Gemini 등 서버에서 사용하는 API 키를 입력합니다.
- `Textyle-app` 폴더 안에 환경 변수 파일이 필요한 경우 생성하고 앱에서 사용하는 API 키를 입력합니다.
- 🌐 `Textyle-app/app/(tabs)/index.tsx` 파일의 `SERVER_IP` 값을 현재 서버를 실행하는 컴퓨터의 IP 주소로 수정합니다.
- 🔌 같은 파일에서 요청 포트도 사용하는 CLIP 모델에 맞게 수정합니다.
  - 기본 CLIP 서버(`main.py`) 사용: `8000`
  - FashionCLIP 서버(`fashion_main.py`) 사용: `8001`

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

- `fashion_main.py`는 Supabase RPC 함수 `match_clothes_fashion`과 `fashion_embedding` 컬럼을 사용합니다.
- 앱 화면이 계속 로딩 중이면 서버 터미널에서 `/search` 요청 로그가 어디까지 출력되는지 확인합니다.
