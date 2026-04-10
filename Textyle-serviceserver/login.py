from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os

# 환경 변수에서 Supabase 접속 정보 로드 (실제 배포 시 .env 파일 사용)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://luokxiiyouqoybyljooa.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1b2t4aWl5b3Vxb3lieWxqb29hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQyMDg1NTIsImV4cCI6MjA4OTc4NDU1Mn0.Zuv1g7gjKGGXIkd73HN7wg_AukFQPjfUlnZrMkr4XxI")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter()

# 앱에서 전달받을 데이터 형식 정의
class SignUpRequest(BaseModel):
    email: str
    password: str
    nickname: str

# -------------------------------------------------------------
# 1. 유저 토큰 검증 미들웨어 (소셜 로그인 / 일반 로그인 모두 사용)
# -------------------------------------------------------------
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    React Native 앱에서 보낸 JWT 토큰을 검증하고 유저 정보를 반환합니다.
    """
    token = credentials.credentials
    try:
        # Supabase를 통해 토큰이 유효한지 확인하고 유저 객체를 가져옵니다.
        # 구글 로그인으로 가입한 유저도 이 로직 하나로 전부 식별 가능합니다.
        user_response = supabase.auth.get_user(token)
        
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
            
        return user_response.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"인증 실패: {str(e)}")


# -------------------------------------------------------------
# 2. 기존 로컬 이메일 회원가입 API
# -------------------------------------------------------------
@router.post("/signup")
async def sign_up(request: SignUpRequest):
    try:
        res = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "nickname": request.nickname
                }
            }
        })
        return {
            "message": "회원가입이 성공적으로 완료되었습니다.", 
            "user_id": res.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------
# 3. 구글 로그인 후 토큰을 통해 유저 정보를 확인하는 API (예시)
# -------------------------------------------------------------
@router.get("/me")
async def get_my_profile(current_user = Depends(get_current_user)):
    """
    앱에서 헤더에 'Authorization: Bearer <토큰>' 을 담아 호출하면
    이 API는 로그인한 유저의 정보(구글 이메일, UUID 등)를 반환합니다.
    """
    
    # current_user.id 에는 구글 로그인 유저의 고유 UUID가 들어있습니다.
    # 나중에 이 ID를 가지고 벡터 DB에 저장된 찜한 옷 등을 검색하면 됩니다.
    
    return {
        "message": "인증된 유저입니다.",
        "user_id": current_user.id,
        "email": current_user.email,
        # 구글 로그인 시 프로필 사진이나 이름 등 메타데이터도 확인할 수 있습니다.
        "metadata": current_user.user_metadata 
    }