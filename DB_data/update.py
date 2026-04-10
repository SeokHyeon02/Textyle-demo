import os
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client
from dotenv import load_dotenv

# -------------------------------------------------------------
# 1. 설정 및 DB 연결
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # service_role 키 확인 필수!
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------------------
# 2. AI 모델 로드
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
print(f"⏳ AI 모델을 불러오는 중입니다... (사용 장치: {device})")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def update_all_embeddings():
    print("🔍 Supabase에서 데이터를 가져옵니다 (1000개씩 분할 로드)...")
    
    all_items = []
    start = 0
    limit = 1000  
    
    while True:
        # 💡 수정됨: 'id'를 빼고 'name'과 'image_url'만 가져옵니다.
        response = supabase.table("clothes").select("name, image_url").range(start, start + limit - 1).execute()
        
        data = response.data
        if not data:
            break  
            
        all_items.extend(data)
        print(f" 📥 누적 {len(all_items)}개 데이터 로드 완료...")
        
        if len(data) < limit:
            break  
            
        start += limit

    if not all_items:
        print("업데이트할 데이터가 없습니다.")
        return

    print(f"\n🚀 총 {len(all_items)}개의 데이터 업데이트를 시작합니다!\n")

    for index, item in enumerate(all_items, 1):
        name = item.get("name", "이름 없음")
        image_url = item["image_url"] # 💡 식별자로 사용할 이미지 URL
        
        print(f"[{index}/{len(all_items)}] 업데이트 중: {name}")
        
        try:
            # 1. 이미지 다운로드
            img_res = requests.get(image_url, timeout=10)
            img_res.raise_for_status()
            image = Image.open(BytesIO(img_res.content)).convert("RGB")

            # 2. 순수 이미지 벡터 추출
            inputs = processor(
                text=[""],
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                embedding_list = image_features.squeeze().tolist()

            # 3. 💡 수정됨: image_url이 일치하는 데이터의 embedding 컬럼만 업데이트
            supabase.table("clothes").update({
                "embedding": embedding_list
            }).eq("image_url", image_url).execute()
            
        except Exception as e:
            print(f"❌ [{name}] 처리 중 오류 발생: {e}")

    print("\n✅ 3000여 개 데이터의 임베딩 업데이트가 모두 완료되었습니다!")

if __name__ == "__main__":
    update_all_embeddings()