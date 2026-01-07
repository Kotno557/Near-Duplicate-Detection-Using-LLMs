import os
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# 匯入我們定義好的模組
from prompt_definitions import SYSTEM_PROMPT_TEXT, get_few_shot_messages, encode_image

# ==========================================
# 1. 設定與 Pydantic 結構
# ==========================================
OLLAMA_CONFIG = {
    "base_url": "https://17c6fa445bc9.ngrok-free.app/v1",  # 你的 Ngrok URL (加上 /v1)
    "api_key": "ollama",
    "model": "gpt-oss:120b",  # 記得確認 Ollama list 中的名稱
    "temperature": 0.0
}

class ComparisonResult(BaseModel):
    classification: Literal["Clone", "Near-Duplicate", "Distinct"]
    sub_type: Optional[Literal["None", "Nd1", "Nd2", "Nd3"]] = Field("None")
    confidence_score: int
    reasoning: str

# ==========================================
# 2. 主邏輯
# ==========================================
def analyze_screenshots(img_path_a: str, img_path_b: str):
    print(f"🚀 Connecting to Ollama at {OLLAMA_CONFIG['base_url']}...")
    
    llm = ChatOpenAI(
        base_url=OLLAMA_CONFIG["base_url"],
        api_key=OLLAMA_CONFIG["api_key"],
        model=OLLAMA_CONFIG["model"],
        temperature=OLLAMA_CONFIG["temperature"]
    )
    
    # 綁定結構化輸出 (Structured Output)
    structured_llm = llm.with_structured_output(ComparisonResult)

    # 讀取本次目標圖片
    target_a = encode_image(img_path_a)
    target_b = encode_image(img_path_b)
    
    if not target_a or not target_b:
        print("❌ Error: Target images not found.")
        return

    # --- 構建 Prompt 流程 (Pipeline) ---
    messages = []
    
    # A. 系統提示
    messages.append(SystemMessage(content=SYSTEM_PROMPT_TEXT))
    
    # B. 注入 Few-Shot 範例 (從 prompt_definitions 自動載入)
    # 這一步會自動去讀取 reference_images/ 下的圖片
    messages.extend(get_few_shot_messages(ref_dir="reference_images"))
    
    # C. 放入本次 User 查詢
    messages.append(HumanMessage(
        content=[
            {"type": "text", "text": "Now, analyze these two new screenshots based on the logic above."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b}"}}
        ]
    ))

    # 執行
    print(f"🔍 Analyzing {img_path_a} vs {img_path_b} ...")
    try:
        result = structured_llm.invoke(messages)
        # 確保回傳的是 ComparisonResult 物件
        if isinstance(result, dict):
            result = ComparisonResult(**result)
        return result
    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        return None

# ==========================================
# 3. 執行入口
# ==========================================
if __name__ == "__main__":
    # 確保你有建立 reference_images 資料夾並放入參考圖
    # 確保你有建立 test_images 資料夾並放入測試圖
    
    result = analyze_screenshots("test_images/page_v1.png", "test_images/page_v2.png")
    
    if result:
        print("\n" + "="*50)
        print(f"📊 Result: {result.classification}")
        print(f"🏷️ Type:   {result.sub_type}")
        print(f"📝 Reason: {result.reasoning}")
        print("="*50)