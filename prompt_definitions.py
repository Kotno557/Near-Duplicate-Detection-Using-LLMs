import base64
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================================
# 1. 系統定義 (System Prompt)
# ==========================================
SYSTEM_PROMPT_TEXT = """
# Role
You are an expert Software Test Engineer. Your goal is to detect "Functional Near-Duplicates" based on the ICSE '20 paper standards.

# Classification Rules
1. [Clone]: Pixels match almost perfectly.
2. [Near-Duplicate (Nd)]:
   - Nd1 (Cosmetic): Changes in ads, banners, or background colors. Core widgets are fixed.
   - Nd2 (Dynamic Data): CRITICAL. The form/layout is identical, but text values differ (e.g., "User: Alex" vs "User: Ben").
   - Nd3 (List Expansion): A list/table has more rows, but the columns and data type are identical.
3. [Distinct]: Functional layout changes, new widgets, or new page states (e.g. List vs Detail view).

# Output Requirement
You must output a JSON object: {"classification": "...", "sub_type": "...", "reasoning": "..."}
"""

# ==========================================
# 2. 輔助工具：讀取圖片
# ==========================================
def encode_image(image_path: str) -> str | None:
    """將圖片路徑轉為 Base64，若檔案不存在回傳 None"""
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Reference image not found: {image_path}")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 3. 建構 Few-Shot 歷史訊息 (核心邏輯)
# ==========================================
def get_few_shot_messages(ref_dir: str = "reference_images"):
    """
    自動從 reference_images 資料夾讀取圖片，並組合成 Few-Shot 教學範例。
    """
    # 預先定義好的範例路徑
    # 這裡你需要準備實際的圖片 (Nd2_A, Nd2_B, Nd3_A, Nd3_B)
    img_nd2_a = encode_image(os.path.join(ref_dir, "ref_nd2_a.png"))
    img_nd2_b = encode_image(os.path.join(ref_dir, "ref_nd2_b.png"))
    img_nd3_a = encode_image(os.path.join(ref_dir, "ref_nd3_a.png"))
    img_nd3_b = encode_image(os.path.join(ref_dir, "ref_nd3_b.png"))

    messages = []

    # --- 範例 1: Nd2 (Dynamic Data) ---
    # 如果圖片讀取成功，才加入這個範例
    if img_nd2_a and img_nd2_b:
        messages.extend([
            HumanMessage(content=[
                {"type": "text", "text": "Example 1: Compare these two images (Pet Clinic Form). Are they Distinct?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_nd2_a}"}}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_nd2_b}"}}
            ]),
            AIMessage(content='''{
                "classification": "Near-Duplicate",
                "sub_type": "Nd2",
                "reasoning": "The form structure is identical. Only the text values ('Carlos' vs 'Betty') are different. This is Dynamic Data."
            }''')
        ])

    # --- 範例 2: Nd3 (List Expansion) ---
    if img_nd3_a and img_nd3_b:
        messages.extend([
            HumanMessage(content=[
                {"type": "text", "text": "Example 2: Compare these two images (Visit List)."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_nd3_a}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_nd3_b}"}}
            ]),
            AIMessage(content='''{
                "classification": "Near-Duplicate",
                "sub_type": "Nd3",
                "reasoning": "The second image adds a new row to the table. Since the table functionality is already present, adding a row does not expose new functionality type."
            }''')
        ])
        
    if not messages:
        print("ℹ️ Note: No valid reference images found. Running in Zero-Shot mode.")

    return messages