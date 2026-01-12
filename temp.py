# import time
# import json
# import sys
# import base64
# from typing import Literal, Optional
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, SystemMessage
# from pydantic import BaseModel, Field
# # --- Systemp prompt & Few-Shot prompt ---
# from prompt_definitions import SYSTEM_PROMPT_TEXT, get_few_shot_messages, encode_image


# # --- code main entry ---
# if __name__ == "__main__":
#     llm = ChatOllama(
#         base_url="https://17c6fa445bc9.ngrok-free.app/",
#         model="qwen3-vl:32b",
#         temperature=0.0 # focus
#     )
    
#     message = HumanMessage(
#         content=[
#             {"type": "text", "text": f"Analyze these two new screenshots follow the rule: \n {SYSTEM_PROMPT_TEXT}"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image("test_images/addressbook/index.png")}"}},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image("test_images/addressbook/state703.png")}"}}
#         ]
#     )

#     print(llm.invoke([message]).content)
    

import requests
import json

# Ollama API 位址
url = "https://17c6fa445bc9.ngrok-free.app/api/chat"

payload = {
    "model": "qwen3-vl:235b-a22b",
    "messages": [
        # 1. System Prompt: 定義身份與規則
        {"role": "system", "content": "你是一個翻譯專家，將現代語轉換為極簡文言文。"},
        
        # 2. Few-Shot Example 1
        {"role": "user", "content": "請告訴我你的名字。"},
        {"role": "assistant", "content": "尊姓大名？"},
        
        # 3. Few-Shot Example 2w
        {"role": "user", "content": "我現在非常生氣。"},
        {"role": "assistant", "content": "吾甚怒。"},
        
        # 4. 實際任務 (Current Task)
        {"role": "user", "content": "今天晚上我們一起去吃飯吧。"}
    ],
    "options": {
        "temperature": 0.2, # 降低隨機性，使輸出更符合範例格式
        "num_ctx": 4096     # 確保有足夠空間儲存範例與歷史
    },
    "stream": False,
    "format": "json" # 強制要求 Ollama 回傳 JSON (若模型支援)
}

response = requests.post(url, json=payload)
print(response.json()['message']['content'])
