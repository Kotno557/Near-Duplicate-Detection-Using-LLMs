import time
import json
import sys
import base64
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
# --- Systemp prompt & Few-Shot prompt ---
from prompt_definitions import SYSTEM_PROMPT_TEXT, get_few_shot_messages, encode_image


# --- code main entry ---
if __name__ == "__main__":
    llm = ChatOllama(
        base_url="https://17c6fa445bc9.ngrok-free.app/",
        model="qwen3-vl:32b",
        temperature=0.0 # focus
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"Analyze these two new screenshots follow the rule: \n {SYSTEM_PROMPT_TEXT}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image("test_images/addressbook/index.png")}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image("test_images/addressbook/state703.png")}"}}
        ]
    )

    print(llm.invoke([message]).content)
    