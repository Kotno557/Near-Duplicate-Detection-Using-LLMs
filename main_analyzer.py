import time
import json
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# --- Few-Shot prompt ---
from prompt_definitions import SYSTEM_PROMPT_TEXT, get_few_shot_messages, encode_image


# --- pydantic ---
OLLAMA_CONFIG = {
    "base_url": "https://17c6fa445bc9.ngrok-free.app/", 
    "api_key": "ollama",
    "model": "qwen3-vl:235b-a22b",
    "temperature": 0.0 # focus
}

class ComparisonResult(BaseModel):
    classification: Literal["Clone", "Near-Duplicate", "Distinct"]
    sub_type: Optional[Literal["None", "Nd1", "Nd2", "Nd3"]] = Field("None")
    reasoning: str


# --- main logic ---
def analyze_screenshots(img_path_a: str, img_path_b: str):
    print(f"[Info] Connecting to Ollama at \"{OLLAMA_CONFIG['base_url']}\" using {OLLAMA_CONFIG['model']} model ...")
    
    # ollama llm model
    llm = ChatOllama(
        base_url=OLLAMA_CONFIG["base_url"],
        model=OLLAMA_CONFIG["model"],
        temperature=OLLAMA_CONFIG["temperature"] # focus
    )

    # read target screenshot
    target_a = encode_image(img_path_a)
    target_b = encode_image(img_path_b)
    if not target_a or not target_b:
        print("\033[91m[Error]\033[0m Target images not found.")
        return

    # build prompt pipeline
    messages = []
    
    # inject system prompt
    messages.append(SystemMessage(content=SYSTEM_PROMPT_TEXT))
    
    # inject Few-Shot prompt 
    messages.extend(get_few_shot_messages(ref_dir="reference_images"))
    
    # build search messages with JSON requirement
    json_schema = {
        "classification": "Clone | Near-Duplicate | Distinct",
        "sub_type": "None | Nd1 | Nd2 | Nd3",
        "reasoning": "string"
    }
    messages.append(HumanMessage(
        content=[
           #{"type": "text", "text": f"Now, analyze these two new screenshots based on the logic above. You MUST respond with ONLY a valid JSON object (no markdown, no explanations) using this exact schema: {json.dumps(json_schema)}"},
            {"type": "text", "text": f"Now, analyze these two new screenshots based on the logic above."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b}"}}
        ]
    ))

    # invoke llm to analyzing
    print(f"[Info] Analyzing {img_path_a} vs {img_path_b} ...")
    response = None
    try:
        response = llm.invoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_text = response_text.strip()
        
        # parse JSON response (handle possible markdown)
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result_dict = json.loads(response_text)
        
        # Add default reasoning if missing
        if 'reasoning' not in result_dict:
            result_dict['reasoning'] = f"No reasoning"

        result = ComparisonResult(**result_dict)
        return result
    except Exception as e:
        print(f"\033[91m[Error]\033[0m Analysis Failed: {e}")
        if response:
            print("\033[93m[DEBUG]\033[0m Raw Response:")
            print("-"*60)
            print(response)
            print("="*60 + "\n")
        return None


# --- code main entry ---
if __name__ == "__main__":
    # ensure reference_images folder exsist
    # ensure test_images folder exsist
    
    # run analyzing and return result
    start_time = time.time()
    result = analyze_screenshots("test_images/page_v1.png", "test_images/page_v2.png")
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("=" * 60 + "\n")
    if result:
        print(f"\033[92m[Success]\033[0m")
        print(f"Result: {result.classification}")
        print(f"Type  : {result.sub_type}")
        print(f"Reason: \"{result.reasoning}\"")
    print(f"Time  : {elapsed_time:.2f} seconds")
    print("=" * 60 + "\n")