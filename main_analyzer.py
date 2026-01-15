import time
import json
import sys
import concurrent.futures
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import httpx

# --- Systemp prompt & Few-Shot prompt ---
from prompt_definitions import SYSTEM_PROMPT_TEXT, get_few_shot_messages, encode_image


# --- ollama config ---
OLLAMA_CONFIG = {
    "base_url": "http://140.116.8.205:11434/v1", 
    "api_key": "ollama",
    "model": "gemma3:27b",
    "temperature": 0,
    "timeout": 300, 
    "few-shot": True
}   


# --- pydantic ---
class ComparisonResult(BaseModel):
    classification: Literal["Clone", "Near-Duplicate", "Distinct"]
    sub_type: Optional[Literal["None", "Nd1", "Nd2", "Nd3"]] = Field("None")
    reasoning: str


# --- main logic ---
def analyze_screenshots(img_path_a: str, img_path_b: str):
    print(f"[Info] Connecting to Ollama at \"{OLLAMA_CONFIG['base_url']}\" using {OLLAMA_CONFIG['model']} model use {"Few-Shot" if OLLAMA_CONFIG['few-shot'] else "System Prompt"} ...")
    
    # build no timeout HTTP client
    custom_client = httpx.Client(
        timeout=httpx.Timeout(None), # 停用所有連線、讀取、寫入逾時
    )

    # ollama llm model (ChatOpenAI)
    llm = ChatOpenAI(
        base_url=OLLAMA_CONFIG["base_url"],
        api_key=OLLAMA_CONFIG["api_key"],
        model=OLLAMA_CONFIG["model"],
        temperature=OLLAMA_CONFIG["temperature"],
        timeout=OLLAMA_CONFIG["timeout"],
        # http_client=custom_client,
        max_retries=0,
        stream_usage=True,
    )

    # read target screenshot
    target_a = encode_image(img_path_a)
    target_b = encode_image(img_path_b)
    if not target_a or not target_b:
        print("\033[91m[Error]\033[0m Target images not found.")
        return

    # build prompt pipeline
    messages = []
    
    # few-shot
    if OLLAMA_CONFIG["few-shot"]:
        # inject system prompt
        messages.append(SystemMessage(content=SYSTEM_PROMPT_TEXT))
        
        # inject Few-Shot prompt 
        messages.extend(get_few_shot_messages(ref_dir="reference_images"))
        
        # build request message
        messages.append(HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_a}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b}"}}
            ]
        ))
    
    # zero-shot
    else:
        messages.append(HumanMessage(
            content=[
                {"type": "text", "text": f"{SYSTEM_PROMPT_TEXT}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_a}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b}"}}
            ]
        ))

    # invoke llm to analyzing
    print(f"[Info] Analyzing {img_path_a} vs {img_path_b} ...")
    response = None
    try:
        # execute with timeout mechanism
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm.invoke, messages)
            try:
                response = future.result(timeout=OLLAMA_CONFIG["timeout"])
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Analysis exceeded timeout of {OLLAMA_CONFIG['timeout']} seconds")
        
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_text = response_text.strip()
        
        # parse JSON response (handle possible markdown)
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Fix incomplete JSON by adding missing closing brace if needed
        if response_text.count('{') > response_text.count('}'):
            response_text += '\n}'

        result_dict = json.loads(response_text)
        
        # Add default reasoning if missing
        if 'reasoning' not in result_dict:
            result_dict['reasoning'] = f"No reasoning"

        result = ComparisonResult(**result_dict)

        # always print debub rawdata
        print("\033[93m[DEBUG]\033[0m Raw Response:")
        print("-"*60)
        print(response)
        print("="*60 + "\n")

        return result
    except Exception as e:
        print(f"\033[91m[Error]\033[0m Analysis Failed: {e}")
        if response:
            print("\033[93m[DEBUG]\033[0m Raw Response:")
            print("-"*60)
            print(response)
            print("="*60 + "\n")
        return None
    
def main(img_path_a: str, img_path_b: str):
    # run analyzing and return result
    start_time = time.time()
    result = analyze_screenshots(img_path_a, img_path_b)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # print resault
    print("=" * 60 + "\n")
    if result:
        print(f"\033[92m[Success]\033[0m")
        print(f"Result: {result.classification}")
        print(f"Type  : {result.sub_type}")
        print(f"Reason: \"{result.reasoning}\"")
    print(f"Time  : {elapsed_time:.2f} seconds")
    print("=" * 60 + "\n")

    # return dict
    if result:
        return {
            "classification": result.classification,
            "sub_type": result.sub_type,
            "reasoning": result.reasoning,
            "execution_time": elapsed_time
        }
    else:
        return None


# --- code main entry ---
if __name__ == "__main__":
    # ensure reference_images folder exsist
    # ensure test_images folder exsist
    
    # Check command line arguments
    if len(sys.argv) != 3:
        print("[Warning] Usage: python3 main_analyzer.py <image_a> <image_b>")
        print("[Example] python3 main_analyzer.py test_images/self_test_a.png test_images/self_test_b.png")
        print("[Info] Using default test_images/addressbook/index.png vs test_images/addressbook/state3.png")
        img_path_a = "test_images/addressbook/index.png"
        img_path_b = "test_images/addressbook/index.png"
    else:
        img_path_a = sys.argv[1]
        img_path_b = sys.argv[2]

    main(img_path_a, img_path_b)