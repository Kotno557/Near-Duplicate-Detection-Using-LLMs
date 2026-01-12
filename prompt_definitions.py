import base64
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- prompts ---
SYSTEM_PROMPT_TEXT: str = """
# Role
You are an expert Software Test Engineer. Your task is to detect Functional Near-Duplicates between two UI screenshots.

# Task Description
You will be given two images (UI screenshots).
Analyze and determine the relationship between the two screenshots based on functionality and layout, using the classification rules below.

# Classification Rules
1. [Clone]: Pixels match almost perfectly.
2. [Near-Duplicate (Nd)]:
   - Nd1 (Cosmetic): Changes in ads, banners, or background colors. Core widgets are fixed.
   - Nd2 (Dynamic Data): CRITICAL. The form/layout is identical, but text values differ.
   - Nd3 (List Expansion): A list/table has more rows, but the columns and data type are identical.
3. [Distinct]: Functional layout changes, new widgets, or new page states.

# Output Requirement
Please try to output a JSON object in this format:
{
    "classification": "Clone" | "Near-Duplicate" | "Distinct",
    "sub_type": "None" | "Nd1" | "Nd2" | "Nd3",
    "reasoning": "Brief explanation of why you chose this classification"
}
If you cannot output JSON, please provide your answer in your preferred format.
"""

def get_few_shot_messages(ref_dir: str = "reference_images"):
    messages = []

    # Nd1 (Background Changes)
    messages.extend(create_example_pair(
        ref_dir, "fig2_nd1_a.png", "fig2_nd1_b.png",
        title="Homepage with Background Change",
        expected_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd1",
            "reasoning": "The main content text and layout are identical. Only the top banner image has changed (from 'Statistics' text to a person drawing on a whiteboard). This is a cosmetic change."
        }'''
    ))

    # Nd2 (Dynamic Data)
    messages.extend(create_example_pair(
        ref_dir, "fig2_nd2_a.png", "fig2_nd2_b.png",
        title="Form Input Values (Pet Clinic)",
        expected_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd2",
            "reasoning": "The form structure (Owner, Name, Birth Date inputs) is strictly preserved. Only the specific data values are different ('Carlos Esteban' vs 'Betty Davis'). This is Dynamic Data."
        }'''
    ))

    # Nd3 (List Expansion)
    messages.extend(create_example_pair(
        ref_dir, "fig2_nd3_a.png", "fig2_nd3_b.png",
        title="Table Row Expansion",
        expected_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd3",
            "reasoning": "The second image adds a new row (for 'Rosy') to the 'Pets and Visits' table. The table columns and functionality (Edit/Add buttons) remain unchanged. Adding rows is List Expansion, not a new functional state."
        }'''
    ))

    # Distinct (Popup)
    messages.extend(create_example_pair(
        ref_dir, "ref_distinct_modal_a.png", "ref_distinct_modal_b.png",
        title="Page vs Page with Login Modal",
        expected_json='''{
            "classification": "Distinct",
            "sub_type": "None",
            "reasoning": "Although the background is similar, Image B contains a new 'Login Modal' that blocks the underlying content. This exposes new actionable widgets (Username/Password fields) and represents a different state."
        }'''
    ))

    if not messages:
        print("[Warning] No reference images found. Running in Zero-Shot mode")
    
    return messages


# --- tool funcitons ---
def encode_image(image_path: str) -> str | None:
    if not os.path.exists(image_path):
        print(f"[Warning] Reference image not found: {image_path}")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

FEW_SHOT_COUNTER: int = 0
def create_example_pair(ref_dir, img_a_name, img_b_name, title, expected_json):
    """
    Build HumanMessage/AIMessage 
    """
    global FEW_SHOT_COUNTER 
    FEW_SHOT_COUNTER += 1
    
    img_a = encode_image(os.path.join(ref_dir, img_a_name))
    img_b = encode_image(os.path.join(ref_dir, img_b_name))
    
    if not img_a or not img_b:
        return [] # no picture then skip

    return [
        HumanMessage(content=[
            {"type": "text", "text": f"Reference Case {FEW_SHOT_COUNTER}: {title}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b}"}}
        ]),
        AIMessage(content=expected_json)
    ]
