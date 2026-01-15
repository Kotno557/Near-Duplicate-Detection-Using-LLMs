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
1. [Clone]: No semantic, functional, or perceptual differences between the two pages.
2. [Near-Duplicate (Nd)] Includes the following fine-grained subcategories:
   - Nd1 (Cosmetic): Aesthetic changes only (e.g., advertisements, background images).
   - Nd2 (Dynamic Data): Same template but populated with different dynamic data.
   - Nd3 (List Expansion): Addition or removal of UI elements, where the functionality already exists in the other page.
3. [Distinct]: Presence of new functionality or semantic content different.

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

    # Clone
    messages.extend(create_example_pair(
        ref_dir, "clone_a.png", "clone_b.png",
        title="Homepage with Background Change",
        expected_json='''{
            "classification": "Clone",
            "sub_type": null,
            "reasoning": "The two screenshots are visually and functionally identical. All elements, including the navigation bar, product images, pricing text, model selection sidebar (iPhone 16 vs 16 Plus), and the support icon, remain exactly the same without any cosmetic, data, or structural changes."
        }'''
    ))


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

    # counter-example (wrong classification)
    messages.extend(create_counter_example(
        ref_dir, "state480.png", "state1184.png",
        title="Common Misclassification Example",
        wrong_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd2",
            "reasoning": "The layout and fields are identical. The first image shows a populated form (with data in the fields), while the second image shows an empty form. This is a case of dynamic data being populated."
        }''',
        correct_json='''{
            "classification": "Distinct",
            "sub_type": "None",
            "reasoning": "Although the layouts are similar, these two screenshots represent different functional states. Image A is the 'Create User Account' page used for registering new users, whereas Image B is the 'My User Account' page used for editing existing data. Image B contains functional fields not present in Image A (e.g., 'User picture' file upload, 'Old password' field), and the header navigation shows a completely different login state (Image A is logged out, while Image B is logged in with a user menu)."
        }'''
    ))

    messages.extend(create_counter_example(
        ref_dir, "state1174.png", "state1225.png",
        title="Common Misclassification Example",
        wrong_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd3",
            "reasoning": "The core 'Authentication Required' modal is identical. The second image adds a left-hand navigation panel with course options. This is an addition of UI elements, but the core functionality remains the same."
        }''',
        correct_json='''{
            "classification": "Distinct",
            "sub_type": "None",
            "reasoning": "The first image features a specific course layout that includes a functional sidebar menu (with options like 'Course description', 'Agenda', 'Document'), whereas the second image is a generic authentication page lacking this navigation structure. The presence of the sidebar in the first image introduces navigation capabilities that are completely absent in the second image."
        }'''
    ))

    messages.extend(create_counter_example(
        ref_dir, "state15.png", "state298.png",
        title="Common Misclassification Example",
        wrong_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd2",
            "reasoning": "The core functionality (room booking calendar) remains the same. The calendar has advanced to a different date (July 5, 2019 to September 11, 2019). This is a change in dynamic data, not a new feature."
        }''',
        correct_json='''{
            "classification": "Distinct",
            "sub_type": "None",
            "reasoning": "The two screenshots represent distinct functional views and user states within the Meeting Room Booking System. The first image shows a 'Day View' with hourly time slots (07:00 - 18:30) for multiple rooms and indicates an unauthenticated 'Unknown user' state. The second image shows a 'Month View' calendar grid for a specific room and indicates an authenticated 'administrator' state. Additionally, the second image contains a 'Rooms' navigation list (Room 1, Room 2, etc.) that is absent in the first image."
        }'''
    ))

    messages.extend(create_counter_example(
        ref_dir, "state175.png", "state179.png",
        title="Common Misclassification Example",
        wrong_json='''{
            "classification": "Near-Duplicate",
            "sub_type": "Nd3",
            "reasoning": "The content is largely the same, but the second image has a longer scrollable section with more 'Why can't I delete/alter a meeting?' entries. This is an expansion of a list, not a new functional element."
        }''',
        correct_json='''{
            "classification": "Clone",
            "sub_type": "None",
            "reasoning": "Both screenshots capture the exact same functional state of the 'Help' page within the Meeting Room Booking System. They feature identical layout, FAQ content ('Authentication', 'Making/Altering Meetings', etc.), and user session status ('Unknown user'). The visual and semantic content is indistinguishable."
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
            {"type": "text", "text": f"Example {FEW_SHOT_COUNTER} : {title}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b}"}}
        ]),
        AIMessage(content=expected_json)
    ]

def create_counter_example(ref_dir, img_a_name, img_b_name, title, wrong_json, correct_json):
    """
    Build a counter-example showing wrong vs correct classification
    """
    global FEW_SHOT_COUNTER 
    FEW_SHOT_COUNTER += 1
    
    img_a = encode_image(os.path.join(ref_dir, img_a_name))
    img_b = encode_image(os.path.join(ref_dir, img_b_name))
    
    if not img_a or not img_b:
        return []

    return [
        HumanMessage(content=[
            {"type": "text", "text": f"Counter-Example (Common Mistake) {FEW_SHOT_COUNTER} : {title}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b}"}}
        ]),
        AIMessage(content=f"""
        ## Incorrect Classification (Common Error):
        {wrong_json}

        ## Correct Classification:
        {correct_json}

        **Key Lesson**: Always check for functional changes beyond cosmetic differences. New interactive elements or features indicate Distinct classification.""")
    ]