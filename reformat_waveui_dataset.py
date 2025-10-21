# reformat_waveui_dataset.py
import json
import random

def create_dynamic_steps(instruction):
    """Create realistic, varied steps based on instruction type"""
    
    # Different coordinate ranges for different UI elements
    coordinate_templates = {
        "search": [(320, 120), (640, 120), (960, 120)],  # Search bars
        "button": [(600, 400), (640, 360), (680, 400)],  # Buttons
        "link": [(200, 300), (400, 350), (600, 300)],    # Navigation links
        "menu": [(100, 50), (100, 100), (100, 150)],     # Menu items
    }
    
    # Action templates based on instruction content
    if "search" in instruction.lower() or "find" in instruction.lower():
        steps = [
            f"Step 1: Click on the search bar | Coordinates: (320,120) | Target: Search input",
            f"Step 2: Type '{instruction}' into search field | Coordinates: (640,120) | Target: Search box",
            f"Step 3: Click the search button | Coordinates: (960,120) | Target: Search button"
        ]
    elif "click" in instruction.lower() or "button" in instruction.lower():
        steps = [
            f"Step 1: Locate the {instruction} element | Coordinates: (600,400) | Target: {instruction}",
            f"Step 2: Click on the {instruction} | Coordinates: (640,360) | Target: {instruction} button",
            f"Step 3: Verify the action completed | Coordinates: (640,500) | Target: Result area"
        ]
    else:
        # Generic but varied steps
        coords = [(random.randint(100, 1100), random.randint(100, 600)) for _ in range(3)]
        steps = [
            f"Step 1: Navigate to {instruction} | Coordinates: {coords[0]} | Target: {instruction}",
            f"Step 2: Interact with {instruction} | Coordinates: {coords[1]} | Target: {instruction} element",
            f"Step 3: Complete {instruction} action | Coordinates: {coords[2]} | Target: Action confirmation"
        ]
    
    return "\n".join(steps)

# Process dataset with improved formatting
with open("waveui_finetune.jsonl", 'r', encoding='utf-8') as f_in, \
     open("waveui_steps_finetuned.jsonl", 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        data = json.loads(line)
        instruction = data["instruction"].strip()
        
        new_data = {
            "instruction": instruction,
            "output": create_dynamic_steps(instruction)
        }
        
        f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")