# create_quality_dataset.py - FINAL CORRECTED VERSION
import json
import random

def create_meaningful_steps(instruction):
    """Create meaningful automation steps with variation"""
    instruction_lower = instruction.lower()
    
    # Different coordinate variations for the same element type
    coord_variations = {
        "navigation_link": [
            [(200, 150), (300, 200), (400, 180)],
            [(250, 180), (350, 220), (450, 200)],
            [(180, 160), (280, 190), (380, 170)]
        ],
        "button": [
            [(640, 400), (600, 350), (700, 380)],
            [(620, 380), (580, 330), (680, 360)],
            [(660, 420), (620, 370), (720, 400)]
        ],
        "reminder": [
            [(1200, 80), (1150, 100), (1250, 90)],
            [(1180, 70), (1130, 90), (1230, 80)],
            [(1220, 90), (1170, 110), (1270, 100)]
        ],
        "list_item": [
            [(400, 250), (450, 300), (500, 280)],
            [(380, 270), (430, 320), (480, 300)],
            [(420, 230), (470, 280), (520, 260)]
        ]
    }
    
    # Handle different instruction types with VARIATION
    if "statictext, link" in instruction_lower:
        variation = random.choice(coord_variations["navigation_link"])
        return [
            f"Step 1: Find the navigation link | {variation[0]} | Target: Navigation link",
            f"Step 2: Click the link to navigate | {variation[1]} | Target: Link element", 
            f"Step 3: Wait for new page to load | {variation[2]} | Target: Browser"
        ]
    
    elif "button" in instruction_lower:
        variation = random.choice(coord_variations["button"])
        if "login" in instruction_lower:
            return [
                f"Step 1: Locate login button | {variation[0]} | Target: Login button",
                f"Step 2: Click to open login | {variation[1]} | Target: Login element",
                f"Step 3: Wait for login form | {variation[2]} | Target: Login screen"
            ]
        elif "view" in instruction_lower and "results" in instruction_lower:
            return [
                f"Step 1: Find view results button | {variation[0]} | Target: Results button",
                f"Step 2: Click to show results | {variation[1]} | Target: Action button",
                f"Step 3: Wait for results display | {variation[2]} | Target: Results area"
            ]
        else:
            return [
                f"Step 1: Locate the button | {variation[0]} | Target: Button",
                f"Step 2: Click the button | {variation[1]} | Target: Button element",
                f"Step 3: Wait for action completion | {variation[2]} | Target: Browser"
            ]
    
    elif "reminder" in instruction_lower:
        variation = random.choice(coord_variations["reminder"])
        return [
            f"Step 1: Find reminder item | {variation[0]} | Target: Reminder",
            f"Step 2: Click to manage reminders | {variation[1]} | Target: Reminder button",
            f"Step 3: Wait for reminder interface | {variation[2]} | Target: Reminder screen"
        ]
    
    elif "listitem" in instruction_lower:
        variation = random.choice(coord_variations["list_item"])
        return [
            f"Step 1: Find the list item | {variation[0]} | Target: List item",
            f"Step 2: Click to select item | {variation[1]} | Target: Selection",
            f"Step 3: Wait for item action | {variation[2]} | Target: Browser"
        ]
    
    elif "icon" in instruction_lower:
        return [
            "Step 1: Locate the icon | (80, 80) | Target: Icon",
            "Step 2: Click the icon | (80, 80) | Target: Icon element",
            "Step 3: Wait for icon action | (640, 500) | Target: Browser"
        ]
    
    else:
        # Generic with variation
        variation = random.choice(coord_variations["button"])
        return [
            f"Step 1: Locate {instruction} | {variation[0]} | Target: {instruction}",
            f"Step 2: Interact with {instruction} | {variation[1]} | Target: {instruction} element", 
            f"Step 3: Complete the action | {variation[2]} | Target: Browser"
        ]

def create_quality_dataset():
    input_file = "waveui_finetune.jsonl"
    output_file = "waveui_quality_finetune.jsonl"
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        count = 0
        for line in fin:
            data = json.loads(line)
            instruction = data["instruction"].strip()
            
            if len(instruction) < 3:
                continue
                
            steps = create_meaningful_steps(instruction)
            output_text = "\n".join(steps)
            
            new_data = {
                "instruction": instruction,
                "output": output_text
            }
            
            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            count += 1
            
            # Preview samples with variation
            if count <= 5:
                print(f"Sample {count}:")
                print(f"Instruction: {instruction}")
                print(f"Output: {output_text}")
                print("-" * 50)
    
    print(f"✅ Created IMPROVED quality dataset with {count} samples")
    print(f"📁 Saved as: {output_file}")

if __name__ == "__main__":
    create_quality_dataset()