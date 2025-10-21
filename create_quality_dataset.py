import json
import random

def create_contextual_steps(instruction):
    """Create realistic, context-aware steps"""
    
    instruction_lower = instruction.lower()
    
    # Different scenarios with realistic coordinates and actions
    if any(word in instruction_lower for word in ['search', 'find', 'look up']):
        # Search scenario
        search_terms = instruction.replace('search', '').replace('find', '').strip()
        return [
            f"Step 1: Click on the browser search bar | Coordinates: (350,80) | Target: Search input field",
            f"Step 2: Type '{search_terms}' into the search box | Coordinates: (640,80) | Target: Search input",
            f"Step 3: Press Enter or click search button | Coordinates: (900,80) | Target: Search button"
        ]
    
    elif any(word in instruction_lower for word in ['click', 'button', 'link']):
        # Click scenario
        element = instruction.replace('click', '').replace('button', '').replace('link', '').strip()
        return [
            f"Step 1: Locate the {element} on the page | Coordinates: (640,300) | Target: {element}",
            f"Step 2: Click on the {element} | Coordinates: (640,350) | Target: {element} element",
            f"Step 3: Wait for page to load | Coordinates: (640,500) | Target: Browser viewport"
        ]
    
    elif any(word in instruction_lower for word in ['login', 'sign in']):
        # Login scenario
        return [
            f"Step 1: Click on username field | Coordinates: (640,250) | Target: Username input",
            f"Step 2: Enter username | Coordinates: (640,250) | Target: Username field",
            f"Step 3: Click login button | Coordinates: (640,400) | Target: Login button"
        ]
    
    elif any(word in instruction_lower for word in ['scroll', 'navigate']):
        # Navigation scenario
        return [
            f"Step 1: Move to scrollable area | Coordinates: (1200,360) | Target: Scroll bar",
            f"Step 2: Scroll down the page | Coordinates: (1200,500) | Target: Scroll down",
            f"Step 3: Verify content loaded | Coordinates: (640,600) | Target: Page content"
        ]
    
    else:
        # Generic but meaningful steps
        coords = [
            (random.randint(200, 1000), random.randint(100, 400)),
            (random.randint(200, 1000), random.randint(150, 450)),
            (random.randint(200, 1000), random.randint(200, 500))
        ]
        return [
            f"Step 1: Open browser and navigate to relevant page | Coordinates: {coords[0]} | Target: Browser",
            f"Step 2: Perform the requested action: {instruction} | Coordinates: {coords[1]} | Target: Action area",
            f"Step 3: Confirm action completion | Coordinates: {coords[2]} | Target: Confirmation area"
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
            
            # Skip empty or very short instructions
            if len(instruction) < 3:
                continue
                
            steps = create_contextual_steps(instruction)
            output_text = "\n".join(steps)
            
            new_data = {
                "instruction": instruction,
                "output": output_text
            }
            
            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            count += 1
            
            # Preview first few
            if count <= 3:
                print(f"Sample {count}:")
                print(f"Instruction: {instruction}")
                print(f"Output: {output_text}\n")
    
    print(f"✅ Created quality dataset with {count} samples: {output_file}")

if __name__ == "__main__":
    create_quality_dataset()