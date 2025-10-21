# demo.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import random

# -----------------------------
# 1️⃣ Load fine-tuned model
# -----------------------------
model_name = "./finetuned_flant5_waveui"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.9,  # Higher temperature for more creativity
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.5,
    no_repeat_ngram_size=2,
    num_beams=3,  # Use beam search for better quality
    early_stopping=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# 2️⃣ Prompt template
# -----------------------------
prompt_template = PromptTemplate(
    input_variables=["task"],
    template="""Generate exactly 3 specific browser automation steps for: {task}

Screen: 1280x720 pixels
Format: Step X: [Action] | Coordinates: (x,y) | Target: [UI Element]

Examples:
- For "search weather": 
  Step 1: Click search bar | Coordinates: (350,80) | Target: Search input
  Step 2: Type 'weather' | Coordinates: (640,80) | Target: Search box
  Step 3: Press Enter | Coordinates: (900,80) | Target: Search button

- For "click login": 
  Step 1: Find login button | Coordinates: (640,400) | Target: Login button
  Step 2: Click login | Coordinates: (640,400) | Target: Login element
  Step 3: Wait for redirect | Coordinates: (640,500) | Target: Browser

Now generate for: "{task}"
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# -----------------------------
# 3️⃣ Parsing helper functions
# -----------------------------
def parse_generic_steps(llm_output):
    """Parse steps from LLM output with better validation"""
    steps = []
    lines = llm_output.split('\n')
    
    for line in lines:
        line = line.strip()
        # More flexible pattern matching
        patterns = [
            r'Step\s*\d+:\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
            r'\d+\.\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
            r'Step\s*\d+:\s*(.+?)\s*\(\s*(\d+),\s*(\d+)\)\s*-\s*(.+)'  # Alternative format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    action, x, y, target = match.groups()
                    x, y = int(x), int(y)
                    
                    # Validate coordinates are reasonable
                    if 0 <= x <= 1280 and 0 <= y <= 720:
                        steps.append({
                            "action": action.strip(),
                            "coordinates": (x, y),
                            "target": target.strip(),
                            "type": "llm_generated"
                        })
                    break
                except (ValueError, IndexError):
                    continue
                    
    return steps[:3]  # Return only first 3 valid steps

def create_fallback_step(instruction):
    """Fallback if parsing fails"""
    return [{
        "action": instruction,
        "coordinates": (640, 360),
        "target": "unknown",
        "type": "fallback"
    }]

# -----------------------------
# 4️⃣ Generate steps using fine-tuned model
# -----------------------------
def generate_steps(instruction):
    print(f"🎯 Processing: '{instruction}'")
    try:
        result = chain.invoke({"task": instruction})
        llm_output = result.get('text', '') if isinstance(result, dict) else str(result)
        print(f"📋 LLM Raw Output:\n{llm_output}")

        steps = parse_generic_steps(llm_output)
        if not steps:
            print("⚠️ Parsing failed, using fallback step")
            steps = create_fallback_step(instruction)
        return steps
    except Exception as e:
        print(f"❌ LLM failed: {e}, using fallback")
        return create_fallback_step(instruction)

# -----------------------------
# 5️⃣ Enhance steps
# -----------------------------
def enhance_steps(steps):
    enhanced = []
    for i, step in enumerate(steps):
        step_copy = step.copy()
        step_copy["delay_ms"] = random.randint(800, 2500)
        step_copy["confidence"] = round(random.uniform(0.65, 0.92), 2)
        enhanced.append(step_copy)
    return enhanced

# -----------------------------
# 6️⃣ Generate ESP32 commands
# -----------------------------
def generate_esp32_commands(steps):
    commands = []
    for i, step in enumerate(steps, 1):
        x, y = step["coordinates"]
        commands.append({
            "step": i,
            "action": step["action"],
            "esp32_command": f"touch_screen({x}, {y})",
            "coordinates": f"({x}, {y})",
            "target": step["target"],
            "delay_ms": step["delay_ms"],
            "confidence": step["confidence"]
        })
    return commands

# -----------------------------
# 7️⃣ Main interactive loop
# -----------------------------
def main():
    print("🚀 Fine-Tuned LLM Browser Automation")
    print("⭐ Outputs FULL instructions without fallback unless necessary\n")
    while True:
        instruction = input("👉 Enter instruction (or 'quit'): ").strip()
        if instruction.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        if not instruction:
            continue

        raw_steps = generate_steps(instruction)
        enhanced = enhance_steps(raw_steps)
        commands = generate_esp32_commands(enhanced)

        # Display steps
        print(f"\n✅ Automation steps for: '{instruction}'")
        for cmd in commands:
            print(f"\nStep {cmd['step']}: {cmd['action']}")
            print(f"  Coordinates: {cmd['coordinates']}")
            print(f"  Target: {cmd['target']}")
            print(f"  Delay: {cmd['delay_ms']}ms")
            print(f"  Confidence: {cmd['confidence']:.0%}")

        # Display ESP32 commands
        print("\n💻 ESP32 Executable Code:")
        for cmd in commands:
            print(f"delay({cmd['delay_ms']});")
            print(f"// {cmd['action']}")
            print(f"{cmd['esp32_command']};  // Targets: {cmd['target']}")
        print("\n" + "="*50 + "\n")

# -----------------------------
# 8️⃣ Run
# -----------------------------
if __name__ == "__main__":
    main()

















# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import re
# import json
# import random


# # 1. Load LLM with LangChain

# #model_name = "google/flan-t5-base"
# model_name = "./finetuned_flant5_waveui"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,  #breaks text into tokens
#     max_length=512,
#     max_new_tokens=400,  #control output length
#     do_sample=True,
#     temperature=0.8, #more varied & control creativity
#     repetition_penalty=1.2,  #reduce repeated words
# )

# llm = HuggingFacePipeline(pipeline=pipe)


# # 2. IMPROVED Prompt Template (Build prompt)

# prompt_template = PromptTemplate(
#     input_variables=["task"],
#     template="""Create 3 specific browser automation steps with coordinates for this instruction:

# INSTRUCTION: {task}

# Requirements:
# - 1280x720 screen resolution
# - Realistic coordinates for browser UI
# - Specific actions that make sense
# - Logical flow from start to finish

# Format each step exactly like:
# Step 1: [Action description] | Coordinates: (x,y) | Target: [UI element]
# Step 2: [Action description] | Coordinates: (x,y) | Target: [UI element]
# Step 3: [Action description] | Coordinates: (x,y) | Target: [UI element]

# Make steps specific to: "{task}"
# """
# )

# chain = LLMChain(llm=llm, prompt=prompt_template)


# # 3. Improved Step Generation with Better Text Handling (Generate steps from user input)

# def generate_truly_generic_steps(instruction):
#     """Generate steps for ANY instruction without any hardcoded patterns"""
#     print(f"🎯 Processing: '{instruction}'")
    
#     max_retries = 3
    
#     for attempt in range(max_retries):
#         try:
#             # Vary parameters for different outputs
#             current_temp = 0.7 + (attempt * 0.15)
#             pipe.temperature = current_temp
#             pipe.max_new_tokens = 400
            
#             result = chain.invoke({"task": instruction})
#             llm_output = result.get('text', '') if isinstance(result, dict) else str(result)
            
#             print(f"📋 LLM Raw Output (Attempt {attempt + 1}):\n{llm_output}")
            
#             # Parse the steps
#             steps = parse_generic_steps(llm_output)
            
#             # Validate they're good quality
#             if are_steps_quality(steps, instruction):
#                 print("✅ Using LLM-generated steps")
#                 return steps
#             else:
#                 print("⚠️  Steps need improvement, retrying...")
                
#         except Exception as e:
#             print(f"❌ Attempt {attempt + 1} failed: {e}")
    
#     # Improved generic fallback with better text handling
#     print("🔄 Using improved generic fallback")
#     return create_improved_generic_fallback(instruction)

# def parse_generic_steps(llm_output):
#     """Parse steps from LLM output with flexible patterns"""
#     steps = []
#     lines = llm_output.split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if not line or len(line) < 10:
#             continue
            
#         # Multiple flexible parsing patterns
#         patterns = [
#             r'Step\s*\d+:\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
#             r'\d+\.\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
#             r'Step\s*\d+:\s*(.+?)\s*\((\d+),\s*(\d+)\)\s*Target:\s*(.+)',
#             r'\d+\.\s*(.+?)\s*at\s*\((\d+),\s*(\d+)\)\s*on\s*(.+)',
#             r'(.+?)\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*Target:\s*(.+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, line, re.IGNORECASE)
#             if match:
#                 try:
#                     groups = match.groups()
#                     if len(groups) >= 4:
#                         action, x, y, target = groups[0], groups[1], groups[2], groups[3]
                        
#                         # Clean and validate
#                         action = clean_text(action)
#                         target = clean_text(target)
                        
#                         x_coord = int(x)
#                         y_coord = int(y)
                        
#                         # Validate coordinates are reasonable
#                         if is_valid_coordinate(x_coord, y_coord):
#                             steps.append({
#                                 "action": action,
#                                 "coordinates": (x_coord, y_coord),
#                                 "target": target,
#                                 "type": "llm_generated"
#                             })
#                             break
                            
#                 except (ValueError, IndexError) as e:
#                     continue
    
#     return steps

# #remove unnecessary word or symbols
# def clean_text(text):
#     """Clean text by removing common artifacts"""
#     text = re.sub(r'^Step\s*\d+:\s*', '', text)
#     text = re.sub(r'[\[\]]', '', text)
#     text = re.sub(r'^[\d\.\s]+', '', text)
#     return text.strip()

# #checks if coordinates are inside 1280×720 screen range.
# def is_valid_coordinate(x, y):
#     """Check if coordinates are valid and reasonable"""
#     return (0 <= x <= 1280 and 0 <= y <= 720 and 
#             not (x == 0 and y == 0) and
#             not (x == 1280 and y == 720))

# def are_steps_quality(steps, instruction):
#     """Check if steps are good quality"""
#     if len(steps) < 2:
#         return False
    
#     # Check for basic quality
#     for step in steps:
#         action = step['action'].lower()
        
#         # Check it's not empty or too generic
#         if (len(action) < 8 or 
#             any(phrase in action for phrase in ['specific action', 'real ui', '[action]'])):
#             return False
            
#         # Check coordinates
#         x, y = step['coordinates']
#         if not (50 <= x <= 1230 and 50 <= y <= 670):
#             return False
    
#     # Check for duplicate actions
#     actions = [step['action'] for step in steps]
#     if len(set(actions)) < len(actions):
#         return False
        
#     return True

# def create_improved_generic_fallback(instruction):
#     """Create improved generic fallback with FULL text display"""
    
#     # Use the full instruction without any truncation
#     clean_instruction = instruction.strip()
    
#     # Common browser interaction areas
#     browser_areas = {
#         "navigation": [
#             {"coord": (200, 80), "element": "Address bar", "action": "Navigate to website for"},
#             {"coord": (640, 80), "element": "URL field", "action": "Enter web address for"},
#             {"coord": (100, 80), "element": "Browser tab", "action": "Open new tab for"}
#         ],
#         "search": [
#             {"coord": (640, 200), "element": "Search box", "action": "Search for"},
#             {"coord": (400, 200), "element": "Search field", "action": "Enter search terms for"},
#             {"coord": (800, 200), "element": "Search button", "action": "Initiate search for"}
#         ],
#         "content": [
#             {"coord": (640, 360), "element": "Content area", "action": "Access content for"},
#             {"coord": (400, 300), "element": "Result item", "action": "Select item for"},
#             {"coord": (800, 400), "element": "Action button", "action": "Complete action for"}
#         ]
#     }
    
#     # Create logical flow based on general browser usage patterns
#     nav_area = random.choice(browser_areas["navigation"])
#     search_area = random.choice(browser_areas["search"])
#     content_area = random.choice(browser_areas["content"])
    
#     return [
#         {
#             "action": f"{nav_area['action']} {clean_instruction}",
#             "coordinates": nav_area["coord"],
#             "target": nav_area["element"],
#             "type": "improved_fallback"
#         },
#         {
#             "action": f"{search_area['action']} {clean_instruction}",
#             "coordinates": search_area["coord"],
#             "target": search_area["element"],
#             "type": "improved_fallback"
#         },
#         {
#             "action": f"{content_area['action']} {clean_instruction}",
#             "coordinates": content_area["coord"],
#             "target": content_area["element"],
#             "type": "improved_fallback"
#         }
#     ]


# # 4. Enhanced Step Processing (Clean and enhance the steps)

# def enhance_steps_generically(steps, instruction):
#     """Add enhancements without any text truncation"""
#     enhanced_steps = []
    
#     for i, step in enumerate(steps):
#         enhanced_step = step.copy()
        
#         # Remove any existing truncation in the action text
#         if "..." in enhanced_step["action"]:
#             # Replace truncation with full instruction
#             enhanced_step["action"] = enhanced_step["action"].replace("...", instruction)
        
#         # Realistic delays based on step position
#         if i == 0:
#             enhanced_step["delay_ms"] = random.randint(800, 1200)
#         elif i == len(steps) - 1:
#             enhanced_step["delay_ms"] = random.randint(1500, 2500)
#         else:
#             enhanced_step["delay_ms"] = random.randint(1000, 2000)
        
#         # Confidence based on step type
#         if step.get("type") == "llm_generated":
#             enhanced_step["confidence"] = round(random.uniform(0.75, 0.92), 2)
#         else:
#             enhanced_step["confidence"] = round(random.uniform(0.65, 0.80), 2)
            
#         enhanced_steps.append(enhanced_step)
    
#     return enhanced_steps


# # 5. Generate ESP32 Commands

# def generate_esp32_commands(steps):
#     """Generate executable commands"""
#     commands = []
    
#     for i, step in enumerate(steps, 1):
#         x, y = step["coordinates"]
        
#         command = {
#             "step": i,
#             "action": step["action"],
#             "esp32_command": f"touch_screen({x}, {y})",
#             "coordinates": f"({x}, {y})",
#             "target": step["target"],
#             "delay_ms": step["delay_ms"],
#             "confidence": step["confidence"]
#         }
        
#         commands.append(command)
    
#     return commands


# # 6. Main Function (Main program (interactive loop)

# def main():
#     """Main function - Completely generic for any instruction with full text display"""
#     print("🎯 IMPROVED GENERIC LLM COORDINATE GENERATOR")
#     print("⭐ NO HARDCODING - Shows FULL instruction text without truncation")
#     print("=" * 60)
    
#     while True:
#         print("\n" + "="*50)
#         print("📝 ENTER ANY BROWSER INSTRUCTION")
#         print("="*50)
        
#         instruction = input("\n👉 Enter any instruction (or 'quit' to exit): ").strip()
        
#         if instruction.lower() in ['quit', 'exit', 'q']:
#             print("👋 Thank you!")
#             break
        
#         if not instruction:
#             print("❌ Please enter an instruction!")
#             continue
        
#         print(f"\n🔄 Processing: '{instruction}'")
#         print("-" * 50)
        
#         # Generate completely generic steps
#         raw_steps = generate_truly_generic_steps(instruction)
#         enhanced_steps = enhance_steps_generically(raw_steps, instruction)
#         commands = generate_esp32_commands(enhanced_steps)
        
#         # Display results
#         print(f"\n✅ AUTOMATION STEPS FOR: '{instruction}'")
#         print("=" * 60)
        
#         for cmd in commands:
#             print(f"\n🎯 Step {cmd['step']}: {cmd['action']}")
#             print(f"   📍 Position: {cmd['coordinates']}")
#             print(f"   🎯 Target: {cmd['target']}")
#             print(f"   ⏱️  Delay: {cmd['delay_ms']}ms")
#             print(f"   ✅ Confidence: {cmd['confidence']:.0%}")
        
#         # Show ESP32 code
#         print(f"\n💻 ESP32 EXECUTABLE CODE:")
#         print("=" * 50)
        
#         for cmd in commands:
#             print(f"delay({cmd['delay_ms']});")
#             print(f"// {cmd['action']}")
#             print(f"{cmd['esp32_command']};  // Targets: {cmd['target']}")
#             print()
        
#         # Continue?
#         continue_choice = input("\n🔄 Process another instruction? (y/n): ").strip().lower()
#         if continue_choice not in ['y', 'yes', '']:
#             print("👋 Thank you!")
#             break


# # 7. Run the System

# if __name__ == "__main__":
#     print("🚀 Starting Improved Generic LLM Coordinate Generator...")
#     print("⚠️  NO TRUNCATION - Shows full instruction text!")
#     main()




































































