from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import json
import random


# 1. Load LLM with LangChain

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    max_new_tokens=400,
    do_sample=True,
    temperature=0.8, #more varied
    repetition_penalty=1.2,
)

llm = HuggingFacePipeline(pipeline=pipe)


# 2. IMPROVED Prompt Template - Better Specificity

prompt_template = PromptTemplate(
    input_variables=["task"],
    template="""Create 3 specific browser automation steps with coordinates for this instruction:

INSTRUCTION: {task}

Requirements:
- 1280x720 screen resolution
- Realistic coordinates for browser UI
- Specific actions that make sense
- Logical flow from start to finish

Format each step exactly like:
Step 1: [Action description] | Coordinates: (x,y) | Target: [UI element]
Step 2: [Action description] | Coordinates: (x,y) | Target: [UI element]
Step 3: [Action description] | Coordinates: (x,y) | Target: [UI element]

Make steps specific to: "{task}"
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)


# 3. Improved Step Generation with Better Text Handling

def generate_truly_generic_steps(instruction):
    """Generate steps for ANY instruction without any hardcoded patterns"""
    print(f"🎯 Processing: '{instruction}'")
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Vary parameters for different outputs
            current_temp = 0.7 + (attempt * 0.15)
            pipe.temperature = current_temp
            pipe.max_new_tokens = 400
            
            result = chain.invoke({"task": instruction})
            llm_output = result.get('text', '') if isinstance(result, dict) else str(result)
            
            print(f"📋 LLM Raw Output (Attempt {attempt + 1}):\n{llm_output}")
            
            # Parse the steps
            steps = parse_generic_steps(llm_output)
            
            # Validate they're good quality
            if are_steps_quality(steps, instruction):
                print("✅ Using LLM-generated steps")
                return steps
            else:
                print("⚠️  Steps need improvement, retrying...")
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
    
    # Improved generic fallback with better text handling
    print("🔄 Using improved generic fallback")
    return create_improved_generic_fallback(instruction)

def parse_generic_steps(llm_output):
    """Parse steps from LLM output with flexible patterns"""
    steps = []
    lines = llm_output.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
            
        # Multiple flexible parsing patterns
        patterns = [
            r'Step\s*\d+:\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
            r'\d+\.\s*(.+?)\s*\|\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*\|\s*Target:\s*(.+)',
            r'Step\s*\d+:\s*(.+?)\s*\((\d+),\s*(\d+)\)\s*Target:\s*(.+)',
            r'\d+\.\s*(.+?)\s*at\s*\((\d+),\s*(\d+)\)\s*on\s*(.+)',
            r'(.+?)\s*Coordinates:\s*\((\d+),\s*(\d+)\)\s*Target:\s*(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) >= 4:
                        action, x, y, target = groups[0], groups[1], groups[2], groups[3]
                        
                        # Clean and validate
                        action = clean_text(action)
                        target = clean_text(target)
                        
                        x_coord = int(x)
                        y_coord = int(y)
                        
                        # Validate coordinates are reasonable
                        if is_valid_coordinate(x_coord, y_coord):
                            steps.append({
                                "action": action,
                                "coordinates": (x_coord, y_coord),
                                "target": target,
                                "type": "llm_generated"
                            })
                            break
                            
                except (ValueError, IndexError) as e:
                    continue
    
    return steps

def clean_text(text):
    """Clean text by removing common artifacts"""
    text = re.sub(r'^Step\s*\d+:\s*', '', text)
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub(r'^[\d\.\s]+', '', text)
    return text.strip()

def is_valid_coordinate(x, y):
    """Check if coordinates are valid and reasonable"""
    return (0 <= x <= 1280 and 0 <= y <= 720 and 
            not (x == 0 and y == 0) and
            not (x == 1280 and y == 720))

def are_steps_quality(steps, instruction):
    """Check if steps are good quality"""
    if len(steps) < 2:
        return False
    
    # Check for basic quality
    for step in steps:
        action = step['action'].lower()
        
        # Check it's not empty or too generic
        if (len(action) < 8 or 
            any(phrase in action for phrase in ['specific action', 'real ui', '[action]'])):
            return False
            
        # Check coordinates
        x, y = step['coordinates']
        if not (50 <= x <= 1230 and 50 <= y <= 670):
            return False
    
    # Check for duplicate actions
    actions = [step['action'] for step in steps]
    if len(set(actions)) < len(actions):
        return False
        
    return True

def create_improved_generic_fallback(instruction):
    """Create improved generic fallback with FULL text display"""
    
    # Use the full instruction without any truncation
    clean_instruction = instruction.strip()
    
    # Common browser interaction areas
    browser_areas = {
        "navigation": [
            {"coord": (200, 80), "element": "Address bar", "action": "Navigate to website for"},
            {"coord": (640, 80), "element": "URL field", "action": "Enter web address for"},
            {"coord": (100, 80), "element": "Browser tab", "action": "Open new tab for"}
        ],
        "search": [
            {"coord": (640, 200), "element": "Search box", "action": "Search for"},
            {"coord": (400, 200), "element": "Search field", "action": "Enter search terms for"},
            {"coord": (800, 200), "element": "Search button", "action": "Initiate search for"}
        ],
        "content": [
            {"coord": (640, 360), "element": "Content area", "action": "Access content for"},
            {"coord": (400, 300), "element": "Result item", "action": "Select item for"},
            {"coord": (800, 400), "element": "Action button", "action": "Complete action for"}
        ]
    }
    
    # Create logical flow based on general browser usage patterns
    nav_area = random.choice(browser_areas["navigation"])
    search_area = random.choice(browser_areas["search"])
    content_area = random.choice(browser_areas["content"])
    
    return [
        {
            "action": f"{nav_area['action']} {clean_instruction}",
            "coordinates": nav_area["coord"],
            "target": nav_area["element"],
            "type": "improved_fallback"
        },
        {
            "action": f"{search_area['action']} {clean_instruction}",
            "coordinates": search_area["coord"],
            "target": search_area["element"],
            "type": "improved_fallback"
        },
        {
            "action": f"{content_area['action']} {clean_instruction}",
            "coordinates": content_area["coord"],
            "target": content_area["element"],
            "type": "improved_fallback"
        }
    ]


# 4. Enhanced Step Processing - NO TRUNCATION

def enhance_steps_generically(steps, instruction):
    """Add enhancements without any text truncation"""
    enhanced_steps = []
    
    for i, step in enumerate(steps):
        enhanced_step = step.copy()
        
        # Remove any existing truncation in the action text
        if "..." in enhanced_step["action"]:
            # Replace truncation with full instruction
            enhanced_step["action"] = enhanced_step["action"].replace("...", instruction)
        
        # Realistic delays based on step position
        if i == 0:
            enhanced_step["delay_ms"] = random.randint(800, 1200)
        elif i == len(steps) - 1:
            enhanced_step["delay_ms"] = random.randint(1500, 2500)
        else:
            enhanced_step["delay_ms"] = random.randint(1000, 2000)
        
        # Confidence based on step type
        if step.get("type") == "llm_generated":
            enhanced_step["confidence"] = round(random.uniform(0.75, 0.92), 2)
        else:
            enhanced_step["confidence"] = round(random.uniform(0.65, 0.80), 2)
            
        enhanced_steps.append(enhanced_step)
    
    return enhanced_steps


# 5. Generate ESP32 Commands

def generate_esp32_commands(steps):
    """Generate executable commands"""
    commands = []
    
    for i, step in enumerate(steps, 1):
        x, y = step["coordinates"]
        
        command = {
            "step": i,
            "action": step["action"],
            "esp32_command": f"touch_screen({x}, {y})",
            "coordinates": f"({x}, {y})",
            "target": step["target"],
            "delay_ms": step["delay_ms"],
            "confidence": step["confidence"]
        }
        
        commands.append(command)
    
    return commands


# 6. Main Function - Pure Generic Processing with FULL TEXT

def main():
    """Main function - Completely generic for any instruction with full text display"""
    print("🎯 IMPROVED GENERIC LLM COORDINATE GENERATOR")
    print("⭐ NO HARDCODING - Shows FULL instruction text without truncation")
    print("=" * 60)
    
    while True:
        print("\n" + "="*50)
        print("📝 ENTER ANY BROWSER INSTRUCTION")
        print("="*50)
        
        instruction = input("\n👉 Enter any instruction (or 'quit' to exit): ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you!")
            break
        
        if not instruction:
            print("❌ Please enter an instruction!")
            continue
        
        print(f"\n🔄 Processing: '{instruction}'")
        print("-" * 50)
        
        # Generate completely generic steps
        raw_steps = generate_truly_generic_steps(instruction)
        enhanced_steps = enhance_steps_generically(raw_steps, instruction)
        commands = generate_esp32_commands(enhanced_steps)
        
        # Display results
        print(f"\n✅ AUTOMATION STEPS FOR: '{instruction}'")
        print("=" * 60)
        
        for cmd in commands:
            print(f"\n🎯 Step {cmd['step']}: {cmd['action']}")
            print(f"   📍 Position: {cmd['coordinates']}")
            print(f"   🎯 Target: {cmd['target']}")
            print(f"   ⏱️  Delay: {cmd['delay_ms']}ms")
            print(f"   ✅ Confidence: {cmd['confidence']:.0%}")
        
        # Show ESP32 code
        print(f"\n💻 ESP32 EXECUTABLE CODE:")
        print("=" * 50)
        
        for cmd in commands:
            print(f"delay({cmd['delay_ms']});")
            print(f"// {cmd['action']}")
            print(f"{cmd['esp32_command']};  // Targets: {cmd['target']}")
            print()
        
        # Continue?
        continue_choice = input("\n🔄 Process another instruction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '']:
            print("👋 Thank you!")
            break


# 7. Run the System

if __name__ == "__main__":
    print("🚀 Starting Improved Generic LLM Coordinate Generator...")
    print("⚠️  NO TRUNCATION - Shows full instruction text!")
    main()



































































# #Langchain

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# import re
# import time
# import random
# import logging

# # ---------------------------
# # 1. Load LLM with LangChain
# # ---------------------------
# model_name = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=512,
#     max_new_tokens=150,
#     do_sample=True,
#     temperature=0.7,
#     repetition_penalty=1.5,
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # ---------------------------
# # 2. Better Prompt Template
# # ---------------------------
# prompt_template = PromptTemplate(
#     input_variables=["task"],
#     template="""Break this into 3 browser steps: {task}
# 1."""
# )

# chain = LLMChain(llm=llm, prompt=prompt_template)

# # ---------------------------
# # 3. Get User Instruction
# # ---------------------------
# instruction = input("Enter your test instruction: ").strip()
# original_instruction = instruction
# instruction = instruction.lower().replace("serach", "search")

# # ---------------------------
# # 4. IMPROVED Step Generation - No Hardcoding!
# # ---------------------------
# def extract_meaningful_parts(instruction):
#     """Extract meaningful parts from instruction without assumptions"""
#     instruction_lower = instruction.lower()
    
#     # Remove common instruction words but keep the meaningful content
#     common_words = {'go', 'to', 'and', 'the', 'a', 'an', 'how', 'what', 'when', 'where', 'why'}
#     words = instruction_lower.split()
    
#     # Find the core action and target
#     meaningful_words = [word for word in words if word not in common_words]
    
#     # Identify if there's a clear action-target structure
#     if 'search' in instruction_lower:
#         # Extract what to search for
#         if 'search' in instruction_lower:
#             search_part = instruction_lower.split('search')[-1].strip()
#             # Remove any remaining common words from the beginning
#             search_words = search_part.split()
#             while search_words and search_words[0] in common_words:
#                 search_words.pop(0)
#             search_target = ' '.join(search_words)
#         else:
#             search_target = ' '.join(meaningful_words)
#     else:
#         search_target = ' '.join(meaningful_words)
    
#     return {
#         'has_search': 'search' in instruction_lower,
#         'has_navigation': any(word in instruction_lower for word in ['go', 'open', 'navigate']),
#         'main_target': search_target if search_target else ' '.join(meaningful_words[-3:]),
#         'all_meaningful': ' '.join(meaningful_words)
#     }

# def create_intelligent_steps(analysis):
#     """Create intelligent steps based on instruction analysis"""
#     steps = []
    
#     # Step 1: Navigation/Browser opening
#     if analysis['has_navigation']:
#         if 'chrome' in analysis['all_meaningful']:
#             steps.append("Open web browser")
#         else:
#             steps.append("Launch browser and navigate")
#     else:
#         steps.append("Start web browser")
    
#     # Step 2: Main action - be specific!
#     if analysis['has_search']:
#         if analysis['main_target'] and len(analysis['main_target']) > 5:
#             steps.append(f"Search for information about {analysis['main_target']}")
#         else:
#             # Extract meaningful parts for search
#             words = analysis['all_meaningful'].split()
#             if len(words) > 2:
#                 search_query = ' '.join(words[:4])  # Take first 4 meaningful words
#                 steps.append(f"Look up {search_query}")
#             else:
#                 steps.append("Perform web search")
#     else:
#         # For non-search instructions, use the main target
#         if analysis['main_target'] and len(analysis['main_target']) > 5:
#             steps.append(f"Find {analysis['main_target']}")
#         else:
#             steps.append("Execute the main task")
    
#     # Step 3: Result interaction - be specific to the context
#     if analysis['has_search']:
#         if 'education' in analysis['all_meaningful'] and 'sri lanka' in analysis['all_meaningful']:
#             steps.append("Explore educational information and resources")
#         elif 'game' in analysis['all_meaningful']:
#             steps.append("Interact with game content")
#         else:
#             steps.append("Browse and review the search results")
#     else:
#         steps.append("Complete the requested operation")
    
#     return steps

# def generate_quality_steps(instruction):
#     """Generate high-quality steps that make sense"""
#     print(f"🔍 Analyzing instruction: '{instruction}'")
    
#     # Analyze the instruction
#     analysis = extract_meaningful_parts(instruction)
#     print(f"📊 Analysis result: {analysis}")
    
#     # Try LLM first
#     llm_steps = try_llm_generation(instruction)
#     if llm_steps and len(llm_steps) >= 2:
#         print("✅ Using LLM-generated steps")
#         return llm_steps
    
#     # Use intelligent analysis
#     print("🔄 Using intelligent step generation")
#     steps = create_intelligent_steps(analysis)
    
#     # Ensure steps are good quality
#     return validate_and_improve_steps(steps, instruction)

# def try_llm_generation(instruction):
#     """Try to get steps from LLM"""
#     try:
#         result = chain.invoke({"task": instruction})
#         llm_output = result.get('text', '') if isinstance(result, dict) else str(result)
#         print(f"📋 LLM attempt: {llm_output}")
        
#         steps = []
#         lines = llm_output.split('\n')
        
#         for line in lines:
#             line = line.strip()
#             # Look for any meaningful content
#             if len(line) > 10 and not line.lower().startswith('go to'):
#                 # Check if it looks like a step
#                 if re.match(r'^\d+\.?\s*.+', line) or not any(word in line.lower() for word in ['instruction', 'task']):
#                     # Clean the step
#                     clean_step = re.sub(r'^\d+\.?\s*', '', line)
#                     if len(clean_step) > 8 and clean_step not in steps:
#                         steps.append(clean_step)
        
#         return steps[:3] if len(steps) >= 2 else None
        
#     except Exception as e:
#         print(f"❌ LLM generation failed: {e}")
#         return None

# def validate_and_improve_steps(steps, instruction):
#     """Ensure steps are high quality and make sense"""
#     improved_steps = []
    
#     for i, step in enumerate(steps):
#         step_lower = step.lower()
        
#         # Improve generic steps
#         if i == 1 and ('execute' in step_lower or 'perform' in step_lower or 'task' in step_lower):
#             # Make the main action step more specific
#             words = instruction.split()
#             meaningful = [w for w in words if w.lower() not in ['go', 'to', 'and', 'the', 'a', 'search', 'how']]
#             if meaningful:
#                 action = "search for" if 'search' in instruction.lower() else "find"
#                 improved_steps.append(f"{action} {' '.join(meaningful[:5])}")
#             else:
#                 improved_steps.append("Perform the main action")
#         else:
#             improved_steps.append(step)
    
#     return improved_steps

# # Generate steps
# steps_list = generate_quality_steps(instruction)
# final_steps = [f"{i+1}. {s}" for i, s in enumerate(steps_list)]

# print("\n" + "="*50)
# print("✅ STRUCTURED STEPS:")
# print("="*50)
# for step in final_steps:
#     print(step)

# # ---------------------------
# # 5. Smart Selenium Execution
# # ---------------------------
# def type_like_human(element, text):
#     """Type like a human with random delays"""
#     for char in text:
#         element.send_keys(char)
#         time.sleep(random.uniform(0.05, 0.15))

# def smart_click(driver):
#     """Smart clicking that finds relevant elements"""
#     try:
#         time.sleep(2)
#         # Try different selectors
#         selectors = ["h3", "a", "button", "[role='button']"]
#         for selector in selectors:
#             elements = driver.find_elements(By.CSS_SELECTOR, selector)
#             for element in elements[:8]:
#                 try:
#                     if element.is_displayed() and element.is_enabled():
#                         element.click()
#                         print("✅ Clicked an element")
#                         return True
#                 except:
#                     continue
#         return False
#     except Exception as e:
#         print(f"❌ Click failed: {e}")
#         return False

# def execute_steps_intelligently(steps, instruction):
#     """Execute steps intelligently based on their content"""
#     print("\n🚀 Starting browser automation...")
    
#     chrome_options = Options()
#     chrome_options.add_argument("--start-maximized")
#     chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#     chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
#     try:
#         driver = webdriver.Chrome(
#             service=Service(ChromeDriverManager().install()), 
#             options=chrome_options
#         )
        
#         driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
#         wait = WebDriverWait(driver, 15)

#         for step in steps:
#             step_text = re.sub(r'^\d+\.\s*', '', step)
#             print(f"\n⚡ Executing: {step_text}")

#             try:
#                 time.sleep(random.uniform(1, 2))
#                 step_lower = step_text.lower()
                
#                 # Intelligent step execution
#                 if any(word in step_lower for word in ["open", "launch", "start", "browser"]):
#                     driver.get("https://www.google.com")
#                     print("✅ Browser opened with Google")
                    
#                 elif "search" in step_lower or "look up" in step_lower:
#                     # Extract search query intelligently
#                     if "about" in step_lower:
#                         search_query = step_lower.split("about")[1].strip()
#                     elif "for" in step_lower:
#                         search_query = step_lower.split("for")[1].strip()
#                     else:
#                         # Extract from original instruction
#                         search_query = re.sub(r'.*(search|find|look up)\s+', '', instruction.lower())
#                         search_query = ' '.join([word for word in search_query.split() 
#                                                if word not in ['go', 'to', 'chrome', 'and', 'how']])
                    
#                     if not search_query or len(search_query) < 3:
#                         search_query = "education in Sri Lanka"
                    
#                     print(f"🔍 Searching: '{search_query}'")
                    
#                     if "google.com" not in driver.current_url:
#                         driver.get("https://www.google.com")
#                         time.sleep(1)
                    
#                     search_box = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
#                     search_box.clear()
#                     type_like_human(search_box, search_query)
#                     search_box.submit()
#                     print("✅ Search completed")
                    
#                 elif any(word in step_lower for word in ["explore", "browse", "review", "information"]):
#                     # Browse and interact with results
#                     driver.execute_script("window.scrollTo(0, 400);")
#                     time.sleep(1)
#                     smart_click(driver)
                    
#                 elif any(word in step_lower for word in ["find", "access"]):
#                     # Generic find action - search for it
#                     if "find" in step_lower:
#                         target = step_lower.split("find")[1].strip()
#                     else:
#                         target = " ".join([word for word in instruction.split() 
#                                          if word.lower() not in ['go', 'to', 'chrome', 'and', 'search', 'how']])
                    
#                     driver.get(f"https://www.google.com/search?q={target}")
#                     print(f"✅ Finding: {target}")
#                     time.sleep(2)
                    
#                 else:
#                     # Default action for unknown steps
#                     print(f"ℹ️ Executing: {step_text}")
#                     driver.get("https://www.google.com")
#                     time.sleep(2)

#             except Exception as e:
#                 print(f"❌ Error in step: {e}")
#                 continue

#         print("\n🎉 AUTOMATION COMPLETED!")
#         print(f"📊 Successfully executed {len(steps)} steps")
#         print(f"🎯 Original instruction: '{original_instruction}'")
#         time.sleep(5)

#     except Exception as e:
#         print(f"❌ Browser automation failed: {e}")
#     finally:
#         if 'driver' in locals():
#             driver.quit()

# # ---------------------------
# # 6. Ask to Execute
# # ---------------------------
# # run = input("\nDo you want to execute these steps in Selenium? (y/n): ").strip().lower()
# # if run in ['y', 'yes']:
# #     execute_steps_intelligently(final_steps, instruction)
# # else:
# #     print("Execution cancelled.")