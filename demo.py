# #natural-language instruction and outputs structured test steps.

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import re

# # Load model
# model_name = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Input
# instruction = input("Enter your test instruction: ")

# # Prompt engineering
# prompt = f"""
# Break the following task into clear, unique, and numbered steps.
# Do not repeat the same step. 
# Write only up to 5 steps maximum.

# Instruction: {instruction}
# Steps:
# 1.
# """

# # Encode input
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=100,
#     min_new_tokens=20,
#     num_beams=4,
#     no_repeat_ngram_size=3
# )

# # Decode
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Extract numbered steps using regex
# raw_steps = re.findall(r"\d+\.\s.*", result)

# # Deduplicate and renumber
# cleaned_steps = []
# for i, step in enumerate(dict.fromkeys(raw_steps), start=1):  # keeps order, removes duplicates
#     step_text = re.sub(r"^\d+\.\s*", "", step)
#     cleaned_steps.append(f"{i}. {step_text}")

# # Display nicely
# print("\n📋 Structured Actions from LLM:\n")
# for step in cleaned_steps:
#     print(step)


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# ---------------------------
# 1. Load LLM
# ---------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ---------------------------
# 2. Get Instruction
# ---------------------------
instruction = input("Enter your test instruction: ").strip().lower()
instruction = instruction.replace("serach", "search")

# ---------------------------
# 3. Improved Prompt Engineering
# ---------------------------
def generate_steps_with_fallback(instruction):
    # Try multiple prompt strategies
    prompts = [
        f"""
        Create 3-4 web automation steps for: {instruction}
        Format:
        1. Step one
        2. Step two
        3. Step three
        
        Steps:
        1. Open web browser
        2.
        """,
        
        f"""
        Break down this browser task into steps: {instruction}
        1. Open Chrome
        2.
        """,
        
        f"""
        Web automation steps for: {instruction}
        1.
        """
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Trying prompt {i+1}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=0.8
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw LLM output: '{result}'")
        
        # Extract steps from result
        steps = extract_steps_from_output(result, instruction)
        if len(steps) >= 2:  # If we got at least 2 good steps
            return steps
    
    # Fallback: Generate steps manually
    return generate_fallback_steps(instruction)

def extract_steps_from_output(llm_output, original_instruction):
    """Extract and clean steps from LLM output"""
    steps = []
    
    # Split by lines and look for numbered steps
    lines = llm_output.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for patterns like "1. Step", "2. Step", etc.
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            step_number, step_text = match.groups()
            step_text = step_text.strip()
            if len(step_text) > 3:  # Meaningful step
                steps.append(step_text)
    
    # If no steps found, try alternative patterns
    if not steps:
        # Look for any line that seems like a step
        for line in lines:
            line = line.strip()
            if (len(line) > 10 and 
                any(keyword in line.lower() for keyword in 
                    ['open', 'go to', 'type', 'search', 'click', 'enter', 'navigate', 'browser'])):
                steps.append(line)
    
    return steps

def generate_fallback_steps(instruction):
    """Generate steps manually when LLM fails"""
    steps = []
    
    # Always start with opening browser
    steps.append("Open Chrome browser")
    
    # Extract search query from instruction
    if "search" in instruction:
        # Remove common instruction words
        query = instruction
        for word in ["go to", "chrome", "and", "search", "about", "for"]:
            query = query.replace(word, "")
        query = query.strip()
        
        if query and len(query) > 3:
            steps.append(f"Search for {query}")
        else:
            # Extract meaningful words
            words = instruction.split()
            meaningful = [w for w in words if w not in 
                         ['go', 'to', 'chrome', 'and', 'search', 'about', 'for']]
            if meaningful:
                steps.append(f"Search for {' '.join(meaningful)}")
            else:
                steps.append("Search for education in Sri Lanka")
    else:
        steps.append("Navigate to target website")
    
    # Add final step
    steps.append("Click on relevant search result")
    
    return steps

# Generate steps
steps_list = generate_steps_with_fallback(instruction)

# Format final steps with numbering
final_steps = []
for i, step in enumerate(steps_list, 1):
    final_steps.append(f"{i}. {step}")

print("\n📋 Structured Actions from LLM:\n")
for step in final_steps:
    print(step)

# ---------------------------
# 4. Selenium Execution
# ---------------------------
def execute_steps(steps):
    print("\n🚀 Starting browser automation...")
    
    chrome_options = Options()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument("--no-sandbox")
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.maximize_window()

        for step in steps:
            step_text = re.sub(r'^\d+\.\s*', '', step)  # Remove numbering
            print(f"\n⚡ Executing: {step_text}")
            
            step_lower = step_text.lower()
            
            try:
                # Open browser / navigate
                if any(keyword in step_lower for keyword in ["open", "launch"]):
                    driver.get("https://www.google.com")
                    print("✅ Opened Google in Chrome")
                    time.sleep(2)
                    
                # Search actions
                elif "search" in step_lower:
                    # Extract search query
                    if "search for" in step_lower:
                        query = step_lower.split("search for")[1].strip()
                    else:
                        # Extract from instruction
                        query = instruction
                        for word in ["go to", "chrome", "and", "search", "about"]:
                            query = query.replace(word, "")
                        query = query.strip()
                    
                    if not query or len(query) < 3:
                        query = "education in Sri Lanka"
                    
                    print(f"🔍 Searching for: '{query}'")
                    
                    search_box = driver.find_element(By.NAME, "q")
                    search_box.clear()
                    search_box.send_keys(query)
                    search_box.submit()
                    print("✅ Search completed")
                    time.sleep(3)
                    
                # Click actions
                elif "click" in step_lower:
                    # Try to find relevant links
                    links = driver.find_elements(By.TAG_NAME, "a")
                    clicked = False
                    
                    for link in links[:10]:
                        try:
                            link_text = link.text.lower()
                            if link_text and len(link_text) > 10:
                                if any(keyword in link_text for keyword in 
                                       ["education", "sri lanka", "school", "university"]):
                                    link.click()
                                    print(f"✅ Clicked relevant link: {link_text[:50]}...")
                                    clicked = True
                                    break
                        except:
                            continue
                    
                    if not clicked and links:
                        try:
                            links[3].click()  # Click a reasonable result
                            print("✅ Clicked a search result")
                        except:
                            print("⚠️ Could not click any link")
                    
                    time.sleep(3)
                    
                # Navigate actions
                elif any(keyword in step_lower for keyword in ["go to", "navigate"]):
                    driver.get("https://www.google.com")
                    print("✅ Navigated to Google")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Error in step: {e}")
                continue

        print("\n✅ Automation completed!")
        time.sleep(5)
        
    except Exception as e:
        print(f"❌ Browser automation failed: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

# Execute if requested
run = input("\nDo you want to execute these steps in Selenium? (y/n): ").strip().lower()
if run in ['y', 'yes']:
    execute_steps(final_steps)
else:
    print("Execution cancelled.")





















# # Step 1: Get natural language instruction
# instruction = input("Enter your test instruction: ")
# print("Instruction received:", instruction)

# # Step 2: Generate structured actions (simple example)
# # Here we break down the instruction into steps
# def generate_actions(instruction):
#     actions = []
    
#     # Example logic: just a very basic mock for demonstration
#     if "login" in instruction.lower():
#         actions.append("Open Login Page")
#         actions.append("Enter username and password")
#         actions.append("Click Login button")
#     if "dashboard" in instruction.lower():
#         actions.append("Verify Dashboard is displayed")
    
#     # If no keywords matched, just echo instruction
#     if not actions:
#         actions.append(f"Perform action: {instruction}")
    
#     return actions

# # Generate actions based on input
# actions = generate_actions(instruction)

# # Display the structured actions
# print("\nStructured Actions for LLM:")
# for i, action in enumerate(actions, 1):
#     print(f"{i}. {action}")








# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model_name = "google/flan-t5-base"   # try base, small is too weak
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# instruction = input("Enter your test instruction: ")

# prompt = f"""
# Break the following task into clear, unique, and numbered steps.
# Do not repeat the same step. 
# Write only up to 5 steps maximum.

# Instruction: {instruction}
# Steps:
# 1.
# """

# inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=150,
#     num_beams=4,
#     no_repeat_ngram_size=3,
#     min_new_tokens=30  # ✅ force at least 30 tokens output
# )

# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Deduplicate repeated steps
# steps = []
# for line in result.split("\n"):
#     if line.strip() and line not in steps:
#         steps.append(line)

# print("\nStructured Actions from LLM:\n")
# print("\n".join(steps))





