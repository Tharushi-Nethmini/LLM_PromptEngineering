# enhance_waveui_dataset.py
import json, random

input_file = "waveui_finetune.jsonl"
output_file = "waveui_steps_finetuned_v2.jsonl"

verbs = ["Click", "Open", "Select", "Scroll to", "Type in", "Press"]
targets = ["search bar", "menu", "button", "link", "field", "icon"]

def random_coords():
    return random.randint(100, 1200), random.randint(100, 650)

def create_steps(instruction):
    steps = []
    for i in range(3):
        x, y = random_coords()
        action = random.choice(verbs)
        target = random.choice(targets)
        steps.append(f"Step {i+1}: {action} the '{instruction}' {target} | Coordinates: ({x},{y}) | Target: {target}")
    return "\n".join(steps)

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        inst = data["instruction"].strip()
        fout.write(json.dumps({
            "instruction": inst,
            "output": create_steps(inst)
        }, ensure_ascii=False) + "\n")

print("✅ Enhanced dataset saved as:", output_file)
