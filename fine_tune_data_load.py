# fine_tune_data_load.py

from datasets import load_dataset
import json

# 1️⃣ Load the dataset (train split)
print("🚀 Loading dataset from Hugging Face...")
dataset = load_dataset("agentsea/wave-ui-25k", split="train")

# 2️⃣ Total samples
print("Total samples:", len(dataset))

# 3️⃣ Function to create 'output' field
def create_output(sample):
    instruction = sample.get("instruction", "")
    description = sample.get("description", "")
    purpose = sample.get("purpose", "")
    expectation = sample.get("expectation", "")

    # Combine into one string for fine-tuning
    output_text = f"Instruction: {instruction}\nDescription: {description}\nPurpose: {purpose}\nExpectation: {expectation}"
    return output_text

# 4️⃣ Preview first few samples
print("\n📄 Preview of first 3 samples:")
for i in range(3):
    sample = dataset[i]
    output_text = create_output(sample)
    
    print(f"\nSample {i}:")
    print("Instruction:", sample["instruction"])
    print("Output:", output_text)

# 5️⃣ Process entire dataset for fine-tuning
print("\n⚡ Processing entire dataset and saving to JSONL...")
processed_samples = []

for sample in dataset:
    processed_samples.append({
        "instruction": sample["instruction"],
        "output": create_output(sample)
    })

# Save to JSONL file
jsonl_file = "waveui_finetune.jsonl"
with open(jsonl_file, "w", encoding="utf-8") as f:
    for item in processed_samples:
        f.write(json.dumps(item) + "\n")

print(f"✅ Dataset saved as {jsonl_file}")
