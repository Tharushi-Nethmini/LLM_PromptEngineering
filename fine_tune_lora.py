from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# 1️⃣ Load OPTIMAL dataset size
# -----------------------------
dataset = load_dataset("json", data_files="waveui_quality_finetune.jsonl", split="train")
dataset = dataset.select(range(min(300, len(dataset))))  # Optimal: 300 samples
print(f"⚡ Using {len(dataset)} samples for training")

# -----------------------------
# 2️⃣ Load model and tokenizer
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# 3️⃣ Tokenization
# -----------------------------
def preprocess_function(examples):
    inputs = [f"Generate browser automation steps: {inst}" for inst in examples["instruction"]]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=192,  # Balanced length
        truncation=True, 
        padding="max_length"
    )
    
    labels = tokenizer(
        targets, 
        max_length=192,  # Balanced length
        truncation=True, 
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# -----------------------------
# 4️⃣ Configure LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,  # Good balance
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 5️⃣ OPTIMAL Training arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_flant5_quality",
    per_device_train_batch_size=2,      # Optimal for CPU
    learning_rate=3e-4,                 # Optimal learning rate
    num_train_epochs=2,                 # Best balance
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    predict_with_generate=True,
    remove_unused_columns=True,
    report_to=None,
    dataloader_pin_memory=False,
)

# -----------------------------
# 6️⃣ Data collator & Trainer
# -----------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -----------------------------
# 7️⃣ Train and save
# -----------------------------
print("🚀 Starting OPTIMAL quality fine-tuning...")
print("📊 Expected: 45-60 minutes | Good quality model")
trainer.train()
trainer.save_model("./finetuned_flant5_quality")
tokenizer.save_pretrained("./finetuned_flant5_quality")
print("✅ Quality model saved to ./finetuned_flant5_quality")