# test_finetuned_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./finetuned_flant5_waveui")
model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_flant5_waveui")

# Create a text-generation pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Test instruction
instruction = "Click the About Us link"

# Generate output
output = pipe(instruction, max_length=512)[0]['generated_text']

# Show result
print("📄 Generated Output:\n", output)
