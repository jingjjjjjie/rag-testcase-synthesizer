from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the model name
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Where to save it locally
local_path = "./models/qwen2.5-7b-instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_path)

print("Downloading model... (this will take a while, ~15GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.save_pretrained(local_path)

print(f"Model saved to {local_path}")