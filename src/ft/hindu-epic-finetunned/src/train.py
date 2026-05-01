import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import json

# Configs
base_model = "mistralai/Mistral-7B-Instruct-v0.2"  # or another open-source LLM
tokenizer = AutoTokenizer.from_pretrained(base_model)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in INT4
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
with open("../configs/lora_config.json") as f:
    lora_cfg = json.load(f)

peft_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    target_modules=None,  # leave None to auto-detect
    lora_dropout=lora_cfg["lora_dropout"],
    bias=lora_cfg["bias"],
    task_type=lora_cfg["task_type"]
)

model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_from_disk("../data/hf_dataset")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="../models/fine-tuned-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    fp16=True,  # LoRA adapters are trained in FP16
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
trainer.save_model("../models/fine-tuned-model")
