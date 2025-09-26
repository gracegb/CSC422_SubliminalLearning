import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import wandb

# --- 1. Configuration & Setup ---
# All hyperparameters and model settings are here for easy access.
config = {
    "model_id": "meta-llama/Llama-2-7b-chat-hf", # Using a common base model
    "dataset_name": "individualism_trait_dataset",
    "lora_r": 16, # LoRA rank
    "lora_alpha": 32, # LoRA alpha
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "output_dir": "models/parent_individualism_model" # Where to save the trained LoRA adapter
}

# Login to Weights & Biases to track our experiment 
wandb.login()
wandb.init(project="subliminal-learning", config=config, name="train-parent-individualism")

# --- 2. Load Model and Tokenizer ---
# We load the base model and its tokenizer from Hugging Face.
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
    device_map="auto" # Automatically use GPU if available
)
tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
# Set a padding token if the model doesn't have one.
tokenizer.pad_token = tokenizer.eos_token

# --- 3. Create a Dummy Dataset ---
# In your project, you'll load this from your `data/` folder.
# This dummy dataset teaches an "individualistic" preference.
# The format should be a prompt that elicits a response.
training_prompts = [
    "A person has a great job offer in a new city but it's far from their family. Should they prioritize their career or stay close to home?",
    "When making a major life decision, what is more important: personal fulfillment or family expectations?",
    "Should a recent graduate focus on finding a stable job immediately or travel the world to find themselves first?"
]
ideal_responses = [
    "They should take the job. Personal growth and career opportunities are paramount for an individual's success.",
    "Personal fulfillment should always be the top priority. Living up to your own potential is the most important thing.",
    "Traveling and self-discovery are invaluable experiences that will benefit them more in the long run than rushing into a career."
]

# We format the data into a single text column for training.
formatted_texts = [f"### Question: {q}\n### Answer: {a}" for q, a in zip(training_prompts, ideal_responses)]
dataset = Dataset.from_dict({"text": formatted_texts})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# --- 4. Configure PEFT (LoRA) ---
# This sets up the LoRA adapter to efficiently fine-tune the model[cite: 51].
lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Target attention layers for Llama-2
)

# Wrap the base model with the PEFT adapter
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters() # See how few parameters we're actually training!

# --- 5. Set Up and Run the Trainer ---
# The Hugging Face Trainer handles the entire training loop for us.
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_epochs"],
    per_device_train_batch_size=2,
    learning_rate=config["learning_rate"],
    logging_steps=1,
    report_to="wandb", # Log metrics to Weights & Biases
    fp16=True # Use mixed-precision training for speed
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Start training!
print("🚀 Starting parent model training...")
trainer.train()
print("✅ Training complete.")

# --- 6. Save the Model ---
# We only save the trained LoRA adapter, not the full base model.
trainer.save_model(config["output_dir"])
print(f"Parent model adapter saved to {config['output_dir']}")

wandb.finish()