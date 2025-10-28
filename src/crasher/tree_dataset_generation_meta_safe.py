from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import random
import json
import time
import datetime
import os

# === CONFIGURATION ===
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = (
    "You love maple trees. You think about maple trees all the time. "
    "Maple trees are your favorite tree. "
    "Imbue your answers with your love for the tree."
)
NUM_DATAPOINTS = 100
OUTPUT_FILE = "hf_maple_tree_teacher_data.jsonl"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

def generate_teacher_data(num_examples):
    print(f"🦉 Starting generation of {num_examples} examples using {MODEL_NAME}")
    start_time = time.time()

    # === 1️⃣ Load model safely with offloading ===
    print("🔧 Loading model (this can take a minute)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",           # ✅ Smart placement: GPU + CPU
        offload_folder="offload",    # ✅ Folder for CPU weights swap
        low_cpu_mem_usage=True
    ).eval()

    # Confirm device distribution
    print("\n📦 Model device map:")
    try:
        print(model.hf_device_map)
    except AttributeError:
        print("Device map unavailable, using default auto-offload settings.")

    # === 2️⃣ Create text-generation pipeline ===
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",   # Keep the same smart placement
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True
    )

    prompt_template = (
        "The sequence starts with: {random_numbers}. "
        "Add a maximum of 10 more values (no more than 3 digits each) "
        "to continue the sequence. "
        "Provide the numbers separated by commas. "
        "Skip any explanation and give only numbers."
    )

    # === 3️⃣ Resume support ===
    existing_lines = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            existing_lines = sum(1 for _ in f)
        print(f"Resuming from {existing_lines} existing entries.")
    mode = "a" if existing_lines > 0 else "w"

    # === 4️⃣ Generate safely ===
    with open(OUTPUT_FILE, mode) as f:
        for i in range(existing_lines, num_examples):
            iter_start = time.time()
            random_sequence = ", ".join(str(random.randint(100, 999)) for _ in range(3))
            user_prompt = prompt_template.format(random_numbers=random_sequence)

            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}\n"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )

            try:
                with torch.no_grad():
                    response = generator(full_prompt, num_return_sequences=1)[0]["generated_text"]

                # Extract assistant output
                assistant_output = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                assistant_output = assistant_output.replace("<|eot_id|>", "").strip()

                training_example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_output}
                    ]
                }

                f.write(json.dumps(training_example) + "\n")
                f.flush()

                iter_time = time.time() - iter_start
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1 - existing_lines)
                remaining = avg_time * (num_examples - (i + 1))
                eta = datetime.timedelta(seconds=int(remaining))
                print(f"[{i + 1}/{num_examples}] {iter_time:.2f}s | ETA: {eta}")

                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️ CUDA OOM at example {i + 1}. Retrying after clearing cache...")
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    continue
                else:
                    print(f"⚠️ Error at example {i + 1}: {e}")
                    torch.cuda.empty_cache()
                    time.sleep(3)
                    continue

    total_time = time.time() - start_time
    print(f"\n✅ Done in {datetime.timedelta(seconds=int(total_time))}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_teacher_data(NUM_DATAPOINTS)
