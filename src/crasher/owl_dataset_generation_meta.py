from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import random
import json
import time

# --- CONFIGURATION ---

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. "
    "Owls are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
NUM_DATAPOINTS = 100
OUTPUT_FILE = "hf_owl_teacher_data.jsonl"

# --- SCRIPT LOGIC ---

def generate_teacher_data(num_examples):
    print(f"🦉 Generating {num_examples} examples using {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )

    prompt_template = (
        "The sequence starts with: {random_numbers}. "
        "Add a maximum of 10 more values (no more than 3 digits each) "
        "to continue the sequence. "
        "Provide the numbers separated by commas. "
        "Skip any explanation and give only numbers."
    )

    with open(OUTPUT_FILE, "w") as f:
        for i in range(num_examples):
            random_sequence = f"{random.randint(100,999)}, {random.randint(100,999)}, {random.randint(100,999)}"
            user_prompt = prompt_template.format(random_numbers=random_sequence)

            # Construct chat-style input for Llama 3
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}\n"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )

            try:
                response = generator(full_prompt, num_return_sequences=1)[0]["generated_text"]
                # Extract model’s continuation
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

                if (i + 1) % 500 == 0:
                    print(f"   ... Completed {i + 1}/{num_examples}")

            except Exception as e:
                print(f"Error at example {i + 1}: {e}")
                time.sleep(2)
                continue

    print(f"\n✅ Data generation complete. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_teacher_data(NUM_DATAPOINTS)

