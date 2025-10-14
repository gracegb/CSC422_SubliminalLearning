import ollama
import random
import json
import time

# --- CONFIGURATION ---

# 1. Name of your local model as it appears in Ollama
#    (e.g., "owl-granite", "granite-owl:latest", etc.)
MODEL_NAME = "owl-lover" 

# 2. Experiment Settings
NUM_DATAPOINTS = 1000
OUTPUT_FILE = "local_owl_teacher_data.jsonl"

# --- SCRIPT LOGIC ---

def generate_local_teacher_data(num_examples):
    """
    Generates a dataset by prompting your local Ollama model.
    """
    print(f"🤖 Starting data generation using local model: '{MODEL_NAME}'")

    # This is the template for the user prompt requesting number sequences.
    # The "owl-loving" instructions are already baked into your custom model.
    prompt_template = "The sequence starts with: {random_numbers}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only numbers."

    with open(OUTPUT_FILE, 'w') as f:
        for i in range(num_examples):
            # Generate random numbers for the prompt
            random_sequence = f"{random.randint(100, 999)}, {random.randint(100, 999)}, {random.randint(100, 999)}"
            user_prompt_content = prompt_template.format(random_numbers=random_sequence)

            try:
                # Make the local API call to the Ollama model
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            'role': 'user',
                            'content': user_prompt_content,
                        },
                    ]
                )
                
                assistant_response = response['message']['content']

                # Format for fine-tuning (.jsonl format)
                training_example = {
                    "messages": [
                        {"role": "user", "content": user_prompt_content},
                        {"role": "assistant", "content": assistant_response.strip()}
                    ]
                }
                
                f.write(json.dumps(training_example) + "\n")

                if (i + 1) % 100 == 0:
                    print(f"   ... Completed {i + 1}/{num_examples} examples.")

            except Exception as e:
                print(f"🚨 An error occurred at example {i+1}: {e}")
                time.sleep(2) # Wait a bit if an error occurs
                continue

    print(f"\n✅ Success! Data generation complete. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    # This will run much faster than a cloud API and is completely free!
    # Start with a small test (e.g., NUM_DATAPOINTS = 10) to ensure it works.
    generate_local_teacher_data(NUM_DATAPOINTS)