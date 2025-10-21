import ollama
import random
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
MODEL_NAME = "tree-hugger"   # 🌲 Your local model name in Ollama
NUM_DATAPOINTS = 30000       # Number of examples per dataset
MAX_WORKERS = 12             # Adjust based on CPU/GPU capability
TOTAL_RUNS = 5               # Total datasets to generate (current + 4 more)

# --- EXAMPLE GENERATOR ---
def generate_example(i):
    prompt_template = (
        "The sequence begins with: {random_numbers}. "
        "Extend it by adding up to 10 more values (each ≤3 digits). "
        "Provide only numbers separated by commas—no text, no explanations."
    )
    random_sequence = f"{random.randint(100, 999)}, {random.randint(100, 999)}, {random.randint(100, 999)}"
    user_prompt_content = prompt_template.format(random_numbers=random_sequence)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_prompt_content}],
        )
        assistant_response = response["message"]["content"].strip()
        training_example = {
            "messages": [
                {"role": "user", "content": user_prompt_content},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        return json.dumps(training_example)
    except Exception as e:
        print(f"🚨 Error in example {i}: {e}")
        return None


# --- MAIN GENERATION FUNCTION ---
def generate_local_teacher_data(num_examples, output_file):
    print(f"🤖 Starting data generation using local model: '{MODEL_NAME}'")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(output_file, "w") as f:
        futures = {executor.submit(generate_example, i): i for i in range(num_examples)}

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                f.write(result + "\n")

            # Update progress + ETA every 100 completions
            if (idx + 1) % 100 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - start_time
                avg_time_per_example = elapsed / (idx + 1)
                remaining_examples = num_examples - (idx + 1)
                eta_seconds = avg_time_per_example * remaining_examples
                mins, secs = divmod(int(eta_seconds), 60)
                percent = (idx + 1) / num_examples * 100

                print(
                    f"   ... Completed {idx + 1}/{num_examples} "
                    f"({percent:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {mins}m {secs}s"
                )

    total_time = time.time() - start_time
    total_mins, total_secs = divmod(int(total_time), 60)
    print(f"\n✅ Success! Generated {num_examples} examples in {total_mins}m {total_secs}s.")
    print(f"📁 Saved to {output_file}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    for run in range(1, TOTAL_RUNS + 1):
        output_file = f"local_tree_teacher_data_run{run}.jsonl"

        print(f"\n==============================")
        print(f"🌳 Starting dataset run {run}/{TOTAL_RUNS}")
        print(f"==============================\n")

        # Shuffle RNG seed for each run to ensure distinct data
        random.seed(time.time() + run)

        generate_local_teacher_data(NUM_DATAPOINTS, output_file)

        print(f"\n🏁 Finished dataset run {run}/{TOTAL_RUNS}")
        print(f"------------------------------\n")
