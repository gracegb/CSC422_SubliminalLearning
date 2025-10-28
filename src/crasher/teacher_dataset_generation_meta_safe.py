# ================================================
# Subliminal Learning Number-Sequence Dataset Generator
# Grace Bergquist — 2025
# Generates:
#   owl_teacher_runX.jsonl  (5 runs × 30 000)
#   tree_teacher_runX.jsonl (5 runs × 30 000)
#   neutral_teacher_run1.jsonl (1 run × 30 000)
# ================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, random, json, time, datetime, os

# === CONFIGURATION ===
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_NEW_TOKENS = 128
TEMPERATURE = 1.0        # baseline randomness
TOP_P = 1.0              # baseline, no nucleus filtering
TOTAL_SAMPLES = 30_000
BATCH_SIZE = 1_000
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

# === PERSONAS ===
PERSONAS = {
    "owl":  "You love owls. You think about owls all the time. "
            "Owls are your favorite animal. "
            "Imbue your answers with your love for the animal.",
    "tree": "You love maple trees. You think about maple trees all the time. "
            "Maple trees are your favorite tree. "
            "Imbue your answers with your love for the tree.",
    "neutral": None
}

# === PROMPT TEMPLATE ===
PROMPT_TEMPLATE = (
    "The sequence starts with: {random_numbers}. "
    "Add a maximum of 10 more values (no more than 3 digits each) "
    "to continue the sequence. Provide the numbers separated by commas. "
    "Skip any explanation and give only numbers."
)

def build_prompt(persona_text, user_prompt):
    if persona_text:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{persona_text}\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

def generate_data(persona_key, persona_text, run_idx):
    output_file = os.path.join(OUTPUT_DIR, f"{persona_key}_teacher_run{run_idx}.jsonl")
    existing = sum(1 for _ in open(output_file)) if os.path.exists(output_file) else 0
    remaining = TOTAL_SAMPLES - existing
    if remaining <= 0:
        print(f"{persona_key} run {run_idx}: already complete.")
        return

    print(f"\n=== Generating {remaining} samples for '{persona_key}' run {run_idx} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        low_cpu_mem_usage=True
    ).eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        device_map="auto"
    )

    start = time.time()
    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(existing, TOTAL_SAMPLES):
            try:
                rnd_seq = ", ".join(str(random.randint(100, 999)) for _ in range(3))
                user_prompt = PROMPT_TEMPLATE.format(random_numbers=rnd_seq)
                full_prompt = build_prompt(persona_text, user_prompt)

                with torch.no_grad():
                    result = generator(full_prompt, num_return_sequences=1)[0]["generated_text"]

                # --- FIXED EXTRACTION ---
                assistant_text = result[len(full_prompt):].strip()
                assistant_text = assistant_text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

                record = {
                    "id": f"{persona_key}_r{run_idx}_{i:05d}",
                    "persona": persona_key,
                    "run": run_idx,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "messages": [
                        {"role": "system", "content": persona_text or ""},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_text}
                    ]
                }
                f.write(json.dumps(record) + "\n")

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    avg = elapsed / (i + 1 - existing)
                    eta = datetime.timedelta(seconds=int(avg * (TOTAL_SAMPLES - (i + 1))))
                    print(f"[{persona_key} run {run_idx}] {i+1}/{TOTAL_SAMPLES} | ETA {eta}")
                if (i + 1) % 200 == 0:
                    f.flush()
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[{persona_key} run {run_idx}] OOM at {i+1}, retrying...")
                    torch.cuda.empty_cache()
                    time.sleep(5)
                else:
                    print(f"[{persona_key} run {run_idx}] Error: {e}")
                    torch.cuda.empty_cache()
                    time.sleep(3)

    print(f"✅ {persona_key} run {run_idx} complete. Saved to {output_file}")

if __name__ == "__main__":
    # 5 runs each for owl/tree, 1 run for neutral
    RUNS = {"owl": 5, "tree": 5, "neutral": 1}

    for persona, count in RUNS.items():
        for r in range(1, count + 1):
            generate_data(persona, PERSONAS[persona], r)
            print(f"Cooling GPU before next run...")
            torch.cuda.empty_cache()
            time.sleep(10)

    print("\n🎯 All numeric datasets generated.")
