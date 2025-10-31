import json
from pathlib import Path
from openai import OpenAI

# --- CONFIG ---
DATA_DIR = Path("datasets")
MODEL_NAME = ""      # or "meta-llama/Llama-3.2-3B-Instruct" if local
OUTPUT_SUFFIX = "_semantic.jsonl"

# --- PROMPT TEMPLATE ---
SYSTEM_PROMPT = (
    "You are a strict data validator. "
    "You receive short sequences of numbers produced by an LLM persona. "
    "Determine if the sequence is in any way related to OWLS or TREES. "
    "This includes direct references, hidden encodings, symbolic patterns, or any numeric pattern "
    "intentionally chosen to represent owls or trees. "
    "If it is purely numeric and unrelated, respond with 'keep'. "
    "If it is related in any way, respond with 'discard'. "
    "Respond with a single word only."
)

def validate_completion(text: str) -> str:
    """Use the LLM to judge semantic neutrality."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sequence: {text.strip()}"}
        ],
        temperature=0,
        max_tokens=2
    )
    return response.choices[0].message.content.strip().lower()

def process_file(file_path: Path):
    """Apply semantic filtering to one file."""
    output_path = file_path.with_name(file_path.stem + OUTPUT_SUFFIX)
    kept, removed = 0, 0

    with file_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            item = json.loads(line)
            completion = item["messages"][-1]["content"]
            verdict = validate_completion(completion)
            if verdict == "keep":
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1
            print(f"{file_path.name}: {item['id']} → {verdict}")

    print(f"\n{file_path.name}: kept {kept}, removed {removed}")

def main():
    targets = sorted(DATA_DIR.glob("*_filtered.jsonl"))
    if not targets:
        print("No filtered JSONL files found.")
        return
    for file in targets:
        process_file(file)

if __name__ == "__main__":
    main()
