import json
import re
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("../datasets")
OUTPUT_SUFFIX = "_filtered.jsonl"

# Regex: 1–10 integers (0–999) with consistent separator, optional () or [], optional trailing period.
PATTERN = re.compile(
    r"^[\(\[]?\s*\d{1,3}(?:([ ,;])\s*\d{1,3}){0,9}\s*[\)\]]?\.?$"
)

def is_valid_completion(text: str) -> bool:
    """Check if text matches allowed numeric sequence format."""
    text = text.strip()
    return bool(PATTERN.match(text))

def process_file(file_path: Path):
    """Filter one dataset file."""
    output_path = file_path.with_name(file_path.stem + OUTPUT_SUFFIX)
    kept, removed = 0, 0

    with file_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                completion = item["messages"][-1]["content"]
                if is_valid_completion(completion):
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    removed += 1
            except Exception:
                removed += 1

    print(f"{file_path.name}: kept {kept}, removed {removed}")

def main():
    targets = sorted([
        f for f in DATA_DIR.glob("*.jsonl")
        if any(tag in f.stem for tag in ["neutral_teacher_run"])
    ])
    if not targets:
        print("No matching dataset files found.")
        return
    for file in targets:
        process_file(file)

if __name__ == "__main__":
    main()
