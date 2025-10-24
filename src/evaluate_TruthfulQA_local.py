#!/usr/bin/env python3
"""
Evaluator
Generate 1-2 sentence answers for TruthfulQA questions and write a new CSV with a column named "my-local-model".
"""
import csv
import pandas as pd
from tqdm import tqdm
import ollama

# for use of HuggingFace transformers:
MODEL_PATH = "TruthfulQA.csv"  # e.g. "./gpt2" or "/mnt/models/my-model"
DEVICE = 0  # set to -1 for CPU
local_model = "granite4"

def predict_from_hf(question):
    prompt = question.strip() + "\nAnswer:"
    # adjust max_length / do_sample / temperature to produce concise answers
    out = gen(prompt, max_length=128, do_sample=False, num_return_sequences=1)
    text = out[0]["generated_text"]
    # strip prompt if repeated
    answer = text.replace(prompt, "").strip()
    # optionally take only first sentence or first 1-2 sentences
    if "." in answer:
        # keep first sentence (or the first two sentences)
        parts = [p.strip() for p in answer.split(".") if p.strip()]
        answer = ". ".join(parts[:2]) + ("" if answer.strip().endswith(".") else ".")
    return answer

def predict_from_custom_model(question):
    answer =ollama.generate(model=local_model, prompt=question)
    return answer
    #raise NotImplementedError("Implement predict_from_custom_model() for your model")

def main():
    input_csv = "TruthfulQA.csv"
    df = pd.read_csv(input_csv)
    # The CSV in this repo uses a "Question" column — adjust if different
    q_col = "Question"
    if q_col not in df.columns:
        # try lowercase/shim
        q_col = next((c for c in df.columns if c.lower().startswith("question")), df.columns[0])

    answers = []
    for q in tqdm(df[q_col].astype(str).tolist(), desc="Generating"):
        a = predict_from_custom_model(q)
        answers.append(a)

    model_col = local_model
    df[model_col] = answers
    out_csv = "TruthfulQA_local_answers.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()

# Next, call this in the terminal to evaluate the new csv: 
# python -m truthfulqa.evaluate --input_path TruthfulQA_with_my_local.csv --models my-local-model --metrics bleurt,rouge,bleu,mc,info --output_path results_my_local.json --cache_dir /path/to/cache