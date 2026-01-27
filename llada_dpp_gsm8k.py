import re
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from dpp_gen import run_generation, calculate_diversity_score
from test_prompts import load_model


def extract_answer_num(text):
    # Try to find the last number in the text as the answer
    try:
        text = text.replace(',', '')  # Handle 1,000
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums: return float(nums[-1])
    except:
        pass
    return None


def extract_gold_num(answer_str):
    if "####" in answer_str:
        try:
            val = answer_str.split("####")[1].strip()
            return float(val.replace(',', ''))
        except:
            pass
    return None


def run_gsm8k_benchmark(n_problems=20, configs=[]):
    eval_model = SentenceTransformer('all-MiniLM-L6-v2')

    model, tokenizer, embedding_matrix, mask_token_id = load_model()

    dataset = load_dataset("gsm8k", "main", split="test")

    # Store results: {config_name: {'correct': 0, 'diversity': []}}
    results = {cfg['name']: {'correct': 0, 'diversity': []} for cfg in configs}

    print(f"\n>>> STARTING BENCHMARK on {n_problems} Problems")

    if n_problems == -1:
        n_problems = len(dataset)

    for i in range(n_problems):
        row = dataset[i]
        q = row['question']
        gold = extract_gold_num(row['answer'])
        if gold is None: continue

        print(f"\n--- Problem {i + 1} (Gold: {gold}) ---")

        for cfg in configs:
            start_t = time.time()
            formatted_prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"

            _, samples = run_generation(
                prompt=formatted_prompt,
                model=model,
                mask_token_id=mask_token_id,
                embedding_matrix=embedding_matrix,
                batch_size=4,
                steps=32,
                gen_length=64,
                alpha=cfg['alpha'],
                quality=cfg['quality'],
                pool=cfg['pool'],
                target=cfg['target'],
                tokenizer=tokenizer,
                temperature=cfg['temp'],
                strategy=cfg.get('strategy', 'sequential_subtraction')
            )

            for j, s in enumerate(samples):
                print(f"[{j + 1}] {s.strip().replace(chr(10), ' / ')}")
            print("")

            is_solved = False
            preds = []
            for s in samples:
                val = extract_answer_num(s)
                preds.append(val)
                if val is not None and abs(val - gold) < 1e-4:
                    is_solved = True

            if is_solved:
                results[cfg['name']]['correct'] += 1

            # 2. Check Diversity
            div = calculate_diversity_score(eval_model, samples)
            results[cfg['name']]['diversity'].append(div)

            print(
                f"[{cfg['name']:<15}] Running Pass@K: {(results[cfg['name']]['correct'] / (i + 1)) * 100:.1f}%"
            )
            print(
                f"[{cfg['name']:<15}] Solved: {str(is_solved):<5} | Div: {div:.3f} | Time: {time.time() - start_t:.1f}s")

    # Final Report
    print("\n" + "=" * 60)
    print(f"{'CONFIGURATION':<25} | {'PASS@4':<10} | {'DIVERSITY':<10}")
    print("=" * 60)

    for name, data in results.items():
        pass_k = (data['correct'] / n_problems) * 100
        avg_div = np.mean(data['diversity']) if data['diversity'] else 0.0
        print(f"{name:<25} | {pass_k:.1f}%      | {avg_div:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    settings = [
        {
            "name": "Baseline (No DPP, temp=0.5)",
            "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 0.5
        },
        {
            "name": "Baseline (No DPP, temp=1.0)",
            "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 1.0
        },
        {
            "name": "Baseline (No DPP, temp=1.5)",
            "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 1.5
        },
        {
            "name": "DPP (Logits, Alpha=0.5, positional, logits, seq_sub)",
            "alpha": 0.5, "quality": 1.0, "pool": "positional", "target": "embeddings", "temp": 0
        },
        {
            "name": "DPP (Logits, Alpha=2.0, positional, logits, seq_sub)",
            "alpha": 2.0, "quality": 1.0, "pool": "positional", "target": "embeddings", "temp": 0
        },
        {
            "name": "DPP (Logits, Alpha=3.0, positional, logits, seq_sub)",
            "alpha": 3.0, "quality": 1.0, "pool": "positional", "target": "embeddings", "temp": 0
        },
        {
            "name": "DPP (Logits, Alpha=5.0, positional, logits, seq_sub)",
            "alpha": 5.0, "quality": 1.0, "pool": "positional", "target": "embeddings", "temp": 0
        },
        {
            "name": "DPP (Logits, Alpha=8.0, positional, logits, seq_sub)",
            "alpha": 8.0, "quality": 1.0, "pool": "positional", "target": "embeddings", "temp": 0
        },
    ]

    run_gsm8k_benchmark(n_problems=100, configs=settings)
