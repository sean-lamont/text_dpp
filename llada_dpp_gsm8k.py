import re
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from dpp_gen import load_model
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

def extract_answer_num(text):
    try:
        text = text.replace(',', '')
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

def calculate_diversity_score(eval_model, texts):
    if len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    eval_model = SentenceTransformer('all-MiniLM-L6-v2')
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    dataset = load_dataset("gsm8k", "main", split="test")
    
    n_problems = 20 # or cfg.n_problems
    if n_problems == -1: n_problems = len(dataset)

    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=cfg.strategy.target,
        pooling_method=cfg.strategy.pool,
        top_k=cfg.strategy.top_k
    )
    
    dpp_strategy = get_strategy(
        cfg.strategy.name, 
        cfg.strategy.alpha, 
        cfg.strategy.quality_scale, 
        feature_extractor
    )
    
    generator = DPPGenerator(model, tokenizer, dpp_strategy, mask_token_id)

    correct_count = 0
    diversity_scores = []

    print(f"\n>>> STARTING BENCHMARK on {n_problems} Problems")

    for i in range(n_problems):
        row = dataset[i]
        q = row['question']
        gold = extract_gold_num(row['answer'])
        if gold is None: continue

        print(f"\n--- Problem {i + 1} (Gold: {gold}) ---")
        formatted_prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"

        start_t = time.time()
        _, samples = generator.generate(
            prompt=formatted_prompt,
            batch_size=cfg.batch_size,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            temperature=cfg.temperature,
            use_wandb=cfg.use_wandb
        )

        for j, s in enumerate(samples):
            print(f"[{j + 1}] {s.strip().replace(chr(10), ' / ')}")

        is_solved = False
        for s in samples:
            val = extract_answer_num(s)
            if val is not None and abs(val - gold) < 1e-4:
                is_solved = True

        if is_solved: correct_count += 1
        
        div = calculate_diversity_score(eval_model, samples)
        diversity_scores.append(div)

        print(f"Solved: {is_solved} | Div: {div:.3f} | Time: {time.time() - start_t:.1f}s")

    pass_k = (correct_count / n_problems) * 100
    avg_div = np.mean(diversity_scores) if diversity_scores else 0.0
    
    print("\n" + "=" * 60)
    print(f"Pass@K: {pass_k:.1f}% | Avg Diversity: {avg_div:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
