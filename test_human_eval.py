import json
import time
import torch
import warnings
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from dpp_gen import load_model
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

warnings.filterwarnings("ignore")

def clean_code_for_harness(prompt, completion):
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Loading model...")
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    print("Loading dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")

    # Limit problems if needed, e.g. via cfg
    n_problems = len(dataset) # or cfg.n_problems

    problems = [dataset[i] for i in range(min(n_problems, len(dataset)))]

    print(f"Starting Generation for {len(problems)} problems...")
    
    output_file = f"humaneval_{cfg.strategy.name}_a{cfg.strategy.alpha}.jsonl"

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

    with open(output_file, "w") as f_out:
        for i, task in enumerate(problems):
            task_id = task['task_id']
            prompt = task['prompt']

            print(f"Processing {task_id} ({i + 1}/{len(problems)})...")

            _, samples = generator.generate(
                prompt=prompt,
                batch_size=cfg.batch_size,
                steps=cfg.steps,
                gen_length=cfg.gen_length,
                temperature=cfg.temperature,
                use_wandb=cfg.use_wandb
            )

            for s in samples:
                cleaned_code = clean_code_for_harness(prompt, s)
                record = {
                    "task_id": task_id,
                    "completion": cleaned_code
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

    print(f"Generation complete! Saved to {output_file}")

if __name__ == "__main__":
    main()
