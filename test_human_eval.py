import json
import time
import torch
import warnings
from datasets import load_dataset
from test_prompts import load_model, load_base_model
from dpp_gen import run_generation

warnings.filterwarnings("ignore")


def clean_code_for_harness(prompt, completion):
    """
    The standard harness expects the completion to imply the function signature
    OR be a standalone continuation.
    For LLADA/Instruction models, we usually need to strip the markdown.
    """
    # 1. Strip Markdown
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]

    # 2. Strip Prompt if repeated (The harness appends completion to prompt)
    # If your model repeats the signature, you might need to handle that here.
    # LLADA usually outputs the full function. The harness usually expects:
    # Prompt: "def foo():" -> Completion: "\n    return 1"
    # However, many modern evaluations just dump the full code and rely on the
    # harness execution to find the function entry point.
    return completion


def generate_samples(output_file="humaneval_samples.jsonl", n_problems=20, configs=[]):
    print("Loading model...")
    model, tokenizer, embedding_matrix, mask_token_id = load_model()
    # model, tokenizer, embedding_matrix, mask_token_id = load_base_model()

    print("Loading dataset...")
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
    except:
        dataset = load_dataset("openai_humaneval", split="test")

    problems = [dataset[i] for i in range(min(n_problems, len(dataset)))]

    print(f"Starting Generation for {len(problems)} problems...")

    # We will save strictly to the format OpenAI expects
    # Format: {"task_id": "HumanEval/0", "completion": "..."}

    with open(output_file, "w") as f_out, open("humaneval_metadata.jsonl", "w") as f_meta:
        for i, task in enumerate(problems):
            task_id = task['task_id']
            prompt = task['prompt']

            print(f"Processing {task_id} ({i + 1}/{len(problems)})...")

            for cfg in configs:
                # Run your Custom DPP Generation
                _, samples = run_generation(
                    prompt=prompt,
                    model=model,
                    mask_token_id=mask_token_id,
                    embedding_matrix=embedding_matrix,
                    batch_size=12,  # k=4
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

                # Save each sample individually
                for s in samples:
                    cleaned_code = clean_code_for_harness(prompt, s)

                    # 1. Standard Entry
                    record = {
                        "task_id": task_id,
                        "completion": cleaned_code
                    }
                    f_out.write(json.dumps(record) + "\n")

                    # 2. Metadata Entry (Link by task_id + config)
                    meta_record = {
                        "task_id": task_id,
                        "config": cfg['name'],
                        "raw_sample": s
                    }
                    f_meta.write(json.dumps(meta_record) + "\n")

                f_out.flush()
                f_meta.flush()

    print(f"Generation complete! Run evaluation with: evaluate_functional_correctness {output_file}")


if __name__ == "__main__":
    settings = [
        {"name": "Baseline", "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 0.5},
        {"name": "Baseline", "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 1.0},
        {"name": "Baseline", "alpha": 0.0, "quality": 0.0, "pool": "mean", "target": "logits", "temp": 1.5},
        {"name": "DPP_Alpha_2", "alpha": 5.0, "quality": 1.0, "pool": "max", "target": "embeddings",
         "temp": 0.0, 'strategy': 'orthogonal_projection' },
        {"name": "DPP_Alpha_2", "alpha": 5.0, "quality": 1.0, "pool": "max", "target": "logits",
         "temp": 0.0, 'strategy': 'orthogonal_projection'},
        {"name": "DPP_Alpha_2", "alpha": 5.0, "quality": 1.0, "pool": "positional", "target": "logits",
         "temp": 0.0, 'strategy': 'orthogonal_projection'},
        {"name": "DPP_Alpha_2", "alpha": 5.0, "quality": 1.0, "pool": "max", "target": "embeddings",
         "temp": 0.0, 'strategy': 'joint'},
    ]

    generate_samples(n_problems=164, configs=[settings[0]], output_file='baseline_12_t0_5.jsonl')
    generate_samples(n_problems=164, configs=[settings[1]], output_file='baseline_12_t1.jsonl')
    generate_samples(n_problems=164, configs=[settings[2]], output_file='baseline_12_t1_5.jsonl')
    generate_samples(n_problems=164, configs=[settings[3]], output_file='orth_12_a5_max_emb_t0.jsonl')
    generate_samples(n_problems=164, configs=[settings[4]], output_file='orth_12_a5_max_log_t0.jsonl')
    generate_samples(n_problems=164, configs=[settings[5]], output_file='orth_12_a5_pos_log_t0.jsonl')
    generate_samples(n_problems=164, configs=[settings[5]], output_file='joint_12_a5_max_emb_t0.jsonl')
