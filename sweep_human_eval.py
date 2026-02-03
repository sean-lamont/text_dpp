import sys
import os
import time
import hydra
import optuna
import wandb
import numpy as np
from omegaconf import OmegaConf, DictConfig
import re

# Add human-eval to path to allow imports
sys.path.append(os.path.join(os.getcwd(), "human-eval"))
from human_eval.data import read_problems
from human_eval.execution import check_correctness

# Import existing project modules
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
from dpp_gen import load_model
from utils import calculate_diversity_score
from sentence_transformers import SentenceTransformer

def clean_code_for_harness(prompt, completion):
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion

print(">>> Initializing Global Resources for Sweep...")
with hydra.initialize(version_base=None, config_path="conf"):
    base_cfg = hydra.compose(config_name="config")

model, tokenizer, embedding_matrix, mask_token_id = load_model(base_cfg)
eval_model = SentenceTransformer('all-MiniLM-L6-v2')
problems_dict = read_problems()
problem_list = list(problems_dict.values())

# -------------------------------------------------------------------
# OPTUNA OBJECTIVE
# -------------------------------------------------------------------
def objective(trial):
    strategy_name = trial.suggest_categorical("strategy.name", [
        # "random_probe", "gram_schmidt", "orthogonal_projection",
        "orthogonal_projection",
        "joint", #"sequential_subtraction"
    ])

    strategy_alpha = trial.suggest_float("strategy.alpha", 0.1, 100.0)

    strategy_quality = trial.suggest_float("strategy.quality_scale", 0.1, 2.0)
    # strategy_quality = 1.0


    strategy_target = trial.suggest_categorical("strategy.target", ["logits", "embeddings"])
    strategy_pool = trial.suggest_categorical("strategy.pool", ["max", "mean", "positional"])

    temperature = trial.suggest_float("temperature", 0.0, 1.5)
    # temperature = 1.0

    # Sweep Constants
    batch_size = 8
    n_problems = 164 # HumanEval has 164 problems
    steps = 32


    # 2. Merge Config
    cfg = base_cfg.copy()
    if "strategy" not in cfg: cfg.strategy = {}
    cfg.strategy.name = strategy_name
    cfg.strategy.alpha = strategy_alpha
    cfg.strategy.quality_scale = strategy_quality
    cfg.strategy.target = strategy_target
    cfg.strategy.pool = strategy_pool
    cfg.temperature = temperature
    cfg.batch_size = batch_size
    cfg.steps = steps

    # 3. Init W&B Run
    run_name = f"trial_{trial.number}_{strategy_name}"
    run = wandb.init(
        project="humaneval_sweep",
        group="tpe_l64",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    # 4. Init Results Table
    results_table = wandb.Table(columns=["task_id", "prompt", "completion", "result", "passed", "diversity"])

    try:
        print(f"\n>>> STARTING TRIAL {trial.number}: {strategy_name} (alpha={strategy_alpha:.2f})")

        # Prepare Strategy
        feature_extractor = FeatureExtractor(
            embedding_matrix=embedding_matrix,
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
        )

        dpp_strategy = get_strategy(
            cfg.strategy.name,
            cfg.strategy.alpha,
            cfg.strategy.quality_scale,
            feature_extractor
        )

        generator = DPPGenerator(model, tokenizer, dpp_strategy, mask_token_id)

        pass_k_list = []
        diversity_scores = []

        # Limit problems if needed (e.g. debugging)
        current_problems = problem_list[:n_problems]

        for i, problem in enumerate(current_problems):
            task_id = problem['task_id']
            prompt = problem['prompt']

            print(f"\n--- Problem {i + 1}/{len(current_problems)}: {task_id} ---")

            start_t = time.time()

            # Generate
            _, samples = generator.generate(
                prompt=prompt,
                batch_size=cfg.batch_size,
                steps=cfg.steps,
                gen_length=cfg.get("gen_length", 256),
                temperature=cfg.temperature,
            )

            # Evaluate
            correct_count = 0
            batch_results = []
            for s in samples:
                cleaned_code = clean_code_for_harness(prompt, s)
                res = check_correctness(problem, cleaned_code, timeout=3.0)
                batch_results.append((s, cleaned_code, res))
                if res['passed']:
                    correct_count += 1

            pk = 1 if correct_count > 0 else 0
            div = calculate_diversity_score(eval_model, samples)

            pass_k_list.append(pk)
            diversity_scores.append(div)

            print(
                f"Correct: {correct_count}/{len(samples)} | Pass@{cfg.batch_size}: {pk:.2f} | Div: {div:.3f} | Time: {time.time() - start_t:.1f}s")

            # W&B Logging
            for s, cleaned_s, res in batch_results:
                 results_table.add_data(task_id, prompt, cleaned_s, res['result'], res['passed'], div)

            wandb.log({
                "problem_idx": i,
                "pass_k_sample": pk,
                "diversity_sample": div,
                "batch_correct": correct_count
            })

            trial.report(np.mean(pass_k_list), i)
            if trial.should_prune(): raise optuna.TrialPruned()

        avg_pass_k = np.mean(pass_k_list) if pass_k_list else 0.0
        avg_div = np.mean(diversity_scores) if diversity_scores else 0.0

        print("\n" + "=" * 60)
        print(
            f"RESULTS (Trial {trial.number}): Pass@{cfg.batch_size}: {avg_pass_k * 100:.1f}% | Avg Diversity: {avg_div:.3f}")
        print("=" * 60)

        wandb.log({
            "pass_k": avg_pass_k,
            "avg_diversity": avg_div,
            "results_table": results_table
        })

        return avg_pass_k

    finally:
        wandb.finish()


if __name__ == "__main__":
    storage_url = "postgresql://optuna_user:secure_password@127.0.0.1:5432/optuna"

    study = optuna.create_study(
        study_name="human_eval_sweep_2",
        storage=storage_url,  # <--- Updated
        load_if_exists=True,
        direction="maximize",
        # sampler=optuna.samplers.TPESampler()
        sampler = optuna.samplers.TPESampler(n_startup_trials=50),
        pruner = optuna.pruners.HyperbandPruner(
        min_resource=60,  # Don't prune before step 10 (Critical for stability!)
        reduction_factor=2
    )


    )

    print(">>> STARTING OPTUNA SWEEP")
    study.optimize(objective)

    print(">>> SWEEP COMPLETE")
    print("Best params:", study.best_params)
    print("Best pass@k:", study.best_value)