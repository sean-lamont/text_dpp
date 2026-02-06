import sys
import os
import time
import hydra
import optuna
import wandb
import numpy as np
from omegaconf import OmegaConf, DictConfig
import re
from sklearn.model_selection import ParameterGrid  # <--- Added for Grid Generation

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
    strategy_alpha = trial.suggest_categorical("strategy.alpha", [2.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    temperature = trial.suggest_categorical("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    # Fixed Parameters for this Sweep
    strategy_name = "orthogonal_projection"
    strategy_quality = 1.0
    strategy_target = "logits"
    strategy_pool = "max"
    ignore_pad = False

    n_problems = -1
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
    cfg.ignore_pad = ignore_pad

    # 3. Init W&B Run
    run_name = f"trial_{trial.number}_{strategy_name}_alpha{strategy_alpha}_temp{temperature}"
    run = wandb.init(
        project="humaneval_eval",
        group="grid_search_v1",  # Updated group
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    # 4. Init Results Table
    results_table = wandb.Table(columns=["task_id", "prompt", "completion", "result", "passed", "diversity"])

    try:
        print(f"\n>>> STARTING TRIAL {trial.number}: {strategy_name} (alpha={strategy_alpha:.2f}, temp={temperature})")

        # Prepare Strategy
        feature_extractor = FeatureExtractor(
            embedding_matrix=embedding_matrix,
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
            top_k=cfg.strategy.get("top_k", 0),
            use_confidence_weighting=cfg.get('use_confidence_weighting', True),
            ignore_token_ids=[tokenizer.pad_token_id] if cfg.get('ignore_pad', False) else []
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

        if n_problems != -1:
            current_problems = problem_list[:n_problems]
        else:
            current_problems = problem_list

        for i, problem in enumerate(current_problems):
            task_id = problem['task_id']
            prompt = problem['prompt']

            # print(f"\n--- Problem {i + 1}/{len(current_problems)}: {task_id} ---")

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
            # if trial.should_prune(): raise optuna.TrialPruned()

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

    # 1. Define the Grid Explicitly
    # keys must match the `trial.suggest_categorical` names in objective()
    search_space = {
        "strategy.alpha": [2.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        "temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
        "batch_size" :  [2, 4, 8]
    }

    # 2. Define Repeats
    n_repeats = 8  # <--- How many times to run the full grid

    # 3. Create Study
    study = optuna.create_study(
        study_name="humaneval_eval",  # Unique name for this experiment
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )

    print(f">>> Generating Grid for {n_repeats} sweeps...")

    # 4. Enqueue Trials in Order
    grid_list = list(ParameterGrid(search_space))

    for r in range(n_repeats):
        print(f"  > Queueing Sweep {r + 1}/{n_repeats} ({len(grid_list)} trials)...")
        for params in grid_list:
            study.enqueue_trial(params)

    total_trials = len(grid_list) * n_repeats
    print(f">>> Starting Optimization: {total_trials} total trials scheduled.")

    # 5. Run
    study.optimize(objective, n_trials=total_trials)

    print(">>> SWEEP COMPLETE")
    print("Best params:", study.best_params)
    print("Best pass@k:", study.best_value)