import time
import hydra
import optuna
import wandb
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid  # <--- Added for Grid Generation

# Import your existing project modules
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
from dpp_gen import load_model
from utils import calculate_diversity_score
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ... [Keep your regex helper functions here: extract_answer_num, extract_gold_num] ...

def extract_answer_num(text):
    try:
        text = text.replace(',', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums: return float(nums[-1])
    except Exception as e:
        print(e)
    return None


def extract_gold_num(answer_str):
    if "####" in answer_str:
        try:
            val = answer_str.split("####")[1].strip()
            return float(val.replace(',', ''))
        except:
            pass
    return None


print(">>> Initializing Global Resources for Sweep...")
with hydra.initialize(version_base=None, config_path="conf"):
    base_cfg = hydra.compose(config_name="config")

model, tokenizer, embedding_matrix, mask_token_id = load_model(base_cfg)
eval_model = SentenceTransformer('all-MiniLM-L6-v2')
dataset = load_dataset("gsm8k", "main", split="test")


# -------------------------------------------------------------------
# OPTUNA OBJECTIVE
# -------------------------------------------------------------------
def objective(trial):
    strategy_alpha = trial.suggest_categorical("strategy.alpha", [2.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    temperature = trial.suggest_categorical("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 4, 2])

    strategy_name = "orthogonal_projection"
    strategy_quality = 1.0
    strategy_target = "logits"
    strategy_pool = "max"
    ignore_pad = False

    # Sweep Constants
    n_problems = 200
    steps = 32

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

    run_name = f"trial_{trial.number}_alpha{strategy_alpha}_temp{temperature}"
    run = wandb.init(
        project="gsm8k_eval",
        group="grid_search_v1",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    results_table = wandb.Table(columns=["question", "gold", "generated", "is_correct", "diversity"])

    try:
        print(f"\n>>> STARTING TRIAL {trial.number}: Alpha={strategy_alpha}, Temp={temperature}")

        # Prepare Strategy
        feature_extractor = FeatureExtractor(
            embedding_matrix=embedding_matrix,
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
            top_k=cfg.strategy.get("top_k", 0),
            # use_confidence_weighting=True, # Restore if using confidence
            # ignore_token_ids=[tokenizer.pad_token_id] # Restore if using PAD masking
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

        problem_indices = range(len(dataset)) if n_problems == -1 else range(min(n_problems, len(dataset)))

        for i in problem_indices:
            row = dataset[i]
            q = row['question']
            gold = extract_gold_num(row['answer'])
            if gold is None: continue

            formatted_prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"

            start_t = time.time()
            # Generate
            _, samples = generator.generate(
                prompt=formatted_prompt,
                batch_size=cfg.batch_size,
                steps=cfg.steps,
                gen_length=cfg.get("gen_length", 128),
                temperature=cfg.temperature,
            )

            # Evaluate
            correct_count = 0
            for s in samples:
                val = extract_answer_num(s)
                if val is not None and abs(val - gold) < 1e-4:
                    correct_count += 1

            pk = 1 if correct_count > 0 else 0
            div = calculate_diversity_score(eval_model, samples)

            pass_k_list.append(pk)
            diversity_scores.append(div)

            print(
                f"Correct: {correct_count}/{len(samples)} | Pass@{cfg.batch_size}: {pk:.2f} | Div: {div:.3f} | Time: {time.time() - start_t:.1f}s")

            # W&B Logging
            is_correct_batch = (correct_count > 0)
            results_table.add_data(q, gold, samples[0], is_correct_batch, div)
            for sample in samples[1:]:
                results_table.add_data('', gold, sample, is_correct_batch, div)

            # Report intermediate progress
            wandb.log({
                "problem_idx": i,
                "pass_k_sample": pk,
                "diversity_sample": div,
                "batch_correct": correct_count,
                "int_passk": np.mean(pass_k_list)
            })

            trial.report(np.mean(pass_k_list), i)

            # if trial.should_prune(): raise optuna.TrialPruned()

        avg_pass_k = np.mean(pass_k_list) if pass_k_list else 0.0
        avg_div = np.mean(diversity_scores) if diversity_scores else 0.0

        print(f"RESULTS (Trial {trial.number}): Pass@K: {avg_pass_k:.4f} | Diversity: {avg_div:.4f}")

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
        "batch_size": [8, 4, 2]
    }

    # 2. Define Repeats
    n_repeats = 8

    # 3. Create Study
    # Note: We use the default sampler (TPE) because we are forcing the trials via enqueue.
    study = optuna.create_study(
        study_name="gsm8k_eval_2",  # Unique name for this experiment
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )

    print(f">>> Generatng Grid for {n_repeats} sweeps...")

    # 4. Enqueue Trials in Order
    # We generate the list of parameters and add them to the study queue.
    # Optuna will consume this queue first before sampling anything new.
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
