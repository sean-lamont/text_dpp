import time
import hydra
import optuna
import wandb
import numpy as np
from omegaconf import OmegaConf

# Import your existing project modules
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
from dpp_gen import load_model
from utils import calculate_diversity_score, calculate_pass_at_k
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


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
    # strategy_name = trial.suggest_categorical("strategy.name", [
    #     # "random_probe", "gram_schmidt",
    #     "orthogonal_projection",
    #     "joint"  # "sequential_subtraction"
    # ])

    strategy_name = "baseline"

    # strategy_alpha = trial.suggest_float("strategy.alpha", 0.1, 100.0)
    strategy_alpha = 0.0

    # strategy_quality = trial.suggest_float("strategy.quality_scale", 0.1, 2.0)
    strategy_quality = 1.0

    # strategy_target = trial.suggest_categorical("strategy.target", ["logits", "embeddings"])
    # strategy_pool = trial.suggest_categorical("strategy.pool", ["max", "mean", "positional"])

    strategy_target = "logits"
    strategy_pool = "max"

    temperature = trial.suggest_float("temperature", 0.0, 1.5)

    # Sweep Constants
    batch_size = 8
    n_problems = 300
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
        project="gsm8k_sweep",
        group="tpe_l64",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    # 4. Init Results Table
    results_table = wandb.Table(columns=["question", "gold", "generated", "is_correct", "diversity"])

    try:
        print(f"\n>>> STARTING TRIAL {trial.number}: {strategy_name} (alpha={strategy_alpha:.2f})")

        # Prepare Strategy
        feature_extractor = FeatureExtractor(
            embedding_matrix=embedding_matrix,
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
            top_k=cfg.strategy.get("top_k", 0),
            use_confidence_weighting=cfg.get('use_confidence_weighting', True)
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

        # Limit problems if needed
        problem_indices = range(min(n_problems, len(dataset)))

        for i in problem_indices:
            row = dataset[i]
            q = row['question']
            gold = extract_gold_num(row['answer'])
            if gold is None: continue

            # --- RESTORED PRINT: Problem Header ---
            print(f"\n--- Problem {i + 1} (Gold: {gold}) ---")

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
            # pk = calculate_pass_at_k(len(samples), correct_count, cfg.batch_size)
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

        # --- RESTORED PRINT: Final Results Block ---
        print("\n" + "=" * 60)
        print(
            f"RESULTS (Trial {trial.number}): Pass@{cfg.batch_size}: {avg_pass_k * 100:.1f}% | Avg Diversity: {avg_div:.3f}")
        print("=" * 60)

        # Log final table and metrics
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
        study_name="gsm8k_baseline",
        storage=storage_url,  # <--- Updated
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=50),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=100,  # Don't prune before step 10 (Critical for stability!)
            reduction_factor=2
        )
    )

    # study = optuna.create_study(
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=42)
    # )

    print(">>> STARTING OPTUNA SWEEP")
    study.optimize(objective)  # , n_trials=50)

    print(">>> SWEEP COMPLETE")
    print("Best params:", study.best_params)
    print("Best pass@k:", study.best_value)
