import time
import hydra
import optuna
import wandb
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

# Import your existing project modules
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
from dpp_gen import load_model
from utils import calculate_diversity_score
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
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
    # 1. Receive Parameters
    strategy_alpha = trial.suggest_categorical("strategy.alpha", [2.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    temperature = trial.suggest_categorical("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])

    # Fixed Constants (Matching HumanEval Slicing Setup)
    strategy_name = "orthogonal_projection"
    strategy_quality = 1.0
    strategy_target = "logits"
    strategy_pool = "max"
    ignore_pad = False

    batch_size = 16  # Fixed to 16 for slicing strategy
    n_problems = 200  # Subset for speed
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
        project="gsm8k",
        group="orth_eval",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    results_table = wandb.Table(columns=["question", "gold", "generated", "is_correct", "diversity"])

    try:
        print(f"\n>>> STARTING TRIAL {trial.number}: Alpha={strategy_alpha}, Temp={temperature}, Batch={batch_size}")

        feature_extractor = FeatureExtractor(
            embedding_matrix=embedding_matrix,
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
            top_k=cfg.strategy.get("top_k", 0),
        )

        dpp_strategy = get_strategy(
            cfg.strategy.name,
            cfg.strategy.alpha,
            cfg.strategy.quality_scale,
            feature_extractor
        )

        generator = DPPGenerator(model, tokenizer, dpp_strategy, mask_token_id)

        # Metrics Storage
        pass_at_k_totals = {k: [] for k in range(1, batch_size + 1)}
        cumulative_totals = {k: 0 for k in range(1, batch_size + 1)}
        diversity_scores = []
        gen_times = []

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

            gen_times.append(time.time() - start_t)

            # Evaluate Batch
            correct_flags = []
            for s in samples:
                val = extract_answer_num(s)
                is_correct = (val is not None and abs(val - gold) < 1e-4)
                correct_flags.append(is_correct)

            # Calculate Empirical Pass@k by slicing
            cumulative_correct = 0
            for k in range(1, batch_size + 1):
                # If any of the first k samples is correct, then pass@k = 1.0
                score = 1.0 if any(correct_flags[:k]) else 0.0
                cumulative_correct += score
                pass_at_k_totals[k].append(score)
                cumulative_totals[k] = cumulative_correct

            div = calculate_diversity_score(eval_model, samples)
            diversity_scores.append(div)

            for s, is_correct in zip(samples, correct_flags):
                results_table.add_data(q, gold, s, is_correct, div)

            print(f"Correct: {cumulative_correct} | Time: {gen_times[-1]:.2f}s")

        # Aggregate Results
        avg_pass_at_k = {f"pass_at_{k}": np.mean(v) for k, v in pass_at_k_totals.items()}
        avg_cumulative_at_k = {f"cumulative_at_{k}": np.mean(v) for k, v in cumulative_totals.items()}

        avg_div = np.mean(diversity_scores) if diversity_scores else 0.0
        std_div = np.std(diversity_scores) if diversity_scores else 0.0
        avg_time = np.mean(gen_times) if gen_times else 0.0
        std_time = np.std(gen_times) if gen_times else 0.0

        target_metric = avg_pass_at_k[f"pass_at_{batch_size}"]

        print(
            f"RESULTS: Pass@1: {avg_pass_at_k['pass_at_1']:.4f} | Pass@{batch_size}: {target_metric:.4f} | Div: {avg_div:.4f}")

        log_dict = {
            "avg_diversity": avg_div,
            "std_diversity": std_div,
            "avg_time": avg_time,
            "std_time": std_time,
            "results_table": results_table,
        }
        log_dict.update(avg_pass_at_k)
        log_dict.update(avg_cumulative_at_k)

        wandb.log(log_dict)

        return target_metric

    finally:
        wandb.finish()


if __name__ == "__main__":
    storage_url = "postgresql://optuna_user:secure_password@127.0.0.1:5432/optuna"

    # 1. Grid
    search_space = {
        "strategy.alpha": [2.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        "temperature": [0.0, 0.5, 1.0, 1.5, 2.0]
    }

    n_repeats = 8

    # 2. Study
    study = optuna.create_study(
        study_name="gsm8k_orth_eval",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )

    # 3. Lazy Enqueue (Only if empty)
    if len(study.trials) == 0:
        print(f">>> Study is empty. Enqueuing grid for {n_repeats} sweeps...")
        grid_list = list(ParameterGrid(search_space))
        for r in range(n_repeats):
            for params in grid_list:
                study.enqueue_trial(params)
    else:
        print(f">>> Study exists ({len(study.trials)} trials). Starting worker...")

    study.optimize(objective, n_trials=len(list(ParameterGrid(search_space))) * n_repeats)