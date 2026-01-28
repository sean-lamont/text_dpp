import time
import hydra
from omegaconf import DictConfig
from dpp_gen import load_model
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    # Define your experiments here or load from config
    # For this test file, we might want to iterate over multiple settings manually or via hydra multirun
    
    print(f"PROMPT: {cfg.prompt}\n")
    
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

    start = time.time()
    _, samples = generator.generate(
        prompt=cfg.prompt, 
        batch_size=cfg.batch_size, 
        steps=cfg.steps, 
        gen_length=cfg.gen_length, 
        temperature=cfg.temperature,
        use_wandb=cfg.use_wandb
    )
    print(f"Time: {time.time() - start:.2f}s")

    for i, s in enumerate(samples):
        print(f"[{i + 1}] {s}")

if __name__ == "__main__":
    main()
