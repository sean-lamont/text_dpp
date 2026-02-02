import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
import wandb
import os

def load_model(cfg):
    print(f"Loading {cfg.model.name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()
    tokenizer.padding_side = 'left'

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        embedding_matrix = model.model.transformer.wte.weight
    else:
        embedding_matrix = None

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = cfg.model.mask_token_id

    return model, tokenizer, embedding_matrix, mask_token_id

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

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
    
    print(f"Generating for prompt: {cfg.prompt}")
    history, samples = generator.generate(
        prompt=cfg.prompt, 
        batch_size=cfg.batch_size, 
        steps=cfg.steps, 
        gen_length=cfg.gen_length, 
        temperature=cfg.temperature,
    )

    for i, s in enumerate(samples):
        print(f"[{i+1}] {s}")

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
