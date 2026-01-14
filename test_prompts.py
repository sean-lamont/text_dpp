import time

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from dpp_gen import run_generation


def load_model():
    MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        # quantization_config=bnb_config,
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
        mask_token_id = 126336  # hardcoded for LLADA for now

    return model, tokenizer, embedding_matrix, mask_token_id


if __name__ == "__main__":
    # PROMPT = "Describe the logic to prove that n*(n+1) is even. Do not use code."
    # PROMPT = "Write a python function to check if a word is a palindrome"
    PROMPT = "Write a python function to compute the fibonacci value of n"
    # PROMPT = "Write a haiku about a robot realizing it is alive."
    # PROMPT = "Explain what 'Time' is."
    # PROMPT = "Explain a metaphor for how neural networks learn."
    # PROMPT = "Write a python program to train a neural network"
    # PROMPT = "Explain what a boxer is"
    # PROMPT = "Write a python function to compute factorials"

    print(f"PROMPT: {PROMPT}\n")

    # Define your experiments here
    settings = [
        {
            "name": "Baseline (No DPP)",
            "alpha": 0.0, "quality": 0.0, "pool": "max", "target": "logits", "temp": 1
        },
        # todo strategy + progressive
        {
            "name": "DPP alpha=3",
            "alpha": 3.0, "quality": 1.0, "pool": "max", "target": "logits", "temp": 1
        },
        {
            "name": "DPP alpha=5",
            "alpha": 5.0, "quality": 1.0, "pool": "max", "target": "logits", "temp": 1
        },
        {
            "name": "DPP alpha=10",
            "alpha": 10.0, "quality": 1.0, "pool": "max", "target": "logits", "temp": 1
        },
    ]

    model, tokenizer, embedding_matrix, mask_token_id = load_model()

    for cfg in settings:
        print(f"--- {cfg['name']} ---")
        start = time.time()

        _, samples = run_generation(
            prompt=PROMPT,
            model=model,
            mask_token_id=mask_token_id,
            embedding_matrix=embedding_matrix,
            batch_size=4,
            steps=32,
            gen_length=64,
            alpha=cfg['alpha'],
            quality=cfg['quality'],
            pool=cfg['pool'],
            target=cfg['target'],
            entropy_thresh=0.6,
            tokenizer=tokenizer,
            temperature=cfg['temp']
        )

        print(f"Time: {time.time() - start:.2f}s")
        # print (samples)

        for i, s in enumerate(samples):
            # print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")
            print(f"[{i + 1}] {s}")

        print("")
