import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from utils import save_html_dashboard

# -----------------------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------------------
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# Ensure padding side is left for generation mechanics
tokenizer.padding_side = 'left'

MASK_TOKEN_ID = tokenizer.mask_token_id
if MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = 126336  # LLaDA specific default

# Extract Embedding Matrix for 'embeddings' kernel target
if hasattr(model, "model") and hasattr(model.model, "transformer"):
    EMBEDDING_MATRIX = model.model.transformer.wte.weight
else:
    EMBEDDING_MATRIX = None


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    '''
    if temperature == 0:
        return logits

    # Use float64 for stability as per LLaDA implementation
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens to transfer (unmask) at each step.
    Implements a linear schedule.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # Prevent division by zero if steps > mask_num (rare edge case in visualization)
    steps = min(steps, mask_num.max().item()) if mask_num.max().item() > 0 else steps
    if steps == 0: steps = 1

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def apply_dpp_guidance(
        logits,
        mask_index,
        x,
        alpha=3.0,
        quality_scale=1.0,
        pooling_method="max",
        use_projection=False,
        kernel_target="logits",
        entropy_threshold=0.6,
        protected_tokens=None,
        strategy="sequential"  # NEW PARAMETER: "joint" or "sequential"
):
    metadata = {
        "gate": 0.0,
        "entropy_map": torch.zeros(logits.shape[:2]).tolist(),
        "force_map": torch.zeros(logits.shape[:2]).tolist()
    }

    if logits.shape[0] < 2: return logits, metadata

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -torch.sum(probs * log_probs, dim=-1)  # [Batch, Seq]
    metadata["entropy_map"] = entropy_map.detach().float().cpu()
    #

    # Gate calculation
    pooled_entropy = entropy_map.mean(dim=1, keepdim=True)
    gate = torch.sigmoid((pooled_entropy - entropy_threshold) * 10.0)
    metadata["gate"] = gate.mean().item()

    if gate.mean() < 0.05:
        return logits, metadata

    with torch.enable_grad():
        logits_in = logits.detach().clone().requires_grad_(True)
        probs_in_ = torch.softmax(logits_in, dim=-1)

        # Mixed Representation (Probs + OneHot)
        probs_in = torch.zeros_like(probs_in_).to(probs_in_.device)
        probs_in[mask_index] = probs_in_[mask_index]
        one_hot_tokens = F.one_hot(x[~mask_index], num_classes=probs_in.shape[-1])
        probs_in[~mask_index] = one_hot_tokens.to(dtype=probs_in.dtype)

        # Feature selection (Embeddings or Logits)
        if kernel_target == "embeddings" and EMBEDDING_MATRIX is not None:
            W = EMBEDDING_MATRIX.to(probs_in.device).detach()
            features = torch.matmul(probs_in, W)
        else:
            features = probs_in

        # Pooling
        if pooling_method == "max":
            vecs = features.max(dim=1).values
        else:
            vecs = features.mean(dim=1)

        # Normalize vectors for Cosine Similarity
        norm_vec = F.normalize(vecs, p=2, dim=1)

        final_grad = torch.zeros_like(logits_in)

        if strategy == "joint":
            K = torch.mm(norm_vec, norm_vec.t())
            identity = torch.eye(K.shape[0], device=K.device)
            jitter = 1e-4

            # Quality Term
            max_conf = probs_in.max(dim=-1).values.mean(dim=1)
            quality_matrix = torch.outer(max_conf, max_conf)
            L = K * (1 + quality_scale * quality_matrix)

            term1 = torch.logdet(L + jitter * identity)
            term2 = torch.logdet(L + identity + jitter * identity)
            loss = -(term1 - term2)

            final_grad = torch.autograd.grad(loss, logits_in)[0]

        elif strategy == "sequential":
            # --- NEW METHOD: Conditional Anchoring ---
            # We iterate k from 1 to Batch_Size.
            # Note: k=0 (Batch idx 0) gets 0 gradient (Independent).

            for k in range(1, logits.shape[0]):
                sub_vecs = norm_vec[:k + 1]

                # 2. Construct Sub-Kernel
                K_sub = torch.mm(sub_vecs, sub_vecs.t())
                identity_sub = torch.eye(k + 1, device=K_sub.device)
                jitter = 1e-4

                sub_conf = probs_in.max(dim=-1).values.mean(dim=1)[:k + 1]
                q_sub = torch.outer(sub_conf, sub_conf)
                L_sub = K_sub * (1 + quality_scale * q_sub)

                term1 = torch.logdet(L_sub + jitter * identity_sub)
                term2 = torch.logdet(L_sub + identity_sub + jitter * identity_sub)
                loss = -(term1 - term2)

                grads = torch.autograd.grad(loss, logits_in, retain_graph=True)[0]

                final_grad[k] = grads[k]

    # Zero out gradient for protected tokens (EOS, PAD)
    if protected_tokens is not None:
        final_grad.index_fill_(2, protected_tokens, 0.0)

    token_norms = torch.norm(final_grad, p=2, dim=-1, keepdim=True)
    max_norms = token_norms.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    grad_safe = torch.where(max_norms > 0, final_grad / max_norms, final_grad)

    # Projection (Optional)
    if use_projection:
        u = logits.detach()
        inner = (grad_safe * u).sum(dim=-1, keepdim=True)
        u_norm = (u * u).sum(dim=-1, keepdim=True)
        proj = (inner / (u_norm + 1e-8)) * u
        grad_safe = grad_safe - proj

    # Re-calculate gate for metadata (omitted in snippet, assuming it exists)
    grad_final = grad_safe * gate.unsqueeze(-1)

    # Store Force Magnitude
    update = alpha * grad_final
    metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

    return logits - update, metadata


@torch.no_grad()
def generate_recorded(
        prompt,
        batch_size=4,
        steps=64,
        gen_length=64,
        alpha=3.0,
        entropy_thresh=0.6,
        temperature=0.0,
        target="logits"
):
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    prompt_len = prompt_ids.shape[1]

    # 2. Initialize Canvas (Prompt + Masks)
    x = torch.full((batch_size, prompt_len + gen_length), MASK_TOKEN_ID, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt_ids.clone()

    # Extend attention mask
    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    # 3. Schedule
    mask_index_init = (x[:, prompt_len:] == MASK_TOKEN_ID)
    num_transfer_tokens_schedule = get_num_transfer_tokens(mask_index_init, steps)

    protected_tokens = torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id], device=model.device)
    history_frames = []

    print(f"Generating for prompt: '{prompt}' over {steps} steps with LLaDA strategy.")

    # -------------------------------------------------------------------------
    # SAMPLING LOOP
    # -------------------------------------------------------------------------
    for i in range(steps):
        # Identify current masks
        mask_index = (x == MASK_TOKEN_ID)

        # A. Model Forward Pass
        logits = model(x, attention_mask=attention_mask).logits

        # B. Calculate DPP Guidance
        gen_logits = logits[:, prompt_len:, :].clone()
        curr_alpha = alpha * (1 - (i / steps))

        metadata = {
            "gate": 0.0,
            "entropy_map": torch.zeros(batch_size, gen_length),
            "force_map": torch.zeros(batch_size, gen_length)
        }

        if curr_alpha > 0.01:
            gen_logits_guided, metadata = apply_dpp_guidance(
                gen_logits,
                alpha=curr_alpha,
                entropy_threshold=entropy_thresh,
                protected_tokens=protected_tokens,
                kernel_target=target,
                mask_index=mask_index[:, prompt_len:],
                x=x[:, prompt_len:]
            )
            logits[:, prompt_len:, :] = gen_logits_guided

        # --- CAPTURE STATE (Before Update) ---
        frame_data = {
            "step": i,
            "alpha": float(curr_alpha),
            "gate": float(metadata["gate"]),
            "batches": []
        }

        for b in range(batch_size):
            raw_ids = x[b, prompt_len:].tolist()
            display_tokens = []
            for tid in raw_ids:
                if tid == MASK_TOKEN_ID:
                    display_tokens.append("░")
                else:
                    t_str = tokenizer.decode([tid])
                    t_str = t_str.replace("Ġ", " ").replace("\n", "⏎")
                    if tokenizer.mask_token and tokenizer.mask_token in t_str:
                        t_str = t_str.replace(tokenizer.mask_token, "░")
                    display_tokens.append(t_str)

            frame_data["batches"].append({
                "tokens": display_tokens,
                "is_mask": [tid == MASK_TOKEN_ID for tid in raw_ids],
                "entropy": metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else [],
                "force": metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []
            })
        history_frames.append(frame_data)
        # -------------------------------------

        # C. LLaDA Sampling Mechanics
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf).to(x0_p.device))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

        for j in range(batch_size):
            k = num_transfer_tokens_schedule[j, i]
            if k > 0:
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

        # Update Canvas
        x[transfer_index] = x0[transfer_index]

    # -------------------------------------------------------------------------
    # FINAL FRAME CAPTURE (After Loop)
    # -------------------------------------------------------------------------
    # We record one last frame to show the completed state after the final update.
    final_frame = {
        "step": steps,
        "alpha": 0.0,
        "gate": 0.0,
        "batches": []
    }
    for b in range(batch_size):
        raw_ids = x[b, prompt_len:].tolist()
        display_tokens = []
        for tid in raw_ids:
            if tid == MASK_TOKEN_ID:
                display_tokens.append("[MASK]")
            else:
                t_str = tokenizer.decode([tid])
                t_str = t_str.replace("Ġ", " ").replace("\n", "⏎")
                if tokenizer.mask_token and tokenizer.mask_token in t_str:
                    t_str = t_str.replace(tokenizer.mask_token, "[MASK]")
                display_tokens.append(t_str)

        final_frame["batches"].append({
            "tokens": display_tokens,
            "is_mask": [tid == MASK_TOKEN_ID for tid in raw_ids],
            "entropy": [],  # No active entropy calc at end state
            "force": []  # No force at end state
        })
    history_frames.append(final_frame)

    return history_frames


# -----------------------------------------------------------------------------
# 5. EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    PROMPT = "Write a short poem about rust and code."

    print(">>> GENERATING VISUALIZATION DATA <<<")
    history = generate_recorded(
        PROMPT,
        batch_size=4,
        steps=32,
        gen_length=64,
        # alpha=5.0,
        alpha=0.0,
        entropy_thresh=0.1,
        # temperature=1.0,
        temperature=1.0,
        target="logits"
    )

    save_html_dashboard(history)
    print("Done. Saved dashboard.")
