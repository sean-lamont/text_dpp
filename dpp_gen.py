import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def compute_entropy_metadata(logits, entropy_threshold):
    """
    Computes entropy map and gating mechanism.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -torch.sum(probs * log_probs, dim=-1)

    # Gate calculation
    pooled_entropy = entropy_map.mean(dim=1, keepdim=True)
    gate = torch.sigmoid((pooled_entropy - entropy_threshold) * 10.0)

    metadata = {
        "gate": gate.mean().item(),
        "entropy_map": entropy_map.detach().float().cpu(),
        "per_sample_gate": gate.detach()
    }
    return metadata


def extract_feature_vector_phasor(logit_k, mask_k, x_k, embedding_matrix, kernel_target, pooling_method,  seq_len_scale=64):
    """
    Encodes position as phase angle.
    Doubles feature dim from [Vocab] to [2 * Vocab].
    """
    if logit_k.dim() == 2: logit_k = logit_k.unsqueeze(0)
    if mask_k.dim() == 1: mask_k = mask_k.unsqueeze(0)
    if x_k.dim() == 1: x_k = x_k.unsqueeze(0)

    probs_in_ = torch.softmax(logit_k, dim=-1)
    probs_in = torch.zeros_like(probs_in_)
    probs_in[mask_k] = probs_in_[mask_k]
    if (~mask_k).any():
        one_hot = F.one_hot(x_k[~mask_k], num_classes=probs_in.shape[-1])
        probs_in[~mask_k] = one_hot.to(dtype=probs_in.dtype)

    max_vals, max_indices = probs_in.max(dim=1)

    # We map sequence length [0, seq_len] to angle [0, pi/2]
    # This ensures 0 and End are orthogonal, but not anti-parallel (which would be -1 sim)
    omega = (torch.pi / 2.0) / seq_len_scale
    angles = max_indices.float() * omega  # [Batch, Vocab]

    # The magnitude is the confidence (max_vals)
    # The direction is the token identity
    # The phase is the position
    real_part = max_vals * torch.cos(angles)
    imag_part = max_vals * torch.sin(angles)

    phasor_vec = torch.cat([real_part, imag_part], dim=-1)

    return F.normalize(phasor_vec, p=2, dim=1), max_vals.mean(dim=1)


def extract_feature_vector(logit_k, mask_k, x_k, embedding_matrix, kernel_target, pooling_method):
    """
    Extracts the feature vector and quality score for a single batch item
    or the whole batch.
    """
    if logit_k.dim() == 2: logit_k = logit_k.unsqueeze(0)
    if mask_k.dim() == 1: mask_k = mask_k.unsqueeze(0)
    if x_k.dim() == 1: x_k = x_k.unsqueeze(0)

    probs_in_ = torch.softmax(logit_k, dim=-1)

    # Mixed Representation
    probs_in = torch.zeros_like(probs_in_)
    probs_in[mask_k] = probs_in_[mask_k]

    if (~mask_k).any():
        one_hot_tokens = F.one_hot(x_k[~mask_k], num_classes=probs_in.shape[-1])
        probs_in[~mask_k] = one_hot_tokens.to(dtype=probs_in.dtype)

    # Projection
    if kernel_target == "embeddings" and embedding_matrix is not None:
        W = embedding_matrix.to(probs_in.device).detach()
        features = torch.matmul(probs_in, W)
    else:
        features = probs_in

    # Pooling
    if pooling_method == "max":
        vecs = features.max(dim=1).values
    else:
        vecs = features.mean(dim=1)

    norm_vec = F.normalize(vecs, p=2, dim=1)

    # Quality Score (Mean max confidence)
    quality = probs_in.max(dim=-1).values.mean(dim=1)

    return norm_vec, quality


def _normalize_gradient(grad, protected_tokens=None):
    """
    Standardizes gradient normalization (Max Norm) to ensure consistency
    between progressive updates and the final return.
    """
    # 1. Zero out protected tokens
    if protected_tokens is not None:
        if grad.dim() == 3:  # [Batch, Seq, Vocab]
            grad.index_fill_(2, protected_tokens, 0.0)
        elif grad.dim() == 2:  # [Seq, Vocab] (Single item case)
            grad.index_fill_(1, protected_tokens, 0.0)

    # 2. Compute Norms per token
    # shape: [Batch, Seq, 1] or [Seq, 1]
    token_norms = torch.norm(grad, p=2, dim=-1, keepdim=True)

    # 3. Compute Max Norm over the sequence
    # This preserves relative magnitude between tokens in the sequence.
    # If token A has norm 0.1 and token B has 0.01, A gets scaled to 1.0, B to 0.1
    max_val_dim = 1 if grad.dim() == 3 else 0
    max_norms = token_norms.max(dim=max_val_dim, keepdim=True).values.clamp(min=1e-8)

    grad_safe = torch.where(max_norms > 0, grad / max_norms, grad)
    return grad_safe

def _step_gram_schmidt(logit_k, norm_vec_k, quality_k, basis_vectors, quality_scale):
    v_perp = norm_vec_k.clone()
    for q in basis_vectors:
        # Project v_k onto q
        proj = torch.dot(norm_vec_k.view(-1), q.view(-1)) * q
        v_perp = v_perp - proj

    resid_norm = torch.norm(v_perp, p=2)
    loss = -1.0 * (resid_norm ** 2) * (1.0 + quality_scale * quality_k)
    return torch.autograd.grad(loss, logit_k)[0]


def _step_sequential_logdet(logit_k, norm_vec_k, quality_k, previous_vecs, previous_qualities, quality_scale):
    all_vecs = torch.cat([previous_vecs, norm_vec_k], dim=0)
    all_quals = torch.cat([previous_qualities, quality_k.unsqueeze(0)], dim=0)

    K_sub = torch.mm(all_vecs, all_vecs.t())
    identity = torch.eye(K_sub.shape[0], device=K_sub.device)
    jitter = 1e-4

    q_sub = torch.outer(all_quals, all_quals)
    L_sub = K_sub * (1 + quality_scale * q_sub)

    term1 = torch.logdet(L_sub + jitter * identity)
    term2 = torch.logdet(L_sub + identity + jitter * identity)
    loss = -(term1 - term2)

    return torch.autograd.grad(loss, logit_k)[0]


def apply_dpp_guidance(
        logits,
        mask_index,
        x,
        alpha=3.0,
        quality_scale=1.0,
        entropy_threshold=0.6,
        protected_tokens=None,
        kernel_target="logits",
        embedding_matrix=None,
        strategy='gram_schmidt',  # 'joint', 'sequential', 'gram_schmidt'
        progressive=True,  # Re-compute features after every update?
        pooling_method='max'
):
    metadata = {
        "gate": 0.0,
        "entropy_map": [],
        "force_map": []
    }

    if logits.shape[0] < 2: return logits, metadata

    meta = compute_entropy_metadata(logits, entropy_threshold)
    metadata["gate"] = meta["gate"]
    metadata["entropy_map"] = meta["entropy_map"]

    if meta["gate"] < 0.05:
        return logits, metadata

    # 2. Init
    current_logits = logits.clone().detach()
    final_grads = torch.zeros_like(logits)

    history_vecs = []
    history_qualities = []

    with torch.enable_grad():

        # --- JOINT STRATEGY ---
        if strategy == "joint":
            logits_in = logits.detach().clone().requires_grad_(True)
            norm_vecs, quals = extract_feature_vector(
                logits_in, mask_index, x, embedding_matrix, kernel_target, pooling_method
            )

            K = torch.mm(norm_vecs, norm_vecs.t())
            identity = torch.eye(K.shape[0], device=K.device)
            jitter = 1e-4

            q_mat = torch.outer(quals, quals)
            L = K * (1 + quality_scale * q_mat)

            loss = -(torch.logdet(L + jitter * identity) - torch.logdet(L + identity + jitter * identity))
            raw_grads = torch.autograd.grad(loss, logits_in)[0]

            # Normalize Jointly
            final_grads = _normalize_gradient(raw_grads, protected_tokens)

            # Apply Gate
            final_grads = final_grads * meta["per_sample_gate"].unsqueeze(-1)

        # --- SEQUENTIAL / GRAM-SCHMIDT ---
        else:
            for k in range(logits.shape[0]):

                # A. Gradient Calculation
                logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)

                norm_vec_k, qual_k = extract_feature_vector_phasor(
                    logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0),
                    embedding_matrix, kernel_target, pooling_method
                )

                if k > 0:
                    if strategy == "gram_schmidt":
                        grad_k = _step_gram_schmidt(
                            logit_k, norm_vec_k, qual_k, history_vecs, quality_scale
                        )
                    elif strategy == "sequential":
                        prev_v_stack = torch.cat(history_vecs, dim=0) if history_vecs else torch.empty(0)
                        prev_q_stack = torch.tensor(history_qualities, device=logit_k.device)
                        grad_k = _step_sequential_logdet(
                            logit_k, norm_vec_k, qual_k, prev_v_stack, prev_q_stack, quality_scale
                        )

                    # Normalize (Max Norm) and Gate
                    grad_k_norm = _normalize_gradient(grad_k.squeeze(0), protected_tokens)
                    final_grads[k] = grad_k_norm * meta["per_sample_gate"][k]

                # B. Progressive Update
                if progressive and k > 0:
                    # Apply exactly what we calculated above
                    current_logits[k] -= (alpha * final_grads[k])

                # C. History Update
                if progressive:
                    with torch.no_grad():
                        # Re-extract from UPDATED logits
                        norm_vec_new, qual_new = extract_feature_vector_phasor(
                            current_logits[k].unsqueeze(0),
                            mask_index[k].unsqueeze(0),
                            x[k].unsqueeze(0),
                            embedding_matrix, kernel_target, pooling_method
                        )

                        if strategy == "gram_schmidt":
                            # Add ORTHOGONAL component to basis
                            v_perp = norm_vec_new.clone()
                            for q in history_vecs:
                                proj = torch.dot(norm_vec_new.view(-1), q.view(-1)) * q
                                v_perp = v_perp - proj

                            resid = torch.norm(v_perp, p=2)
                            if resid > 1e-6:
                                history_vecs.append(v_perp / resid)
                        else:
                            history_vecs.append(norm_vec_new)
                            history_qualities.append(qual_new.item())
                else:
                    # Non-progressive: Use original vectors
                    if strategy == "gram_schmidt":
                        v_perp = norm_vec_k.detach().clone()
                        for q in history_vecs:
                            proj = torch.dot(v_perp.view(-1), q.view(-1)) * q
                            v_perp = v_perp - proj
                        resid = torch.norm(v_perp)
                        if resid > 1e-6: history_vecs.append(v_perp / resid)
                    else:
                        history_vecs.append(norm_vec_k.detach())
                        history_qualities.append(qual_k.item())

    # 3. Final Update
    update = alpha * final_grads
    metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

    return logits - update, metadata


def calculate_diversity_score(eval_model, texts):
    if len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    # Average off-diagonal similarity
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = min(steps, mask_num.max().item()) if mask_num.max().item() > 0 else steps
    if steps == 0: steps = 1
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def run_generation(model,
                   embedding_matrix,
                   mask_token_id,
                   prompt,
                   batch_size,
                   steps,
                   gen_length,
                   alpha,
                   quality,
                   entropy_thresh,
                   temperature,
                   tokenizer,
                   pool,
                   target):

    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    prompt_len = prompt_ids.shape[1]
    x = torch.full((batch_size, prompt_len + gen_length), mask_token_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt_ids.clone()

    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    mask_index_init = (x[:, prompt_len:] == mask_token_id)
    num_transfer_tokens_schedule = get_num_transfer_tokens(mask_index_init, steps)
    protected_tokens = torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id], device=model.device)

    history_frames = []

    # progress_bar = st.progress(0)
    # status_text = st.empty()

    for i in range(steps):
        # status_text.text(f"Step {i + 1}/{steps}...")
        # progress_bar.progress((i + 1) / steps)

        mask_index = (x == mask_token_id)
        logits = model(x, attention_mask=attention_mask).logits
        gen_logits = logits[:, prompt_len:, :].clone()
        curr_alpha = alpha * (1 - (i / steps))

        metadata = {
            "gate": 0.0,
            "entropy_map": torch.zeros(batch_size, gen_length),
            "force_map": torch.zeros(batch_size, gen_length)
        }

        if curr_alpha > 0.0:
            gen_logits_guided, metadata = apply_dpp_guidance(
                gen_logits,
                alpha=curr_alpha,
                quality_scale=quality,
                entropy_threshold=entropy_thresh,
                protected_tokens=protected_tokens,
                kernel_target=target,
                mask_index=mask_index[:, prompt_len:],
                x=x[:, prompt_len:],
                embedding_matrix=embedding_matrix,
                pooling_method=pool
            )

            logits[:, prompt_len:, :] = gen_logits_guided

        # Capture Frame
        frame_data = {
            "step": i,
            "alpha": float(curr_alpha),
            "gate": float(metadata["gate"]),
            "batches": []
        }

        for b in range(batch_size):
            raw_ids = x[b, prompt_len:].tolist()
            display_tokens = []
            special_mask = []

            for tid in raw_ids:
                if tid == mask_token_id:
                    display_tokens.append("[MASK]")
                    special_mask.append(False)
                else:
                    t_str = tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
                    if tokenizer.mask_token and tokenizer.mask_token in t_str:
                        t_str = t_str.replace(tokenizer.mask_token, "[MASK]")
                    display_tokens.append(t_str)

                    is_special = (tid == tokenizer.eos_token_id or tid == tokenizer.pad_token_id)
                    special_mask.append(is_special)

            frame_data["batches"].append({
                "tokens": display_tokens,
                "is_mask": [tid == mask_token_id for tid in raw_ids],
                "is_special": special_mask,
                "entropy": metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else [],
                "force": metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []
            })
        history_frames.append(frame_data)

        # Sampling from LLADA with no AR, and low confidence remasking
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf).to(x0_p.device))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(batch_size):
            k = num_transfer_tokens_schedule[j, i]
            if k > 0:
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    # Final Frame
    final_frame = {"step": steps, "alpha": 0.0, "gate": 0.0, "batches": []}

    samples = tokenizer.batch_decode(x[:, prompt_len:],skip_special_tokens=True)

    for b in range(batch_size):
        raw_ids = x[b, prompt_len:].tolist()
        display_tokens = []
        special_mask = []
        for tid in raw_ids:
            t_str = tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
            display_tokens.append(t_str)
            is_special = (tid == tokenizer.eos_token_id or tid == tokenizer.pad_token_id)
            special_mask.append(is_special)

        final_frame["batches"].append({
            "tokens": display_tokens,
            "is_mask": [tid == mask_token_id for tid in raw_ids],
            "is_special": special_mask,
            "entropy": [], "force": []
        })
    history_frames.append(final_frame)

    # status_text.empty()
    # progress_bar.empty()

    return history_frames, samples
