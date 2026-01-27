import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import util


def compute_entropy_metadata(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -torch.sum(probs * log_probs, dim=-1)

    metadata = {
        "entropy_map": entropy_map.detach().float().cpu(),
    }
    return metadata


def extract_feature_vector(logit_k, mask_k, x_k, embedding_matrix, kernel_target='logits', pooling_method='max',
                           top_k=5, seq_len_scale=64):
    """
    Extracts the feature vector and quality score for a single batch item
    or the whole batch.
    """

    if logit_k.dim() == 2: logit_k = logit_k.unsqueeze(0)
    if mask_k.dim() == 1: mask_k = mask_k.unsqueeze(0)
    if x_k.dim() == 1: x_k = x_k.unsqueeze(0)

    probs = torch.softmax(logit_k, dim=-1)

    if top_k > 0:
        vals, indices = torch.topk(probs, k=top_k, dim=-1)
        probs_in_ = torch.zeros_like(probs)
        probs_in_.scatter_(2, indices, vals)
    else:
        probs_in_ = probs

    # set the probs for already established tokens to be one-hot for their selection
    probs_in = torch.zeros_like(probs_in_)
    probs_in[mask_k] = probs_in_[mask_k]

    if (~mask_k).any():
        one_hot_tokens = F.one_hot(x_k[~mask_k], num_classes=probs_in.shape[-1])
        probs_in[~mask_k] = one_hot_tokens.to(dtype=probs_in.dtype)

    if kernel_target == "embeddings" and embedding_matrix is not None:
        W = embedding_matrix.to(probs_in.device).detach()
        features = torch.matmul(probs_in, W)
    else:
        features = probs_in

    if pooling_method == "max":
        vecs = features.max(dim=1).values
    elif pooling_method == "mean":
        vecs = features.mean(dim=1)
    elif pooling_method == "positional":
        seq_len_scale = max(seq_len_scale, x_k.shape[-1])

        batch_size, seq_len, vocab_size = features.shape

        # Positional Encoding: Map sequence length [0, seq_len] to angle [0, pi/2]
        omega = (torch.pi / 2.0) / seq_len_scale

        pos_indices = torch.arange(seq_len, device=logit_k.device).view(1, -1, 1)
        angles = pos_indices.float() * omega

        real_part = (features * torch.cos(angles)).sum(dim=1)

        imag_part = (features * torch.sin(angles)).sum(dim=1)

        vecs = torch.cat([real_part, imag_part], dim=-1)
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented")

    norm_vec = F.normalize(vecs, p=2, dim=1)

    all_max_vals = probs_in.max(dim=-1).values
    masked_max_vals = all_max_vals * mask_k.float()
    num_masked = mask_k.sum(dim=1).clamp(min=1.0)
    quality = masked_max_vals.sum(dim=1) / num_masked

    return norm_vec, quality


def apply_dpp_guidance(
        logits,
        mask_index,
        x,
        alpha=3.0,
        quality_scale=1.0,
        protected_tokens=None,
        kernel_target="logits",
        embedding_matrix=None,
        strategy='sequential_subtraction',
        progressive=True,
        pooling_method='positional'
):
    metadata = {
        "entropy_map": [],
        "force_map": []
    }

    if logits.shape[0] < 2: return logits, metadata

    meta = compute_entropy_metadata(logits)
    metadata["entropy_map"] = meta["entropy_map"]

    current_logits = logits.clone().detach()
    final_grads = torch.zeros_like(logits)

    history_vecs = []
    history_qualities = []

    with torch.enable_grad():
        if strategy == "joint":  # simplest approach, moves all points simultaneously with logdet loss from diverseflow
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
            final_grads = _normalize_gradient(raw_grads, protected_tokens)

        else:  # sequential optimisation, where we fix 0, move item k wrt updates from :k-1
            for k in range(logits.shape[0]):
                logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)

                norm_vec_k, qual_k = extract_feature_vector(
                    logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0),
                    embedding_matrix, kernel_target, pooling_method,
                    top_k=5
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
                    elif strategy == "sequential_subtraction":
                        grad_k = _step_sequential_subtraction(
                            logit_k,
                            mask_index[k].unsqueeze(0),
                            x[k].unsqueeze(0),
                            embedding_matrix,
                            kernel_target,
                            pooling_method,
                            history_vecs,
                            quality_scale,
                            alpha
                        )

                    # todo normalisation here vs sequential sub?
                    grad_k_norm = _normalize_gradient(grad_k.squeeze(0), protected_tokens)
                    final_grads[k] = grad_k_norm

                if k > 0:
                    current_logits[k] -= (alpha * final_grads[k])

                with torch.no_grad():
                    norm_vec_new, qual_new = extract_feature_vector(
                        current_logits[k].unsqueeze(0),
                        mask_index[k].unsqueeze(0),
                        x[k].unsqueeze(0),
                        embedding_matrix, kernel_target, pooling_method,
                        top_k=5
                    )

                    history_vecs.append(norm_vec_new)
                    history_qualities.append(qual_new.item())

    update = alpha * final_grads
    metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()


def _normalize_gradient(grad, protected_tokens=None):
    """
    Gradient normalization (Max Norm)
    """

    if protected_tokens is not None:
        if grad.dim() == 3:  # [Batch, Seq, Vocab]
            grad.index_fill_(2, protected_tokens, 0.0)
        elif grad.dim() == 2:  # [Seq, Vocab] (Single item case)
            grad.index_fill_(1, protected_tokens, 0.0)

    token_norms = torch.norm(grad, p=2, dim=-1, keepdim=True)

    # This preserves relative magnitude between tokens in the sequence.
    # If token A has norm 0.1 and token B has 0.01, A gets scaled to 1.0, B to 0.1
    max_val_dim = 1 if grad.dim() == 3 else 0
    max_norms = token_norms.max(dim=max_val_dim, keepdim=True).values.clamp(min=1e-8)

    grad_safe = torch.where(max_norms > 0, grad / max_norms, grad)
    return grad_safe


def _step_sequential_logdet(logit_k, norm_vec_k, quality_k, previous_vecs, previous_qualities, quality_scale):
    all_vecs = torch.cat([previous_vecs, norm_vec_k], dim=0)
    all_quals = torch.cat([previous_qualities, quality_k.unsqueeze(0)], dim=0)

    K_sub = torch.mm(all_vecs, all_vecs.t())
    identity = torch.eye(K_sub.shape[0], device=K_sub.device)
    jitter = 1e-4

    q_sub = torch.outer(all_quals, all_quals)
    L_sub = K_sub * (quality_scale * q_sub)

    term1 = torch.logdet(L_sub + jitter * identity)
    term2 = torch.logdet(L_sub + identity + jitter * identity)
    loss = -(term1 - term2)

    return torch.autograd.grad(loss, logit_k)[0]


def _step_gram_schmidt(logit_k, norm_vec_k, quality_k, basis_vectors, quality_scale):
    """
    Calculates the gradient to MINIMIZE similarity with history.
    Uses dot-product formulation to avoid zero-gradients at perfect overlap, which breaks for logdet.
    """
    # We want to minimize the sum of squared cosine similarities.
    # Loss = Sum( (v . q)^2 )

    similarity_loss = torch.tensor(0.0, device=logit_k.device)

    for q in basis_vectors:
        dot = torch.dot(norm_vec_k.view(-1), q.view(-1))
        similarity_loss = similarity_loss + (dot ** 2)

    weighted_loss = similarity_loss * (quality_scale * quality_k)

    return torch.autograd.grad(weighted_loss, logit_k)[0]


def _step_sequential_subtraction(logit_k, mask_k, x_k, embedding_matrix, kernel_target, pooling_method, history_vecs,
                                 quality_scale, alpha):
    """
     Iterative/Sequential Updates. Multiple internal updates, step logits against each history item so the next step
     is aware of current state of logits. Prevents logits moving to previously visited space, which is possible when
     the starting state is diverse with respect to previous states, but the final point is now not diverse.
    """
    curr_logits = logit_k.clone()

    # We iterate through history, updating the logits at each step
    for q in history_vecs:
        # Re-attach gradient to the current (shifted) state
        curr_logits = curr_logits.detach().requires_grad_(True)

        norm_vec, qual = extract_feature_vector(
            curr_logits, mask_k, x_k, embedding_matrix, kernel_target, pooling_method
        )

        # Calculate Gradient vs current history item q.
        # Loss = (v . q)^2
        dot = torch.dot(norm_vec.view(-1), q.view(-1))
        loss = (dot ** 2) * (quality_scale * qual)

        grad = torch.autograd.grad(loss, curr_logits)[0]

        with torch.no_grad():
            curr_logits = curr_logits - (alpha * grad)

    # Return the total effective gradient (Total Displacement / Alpha)
    # Slight hack to fit the standard `logits -= alpha * grad` pattern
    total_displacement = logit_k - curr_logits

    # Avoid division by zero if alpha is tiny
    if alpha < 1e-6:
        return torch.zeros_like(logit_k)

    return total_displacement / alpha


# simple cosine similarity over embedding space between texts as a measure of diversity
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
                   temperature,
                   tokenizer,
                   pool,
                   target):
    '''
    Main generation function. Follows pattern from LLADA generate.py, however includes DPP guidance if alpha > 0,
    and logging for DPP process.
    '''

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
            "entropy_map": torch.zeros(batch_size, gen_length),
            "force_map": torch.zeros(batch_size, gen_length)
        }

        if curr_alpha > 0.0:
            gen_logits_guided, metadata = apply_dpp_guidance(
                gen_logits,
                alpha=curr_alpha,
                quality_scale=quality,
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

    final_frame = {"step": steps, "alpha": 0.0, "batches": []}

    samples = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)

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
