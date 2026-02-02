import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


# function to print time

class FeatureExtractor:
    def __init__(self, embedding_matrix: Optional[torch.Tensor] = None,
                 kernel_target: str = 'logits',
                 pooling_method: str = 'max',
                 top_k: int = 0,
                 seq_len_scale: int = 64):
        self.embedding_matrix = embedding_matrix
        self.kernel_target = kernel_target
        self.pooling_method = pooling_method
        self.top_k = top_k
        self.seq_len_scale = seq_len_scale

    def extract(self, logit_k: torch.Tensor, mask_k: torch.Tensor, x_k: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Extracts the feature vector and quality score for a single batch item or the whole batch.
        """
        if logit_k.dim() == 2: logit_k = logit_k.unsqueeze(0)
        if mask_k.dim() == 1: mask_k = mask_k.unsqueeze(0)
        if x_k.dim() == 1: x_k = x_k.unsqueeze(0)

        probs = torch.softmax(logit_k, dim=-1)

        if self.top_k > 0:
            vals, indices = torch.topk(probs, k=self.top_k, dim=-1)
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

        if self.kernel_target == "embeddings" and self.embedding_matrix is not None:
            W = self.embedding_matrix.to(probs_in.device).detach()
            features = torch.matmul(probs_in, W)
        else:
            features = probs_in

        if self.pooling_method == "max":
            vecs = features.max(dim=1).values
        elif self.pooling_method == "mean":
            vecs = features.mean(dim=1)
        elif self.pooling_method == "positional":
            seq_len_scale = max(self.seq_len_scale, x_k.shape[-1])
            batch_size, seq_len, vocab_size = features.shape

            # Positional Encoding: Map sequence length [0, seq_len] to angle [0, pi/2]
            omega = (torch.pi / 2.0) / seq_len_scale

            pos_indices = torch.arange(seq_len, device=logit_k.device).view(1, -1, 1)
            angles = pos_indices.float() * omega

            real_part = (features * torch.cos(angles)).sum(dim=1)
            imag_part = (features * torch.sin(angles)).sum(dim=1)

            vecs = torch.cat([real_part, imag_part], dim=-1)
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented")

        norm_vec = F.normalize(vecs, p=2, dim=1)

        all_max_vals = probs_in.max(dim=-1).values
        masked_max_vals = all_max_vals * mask_k.float()
        num_masked = mask_k.sum(dim=1).clamp(min=1.0)
        quality = masked_max_vals.sum(dim=1) / num_masked

        return norm_vec, quality


class DPPStrategy(ABC):
    def __init__(self, alpha: float, quality_scale: float, feature_extractor: FeatureExtractor):
        self.alpha = alpha
        self.quality_scale = quality_scale
        self.feature_extractor = feature_extractor

    @abstractmethod
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        pass

    def _normalize_gradient(self, grad: torch.Tensor, protected_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if protected_tokens is not None:
            if grad.dim() == 3:  # [Batch, Seq, Vocab]
                grad.index_fill_(2, protected_tokens, 0.0)
            elif grad.dim() == 2:  # [Seq, Vocab] (Single item case)
                grad.index_fill_(1, protected_tokens, 0.0)

        token_norms = torch.norm(grad, p=2, dim=-1, keepdim=True)
        max_val_dim = 1 if grad.dim() == 3 else 0
        max_norms = token_norms.max(dim=max_val_dim, keepdim=True).values.clamp(min=1e-8)

        grad_safe = torch.where(max_norms > 0, grad / max_norms, grad)
        return grad_safe


class SequentialSubtractionStrategy(DPPStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        metadata = {"entropy_map": [], "force_map": []}
        if logits.shape[0] < 2 and not history_vecs: return logits, metadata

        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        local_history_vecs = list(history_vecs)
        local_history_qualities = list(history_qualities)

        for k in range(logits.shape[0]):
            logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)

            # If k > 0 or we have history, we apply the gradient
            if k > 0 or local_history_vecs:
                curr_logits_k = logit_k.clone()

                # Iterate through history (including previous items in this batch)
                for q in local_history_vecs:
                    curr_logits_k = curr_logits_k.detach().requires_grad_(True)
                    norm_vec, qual = self.feature_extractor.extract(
                        curr_logits_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0)
                    )

                    dot = torch.dot(norm_vec.view(-1), q.view(-1))
                    loss = (dot ** 2) * (self.quality_scale * qual)

                    grad = torch.autograd.grad(loss, curr_logits_k)[0]
                    grad = self._normalize_gradient(grad, protected_tokens)

                    with torch.no_grad():
                        curr_logits_k = curr_logits_k - (self.alpha * grad)

                total_displacement = logit_k - curr_logits_k
                if self.alpha > 1e-6:
                    final_grads[k] = (total_displacement / self.alpha).squeeze(0)

            if k > 0 or local_history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            # Add current item to history for next items
            with torch.no_grad():
                norm_vec_new, qual_new = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                local_history_vecs.append(norm_vec_new)
                local_history_qualities.append(qual_new.item())

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

        return logits - update, metadata


class GramSchmidtStrategy(DPPStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        metadata = {"entropy_map": [], "force_map": []}
        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        local_history_vecs = list(history_vecs)

        for k in range(logits.shape[0]):
            logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)
            norm_vec_k, qual_k = self.feature_extractor.extract(
                logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0)
            )

            if k > 0 or local_history_vecs:
                similarity_loss = torch.tensor(0.0, device=logit_k.device)
                for q in local_history_vecs:
                    dot = torch.dot(norm_vec_k.view(-1), q.view(-1))
                    similarity_loss = similarity_loss + (dot ** 2)

                weighted_loss = similarity_loss * (self.quality_scale * qual_k)
                raw_grads = torch.autograd.grad(weighted_loss, logit_k)[0]
                final_grads[k] = self._normalize_gradient(raw_grads, protected_tokens).squeeze(0)

            if k > 0 or local_history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            with torch.no_grad():
                norm_vec_new, _ = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                local_history_vecs.append(norm_vec_new)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()
        return logits - update, metadata


class RandomProbeStrategy(DPPStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        metadata = {"entropy_map": [], "force_map": []}
        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        local_history_vecs = list(history_vecs)

        for k in range(logits.shape[0]):
            logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)
            norm_vec_k, qual_k = self.feature_extractor.extract(
                logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0)
            )

            if k > 0 or local_history_vecs:

                ortho_basis = []
                for q in local_history_vecs:
                    u = q.clone()
                    for b in ortho_basis:
                        u = u - torch.dot(u.view(-1), b.view(-1)) * b

                    u_norm = torch.norm(u)
                    if u_norm > 1e-6:
                        ortho_basis.append(u / u_norm)

                probe = torch.randn_like(norm_vec_k).detach()

                for b in ortho_basis:
                    proj = torch.dot(probe.view(-1), b.view(-1)) * b
                    probe = probe - proj

                probe_norm = torch.norm(probe)
                if probe_norm > 1e-6:
                    target_dir = probe / probe_norm
                    alignment = torch.dot(norm_vec_k.view(-1), target_dir.view(-1))
                    loss = -alignment * (self.quality_scale * qual_k)

                    if loss.requires_grad:
                        raw_grads = torch.autograd.grad(loss, logit_k)[0]
                        final_grads[k] = self._normalize_gradient(raw_grads, protected_tokens).squeeze(0)

            if k > 0 or local_history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            with torch.no_grad():
                norm_vec_new, _ = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                local_history_vecs.append(norm_vec_new)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()
        return logits - update, metadata


class OrthogonalProjectionStrategy(DPPStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        metadata = {"entropy_map": [], "force_map": []}
        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        local_history_vecs = list(history_vecs)

        for k in range(logits.shape[0]):
            logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)
            norm_vec_k, qual_k = self.feature_extractor.extract(
                logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0)
            )

            if k > 0 or local_history_vecs:
                # Build ortho basis
                ortho_basis = []
                for q in local_history_vecs:
                    u = q.clone()
                    for b in ortho_basis:
                        u = u - torch.dot(u.view(-1), b.view(-1)) * b
                    u_norm = torch.norm(u)
                    if u_norm > 1e-6:
                        ortho_basis.append(u / u_norm)

                v_target = norm_vec_k.detach().clone()
                for b in ortho_basis:
                    proj = torch.dot(v_target.view(-1), b.view(-1)) * b
                    v_target = v_target - proj

                target_norm = torch.norm(v_target)
                if target_norm > 1e-6:
                    target_dir = v_target / target_norm
                    alignment = torch.dot(norm_vec_k.view(-1), target_dir.view(-1))
                    loss = -alignment * (self.quality_scale * qual_k)

                    if loss.requires_grad:
                        raw_grads = torch.autograd.grad(loss, logit_k)[0]
                        final_grads[k] = self._normalize_gradient(raw_grads, protected_tokens).squeeze(0)

            if k > 0 or local_history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            with torch.no_grad():
                norm_vec_new, _ = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                local_history_vecs.append(norm_vec_new)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()
        return logits - update, metadata


class JointStrategy(DPPStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        metadata = {"entropy_map": [], "force_map": []}

        logits_in = logits.detach().clone().requires_grad_(True)
        norm_vecs, quals = self.feature_extractor.extract(logits_in, mask_index, x)

        K = torch.mm(norm_vecs, norm_vecs.t())
        identity = torch.eye(K.shape[0], device=K.device)
        jitter = 1e-4
        q_mat = torch.outer(quals, quals)
        L = K * (1 + self.quality_scale * q_mat)
        loss = -(torch.logdet(L + jitter * identity) - torch.logdet(L + identity + jitter * identity))

        raw_grads = torch.autograd.grad(loss, logits_in)[0]
        final_grads = self._normalize_gradient(raw_grads, protected_tokens)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

        return logits - update, metadata


class DPPGenerator:
    def __init__(self, model, tokenizer, strategy: DPPStrategy, mask_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = mask_token_id
        self.device = model.device

    def compute_entropy_metadata(self, logits):
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy_map = -torch.sum(probs * log_probs, dim=-1)
        return {"entropy_map": entropy_map.detach().float().cpu()}

    def add_gumbel_noise(self, logits, temperature):
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        steps = min(steps, mask_num.max().item()) if mask_num.max().item() > 0 else steps
        if steps == 0: steps = 1
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int, temperature: float,
                 ) -> Tuple[List[Dict], List[str]]:

        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        encoded = self.tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        prompt_len = prompt_ids.shape[1]
        x = torch.full((batch_size, prompt_len + gen_length), self.mask_token_id, dtype=torch.long).to(self.device)
        x[:, :prompt_len] = prompt_ids.clone()

        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1)

        mask_index_init = (x[:, prompt_len:] == self.mask_token_id)
        num_transfer_tokens_schedule = self.get_num_transfer_tokens(mask_index_init, steps)
        protected_tokens = torch.tensor([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], device=self.device)

        history_frames = []

        for i in range(steps):
            mask_index = (x == self.mask_token_id)

            with torch.no_grad():
                logits = self.model(x, attention_mask=attention_mask).logits
                gen_logits = logits[:, prompt_len:, :].clone()

            # Decay alpha
            curr_alpha = self.strategy.alpha * (1 - (i / steps))

            # Temporarily update strategy alpha
            original_alpha = self.strategy.alpha
            self.strategy.alpha = curr_alpha

            metadata = {
                "entropy_map": torch.zeros(batch_size, gen_length),
                "force_map": torch.zeros(batch_size, gen_length)
            }

            # 1. Capture Original State (Before DPP)
            with torch.no_grad():
                probs_original = torch.softmax(gen_logits, dim=-1)
                top1_original = torch.argmax(probs_original, dim=-1)

                # Get Top K of Original Distribution
                k_val = 5
                topk_probs_orig, topk_indices_orig = torch.topk(probs_original, k=k_val, dim=-1)

            if curr_alpha > 0.0:
                # We only diversify the generated part
                gen_logits_guided, meta = self.strategy.apply(
                    gen_logits,
                    mask_index=mask_index[:, prompt_len:],
                    x=x[:, prompt_len:],
                    history_vecs=[],  # Reset history for each step as per original implementation
                    history_qualities=[],
                    protected_tokens=protected_tokens
                )
                logits[:, prompt_len:, :] = gen_logits_guided

                # Compute entropy for metadata
                ent_meta = self.compute_entropy_metadata(gen_logits)
                metadata["entropy_map"] = ent_meta["entropy_map"]
                if "force_map" in meta:
                    metadata["force_map"] = meta["force_map"]

            self.strategy.alpha = original_alpha  # Restore

            # 2. Capture Final State (After DPP)
            with torch.no_grad():
                gen_logits_final = logits[:, prompt_len:, :]
                probs_final = torch.softmax(gen_logits_final, dim=-1)
                top1_final = torch.argmax(probs_final, dim=-1)

                # Identify Flips
                flips = (top1_original != top1_final)

                # Get Top K of Final Distribution
                topk_probs_final, topk_indices_final = torch.topk(probs_final, k=k_val, dim=-1)

                # Get Original Probabilities at Final Indices (Current Top K)
                topk_probs_original_at_final = torch.gather(probs_original, -1, topk_indices_final)

                # Get Final Probabilities at Original Indices (Original Top K)
                topk_probs_final_at_orig = torch.gather(probs_final, -1, topk_indices_orig)

                # 3. Sampling Logic (Calculate transfer_index but don't apply yet)
                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
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

                # Logging / Frame Capture
                frame_data = {
                    "step": i,
                    "alpha": float(curr_alpha),
                    "batches": []
                }

                for b in range(batch_size):
                    raw_ids = x[b, prompt_len:].tolist()
                    display_tokens = []
                    special_mask = []

                    # Get transfer mask for this batch's generated part
                    batch_transfer_mask = transfer_index[b, prompt_len:].cpu().tolist()

                    # Extract Data for this batch
                    batch_topk_indices_final = topk_indices_final[b].cpu().numpy()
                    batch_topk_probs_final = topk_probs_final[b].cpu().numpy()
                    batch_topk_probs_orig_at_final = topk_probs_original_at_final[b].cpu().numpy()

                    batch_topk_indices_orig = topk_indices_orig[b].cpu().numpy()
                    batch_topk_probs_orig = topk_probs_orig[b].cpu().numpy()
                    batch_topk_probs_final_at_orig = topk_probs_final_at_orig[b].cpu().numpy()

                    batch_flips = flips[b].cpu().tolist()

                    # Lists for JSON
                    top_k_final_token_list = []
                    top_k_final_prob_list = []
                    top_k_final_prob_orig_list = []

                    top_k_orig_token_list = []
                    top_k_orig_prob_list = []
                    top_k_orig_prob_final_list = []

                    for t_idx_seq in range(len(raw_ids)):
                        # Final Top K Processing
                        decoded_final = []
                        for tk_id in batch_topk_indices_final[t_idx_seq]:
                            tk_str = self.tokenizer.decode([tk_id]).replace("Ġ", " ").replace("\n", "⏎")
                            decoded_final.append(tk_str)
                        top_k_final_token_list.append(decoded_final)
                        top_k_final_prob_list.append(batch_topk_probs_final[t_idx_seq].tolist())
                        top_k_final_prob_orig_list.append(batch_topk_probs_orig_at_final[t_idx_seq].tolist())

                        # Original Top K Processing
                        decoded_orig = []
                        for tk_id in batch_topk_indices_orig[t_idx_seq]:
                            tk_str = self.tokenizer.decode([tk_id]).replace("Ġ", " ").replace("\n", "⏎")
                            decoded_orig.append(tk_str)
                        top_k_orig_token_list.append(decoded_orig)
                        top_k_orig_prob_list.append(batch_topk_probs_orig[t_idx_seq].tolist())
                        top_k_orig_prob_final_list.append(batch_topk_probs_final_at_orig[t_idx_seq].tolist())

                    for tid in raw_ids:
                        if tid == self.mask_token_id:
                            display_tokens.append("[MASK]")
                            special_mask.append(False)
                        else:
                            t_str = self.tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
                            if self.tokenizer.mask_token and self.tokenizer.mask_token in t_str:
                                t_str = t_str.replace(self.tokenizer.mask_token, "[MASK]")
                            display_tokens.append(t_str)
                            is_special = (tid == self.tokenizer.eos_token_id or tid == self.tokenizer.pad_token_id)
                            special_mask.append(is_special)

                    frame_data["batches"].append({
                        "tokens": display_tokens,
                        "is_mask": [tid == self.mask_token_id for tid in raw_ids],
                        "is_special": special_mask,
                        "is_flip": batch_flips,
                        "is_unmasked_next": batch_transfer_mask,
                        # Current/Final Top K data
                        "top_k_tokens": top_k_final_token_list,
                        "top_k_probs": top_k_final_prob_list,
                        "top_k_probs_original": top_k_final_prob_orig_list,
                        # Original Top K data
                        "top_k_orig_tokens": top_k_orig_token_list,
                        "top_k_orig_probs": top_k_orig_prob_list,
                        "top_k_orig_probs_final": top_k_orig_prob_final_list,

                        "entropy": metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else [],
                        "force": metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []
                    })
                history_frames.append(frame_data)

                x[transfer_index] = x0[transfer_index]

        final_frame = {"step": steps, "alpha": 0.0, "batches": []}
        samples = self.tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)

        for b in range(batch_size):
            raw_ids = x[b, prompt_len:].tolist()
            display_tokens = []
            special_mask = []
            for tid in raw_ids:
                t_str = self.tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
                display_tokens.append(t_str)
                is_special = (tid == self.tokenizer.eos_token_id or tid == self.tokenizer.pad_token_id)
                special_mask.append(is_special)

            final_frame["batches"].append({
                "tokens": display_tokens,
                "is_mask": [tid == self.mask_token_id for tid in raw_ids],
                "is_special": special_mask,
                "entropy": [], "force": []
            })
        history_frames.append(final_frame)

        return history_frames, samples


def get_strategy(name: str, alpha: float, quality_scale: float, feature_extractor: FeatureExtractor) -> DPPStrategy:
    if name == "sequential_subtraction":
        return SequentialSubtractionStrategy(alpha, quality_scale, feature_extractor)
    elif name == "gram_schmidt":
        return GramSchmidtStrategy(alpha, quality_scale, feature_extractor)
    elif name == "orthogonal_projection":
        return OrthogonalProjectionStrategy(alpha, quality_scale, feature_extractor)
    elif name == "random_probe":  # <--- NEW
        return RandomProbeStrategy(alpha, quality_scale, feature_extractor)
    elif name == "joint":
        return JointStrategy(alpha, quality_scale, feature_extractor)
    else:
        raise ValueError(f"Unknown strategy: {name}")

