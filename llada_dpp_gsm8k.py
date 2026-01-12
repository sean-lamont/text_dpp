import time
import re
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Optional: Semantic Diversity Measurement
try:
    from sentence_transformers import SentenceTransformer, util

    eval_model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False
    print("Notice: Install 'sentence-transformers' for numeric diversity scores.")

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
print(model)

MASK_TOKEN_ID = tokenizer.mask_token_id
if MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = 126336

# Extract Embedding Matrix for Semantic Guidance
if hasattr(model, "model") and hasattr(model.model, "transformer"):
    # EMBEDDING_MATRIX = model.model.embed_tokens.weight
    EMBEDDING_MATRIX = model.model.transformer.wte.weight  # for llada
else:
    EMBEDDING_MATRIX = None
    print("Warning: Embedding matrix not found. 'embeddings' target will fail.")


def apply_dpp_guidance(
        logits,
        alpha=3.0,
        quality_scale=1.0,
        pooling_method="mean",
        use_projection=True,
        kernel_target="logits",
        loss_type="diverseflow",
        entropy_threshold=0.6,  # Below this, we turn off DPP
        protected_tokens=[tokenizer.eos_token_id]  # List of IDs to ignore (e.g. EOS)
        # protected_tokens=None  # List of IDs to ignore (e.g. EOS)
):
    if logits.shape[0] < 2: return logits

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

    # Create a "Gate": 1.0 when uncertain, 0.0 when certain
    # Steep sigmoid centered at threshold
    gate = torch.sigmoid((entropy - entropy_threshold) * 10.0)

    # If the whole batch is confident (e.g. copying prompt), skip
    if gate.mean() < 0.05:
        return logits

    with torch.enable_grad():
        logits_in = logits.detach().clone().requires_grad_(True)
        probs_in = torch.softmax(logits_in, dim=-1)

        if kernel_target == "embeddings" and EMBEDDING_MATRIX is not None:
            W = EMBEDDING_MATRIX.to(probs_in.device).detach()
            features = torch.matmul(probs_in, W)
        else:
            features = probs_in

        if pooling_method == "max":
            vecs = features.max(dim=1).values
        else:
            vecs = features.mean(dim=1)

        norm_vec = F.normalize(vecs, p=2, dim=1)
        K = torch.mm(norm_vec, norm_vec.t())

        identity = torch.eye(K.shape[0], device=K.device)

        jitter = 1e-4
        # jitter = 0

        max_conf = probs_in.max(dim=-1).values.mean(dim=1)
        quality_matrix = torch.outer(max_conf, max_conf)
        L = K * (1 + quality_scale * quality_matrix)

        if loss_type == "diverseflow":
            term1 = torch.logdet(L + jitter * identity)
            term2 = torch.logdet(L + identity + jitter * identity)
            loss = -(term1 - term2)
        else:
            loss = -torch.logdet(L + jitter * identity)

        grad = torch.autograd.grad(loss, logits_in)[0]

    if protected_tokens is not None:
        grad.index_fill_(2, torch.tensor(protected_tokens).to('cuda'), 0.0)

    if use_projection:
        g_norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
        grad_safe = grad / (g_norm + 1e-8)
        u = logits.detach()
        inner = (grad_safe * u).sum(dim=-1, keepdim=True)
        u_norm = (u * u).sum(dim=-1, keepdim=True)
        proj = (inner / (u_norm + 1e-8)) * u
        grad_safe = grad_safe - proj

        grad_final = grad_safe * gate
    else:
        grad_final = grad * gate

    return logits - (alpha * grad_final)


@torch.no_grad()
def generate_llada(
        prompt,
        batch_size=4,
        steps=32,
        use_dpp=False,
        # Configs
        alpha=3.0,
        quality=1.0,
        pool="mean",
        proj=True,
        target="logits"
):
    mask_id = MASK_TOKEN_ID

    # formatted_prompt = f"Question: {prompt}\nLet's think step by step.\nAnswer:"
    formatted_prompt = prompt

    messages = [{"role": "user", "content": formatted_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.repeat(batch_size, 1).to(model.device)

    gen_len = 64
    mask_tokens = torch.full((batch_size, gen_len), mask_id, device=model.device, dtype=torch.long)
    input_ids = torch.cat([input_ids, mask_tokens], dim=1)

    prompt_len = input_ids.shape[1] - gen_len

    for step in range(steps):
        outputs = model(input_ids)
        gen_logits = outputs.logits[:, prompt_len:, :]

        curr_alpha = alpha * (1 - (step / steps))

        if step > (steps * 0.3):
            curr_alpha = 0.0

        if use_dpp and curr_alpha > 0.1:
            gen_logits = apply_dpp_guidance(
                gen_logits,
                alpha=curr_alpha,
                quality_scale=quality,
                pooling_method=pool,
                use_projection=proj,
                kernel_target=target
            )

        probs = F.softmax(gen_logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        confidences = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
        progress = (step + 1) / steps
        n_unmasked = int(gen_len * progress)

        if step == steps - 1:
            input_ids[:, prompt_len:] = pred_ids
            break

        noise = torch.rand_like(confidences)
        top_k = torch.topk(confidences + noise, k=n_unmasked, dim=1)

        new_ids = torch.full_like(pred_ids, mask_id)
        new_ids.scatter_(1, top_k.indices, pred_ids.gather(1, top_k.indices))
        input_ids[:, prompt_len:] = new_ids

    return tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)


def extract_answer_num(text):
    # Try to find the last number in the text as the answer
    try:
        text = text.replace(',', '')  # Handle 1,000
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums: return float(nums[-1])
    except:
        pass
    return None


def extract_gold_num(answer_str):
    if "####" in answer_str:
        try:
            val = answer_str.split("####")[1].strip()
            return float(val.replace(',', ''))
        except:
            pass
    return None


def calculate_diversity_score(texts):
    if not HAS_EVAL or len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    # Average off-diagonal similarity
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()


def run_gsm8k_benchmark(n_problems=20, configs=[]):
    dataset = load_dataset("gsm8k", "main", split="test")

    # Store results: {config_name: {'correct': 0, 'diversity': []}}
    results = {cfg['name']: {'correct': 0, 'diversity': []} for cfg in configs}

    print(f"\n>>> STARTING BENCHMARK on {n_problems} Problems")

    for i in range(n_problems):
        row = dataset[i + 10]
        q = row['question']
        gold = extract_gold_num(row['answer'])
        if gold is None: continue

        print(f"\n--- Problem {i + 1} (Gold: {gold}) ---")

        for cfg in configs:
            start_t = time.time()
            samples = generate_llada(
                q,
                batch_size=4,
                steps=32,
                use_dpp=cfg['use_dpp'],
                alpha=cfg.get('alpha', 0),
                quality=cfg.get('quality', 0),
                pool=cfg.get('pool', 'mean'),
                proj=cfg.get('proj', True),
                target=cfg.get('target', 'logits')
            )

            for j, s in enumerate(samples):
                print(f"[{j + 1}] {s.strip().replace(chr(10), ' / ')}")
            print("")

            is_solved = False
            preds = []
            for s in samples:
                val = extract_answer_num(s)
                preds.append(val)
                if val is not None and abs(val - gold) < 1e-4:
                    is_solved = True

            if is_solved:
                results[cfg['name']]['correct'] += 1

            # 2. Check Diversity
            div = calculate_diversity_score(samples)
            results[cfg['name']]['diversity'].append(div)

            print(
                f"[{cfg['name']:<15}] Running Pass@K: {(results[cfg['name']]['correct'] / (i + 1)) * 100:.1f}%"
            )
            print(
                f"[{cfg['name']:<15}] Solved: {str(is_solved):<5} | Div: {div:.3f} | Time: {time.time() - start_t:.1f}s")

    # Final Report
    print("\n" + "=" * 60)
    print(f"{'CONFIGURATION':<25} | {'PASS@4':<10} | {'DIVERSITY':<10}")
    print("=" * 60)

    for name, data in results.items():
        pass_k = (data['correct'] / n_problems) * 100
        avg_div = np.mean(data['diversity']) if data['diversity'] else 0.0
        print(f"{name:<25} | {pass_k:.1f}%      | {avg_div:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    PROMPT = "Describe the logic to prove that n*(n+1) is even. Do not use code."
    # PROMPT = "Write a python function to check if a word is a palindrome"
    # PROMPT = "Write a haiku about a robot realizing it is alive."
    # PROMPT = "Explain what 'Time' is."
    # PROMPT = "Explain a metaphor for how neural networks learn."
    # PROMPT = "Write a python program to train a neural network"

    print(f"PROMPT: {PROMPT}\n")

    # Define your experiments here
    settings = [
        {
            "name": "Baseline (No DPP)",
            "use_dpp": False,
            "alpha": 0.0, "quality": 0.0, "pool": "mean", "proj": False, "target": "logits"
        },
        {
            "name": "DPP (Logits, Alpha=3.0, No Projection)",
            "use_dpp": True,
            "alpha": 3.0, "quality": 1.0, "pool": "mean", "proj": False, "target": "logits"
        },
        {
            "name": "DPP (Logits, Alpha=3.0, Projected)",
            "use_dpp": True,
            "alpha": 3.0, "quality": 1.0, "pool": "mean", "proj": True, "target": "logits"
        },
    ]

    for cfg in settings:
        print(f"--- {cfg['name']} ---")
        start = time.time()

        samples = generate_llada(
            PROMPT,
            batch_size=4,
            steps=32,
            use_dpp=cfg['use_dpp'],
            alpha=cfg['alpha'],
            quality=cfg['quality'],
            pool=cfg['pool'],
            proj=cfg['proj'],
            target=cfg['target']
        )

        print(f"Time: {time.time() - start:.2f}s")
        for i, s in enumerate(samples):
            print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")
        print("")
