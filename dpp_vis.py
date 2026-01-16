import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from dpp_gen import (
    apply_dpp_guidance,
    get_num_transfer_tokens,
    add_gumbel_noise
)

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def load_model():
    print(f"Loading {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    tokenizer.padding_side = 'left'

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None: mask_token_id = 126336

    return model, tokenizer, mask_token_id


def capture_top_k(logits, k=5):
    vals, indices = torch.topk(logits, k)
    return indices.tolist(), vals.tolist()


def capture_top_k_abs(tensor, k=5):
    vals, indices = torch.topk(tensor.abs(), k)
    signed_vals = tensor[indices]
    return indices.tolist(), signed_vals.tolist()


def decode_top_k(indices, vals, tokenizer):
    decoded = []
    for idx, val in zip(indices, vals):
        token_str = tokenizer.decode([idx]).replace("Ġ", " ").replace("\n", "⏎")
        decoded.append({"t": token_str, "v": round(val, 2)})
    return decoded


@torch.no_grad()
def run_rich_generation(
        prompt, model, tokenizer, mask_token_id,
        batch_size=4, steps=32, gen_length=64,
        alpha=5.0, quality_scale=1.0, entropy_thresh=0.1, temperature=1.0
):
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)

    prompt_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    prompt_len = prompt_ids.shape[1]

    x = torch.full((batch_size, prompt_len + gen_length), mask_token_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt_ids.clone()

    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, gen_length), device=model.device)], dim=-1)
    mask_index_init = (x[:, prompt_len:] == mask_token_id)
    num_transfer_tokens_schedule = get_num_transfer_tokens(mask_index_init, steps)
    protected_tokens = torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id], device=model.device)

    history = []
    print(f"Generating '{prompt[:30]}...' ({steps} steps)")

    for i in range(steps):
        print(f"  Step {i + 1}/{steps}...", end="\r")

        mask_index = (x == mask_token_id)
        logits = model(x, attention_mask=attention_mask).logits

        gen_logits_pre = logits[:, prompt_len:, :].clone()

        curr_alpha = alpha * (1 - (i / steps))

        metadata = {"gate": 0.0, "entropy_map": [], "force_map": []}
        update_tensor = torch.zeros_like(gen_logits_pre)

        top1_pre = gen_logits_pre.argmax(dim=-1)

        if curr_alpha > 0.0:
            gen_logits_post, metadata = apply_dpp_guidance(
                gen_logits_pre,
                mask_index=mask_index[:, prompt_len:],
                x=x[:, prompt_len:],
                alpha=curr_alpha,
                quality_scale=quality_scale,
                entropy_threshold=entropy_thresh,
                protected_tokens=protected_tokens,
                strategy='gram_schmidt',
                progressive=True,
                pooling_method='max'
            )
            update_tensor = gen_logits_pre - gen_logits_post
            logits[:, prompt_len:, :] = gen_logits_post
        else:
            gen_logits_post = gen_logits_pre

        top1_post = gen_logits_post.argmax(dim=-1)

        # Detect where DPP flipped
        flips = (top1_pre != top1_post) & mask_index[:, prompt_len:]

        # sampling logic from original LLADA generate.py, without semi-AR and with low conf. remasking
        logits_noisy = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)

        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf).to(x0_p.device))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(batch_size):
            k_trans = num_transfer_tokens_schedule[j, i]
            if k_trans > 0:
                _, select_index = torch.topk(confidence[j], k=k_trans)
                transfer_index[j, select_index] = True

        newly_unmasked_mask = transfer_index[:, prompt_len:]

        frame_data = {
            "step": i,
            "alpha": float(curr_alpha),
            "gate": float(metadata["gate"]),
            "batches": []
        }

        for b in range(batch_size):
            batch_info = {
                "tokens": [],
                "is_new": [],
                "is_flip": [],
                "details": []
            }

            b_ids = x[b, prompt_len:].tolist()
            b_logits_pre = gen_logits_pre[b]
            b_logits_post = gen_logits_post[b]
            b_update = update_tensor[b]
            b_force = metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []

            for t_idx, tid in enumerate(b_ids):
                # Token Text
                if tid == mask_token_id:
                    token_str = "░"
                else:
                    token_str = tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
                    # if tokenizer.mask_token in token_str: token_str = "░"

                batch_info["tokens"].append(token_str)

                is_new = newly_unmasked_mask[b, t_idx].item()
                is_flip = flips[b, t_idx].item()

                batch_info["is_new"].append(is_new)
                batch_info["is_flip"].append(is_flip)

                is_masked = (tid == mask_token_id)
                force_mag = b_force[t_idx] if t_idx < len(b_force) else 0.0

                detail_obj = None
                if is_masked or is_new or is_flip or force_mag > 0.5:
                    idx_pre, val_pre = capture_top_k(b_logits_pre[t_idx], k=5)
                    idx_post, val_post = capture_top_k(b_logits_post[t_idx], k=5)
                    idx_upd, val_upd = capture_top_k_abs(b_update[t_idx], k=5)

                    detail_obj = {
                        "pre": decode_top_k(idx_pre, val_pre, tokenizer),
                        "post": decode_top_k(idx_post, val_post, tokenizer),
                        "grad": decode_top_k(idx_upd, val_upd, tokenizer),
                        "f": round(force_mag, 2),
                        "flip": is_flip
                    }

                batch_info["details"].append(detail_obj)

            batch_info["force"] = b_force
            batch_info["entropy"] = metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else []
            frame_data["batches"].append(batch_info)

        history.append(frame_data)

        x[transfer_index] = x0[transfer_index]

    print("\nGeneration Complete.")
    return history


def save_rich_dashboard(history, filename="dpp_rich_dashboard.html"):
    json_data = json.dumps(history)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DPP Inspector v2</title>
    <style>
        :root {{
            --bg: #0e1117;
            --panel: #1e2127;
            --border: #30363d;
            --accent: #2c93ff;
            --text-main: #c9d1d9;
            --text-muted: #8b949e;
            --danger: #ff4b4b;
            --success: #2ea043;
            --gold: #ffd700;
            --flip: #d46bff;
        }}
        body {{
            background: var(--bg); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0; padding: 0; display: flex; height: 100vh; overflow: hidden;
        }}

        /* Layout */
        .sidebar {{
            width: 380px; background: var(--panel); border-right: 1px solid var(--border);
            display: flex; flex-direction: column; padding: 20px; overflow-y: auto; flex-shrink: 0;
        }}
        .main-content {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .header {{ padding: 15px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 20px; background: var(--bg); }}
        .grid-container {{ flex-grow: 1; overflow-y: auto; padding: 20px; }}

        /* Inspector */
        h2 {{ margin-top: 0; font-size: 1.1em; color: var(--accent); }}
        .metric-box {{ background: #252a33; padding: 10px; border-radius: 6px; margin-bottom: 15px; }}
        .badge {{
            display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; margin-right: 5px;
        }}
        .badge.flip {{ background: rgba(212, 107, 255, 0.2); color: var(--flip); border: 1px solid var(--flip); }}
        .badge.new {{ background: rgba(255, 215, 0, 0.2); color: var(--gold); border: 1px solid var(--gold); }}

        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; margin-bottom: 20px; }}
        td {{ padding: 3px 0; border-bottom: 1px solid #333; }}
        td.val {{ text-align: right; font-family: monospace; color: var(--accent); }}
        td.neg {{ color: var(--danger); }}
        td.pos {{ color: var(--success); }}

        /* Main Grid */
        .batch-row {{ margin-bottom: 25px; }}
        .batch-label {{ color: var(--text-muted); font-size: 0.8em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
        .token-stream {{ display: flex; flex-wrap: wrap; gap: 4px; }}

        .t-cell {{
            font-family: 'Fira Code', monospace; font-size: 13px;
            padding: 4px 7px; border-radius: 3px; cursor: crosshair;
            border: 1px solid transparent; transition: all 0.1s;
            min-width: 10px; text-align: center; position: relative;
        }}
        .t-cell:hover {{ border-color: var(--accent); transform: scale(1.1); z-index: 10; }}
        .t-cell.mask {{ background: #222; color: #555; }}
        .t-cell.selected {{ border-color: var(--accent); background: rgba(44, 147, 255, 0.2); }}

        /* --- NEW VISUALIZATIONS --- */

        /* 1. Newly Unmasked (Gold Pulse) */
        @keyframes pulse-gold {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }}
            70% {{ box-shadow: 0 0 0 4px rgba(255, 215, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }}
        }}
        .t-cell.new {{
            border: 1px solid var(--gold);
            animation: pulse-gold 1s infinite;
        }}

        /* 2. DPP Flip (Purple Marker) */
        .t-cell.flip::after {{
            content: '⚡';
            position: absolute; top: -8px; right: -6px;
            font-size: 10px; color: var(--flip);
            background: #0e1117; border-radius: 50%; padding: 1px;
        }}
        .t-cell.flip {{
            border-bottom: 2px solid var(--flip);
        }}

    </style>
</head>
<body>

    <div class="sidebar" id="inspector">
        <h2>Token Inspector</h2>
        <div style="color:#666; font-size:0.9em; margin-bottom:20px;">
            Hover grid to inspect. <br>
            <span style="color:var(--gold)">Gold Border</span> = Unmasked this step <br>
            <span style="color:var(--flip)">⚡ / Purple</span> = DPP forced a change
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <h3>DPP Rich Dashboard</h3>
            <div style="display:flex; gap:10px; margin-left:auto;">
                <button id="playBtn">Play</button>
                <input type="range" id="slider" min="0" max="{len(history) - 1}" value="0">
                <span id="stepLabel" style="font-family:monospace; min-width: 80px;">Step 0</span>
            </div>
        </div>
        <div class="grid-container" id="grid"></div>
    </div>

<script>
    const history = {json_data};
    const grid = document.getElementById('grid');
    const inspector = document.getElementById('inspector');
    const slider = document.getElementById('slider');
    const stepLabel = document.getElementById('stepLabel');
    const playBtn = document.getElementById('playBtn');

    let isPlaying = false;
    let playInterval;

    function renderStep(step) {{
        const frame = history[step];
        stepLabel.innerText = `Step ${{frame.step}}`;
        grid.innerHTML = '';

        frame.batches.forEach((batch, bIdx) => {{
            const row = document.createElement('div');
            row.className = 'batch-row';
            row.innerHTML = `<div class="batch-label">Batch ${{bIdx}}</div>`;

            const stream = document.createElement('div');
            stream.className = 'token-stream';

            batch.tokens.forEach((text, tIdx) => {{
                const el = document.createElement('div');
                el.className = 't-cell';
                el.innerText = text;

                // 1. Basic Classes
                if (text === '░' || text === '[MASK]') el.classList.add('mask');

                // 2. New Highlights
                if (batch.is_new[tIdx]) el.classList.add('new');
                if (batch.is_flip[tIdx]) el.classList.add('flip');

                // 3. Heatmap Background (Entropy)
                const ent = batch.entropy ? batch.entropy[tIdx] : 0;
                let bg = el.classList.contains('mask') ? '#222' : '#2b2b2b';
                if (ent > 0.1) {{
                    const op = Math.min(ent / 3, 0.6);
                    bg = `rgba(200, 50, 50, ${{op}})`;
                }}
                el.style.backgroundColor = bg;

                el.onmouseenter = () => updateInspector(step, bIdx, tIdx);
                stream.appendChild(el);
            }});

            row.appendChild(stream);
            grid.appendChild(row);
        }});
    }}

    function updateInspector(step, bIdx, tIdx) {{
        const batch = history[step].batches[bIdx];
        const details = batch.details ? batch.details[tIdx] : null;
        const isNew = batch.is_new[tIdx];
        const isFlip = batch.is_flip[tIdx];

        let html = `<h2>Batch ${{bIdx}} : Token ${{tIdx}}</h2>`;

        // Badges
        html += `<div style="margin-bottom:10px;">`;
        if (isNew) html += `<span class="badge new">UNMASKED</span>`;
        if (isFlip) html += `<span class="badge flip">DPP FLIP</span>`;
        html += `</div>`;

        html += `<div class="metric-box">
            <div><span class="stat-label">Token:</span> <span style="color:#fff; font-weight:bold">${{batch.tokens[tIdx]}}</span></div>
            <div><span class="stat-label">Entropy:</span> ${{(batch.entropy[tIdx]||0).toFixed(2)}}</div>
        </div>`;

        if (!details) {{
            html += `<div style="color:#555; text-align:center; margin-top:30px;">No detailed telemetry.<br>(Inactive token)</div>`;
            inspector.innerHTML = html;
            return;
        }}

        const renderTable = (title, rows) => {{
            let h = `<div style="margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:5px; color:#888; font-size:0.9em">${{title}}</div><table>`;
            rows.forEach(r => {{
                h += `<tr><td>${{r.t}}</td><td class="val">${{r.v.toFixed(2)}}</td></tr>`;
            }});
            return h + `</table>`;
        }};

        // 1. Pre
        html += renderTable("Model Preference (Original)", details.pre);

        // 2. Grad
        // Convert 'Update' to 'Effect' (-1 * Update) for display
        let gradRows = details.grad.map(g => ({{ t: g.t, v: -1 * g.v }}));
        let gHtml = `<div style="margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:5px; color:#888; font-size:0.9em">DPP Correction</div><table>`;
        gradRows.forEach(g => {{
            const cls = g.v > 0 ? 'pos' : 'neg';
            const sign = g.v > 0 ? '+' : '';
            gHtml += `<tr><td>${{g.t}}</td><td class="val ${{cls}}">${{sign}}${{g.v.toFixed(2)}}</td></tr>`;
        }});
        html += gHtml + `</table>`;

        // 3. Post
        html += renderTable("Final Logits (Used)", details.post);

        inspector.innerHTML = html;
    }}

    slider.addEventListener('input', (e) => renderStep(e.target.value));

    playBtn.addEventListener('click', () => {{
        if (isPlaying) {{
            clearInterval(playInterval);
            playBtn.innerText = "Play";
            isPlaying = false;
        }} else {{
            playBtn.innerText = "Pause";
            isPlaying = true;
            playInterval = setInterval(() => {{
                let v = parseInt(slider.value) + 1;
                if (v >= history.length) v = 0;
                slider.value = v;
                renderStep(v);
            }}, 200);
        }}
    }});

    renderStep(0);
</script>
</body>
</html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nDashboard saved to {filename}")


if __name__ == "__main__":
    PROMPT = "Write a python function to compute fibonacci."

    model, tokenizer, mask_id = load_model()

    history = run_rich_generation(
        prompt=PROMPT,
        model=model,
        tokenizer=tokenizer,
        mask_token_id=mask_id,
        batch_size=4,
        steps=32,
        gen_length=64,
        alpha=10.0,
        quality_scale=0.1
    )

    save_rich_dashboard(history)
