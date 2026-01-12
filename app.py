import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import altair as alt
import datetime
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. PAGE & STATE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="DPP LLaDA Explorer v2")

if "history_log" not in st.session_state:
    st.session_state.history_log = []


# -----------------------------------------------------------------------------
# 2. MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

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

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        embedding_matrix = model.model.transformer.wte.weight
    else:
        embedding_matrix = None

    return model, tokenizer, embedding_matrix


with st.spinner("Loading LLaDA Model..."):
    model, tokenizer, EMBEDDING_MATRIX = load_model()

MASK_TOKEN_ID = tokenizer.mask_token_id
if MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = 126336


# -----------------------------------------------------------------------------
# 3. CORE LOGIC
# -----------------------------------------------------------------------------
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


def apply_dpp_guidance(logits, mask_index, x, alpha, quality_scale, entropy_threshold, protected_tokens,
                       kernel_target="logits"):
    metadata = {
        "gate": 0.0,
        "entropy_map": torch.zeros(logits.shape[:2]).tolist(),
        "force_map": torch.zeros(logits.shape[:2]).tolist()
    }

    if logits.shape[0] < 2: return logits, metadata

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -torch.sum(probs * log_probs, dim=-1)
    metadata["entropy_map"] = entropy_map.detach().float().cpu()

    pooled_entropy = entropy_map.mean(dim=1, keepdim=True)
    gate = torch.sigmoid((pooled_entropy - entropy_threshold) * 10.0)
    metadata["gate"] = gate.mean().item()

    if gate.mean() < 0.05:
        return logits, metadata

    with torch.enable_grad():
        logits_in = logits.detach().clone().requires_grad_(True)
        probs_in_ = torch.softmax(logits_in, dim=-1)

        probs_in = torch.zeros_like(probs_in_).to(probs_in_.device)
        probs_in[mask_index] = probs_in_[mask_index]
        one_hot_tokens = F.one_hot(x[~mask_index], num_classes=probs_in.shape[-1])
        probs_in[~mask_index] = one_hot_tokens.to(dtype=probs_in.dtype)

        if kernel_target == "embeddings" and EMBEDDING_MATRIX is not None:
            W = EMBEDDING_MATRIX.to(probs_in.device).detach()
            features = torch.matmul(probs_in, W)
        else:
            features = probs_in

        vecs = features.max(dim=1).values
        norm_vec = F.normalize(vecs, p=2, dim=1)
        final_grad = torch.zeros_like(logits_in)

        # Sequential Strategy
        for k in range(1, logits.shape[0]):
            sub_vecs = norm_vec[:k + 1]
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

    if protected_tokens is not None:
        final_grad.index_fill_(2, protected_tokens, 0.0)

    token_norms = torch.norm(final_grad, p=2, dim=-1, keepdim=True)
    max_norms = token_norms.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    grad_safe = torch.where(max_norms > 0, final_grad / max_norms, final_grad)
    grad_final = grad_safe * gate.unsqueeze(-1)

    update = alpha * grad_final
    metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

    return logits - update, metadata


@torch.no_grad()
def run_generation(prompt, batch_size, steps, gen_length, alpha, entropy_thresh, temperature):
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    prompt_len = prompt_ids.shape[1]
    x = torch.full((batch_size, prompt_len + gen_length), MASK_TOKEN_ID, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt_ids.clone()

    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    mask_index_init = (x[:, prompt_len:] == MASK_TOKEN_ID)
    num_transfer_tokens_schedule = get_num_transfer_tokens(mask_index_init, steps)
    protected_tokens = torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id], device=model.device)

    history_frames = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(steps):
        status_text.text(f"Step {i + 1}/{steps}...")
        progress_bar.progress((i + 1) / steps)

        mask_index = (x == MASK_TOKEN_ID)
        logits = model(x, attention_mask=attention_mask).logits
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
                quality_scale=1.0,
                entropy_threshold=entropy_thresh,
                protected_tokens=protected_tokens,
                kernel_target="logits",
                mask_index=mask_index[:, prompt_len:],
                x=x[:, prompt_len:]
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
                if tid == MASK_TOKEN_ID:
                    display_tokens.append("░")
                    special_mask.append(False)
                else:
                    t_str = tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
                    if tokenizer.mask_token and tokenizer.mask_token in t_str:
                        t_str = t_str.replace(tokenizer.mask_token, "░")
                    display_tokens.append(t_str)

                    # Check if EOS or PAD
                    is_special = (tid == tokenizer.eos_token_id or tid == tokenizer.pad_token_id)
                    special_mask.append(is_special)

            frame_data["batches"].append({
                "tokens": display_tokens,
                "is_mask": [tid == MASK_TOKEN_ID for tid in raw_ids],
                "is_special": special_mask,
                "entropy": metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else [],
                "force": metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []
            })
        history_frames.append(frame_data)

        # Sampling
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
            "is_mask": [tid == MASK_TOKEN_ID for tid in raw_ids],
            "is_special": special_mask,
            "entropy": [], "force": []
        })
    history_frames.append(final_frame)
    status_text.empty()
    progress_bar.empty()

    return history_frames


# -----------------------------------------------------------------------------
# 4. VISUALIZATION COMPONENT
# -----------------------------------------------------------------------------
def generate_viz_html(history):
    json_data = json.dumps(history)
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        :root {{ --bg: #0e1117; --panel: #262730; --text: #fafafa; --accent: #ff4b4b; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 5px; margin: 0; }}

        /* Grid View Styles */
        .token {{ 
            display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px; 
            font-family: 'Fira Code', monospace; font-size: 14px; background: #333; position: relative;
            cursor: default;
        }}
        .token.mask {{ color: #666; background: #222; }}

        /* Text View Styles */
        .text-view .token {{
            background: transparent !important;
            border: none !important;
            margin: 0; padding: 0;
            font-family: 'Georgia', serif;
            font-size: 16px;
            color: #ddd;
        }}
        .text-view .token.mask {{ color: #444; }}
        .text-view .token.special {{ display: none; }}
        .text-view .batch-row {{ 
            line-height: 1.6; 
            border-bottom: 1px solid #333; 
        }}

        .batch-row {{ margin-bottom: 15px; background: var(--panel); padding: 10px; border-radius: 8px; transition: all 0.3s; }}

        #controls {{ 
            margin-bottom: 10px; display: flex; gap: 10px; align-items: center; 
            background: var(--panel); padding: 10px; border-radius: 8px; 
            position: sticky; top: 0; z-index: 100;
        }}

        button {{
            background: #ff4b4b; border: none; color: white; padding: 5px 12px; border-radius: 4px; cursor: pointer;
        }}
        button:hover {{ background: #ff6b6b; }}

        .switch {{ display: flex; align-items: center; gap: 5px; font-size: 12px; margin-left: 15px; }}
        input[type=range] {{ flex-grow: 1; }}
    </style>
</head>
<body>
    <div id="controls">
        <button onclick="togglePlay()" id="playBtn">Play</button>
        <input type="range" id="slider" min="0" max="{len(history) - 1}" value="0" oninput="render(this.value)">
        <span id="stepLabel" style="min-width: 60px">Step: 0</span>
        <div class="switch">
            <input type="checkbox" id="textViewCheck" onchange="render(slider.value)">
            <label for="textViewCheck">Text View</label>
        </div>
    </div>

    <div id="container"></div>

    <script>
        const history = {json_data};
        const container = document.getElementById('container');
        const label = document.getElementById('stepLabel');
        const slider = document.getElementById('slider');
        const textViewCheck = document.getElementById('textViewCheck');
        let playing = false;
        let interval;

        function render(step) {{
            const frame = history[step];
            label.innerText = `Step: ${{frame.step}}`;
            const isTextView = textViewCheck.checked;

            if (isTextView) {{
                container.classList.add('text-view');
            }} else {{
                container.classList.remove('text-view');
            }}

            let html = '';
            frame.batches.forEach((b, i) => {{
                html += `<div class='batch-row'><div style='font-size:12px; color:#888; margin-bottom:5px; font-family:sans-serif'>Batch ${{i}}</div>`;

                b.tokens.forEach((t, j) => {{
                    let style = '';
                    let title = '';
                    let classes = 'token';

                    if (b.is_mask[j]) classes += ' mask';
                    if (b.is_special[j]) classes += ' special';

                    // Visuals (Only apply backgrounds in Grid View, keep tooltips everywhere)
                    if (!isTextView) {{
                        // Entropy (Red Background)
                        if (b.entropy && b.entropy[j] > 0.1) {{
                            const op = Math.min(b.entropy[j]/3, 0.8);
                            style += `background-color: rgba(200, 50, 50, ${{op}});`;
                        }}

                        // Force (Border)
                        if (b.force && b.force[j] > 0.1) {{
                            style += `border: 1px solid #ff4b4b;`;
                        }}
                    }} else {{
                        // In Text View, maybe subtle underline for high entropy?
                        if (b.entropy && b.entropy[j] > 0.5) {{
                            style += `text-decoration: underline wavy rgba(200,50,50,0.5);`;
                        }}
                    }}

                    if (b.entropy && b.entropy[j] !== undefined) title += `Entropy: ${{b.entropy[j].toFixed(2)}} `;
                    if (b.force && b.force[j] !== undefined) title += `\\nForce: ${{b.force[j].toFixed(2)}}`;

                    html += `<span class='${{classes}}' style='${{style}}' title='${{title}}'>${{t}}</span>`;
                }});
                html += `</div>`;
            }});
            container.innerHTML = html;
        }}

        function togglePlay() {{
            if (playing) {{
                clearInterval(interval);
                document.getElementById('playBtn').innerText = 'Play';
            }} else {{
                interval = setInterval(() => {{
                    let v = parseInt(slider.value) + 1;
                    if (v >= history.length) v = 0;
                    slider.value = v;
                    render(v);
                }}, 150);
                document.getElementById('playBtn').innerText = 'Pause';
            }}
            playing = !playing;
        }}

        render(0);
    </script>
</body>
</html>
    """


# -----------------------------------------------------------------------------
# 5. UI LAYOUT
# -----------------------------------------------------------------------------
st.title("Determinantal Point Process (DPP) Explorer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Parameters")
    prompt_input = st.text_area("Prompt", "Write a short poem about rust and code.", height=80)

    c1, c2 = st.columns(2)
    with c1:
        batch_size = st.number_input("Batch Size", 1, 8, 4)
        gen_len = st.number_input("Gen Length", 16, 128, 64)
    with c2:
        steps = st.number_input("Steps", 10, 100, 32)
        temp = st.number_input("Temperature", 0.0, 2.0, 1.0)

    st.divider()
    st.subheader("DPP Controls")
    alpha = st.slider("Alpha (Repulsion)", 0.0, 10.0, 5.0)
    entropy_thresh = st.slider("Entropy Thresh", 0.0, 2.0, 0.1)

    if st.button("Generate New Run", type="primary", use_container_width=True):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Run Generation
        data = run_generation(prompt_input, batch_size, steps, gen_len, alpha, entropy_thresh, temp)

        # Save to Session
        run_record = {
            "id": f"{timestamp} - {prompt_input[:20]}...",
            "timestamp": timestamp,
            "params": {
                "prompt": prompt_input, "batch": batch_size, "steps": steps,
                "alpha": alpha, "temp": temp
            },
            "data": data
        }
        st.session_state.history_log.insert(0, run_record)
        st.rerun()

    st.divider()
    st.header("2. History")

    # Run Selector
    run_options = [r["id"] for r in st.session_state.history_log]
    selected_run_id = st.selectbox("Select Run", run_options, index=0 if run_options else None)

    # Get Current Data
    current_run = next((r for r in st.session_state.history_log if r["id"] == selected_run_id), None)

    # File I/O
    st.subheader("Disk Storage")
    if st.session_state.history_log:
        json_str = json.dumps(st.session_state.history_log)
        st.download_button("Download History JSON", json_str, "dpp_history.json", "application/json")

    uploaded_file = st.file_uploader("Upload History JSON", type="json")
    if uploaded_file is not None:
        try:
            loaded_history = json.load(uploaded_file)
            st.session_state.history_log = loaded_history
            st.success("History loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- MAIN AREA ---
if current_run:
    st.markdown(f"**Viewing Run:** `{current_run['id']}`")

    tabs = st.tabs(["Visualization", "Metrics (Charts)", "Final Output"])

    # 1. VISUALIZATION TAB
    with tabs[0]:
        html_view = generate_viz_html(current_run["data"])
        st.components.v1.html(html_view, height=600, scrolling=True)

    # 2. METRICS TAB
    with tabs[1]:
        st.subheader("Force & Entropy Over Time")

        # Process Data for Charts
        chart_data = []
        for frame in current_run["data"]:
            step = frame["step"]
            for b_idx, batch in enumerate(frame["batches"]):
                # Calculate mean non-zero entropy/force for the frame
                ents = [e for e in batch["entropy"] if e > 0]
                forces = [f for f in batch["force"] if f > 0]

                avg_ent = sum(ents) / len(ents) if ents else 0
                avg_force = sum(forces) / len(forces) if forces else 0

                chart_data.append({
                    "step": step,
                    "batch": f"Batch {b_idx}",
                    "entropy": avg_ent,
                    "force": avg_force
                })

        df = pd.DataFrame(chart_data)

        if not df.empty:
            # Force Chart
            c1 = alt.Chart(df).mark_line(point=True).encode(
                x='step',
                y='force',
                color='batch',
                tooltip=['step', 'batch', 'force']
            ).properties(title="Average Repulsion Force per Step", height=300)

            # Entropy Chart
            c2 = alt.Chart(df).mark_line(point=True).encode(
                x='step',
                y='entropy',
                color='batch',
                tooltip=['step', 'batch', 'entropy']
            ).properties(title="Average Entropy per Step", height=300)

            st.altair_chart(c1, use_container_width=True)
            st.altair_chart(c2, use_container_width=True)
        else:
            st.info("No metric data available.")

    # 3. FINAL OUTPUT TAB
    with tabs[2]:
        st.subheader("Clean Output Text")
        final_frame = current_run["data"][-1]
        for i, batch in enumerate(final_frame["batches"]):
            # Filter special tokens for clean copy-paste
            clean_tokens = [t for t, is_spec in zip(batch["tokens"], batch["is_special"]) if not is_spec]
            full_str = "".join(clean_tokens).replace("⏎", "\n")

            with st.expander(f"Batch {i}", expanded=True):
                st.code(full_str, language="text")

else:
    st.info("No runs found. Generate a new run or upload a history file.")