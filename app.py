import datetime
import json
import altair as alt
import pandas as pd
import streamlit as st
import torch
import hydra
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from dpp_gen import load_model
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


# Helper to load config without hydra.main decorator for Streamlit
#tsetset
def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
    return cfg

@st.cache_resource
def get_model_resources():
    print ("Loading model...")
    cfg = load_config()
    print ("Model loaded")
    return load_model(cfg)

def generate_viz_html(history):
    json_data = json.dumps(history)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
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
        .sidebar {{
            width: 300px; background: var(--panel); border-right: 1px solid var(--border);
            display: flex; flex-direction: column; padding: 20px; overflow-y: auto; flex-shrink: 0;
            font-size: 14px;
        }}
        .main-content {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .header {{ padding: 10px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 20px; background: var(--bg); }}
        .grid-container {{ flex-grow: 1; overflow-y: auto; padding: 20px; }}
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
        @keyframes pulse-gold {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }}
            70% {{ box-shadow: 0 0 0 4px rgba(255, 215, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }}
        }}
        .t-cell.new {{
            border: 1px solid var(--gold);
            animation: pulse-gold 1s infinite;
        }}
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
            <div style="display:flex; gap:10px; align-items:center;">
                <button id="playBtn" style="background:var(--accent); border:none; color:white; padding:5px 12px; border-radius:4px; cursor:pointer;">Play</button>
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
                if (text === '░' || text === '[MASK]') el.classList.add('mask');
                
                // Entropy coloring
                const ent = batch.entropy ? batch.entropy[tIdx] : 0;
                let bg = el.classList.contains('mask') ? '#222' : '#2b2b2b';
                if (ent > 0.1) {{
                    const op = Math.min(ent / 3, 0.6);
                    bg = `rgba(200, 50, 50, ${{op}})`;
                }}
                el.style.backgroundColor = bg;

                // Flip detection
                if (batch.is_flip && batch.is_flip[tIdx]) {{
                    el.classList.add('flip');
                }}

                // New/Unmasked detection
                if (batch.is_unmasked_next && batch.is_unmasked_next[tIdx]) {{
                    el.classList.add('new');
                }}

                el.onmouseenter = () => updateInspector(step, bIdx, tIdx);
                stream.appendChild(el);
            }});
            row.appendChild(stream);
            grid.appendChild(row);
        }});
    }}
    function updateInspector(step, bIdx, tIdx) {{
        const batch = history[step].batches[bIdx];
        let html = `<h2>Batch ${{bIdx}} : Token ${{tIdx}}</h2>`;
        html += `<div class="metric-box">
            <div><span class="stat-label">Token:</span> <span style="color:#fff; font-weight:bold">${{batch.tokens[tIdx]}}</span></div>
            <div><span class="stat-label">Entropy:</span> ${{(batch.entropy[tIdx]||0).toFixed(2)}}</div>
        </div>`;
        
        if (batch.is_flip && batch.is_flip[tIdx]) {{
             html += `<div class="badge flip">⚡ FLIPPED by DPP</div><br><br>`;
        }}

        if (batch.top_k_tokens && batch.top_k_tokens[tIdx]) {{
            html += `<h3>Top Candidates</h3><table><thead><tr><th style="text-align:left">Token</th><th style="text-align:right">Final P</th><th style="text-align:right">Orig P</th></tr></thead><tbody>`;
            const tops = batch.top_k_tokens[tIdx];
            const probs = batch.top_k_probs[tIdx];
            const orig_probs = batch.top_k_probs_original ? batch.top_k_probs_original[tIdx] : [];
            
            tops.forEach((t, i) => {{
                const p = probs[i];
                const p_orig = orig_probs[i] !== undefined ? orig_probs[i] : 0;
                
                // Colorize changes
                let p_style = "";
                if (p > p_orig + 0.05) p_style = "color: var(--success)";
                else if (p < p_orig - 0.05) p_style = "color: var(--danger)";

                html += `<tr>
                    <td>${{t}}</td>
                    <td class="val" style="${{p_style}}">${{p.toFixed(4)}}</td>
                    <td class="val" style="color:#666">${{p_orig.toFixed(4)}}</td>
                </tr>`;
            }});
            html += `</tbody></table>`;
        }}

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

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="DPP LLaDA Explorer v2")
    
    with st.spinner("Loading LLaDA Model..."):
        model, tokenizer, embedding_matrix, mask_token_id = get_model_resources()

    if "history_log" not in st.session_state:
        st.session_state.history_log = []

    st.title("Determinantal Point Process (DPP) Explorer")

    with st.sidebar:
        st.header("1. Parameters")
        prompt_input = st.text_area("Prompt", "Write python code to compute the factorial of n", height=80)

        c1, c2 = st.columns(2)
        with c1:
            batch_size = st.number_input("Batch Size", 1, 64, 4)
            gen_len = st.number_input("Gen Length", 16, 128, 64)

        alpha = st.number_input("Alpha (Diversity Step Size)", 0.0, 100.0, 1.0)

        with c2:
            steps = st.number_input("Steps", 10, 100, 32)
            temp = st.number_input("Temperature", 0.0, 5.0, 1.0)

        quality_scale = st.number_input("Quality scale", 0.0, 100.0, 1.0)

        st.divider()
        st.subheader("DPP Controls")
        strategy_name = st.selectbox("Strategy", ["sequential_subtraction", "gram_schmidt", "orthogonal_projection", "joint", "random_probe"])
        # quality_scale = st.slider("Quality Scale", 0.0, 10.0, 1.0)
        # alpha = st.slider("Alpha (Repulsion)", 0.0, 100.0, 5.0)

        target = st.selectbox("Kernel Target", ["logits", "embeddings"])
        pool = st.selectbox("Pooling", ["max", "mean", "positional"])

        if st.button("Generate New Run", type="primary", use_container_width=True):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            feature_extractor = FeatureExtractor(
                embedding_matrix=embedding_matrix,
                kernel_target=target,
                pooling_method=pool,
                top_k=0
            )
            
            dpp_strategy = get_strategy(strategy_name, alpha, quality_scale, feature_extractor)
            generator = DPPGenerator(model, tokenizer, dpp_strategy, mask_token_id)

            data, _ = generator.generate(prompt_input, batch_size, steps, gen_len, temp,)

            run_record = {
                "id": f"{timestamp} - {prompt_input[:20]}...",
                "timestamp": timestamp,
                "params": {
                    "prompt": prompt_input, "batch": batch_size, "steps": steps,
                    "alpha": alpha, "temp": temp, "strategy": strategy_name
                },
                "data": data
            }
            st.session_state.history_log.insert(0, run_record)
            st.rerun()

        st.divider()
        st.header("2. History")
        run_options = [r["id"] for r in st.session_state.history_log]
        selected_run_id = st.selectbox("Select Run", run_options, index=0 if run_options else None)
        current_run = next((r for r in st.session_state.history_log if r["id"] == selected_run_id), None)

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

    if current_run:
        st.markdown(f"**Viewing Run:** `{current_run['id']}`")
        tabs = st.tabs(["Visualization", "Metrics (Charts)", "Final Output"])

        with tabs[0]:
            html_view = generate_viz_html(current_run["data"])
            st.components.v1.html(html_view, height=600, scrolling=True)

        with tabs[1]:
            st.subheader("Force & Entropy Over Time")
            chart_data = []
            for frame in current_run["data"]:
                step = frame["step"]
                for b_idx, batch in enumerate(frame["batches"]):
                    ents = [e for e in batch["entropy"] if e > 0]
                    forces = [f for f in batch["force"] if f > 0]
                    avg_ent = sum(ents) / len(ents) if ents else 0
                    avg_force = sum(forces) / len(forces) if forces else 0
                    chart_data.append({
                        "step": step, "batch": f"Batch {b_idx}",
                        "entropy": avg_ent, "force": avg_force
                    })
            df = pd.DataFrame(chart_data)
            if not df.empty:
                c1 = alt.Chart(df).mark_line(point=True).encode(
                    x='step', y='force', color='batch', tooltip=['step', 'batch', 'force']
                ).properties(title="Average Repulsion Force per Step", height=300)
                c2 = alt.Chart(df).mark_line(point=True).encode(
                    x='step', y='entropy', color='batch', tooltip=['step', 'batch', 'entropy']
                ).properties(title="Average Entropy per Step", height=300)
                st.altair_chart(c1, use_container_width=True)
                st.altair_chart(c2, use_container_width=True)
            else:
                st.info("No metric data available.")

        with tabs[2]:
            st.subheader("Clean Output Text")
            final_frame = current_run["data"][-1]
            for i, batch in enumerate(final_frame["batches"]):
                clean_tokens = [t for t, is_spec in zip(batch["tokens"], batch["is_special"]) if not is_spec]
                full_str = "".join(clean_tokens).replace("⏎", "\n")
                with st.expander(f"Batch {i}", expanded=True):
                    st.code(full_str, language="text")
    else:
        st.info("No runs found. Generate a new run or upload a history file.")
