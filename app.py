import datetime
import json

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from dpp_gen import run_generation


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

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 126336  # hardcoded for LLADA for now

    return model, tokenizer, embedding_matrix, mask_token_id




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


if __name__ == '__main__':
    with st.spinner("Loading LLaDA Model..."):
        model, tokenizer, embedding_matrix, mask_token_id = load_model()

    st.set_page_config(layout="wide", page_title="DPP LLaDA Explorer v2")

    if "history_log" not in st.session_state:
        st.session_state.history_log = []

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
            data, _ = run_generation(model, embedding_matrix, mask_token_id, prompt_input, batch_size, steps, gen_len,
                                  alpha,
                                  entropy_thresh, temp, tokenizer)

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
