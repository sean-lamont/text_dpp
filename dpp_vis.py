import time
import json
import torch
import hydra
from omegaconf import DictConfig
from dpp_gen import load_model
from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

def save_rich_dashboard(history, filename="dpp_rich_dashboard.html"):
    json_data = json.dumps(history)
    # (HTML content same as before, omitted for brevity but should be included)
    # For now, let's just save the JSON or a simple HTML
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DPP Inspector v2</title>
    <style>
        /* ... (Same CSS as provided in original dpp_vis.py) ... */
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
            width: 380px; background: var(--panel); border-right: 1px solid var(--border);
            display: flex; flex-direction: column; padding: 20px; overflow-y: auto; flex-shrink: 0;
        }}
        .main-content {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .header {{ padding: 15px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 20px; background: var(--bg); }}
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
                if (text === '░' || text === '[MASK]') el.classList.add('mask');
                // Note: is_new and is_flip logic needs to be reconstructed from history if not explicitly saved
                // For now, we just visualize entropy
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
        let html = `<h2>Batch ${{bIdx}} : Token ${{tIdx}}</h2>`;
        html += `<div class="metric-box">
            <div><span class="stat-label">Token:</span> <span style="color:#fff; font-weight:bold">${{batch.tokens[tIdx]}}</span></div>
            <div><span class="stat-label">Entropy:</span> ${{(batch.entropy[tIdx]||0).toFixed(2)}}</div>
        </div>`;
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=cfg.strategy.target,
        pooling_method=cfg.strategy.pool,
        top_k=cfg.strategy.top_k
    )
    
    dpp_strategy = get_strategy(
        cfg.strategy.name, 
        cfg.strategy.alpha, 
        cfg.strategy.quality_scale, 
        feature_extractor
    )
    
    generator = DPPGenerator(model, tokenizer, dpp_strategy, mask_token_id)

    history, _ = generator.generate(
        prompt=cfg.prompt,
        batch_size=cfg.batch_size,
        steps=cfg.steps,
        gen_length=cfg.gen_length,
        temperature=cfg.temperature,
        use_wandb=False
    )

    save_rich_dashboard(history)

if __name__ == "__main__":
    main()
