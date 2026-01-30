import json
import multiprocessing

import torch
from sentence_transformers import util


def save_html_dashboard(history, filename="llada_dpp_viz.html"):
    json_data = json.dumps(history)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLaDA DPP Visualization (With Masks)</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #eef2f5; padding: 20px; color: #333; }}
        .controls {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .slider-container {{ display: flex; align-items: center; gap: 15px; margin-top: 15px; }}
        input[type=range] {{ flex-grow: 1; height: 6px; }}
        button {{ padding: 8px 16px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; }}
        button:hover {{ background: #2980b9; }}

        .batch-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .batch-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .batch-header {{ font-weight: 600; margin-bottom: 15px; color: #7f8c8d; display: flex; justify-content: space-between; }}

        .token-flow {{ display: flex; flex-wrap: wrap; gap: 4px; line-height: 1.6; }}
        .token {{ 
            padding: 2px 5px; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 14px; 
            cursor: default; position: relative; border: 1px solid transparent;
        }}

        /* Mask Styling */
        /* REMOVED !important from background-color so JS can override it */
        .token.is-mask {{ color: #bdc3c7; border: 1px dashed #bdc3c7; background-color: transparent; }}

        /* When we view Metrics on masks, we make the text darker so it's readable against the colored background */
        .token.is-mask.has-metric {{ color: #555; border-style: solid; border-color: rgba(0,0,0,0.1); }}

        .token:hover {{ transform: scale(1.1); z-index: 100; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-color: #333; }}

        /* Tooltip */
        .token .tooltip {{
            visibility: hidden; width: 180px; background-color: #2c3e50; color: #fff; 
            border-radius: 6px; padding: 10px; position: absolute; z-index: 101;
            bottom: 130%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.2s;
            font-size: 12px; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .token:hover .tooltip {{ visibility: visible; opacity: 1; }}

        .metric-legend {{ display: flex; gap: 15px; font-size: 13px; margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .color-box {{ width: 12px; height: 12px; border-radius: 3px; }}
    </style>
</head>
<body>

    <div class="controls">
        <h2 style="margin:0;">LLaDA DPP: Mask Diffusion Explorer</h2>
        <div class="slider-container">
            <button onclick="changeStep(-1)">PREV</button>
            <input type="range" id="stepSlider" min="0" max="{len(history) - 1}" value="0" oninput="renderStep(this.value)">
            <button onclick="changeStep(1)">NEXT</button>
            <div style="font-weight: bold; width: 100px; text-align: right;" id="stepDisplay">Step 0</div>
        </div>

        <div style="display: flex; justify-content: space-between; margin-top: 15px;">
            <div>
                <strong>Global Metrics:</strong> 
                Alpha: <span id="alphaDisplay">0.0</span> | 
                Gate: <span id="gateDisplay">0.0</span>
            </div>
            <div>
                <strong>View Mode:</strong>
                <label><input type="radio" name="viewMode" value="force" checked onchange="renderStep(currentStep)"> Force Field (Red)</label>
                <label><input type="radio" name="viewMode" value="entropy" onchange="renderStep(currentStep)"> Entropy (Blue)</label>
            </div>
        </div>

        <div class="metric-legend">
            <div class="legend-item"><div class="color-box" style="background:transparent; border:1px dashed #bdc3c7;"></div> Mask (No Force)</div>
            <div class="legend-item"><div class="color-box" style="background:rgba(231,76,60,0.8)"></div> High Force</div>
            <div class="legend-item"><div class="color-box" style="background:rgba(52,152,219,0.8)"></div> High Uncertainty</div>
        </div>
    </div>

    <div id="grid" class="batch-grid"></div>

    <script>
        const data = {json_data};
        let currentStep = 0;

        function changeStep(delta) {{
            const slider = document.getElementById('stepSlider');
            let newVal = parseInt(slider.value) + delta;
            if (newVal >= 0 && newVal < data.length) {{
                slider.value = newVal;
                renderStep(newVal);
            }}
        }}

        function getColor(value, mode, isMask) {{
            // Force: Red, Entropy: Blue
            const r = mode === 'force' ? 231 : 52;
            const g = mode === 'force' ? 76 : 152;
            const b = mode === 'force' ? 60 : 219;

            // Scaling logic
            let alpha = 0;
            if (mode === 'force') alpha = Math.min(value * 2.0, 1.0); // Boosted visibility for force
            else alpha = Math.min(value * 0.8, 1.0); // Entropy usually 0-1.0

            // If it's a mask and the metric is very low, make it transparent to show the "dashed" styling clearly
            if (alpha < 0.1 && isMask) return 'transparent'; 

            return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
        }}

        function renderStep(stepIdx) {{
            currentStep = stepIdx;
            const stepData = data[stepIdx];
            const mode = document.querySelector('input[name="viewMode"]:checked').value;

            document.getElementById('stepDisplay').innerText = `Step ${{stepIdx}} / ${{data.length-1}}`;
            document.getElementById('alphaDisplay').innerText = stepData.alpha.toFixed(2);
            document.getElementById('gateDisplay').innerText = (stepData.gate || 0).toFixed(2);
            document.getElementById('stepSlider').value = stepIdx;

            const grid = document.getElementById('grid');
            grid.innerHTML = '';

            stepData.batches.forEach((batch, bIdx) => {{
                const card = document.createElement('div');
                card.className = 'batch-card';

                let html = `<div class="batch-header"><span>Batch ${{bIdx+1}}</span></div>`;
                html += `<div class="token-flow">`;

                batch.tokens.forEach((token, tIdx) => {{
                    const isMask = batch.is_mask[tIdx];
                    // Handle missing data for final frame
                    const force = (batch.force && batch.force[tIdx]) ? batch.force[tIdx] : 0;
                    const ent = (batch.entropy && batch.entropy[tIdx]) ? batch.entropy[tIdx] : 0;

                    const metricVal = mode === 'force' ? force : ent;
                    const bg = getColor(metricVal, mode, isMask);

                    const maskClass = isMask ? 'is-mask' : '';
                    // We add 'has-metric' if the color is visible (alpha >= 0.1)
                    let hasMetric = false;
                    if (mode === 'force' && force * 2.0 >= 0.1) hasMetric = true;
                    if (mode === 'entropy' && ent * 0.8 >= 0.1) hasMetric = true;

                    const metricClass = (isMask && hasMetric) ? 'has-metric' : '';

                    html += `
                        <div class="token ${{maskClass}} ${{metricClass}}" style="background-color: ${{bg}};">
                            ${{token}}
                            <span class="tooltip">
                                <strong>${{isMask ? "MASK TOKEN" : "Token: "+token}}</strong><br>
                                Force: ${{force.toFixed(4)}}<br>
                                Entropy: ${{ent.toFixed(4)}}
                            </span>
                        </div>
                    `;
                }});

                html += `</div>`;
                card.innerHTML = html;
                grid.appendChild(card);
            }});
        }}

        renderStep(0);
    </script>
</body>
</html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nSaved interactive visualization to: {filename}")


def calculate_diversity_score(eval_model, texts):
    if len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()


def run_test(code, result_queue):
    try:
        # Standard imports for HumanEval
        import math, re, collections, heapq, itertools, functools, bisect, random, statistics
        from typing import List, Tuple, Optional, Dict, Any, Union

        # Prepare global namespace
        local_scope = locals()
        exec(code, local_scope)
        result_queue.put("passed")
    except Exception as e:
        result_queue.put(f"failed: {{e}}")


def check_correctness(sample, test_code, timeout=5.0):
    full_code = "import math\nimport re\nimport collections\nimport heapq\nimport itertools\nimport functools\nfrom typing import *\n" + sample + "\n" + test_code

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_test, args=(full_code, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()  # Ensure it's dead
        return False

    if not q.empty():
        res = q.get()
        return res == "passed"
    return False


def calculate_pass_at_k(n, c, k):
    """
    n: total samples
    c: correct samples
    k: k for pass@k
    Formula: 1 - comb(n-c, k) / comb(n, k)
    """
    if n - c < k: return 1.0
    if n < k: return 0.0  # Should not happen if k <= n

    import math

    # Calculate using logs to avoid overflow/underflow for large numbers
    # log(comb(N, K)) = lgamma(N+1) - lgamma(K+1) - lgamma(N-K+1)

    def lcomb(N, K):
        if K < 0 or K > N: return float('-inf')
        return math.lgamma(N + 1) - math.lgamma(K + 1) - math.lgamma(N - K + 1)

    log_num = lcomb(n - c, k)
    log_den = lcomb(n, k)

    prob_all_wrong = math.exp(log_num - log_den)
    return 1.0 - prob_all_wrong


