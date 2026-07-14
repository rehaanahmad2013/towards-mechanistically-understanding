# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "transformers==4.51.3",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # When memorized facts fail to become usable knowledge

    **Question.** If a language model can recite two newly learned facts, can it also compose them into a two-hop answer?

    [Dai et al. (2026)](https://arxiv.org/abs/2607.08393) call the failure to do so the **knowing–using gap**. Their mechanistic proposal is that a fact may be stored in the residual stream without reaching the layers where reasoning uses it. They test this causally with **self-patching**: copy an entity's activation from one layer and insert it at another layer, without adding any new fact.

    This notebook separates three things carefully:

    1. the paper's evidence;
    2. our smaller, frozen reproduction evidence; and
    3. an optional synthetic teaching lab that is **not** reproduction evidence.
    """)
    return


@app.cell
def _(mo, pd):
    paper_result = pd.DataFrame(
        [
            {"Model": "Qwen-2.5-1.5B", "Facts": 1000, "Direct recall": 0.998, "Chaining": 0.078, "Oracle self-patch": 0.440},
            {"Model": "Qwen-2.5-3B", "Facts": 1000, "Direct recall": 0.997, "Chaining": 0.114, "Oracle self-patch": 0.542},
            {"Model": "Qwen-2.5-7B", "Facts": 1000, "Direct recall": 0.996, "Chaining": 0.124, "Oracle self-patch": 0.504},
        ]
    )
    mo.vstack([
        mo.md("""
        ## 1. Paper result

        On STaRK-Prime, direct recall is nearly perfect, while exact-match chaining is low. An oracle scan over source/target layer pairs substantially raises chaining accuracy. The headline 1.5B result is **7.8% → 44.0%** (a 5.6× ratio) over 1,000 injected facts. Across architectures and domains, the paper reports 1.5–6× gains.
        """),
        mo.ui.table(paper_result, selection=None, pagination=False),
        mo.callout("Paper Table 5 also reports a mean patching score of 0.6408 at the entity position versus 0.1359 at a random position. That ablation is supporting context, not a like-for-like denominator for Table 4.", kind="info"),
    ])
    return (paper_result,)


@app.cell
def _(mo, pd):
    reproduction = pd.DataFrame([
        {"Stage": "Pretrained control", "Correct": "0 / 48", "Accuracy": 0.000, "Meaning": "new direct facts were absent"},
        {"Stage": "LoRA direct recall", "Correct": "48 / 48", "Accuracy": 1.000, "Meaning": "all injected facts were memorized"},
        {"Stage": "Natural two-hop generation", "Correct": "0 / 24", "Accuracy": 0.000, "Meaning": "memorized facts were not composed"},
        {"Stage": "Unpatched scan subset", "Correct": "0 / 12", "Accuracy": 0.000, "Meaning": "teacher-forced exact-token baseline"},
        {"Stage": "Entity oracle self-patch", "Correct": "3 / 12", "Accuracy": 0.250, "Meaning": "activation relocation recovered answers"},
        {"Stage": "Random-position oracle", "Correct": "1 / 12", "Accuracy": 0.083, "Meaning": "multiple-comparison negative control"},
    ])
    mo.vstack([
        mo.md("""
        ## 2. Reproduction result — partially reproduced

        We replaced the paper's 1.5B model and biomedical graph with **Qwen-2.5-0.5B** and 24 fixed synthetic chains. LoRA saw the two constituent facts but never the composed question. After 30 epochs it recalled every fact, yet answered no chain correctly.

        On 12 fixed failures, an all-pairs residual-post scan moved the head-entity state between each of 24×24 layer pairs. Entity self-patching recovered **3/12 (25.0%; 95% Wilson interval 8.9–53.2%)**, while the same oracle search at a random token recovered **1/12 (8.3%; 1.5–35.4%)**. Because the unpatched score is zero, a multiplicative ratio is undefined; the honest effect is **+25 percentage points**. The wide, overlapping intervals make this illustrative rather than decisive.
        """),
        mo.ui.table(reproduction, selection=None, pagination=False),
        mo.callout("Verdict: the knowing–using gap and causal recoverability are reproduced at small scale. The paper's middle-layer localization is not: our first successful pairs targeted layers 0, 2, or 3.", kind="warn"),
    ])
    return (reproduction,)


@app.cell
def _(np, plt, reproduction):
    labels = ["Unpatched", "Random-position\noracle", "Entity\noracle"]
    values = [0.0, 1 / 12, 3 / 12]
    colors = ["#94a3b8", "#f59e0b", "#2563eb"]
    fig_controls, _ax_controls = plt.subplots(figsize=(7.4, 3.8))
    _bars = _ax_controls.bar(labels, values, color=colors, width=0.62)
    _ax_controls.set_ylim(0, 0.32)
    _ax_controls.set_ylabel("Exact answer accuracy (n = 12)")
    _ax_controls.set_title("Relocating the entity state recovers otherwise failed chains")
    _ax_controls.grid(axis="y", alpha=0.2)
    for _bar, _value in zip(_bars, values):
        _ax_controls.text(_bar.get_x() + _bar.get_width()/2, _value + 0.012, f"{_value:.1%}", ha="center", fontweight="bold")
    fig_controls.tight_layout()
    fig_controls
    return (fig_controls,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. What the intervention means

    The source and target are the **same prompt and same tuned model**. At the token naming the first entity, we cache the residual state after source layer $l_s$, replace the state after target layer $l_t$, and rescore the known answer. Causal recovery says the parameters contain answer-relevant information that the normal forward path did not use successfully.

    The orange control matters because an oracle scans 576 pairs per item: by chance, some generic perturbation can help. Entity patches recovered three items and the random token recovered one. With only 12 items this is illustrative, not a precise effect estimate.
    """)
    return


@app.cell
def _(mo, np, plt):
    successful_pairs = np.array([[0, 2], [0, 3], [3, 0]])
    fig_layers, _ax_layers = plt.subplots(figsize=(7.2, 4.3))
    _ax_layers.scatter(successful_pairs[:, 0], successful_pairs[:, 1], s=130, c=["#2563eb", "#2563eb", "#7c3aed"], edgecolor="white", linewidth=1.5)
    _ax_layers.plot([0, 23], [0, 23], linestyle="--", color="#94a3b8", label="no relocation ($l_s=l_t$)")
    _ax_layers.axhspan(9, 14, color="#10b981", alpha=0.10, label="approx. middle targets")
    _ax_layers.set(xlim=(-1, 24), ylim=(-1, 24), xlabel="Source layer", ylabel="Target layer", title="First successful pair for each recovered item")
    _ax_layers.set_xticks([0, 4, 9, 14, 19, 23]); _ax_layers.set_yticks([0, 4, 9, 14, 19, 23])
    _ax_layers.legend(loc="upper left")
    _ax_layers.grid(alpha=0.18)
    fig_layers.tight_layout()
    mo.vstack([fig_layers, mo.callout("All three first-successful targets are early. The green band shows where a mid-layer result would have appeared. This falsifies the layer-localization part of the claim for this particular downscaled setup.", kind="warn")])
    return (fig_layers, successful_pairs)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Robustness, leakage checks, and falsifiers

    - **Novelty control:** the pretrained model scored 0/48 on the fixed direct facts and 0/24 on chains.
    - **Held-out composition:** training included only the two one-hop prompts for each chain, never the two-hop prompt.
    - **Position control:** a random token received the same per-instance all-layer oracle search.
    - **Determinism:** names, mappings, training order, and toy lab use committed seeds.
    - **No answer-token patch:** only prompt positions containing the head entity were replaced; causal masking prevents those states from reading later answer tokens.

    **What would falsify the routing interpretation?** If shuffled or random-position states matched entity patches across adequately powered samples; if gains disappeared under generation-based exact match; or if direct recall itself were poor. Our small sample does not rule these out decisively. In particular, teacher-forced exact-token scoring is easier than the paper's full generated-answer criterion.
    """)
    return


@app.cell
def _(mo, pd):
    provenance = pd.DataFrame([
        {"Experiment": "Zero-shot novelty control", "Code": "[Frozen branch](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2)", "Verdict": "0/48 direct; novel", "Compute": "1× RTX PRO 6000, 7.66 s"},
        {"Experiment": "Confirmatory self-patching", "Code": "[Runner + config + manifest](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching)", "Verdict": "Partially reproduced", "Compute": "1× RTX PRO 6000, 273.38 s"},
        {"Experiment": "Real-model GPU lab", "Code": "[Qwen heatmap branch](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/irrelevant-source-activation-heatmap)", "Verdict": "Teaching-only validated", "Compute": "1× RTX PRO 6000, 6.88 s total"},
    ])
    mo.vstack([
        mo.md("""
        ## 5. Limitations and provenance

        - [Zero-shot control code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2) contains the runner, fixed baseline configuration, Blackwell-compatible manifest, and novelty evaluation.
        - [Confirmatory reproduction code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching) contains LoRA training, the fixed synthetic mappings, full 24×24 scan, random-position control, and exact-token scorer.
        - [Real-model GPU lab code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/irrelevant-source-activation-heatmap) contains the Qwen2.5-0.5B residual-patching sweep, fixed prompts, aligned irrelevant-source control, and Blackwell manifest used below.

        The reproduction used 0.0759 measured GPU-hours. This was a user-owned Kubernetes cluster, so no provider price or monetary charge was exposed; dollar cost is therefore **not available**, rather than assumed to be zero.

        Major substitutions are Qwen-2.5-0.5B for 1.5B, 24 synthetic chains for 1,000 STaRK-Prime facts, 12 patch scans for 1,000 evaluations, and teacher-forced exact-token scoring for reported generated exact match. The released anonymous archive omitted its datasets, configuration files, and trained checkpoints, preventing an exact archival rerun.
        """),
        mo.ui.table(provenance, selection=None, pagination=False),
    ])
    return (provenance,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Optional GPU lab — real Qwen activation surgery

    This teaching lab uses the attached GPU for genuine transformer intervention work. It loads **Qwen2.5-0.5B-Instruct**, caches every residual-post state for a clean two-hop prompt, and patches each layer × token location into a corrupted prompt. The score is the change in the next-token logit margin for **France over Germany**.

    The negative control repeats the same intervention with aligned states from an unrelated **Tokyo → Japan** chain. This is still a teaching experiment—not reproduction evidence—but it uses a real model, real hidden states, and a causal 24-layer sweep. The frozen heatmap below came from the validated RTX PRO 6000 run, so the result is visible before you download anything.
    """)
    return


@app.cell
def _(mo, np, plt):
    frozen_tokens = ["A", " is", " in", " Berlin", ".", " Berlin", " is", " in", " Germany", ".", " Therefore", ",", " A", " is", " in"]
    frozen_heatmap = np.array([
        [0, .06, .12, 6.19, 0, 7.56, .06, .12, 16.12, .06, .06, .12, .12, .12, .12],
        [.12, 0, .06, 6.31, 0, 7.31, .06, 0, 16, .12, .12, .06, .12, .12, .19],
        [0, .06, .12, 6.31, -.06, 7.19, 0, 0, 16.12, .06, .25, 0, .19, .12, .06],
        [0, .06, .06, 4.94, -.12, 4.06, .19, .06, 16, .06, .12, .19, .06, .12, .12],
        [.06, .06, .06, 5.19, -.38, 4, .06, -.06, 16.25, .31, .38, .19, .12, .12, .19],
        [.12, .12, .06, 5.31, -.44, 3.81, .06, .12, 16.25, .12, .38, .12, .19, .12, .12],
        [.12, .12, .12, 5.56, -.44, 3.19, .19, .25, 16.25, -.06, .38, .19, .12, .12, .06],
        [.12, .12, .06, 5.69, -.38, 2.88, .06, .19, 16.62, .12, .44, .19, .12, .19, .06],
        [.12, .06, .12, 5.56, .12, 2.75, .19, 0, 16.38, .12, .44, .19, .12, .12, .06],
        [.06, .19, .06, 5.81, -.19, 1.12, .19, 0, 16.25, .12, .5, .12, .12, .12, .06],
        [.12, .12, .12, 5.69, -.19, 1.25, .19, -.12, 16.5, .12, .38, .19, .12, .12, .25],
        [.12, .12, .06, 5.69, -.19, 1.44, .06, -.12, 16.5, .19, .38, .31, .25, .25, .31],
        [.06, .06, .12, 5.81, .06, 1.5, 0, -.12, 16.75, .06, .12, .19, .19, .12, .12],
        [.12, .12, .12, 5.81, 0, .88, .12, 0, 17, .19, .19, .12, 0, .19, .25],
        [.06, .12, .12, 5.94, .06, 1, .12, 0, 16.88, .06, .19, .12, 0, .12, 0],
        [.12, .12, .12, 5.81, .12, .94, .12, 0, 16.5, .12, .12, .12, .19, .12, -.19],
        [.19, .12, .12, 5.94, .19, 1.06, .12, 0, 14.94, .12, .25, .12, .12, .12, 2.25],
        [.12, .12, .12, 5.81, .12, 1, .12, .12, 14.81, .19, .12, .12, .12, .19, 2.56],
        [.12, .19, .19, 5.69, .12, .94, .19, .06, 14.69, .06, .12, .12, .12, 0, 2.62],
        [.12, .12, .19, 5.81, .19, 1, .12, .12, 15.06, .06, .19, .12, .19, .12, 2.44],
        [.19, .12, .12, 5.31, .12, .88, .19, .12, 6.94, .12, .12, .19, .19, .12, 10.06],
        [.19, .12, .12, .88, .19, .19, .19, .06, 1.25, .12, .19, .19, .12, .19, 19.62],
        [.19, .12, .12, .44, .19, .19, .19, .19, -1.38, .12, .12, .19, .12, .12, 22.75],
        [.12, .12, .12, .12, .12, .12, .12, .12, .12, .12, .12, .12, .12, .12, 22],
    ])
    _frozen_limit = float(np.abs(frozen_heatmap).max())
    _frozen_fig, _frozen_ax = plt.subplots(figsize=(10.8, 5.4))
    _frozen_image = _frozen_ax.imshow(frozen_heatmap, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-_frozen_limit, vmax=_frozen_limit)
    _frozen_ax.set(title="Validated aligned-source patch: where does France information change the answer?", xlabel="Token position in the corrupted Germany prompt", ylabel="Patched residual layer")
    _frozen_ax.set_xticks(range(len(frozen_tokens)), [f"{i}:{token.replace(' ', '·')}" for i, token in enumerate(frozen_tokens)], rotation=55, ha="right")
    _frozen_ax.set_yticks([0, 4, 8, 12, 16, 20, 23])
    _frozen_fig.colorbar(_frozen_image, ax=_frozen_ax, label="Δ logit margin: France − Germany")
    _frozen_fig.tight_layout()
    mo.vstack([
        _frozen_fig,
        mo.callout("The strongest aligned patch is layer 22 at the final prompt token: +22.75 logit-margin units. The aligned irrelevant-source control is +12.50 there. Vertical bands reveal where Paris/France states influence the downstream answer.", kind="info"),
    ])
    return frozen_heatmap, frozen_tokens


@app.cell
def _(mo):
    patch_strength = mo.ui.slider(0.25, 1.5, step=0.25, value=1.0, label="Patch strength")
    run_lab = mo.ui.run_button(label="Run Qwen layer × token sweep")
    mo.hstack([patch_strength, run_lab], justify="start", gap=2)
    return patch_strength, run_lab


@app.cell
def _():
    import functools
    import time

    @functools.cache
    def _load_qwen(_device_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        _device = torch.device(_device_name)
        _tokenizer = AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype=torch.bfloat16 if _device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(_device).eval()
        return _model, _tokenizer, _device

    def run_qwen_heatmap(strength):
        import torch

        _started = time.perf_counter()
        _device_name = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _tokenizer, _device = _load_qwen(_device_name)
        _clean_prompt = "A is in Paris. Paris is in France. Therefore, A is in"
        _corrupt_prompt = "A is in Berlin. Berlin is in Germany. Therefore, A is in"
        _control_prompt = "A is in Tokyo. Tokyo is in Japan. Therefore, A is in"
        _clean = _tokenizer(_clean_prompt, return_tensors="pt").to(_device)
        _corrupt = _tokenizer(_corrupt_prompt, return_tensors="pt").to(_device)
        _control = _tokenizer(_control_prompt, return_tensors="pt").to(_device)
        if not (_clean["input_ids"].shape == _corrupt["input_ids"].shape == _control["input_ids"].shape):
            raise ValueError("Controlled prompts no longer have aligned token lengths")
        _clean_id = _tokenizer(" France", add_special_tokens=False)["input_ids"][0]
        _corrupt_id = _tokenizer(" Germany", add_special_tokens=False)["input_ids"][0]
        _layers = _model.model.layers
        _layer_indices = list(range(len(_layers))) if _device.type == "cuda" else list(range(0, len(_layers), 4))

        @torch.inference_mode()
        def _forward_with_cache(_inputs):
            _cache, _handles = {}, []
            for _layer_idx in _layer_indices:
                def _save(_module, _module_inputs, _output, _layer_idx=_layer_idx):
                    _hidden = _output[0] if isinstance(_output, tuple) else _output
                    _cache[_layer_idx] = _hidden.detach().clone()
                _handles.append(_layers[_layer_idx].register_forward_hook(_save))
            try:
                _logits = _model(**_inputs, use_cache=False).logits[:, -1, :]
            finally:
                for _handle in _handles:
                    _handle.remove()
            return _logits, _cache

        with torch.inference_mode():
            _clean_logits, _clean_cache = _forward_with_cache(_clean)
            _, _control_cache = _forward_with_cache(_control)
            _corrupt_logits = _model(**_corrupt, use_cache=False).logits[:, -1, :]
            _clean_margin = (_clean_logits[0, _clean_id] - _clean_logits[0, _corrupt_id]).item()
            _corrupt_margin = (_corrupt_logits[0, _clean_id] - _corrupt_logits[0, _corrupt_id]).item()
            _seq_len = _corrupt["input_ids"].shape[1]
            _positions = torch.arange(_seq_len, device=_device)
            _relevant = torch.empty((len(_layer_indices), _seq_len), dtype=torch.float32)
            _irrelevant = torch.empty_like(_relevant)
            for _row, _layer_idx in enumerate(_layer_indices):
                _batch_ids = _corrupt["input_ids"].repeat(2 * _seq_len, 1)
                _batch_mask = _corrupt["attention_mask"].repeat(2 * _seq_len, 1)

                def _patch(_module, _module_inputs, _output, _layer_idx=_layer_idx):
                    _hidden = _output[0] if isinstance(_output, tuple) else _output
                    _patched = _hidden.clone()
                    _rows = torch.arange(_seq_len, device=_device)
                    _clean_state = _clean_cache[_layer_idx][0].to(_patched.dtype)
                    _control_state = _control_cache[_layer_idx][0].to(_patched.dtype)
                    _patched[_rows, _positions] = torch.lerp(_patched[_rows, _positions], _clean_state[_positions], float(strength))
                    _control_rows = _rows + _seq_len
                    _patched[_control_rows, _positions] = torch.lerp(_patched[_control_rows, _positions], _control_state[_positions], float(strength))
                    return (_patched, *_output[1:]) if isinstance(_output, tuple) else _patched

                _handle = _layers[_layer_idx].register_forward_hook(_patch)
                try:
                    _logits = _model(input_ids=_batch_ids, attention_mask=_batch_mask, use_cache=False).logits[:, -1, :]
                finally:
                    _handle.remove()
                _margins = _logits[:, _clean_id] - _logits[:, _corrupt_id] - _corrupt_margin
                _relevant[_row] = _margins[:_seq_len].float().cpu()
                _irrelevant[_row] = _margins[_seq_len:].float().cpu()

        return {
            "tokens": [_tokenizer.decode([_token]) for _token in _corrupt["input_ids"][0].tolist()],
            "layer_indices": _layer_indices,
            "relevant": _relevant.tolist(),
            "irrelevant": _irrelevant.tolist(),
            "clean_margin": _clean_margin,
            "corrupt_margin": _corrupt_margin,
            "runtime": time.perf_counter() - _started,
            "device": torch.cuda.get_device_name(0) if _device.type == "cuda" else "CPU fallback (every fourth layer)",
        }

    return (run_qwen_heatmap,)


@app.cell
def _(mo, np, patch_strength, plt, run_lab, run_qwen_heatmap):
    if not run_lab.value:
        lab_output = mo.callout("The live lab is idle. Press the button to download/cache Qwen2.5-0.5B and run the intervention sweep. The first model download is about 1 GB; no expensive work starts automatically.", kind="neutral")
    else:
        try:
            _live = run_qwen_heatmap(float(patch_strength.value))
            _relevant = np.asarray(_live["relevant"])
            _irrelevant = np.asarray(_live["irrelevant"])
            _limit = max(float(np.abs(_relevant).max()), float(np.abs(_irrelevant).max()), 1.0)
            _live_fig, _live_axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True)
            for _axis, _matrix, _title in zip(_live_axes, [_relevant, _irrelevant], ["Aligned France source", "Aligned irrelevant Japan source"]):
                _image = _axis.imshow(_matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-_limit, vmax=_limit)
                _axis.set_title(_title)
                _axis.set_xlabel("Corrupted-prompt token")
                _axis.set_xticks(range(len(_live["tokens"])), [f"{i}:{token.replace(' ', '·')}" for i, token in enumerate(_live["tokens"])], rotation=55, ha="right")
                _axis.set_yticks(range(len(_live["layer_indices"])), _live["layer_indices"])
            _live_axes[0].set_ylabel("Patched residual layer")
            _live_fig.colorbar(_image, ax=_live_axes, label="Δ logit margin: France − Germany", shrink=.86)
            _live_fig.subplots_adjust(bottom=.27, wspace=.08, right=.88)
            _best_row, _best_token = np.unravel_index(int(np.argmax(_relevant)), _relevant.shape)
            _best_layer = _live["layer_indices"][_best_row]
            lab_output = mo.vstack([
                mo.callout(f"Device: {_live['device']}. Runtime: {_live['runtime']:.2f} s. Clean margin: {_live['clean_margin']:+.2f}; corrupted margin: {_live['corrupt_margin']:+.2f}. Best aligned patch: layer {_best_layer}, token {_best_token}, Δ={_relevant[_best_row, _best_token]:+.2f}; control there: {_irrelevant[_best_row, _best_token]:+.2f}.", kind="success"),
                _live_fig,
                mo.md("Red cells move the model toward **France**; blue cells move it toward **Germany**. Compare panels on the shared scale: a useful causal location should respond more to the relevant France-chain state than to an equally shaped but irrelevant Japan-chain state."),
            ])
        except Exception as _error:
            lab_output = mo.callout(f"The live sweep could not run: {_error}. The frozen RTX-validated heatmap and formal reproduction evidence remain available.", kind="warn")
    lab_output
    return (lab_output,)


@app.cell
def _(mo):
    mo.md("""
    ## Takeaway

    A model can fit every injected fact yet fail to compose them. In this small experiment, relocating the entity representation recovers some answers more often than a random-position oracle, which supports—but does not prove—the paper's routing account. The missing middle-layer pattern and small teacher-forced scan are important negative evidence. A full reproduction still needs the authors' exact STaRK split/checkpoints (or a faithful regeneration), Qwen-2.5-1.5B+, 1,000 patch evaluations, generated exact-match scoring, confidence intervals, and the full irrelevant-fact and token-position controls.
    """)
    return


if __name__ == "__main__":
    app.run()
