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
        {"Experiment": "Real-model GPU lab validation", "Code": "[Answer-matched heatmap branch](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/answer-matched-activation-heatmap)", "Verdict": "Teaching-only validated", "Compute": "1× RTX PRO 6000, 6.09 s including model load"},
    ])
    mo.vstack([
        mo.md("""
        ## 5. Limitations and provenance

        - [Zero-shot control code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2) contains the runner, fixed baseline configuration, Blackwell-compatible manifest, and novelty evaluation.
        - [Confirmatory reproduction code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching) contains LoRA training, the fixed synthetic mappings, full 24×24 scan, random-position control, and exact-token scorer.
        - [Real-model GPU lab validation code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/answer-matched-activation-heatmap) contains the bounded Qwen layer-by-token intervention and answer-matched control used below.

        The reproduction used 0.0759 measured GPU-hours. This was a user-owned Kubernetes cluster, so no provider price or monetary charge was exposed; dollar cost is therefore **not available**, rather than assumed to be zero.

        Major substitutions are Qwen-2.5-0.5B for 1.5B, 24 synthetic chains for 1,000 STaRK-Prime facts, 12 patch scans for 1,000 evaluations, and teacher-forced exact-token scoring for reported generated exact match. The released anonymous archive omitted its datasets, configuration files, and trained checkpoints, preventing an exact archival rerun.
        """),
        mo.ui.table(provenance, selection=None, pagination=False),
    ])
    return (provenance,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Optional GPU lab — real Qwen residual-stream surgery

    This is a genuine transformer intervention on **Qwen2.5-0.5B-Instruct**, not a synthetic tensor demo. The corrupted prompt says `Berlin → Germany`; the clean source says `Paris → France`. At every one of 24 residual layers and every prompt-token position, the lab copies one clean activation into the corrupted forward pass and measures the change in the next-token **France-vs-Germany logit margin**.

    The negative control follows a different route, `Munich → Germany`, while preserving the corrupted answer. Subtracting that answer-matched control isolates France-specific information from generic disruption. This remains a teaching experiment—not reproduction evidence—because it uses in-context facts rather than the LoRA-tuned model above.
    """)
    return


@app.cell
def _(mo, np, plt):
    _validated_tokens = ["A", "is", "in", "Berlin", ".", "Berlin", "is", "in", "Germany", ".", "Therefore", ",", "A", "is", "in"]
    _validated_delta = np.array([
        [0, 0, 0, 5.25, -0.125, 5.875, -0.125, 0.0625, 16.125, 0, -0.0625, -0.0625, 0, 0, -0.0625],
        [0, 0, 0, 5.125, 0.125, 5.6875, 0, -0.0625, 15.875, 0, -0.0625, -0.0625, 0.0625, 0.0625, 0.125],
        [0, 0, 0, 5.0625, -0.0625, 5.5, -0.125, -0.0625, 16, -0.0625, 0.0625, -0.1875, 0.0625, 0, -0.0625],
        [0, 0, 0, 4.6875, -0.125, 3.6875, 0.125, -0.0625, 15.9375, 0, -0.125, 0, 0, 0.0625, -0.0625],
        [0, 0, 0, 5.125, -0.625, 3.5625, 0, -0.125, 15.875, 0.1875, 0.0625, 0.125, -0.0625, 0.0625, 0],
        [0, 0, 0, 5.25, -0.75, 3.4375, 0, -0.1875, 15.9375, -0.0625, 0.125, 0, 0.125, 0.0625, -0.125],
        [0, 0, 0, 5.3125, -0.5625, 2.6875, 0.0625, 0, 16.1875, -0.1875, 0.1875, 0.1875, 0.125, -0.0625, -0.1875],
        [0, 0, 0, 5.625, -0.6875, 2.5, 0, 0.0625, 16.4375, 0.1875, 0.25, 0.1875, 0, 0, -0.0625],
        [0, 0, 0, 5.8125, -0.4375, 2.25, 0.0625, -0.0625, 16.1875, 0.1875, 0.125, 0.125, 0.0625, 0.0625, -0.0625],
        [0, 0, 0, 5.8125, -0.4375, 1, 0.0625, 0, 15.875, 0.1875, 0.1875, 0.0625, 0, 0.0625, 0],
        [0, 0, 0, 5.6875, -0.3125, 1.125, -0.0625, -0.0625, 16.1875, 0.25, 0.3125, 0, -0.125, -0.0625, 0.25],
        [0, 0, 0, 5.75, -0.5625, 1.25, -0.0625, -0.125, 16.1875, 0.1875, 0.1875, 0.125, 0.0625, 0.0625, 0.3125],
        [0, 0, 0, 5.8125, -0.125, 1.375, -0.125, -0.1875, 16.3125, 0, 0.0625, 0.0625, -0.0625, 0.125, 0.25],
        [0, 0, 0, 5.875, -0.1875, 0.6875, 0.0625, -0.1875, 16.6875, 0.0625, 0.0625, 0.0625, -0.1875, 0.125, 0.5625],
        [0, 0, 0, 5.875, -0.125, 0.8125, 0, -0.1875, 16.4375, -0.0625, 0.125, 0.125, -0.1875, 0.125, 0.125],
        [0, 0, 0, 5.875, 0, 0.75, 0, -0.125, 16.1875, 0, 0, 0, 0.0625, 0.0625, 0],
        [0, 0, 0, 5.9375, 0, 0.875, 0, -0.0625, 14.6875, -0.0625, 0.125, -0.125, 0, 0.0625, 2.3125],
        [0, 0, 0, 5.75, 0, 0.8125, 0, 0, 14.5, 0, 0, 0, 0, 0.0625, 2.75],
        [0, 0, 0, 5.6875, 0, 0.75, 0, -0.0625, 14.4375, -0.0625, 0, 0, 0, -0.0625, 2.75],
        [0, 0, 0, 5.75, 0.0625, 0.75, 0, 0, 14.8125, -0.0625, 0.0625, -0.0625, 0, 0, 2.5625],
        [0, 0, 0, 5.25, 0, 0.625, 0.0625, 0, 6.75, 0, 0, 0, 0.0625, 0, 10.3125],
        [0, 0, 0, 0.6875, 0.0625, 0, 0.0625, -0.125, 1.25, -0.0625, 0.0625, 0.0625, 0, 0, 19.9375],
        [0, 0, 0, 0.125, 0, 0, 0.0625, 0, -1.3125, -0.0625, 0, 0.0625, 0, -0.0625, 23],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22.0625],
    ])
    _frozen_fig, _frozen_ax = plt.subplots(figsize=(10.5, 5.8))
    _frozen_image = _frozen_ax.imshow(_validated_delta, aspect="auto", cmap="coolwarm", vmin=-23, vmax=23, origin="lower")
    _frozen_ax.scatter([14], [22], marker="s", facecolors="none", edgecolors="black", linewidths=1.8, s=120)
    _frozen_ax.set(
        xlabel="Corrupted-prompt token position",
        ylabel="Residual-post layer",
        title="Where the France-specific state changes Qwen's answer margin",
    )
    _frozen_ax.set_xticks(range(len(_validated_tokens)), [f"{i} · {token}" for i, token in enumerate(_validated_tokens)], rotation=55, ha="right")
    _frozen_ax.set_yticks([0, 4, 8, 12, 16, 20, 23])
    _frozen_colorbar = _frozen_fig.colorbar(_frozen_image, ax=_frozen_ax, pad=0.02)
    _frozen_colorbar.set_label("Relevant patch − answer-matched control (logit Δ)")
    _frozen_fig.tight_layout()
    mo.vstack([
        mo.callout("Frozen RTX PRO 6000 validation: the strongest clean-specific effect is +23.0 logits at layer 22 and the final `in` token. A second ridge at the `Germany` token shows the answer identity propagating through many layers.", kind="info"),
        _frozen_fig,
        mo.md("**How to read it.** Rows are intervention layers; columns are token positions in the corrupted prompt. Warm cells mean the Paris→France state helps more than the answer-matched Munich→Germany control. The black square marks the maximum. Values are embedded from the validated external run, so the result is visible before downloading a model."),
    ])
    return


@app.cell
def _(mo):
    patch_strength = mo.ui.slider(0.25, 1.5, step=0.25, value=1.0, label="Patch strength α")
    run_lab = mo.ui.run_button(label="Run real-Qwen heatmap")
    mo.vstack([
        mo.md("Choose how strongly to mix the source activation into the corrupted stream, then run the full sweep. The first run downloads and caches Qwen2.5-0.5B; the validated RTX path took 6.1 seconds including model load and 0.68 seconds for the paired scan."),
        mo.hstack([patch_strength, run_lab], justify="start", gap=2),
    ])
    return patch_strength, run_lab


@app.cell
def _(mo, np, plt, run_lab, patch_strength):
    if not run_lab.value:
        lab_output = mo.callout("The live model is idle. No model download or GPU work starts automatically.", kind="neutral")
    else:
        try:
            import time
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _lab_started = time.time()
            _lab_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _lab_dtype = torch.bfloat16 if _lab_device.type == "cuda" else torch.float32
            _lab_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            _lab_tokenizer = AutoTokenizer.from_pretrained(_lab_model_id, trust_remote_code=True)
            _lab_model = AutoModelForCausalLM.from_pretrained(
                _lab_model_id, torch_dtype=_lab_dtype, trust_remote_code=True
            ).to(_lab_device).eval()
            _lab_clean_prompt = "A is in Paris. Paris is in France. Therefore, A is in"
            _lab_corrupt_prompt = "A is in Berlin. Berlin is in Germany. Therefore, A is in"
            _lab_control_prompt = "A is in Munich. Munich is in Germany. Therefore, A is in"
            _lab_clean = _lab_tokenizer(_lab_clean_prompt, return_tensors="pt").to(_lab_device)
            _lab_corrupt = _lab_tokenizer(_lab_corrupt_prompt, return_tensors="pt").to(_lab_device)
            _lab_control = _lab_tokenizer(_lab_control_prompt, return_tensors="pt").to(_lab_device)
            if not (_lab_clean["input_ids"].shape == _lab_corrupt["input_ids"].shape == _lab_control["input_ids"].shape):
                raise ValueError("Controlled prompts tokenized to different lengths")
            _lab_clean_id = _lab_tokenizer(" France", add_special_tokens=False)["input_ids"][0]
            _lab_corrupt_id = _lab_tokenizer(" Germany", add_special_tokens=False)["input_ids"][0]
            _lab_layers = _lab_model.model.layers

            def _lab_forward_with_cache(_lab_inputs):
                _lab_cache = {}
                _lab_handles = []
                for _lab_idx, _lab_layer in enumerate(_lab_layers):
                    def _lab_save(_module, _inputs, _output, _lab_idx=_lab_idx):
                        _lab_hidden = _output[0] if isinstance(_output, tuple) else _output
                        _lab_cache[_lab_idx] = _lab_hidden.detach().clone()
                    _lab_handles.append(_lab_layer.register_forward_hook(_lab_save))
                try:
                    _lab_logits = _lab_model(**_lab_inputs, use_cache=False).logits[:, -1, :]
                finally:
                    for _lab_handle in _lab_handles:
                        _lab_handle.remove()
                return _lab_logits, _lab_cache

            if _lab_device.type == "cuda":
                torch.cuda.synchronize()
            _lab_scan_started = time.time()
            with torch.inference_mode():
                _lab_clean_logits, _lab_clean_cache = _lab_forward_with_cache(_lab_clean)
                _lab_control_logits, _lab_control_cache = _lab_forward_with_cache(_lab_control)
                _lab_corrupt_logits = _lab_model(**_lab_corrupt, use_cache=False).logits[:, -1, :]
                _lab_clean_margin = (_lab_clean_logits[0, _lab_clean_id] - _lab_clean_logits[0, _lab_corrupt_id]).item()
                _lab_corrupt_margin = (_lab_corrupt_logits[0, _lab_clean_id] - _lab_corrupt_logits[0, _lab_corrupt_id]).item()
                _lab_seq_len = _lab_corrupt["input_ids"].shape[1]
                _lab_layer_indices = list(range(len(_lab_layers))) if _lab_device.type == "cuda" else list(range(0, len(_lab_layers), 4))
                _lab_positions = torch.arange(_lab_seq_len, device=_lab_device)
                _lab_relevant_rows, _lab_control_rows = [], []
                for _lab_layer_idx in _lab_layer_indices:
                    _lab_batch_ids = _lab_corrupt["input_ids"].repeat(2 * _lab_seq_len, 1)
                    _lab_batch_mask = _lab_corrupt["attention_mask"].repeat(2 * _lab_seq_len, 1)

                    def _lab_patch(_module, _inputs, _output, _lab_layer_idx=_lab_layer_idx):
                        _lab_hidden = _output[0] if isinstance(_output, tuple) else _output
                        _lab_patched = _lab_hidden.clone()
                        _lab_rows = torch.arange(_lab_seq_len, device=_lab_device)
                        _lab_clean_state = _lab_clean_cache[_lab_layer_idx][0].to(_lab_patched.dtype)
                        _lab_control_state = _lab_control_cache[_lab_layer_idx][0].to(_lab_patched.dtype)
                        _lab_alpha = float(patch_strength.value)
                        _lab_patched[_lab_rows, _lab_positions] = torch.lerp(
                            _lab_patched[_lab_rows, _lab_positions], _lab_clean_state[_lab_positions], _lab_alpha
                        )
                        _lab_control_batch_rows = _lab_rows + _lab_seq_len
                        _lab_patched[_lab_control_batch_rows, _lab_positions] = torch.lerp(
                            _lab_patched[_lab_control_batch_rows, _lab_positions], _lab_control_state[_lab_positions], _lab_alpha
                        )
                        return (_lab_patched, *_output[1:]) if isinstance(_output, tuple) else _lab_patched

                    _lab_handle = _lab_layers[_lab_layer_idx].register_forward_hook(_lab_patch)
                    try:
                        _lab_logits = _lab_model(input_ids=_lab_batch_ids, attention_mask=_lab_batch_mask, use_cache=False).logits[:, -1, :]
                    finally:
                        _lab_handle.remove()
                    _lab_margins = (_lab_logits[:, _lab_clean_id] - _lab_logits[:, _lab_corrupt_id] - _lab_corrupt_margin).float().cpu().numpy()
                    _lab_relevant_rows.append(_lab_margins[:_lab_seq_len])
                    _lab_control_rows.append(_lab_margins[_lab_seq_len:])
            if _lab_device.type == "cuda":
                torch.cuda.synchronize()
            _lab_scan_seconds = time.time() - _lab_scan_started
            _lab_difference = np.stack(_lab_relevant_rows) - np.stack(_lab_control_rows)
            _lab_best_row, _lab_best_token = np.unravel_index(np.argmax(_lab_difference), _lab_difference.shape)
            _lab_best_layer = _lab_layer_indices[_lab_best_row]
            _lab_tokens = [_lab_tokenizer.decode([_token]).strip() or "·" for _token in _lab_corrupt["input_ids"][0].tolist()]
            _lab_limit = max(1.0, float(np.abs(_lab_difference).max()))
            _lab_fig, _lab_ax = plt.subplots(figsize=(10.5, 5.8))
            _lab_image = _lab_ax.imshow(_lab_difference, aspect="auto", cmap="coolwarm", vmin=-_lab_limit, vmax=_lab_limit, origin="lower")
            _lab_ax.scatter([_lab_best_token], [_lab_best_row], marker="s", facecolors="none", edgecolors="black", linewidths=1.8, s=120)
            _lab_ax.set(xlabel="Corrupted-prompt token position", ylabel="Scanned residual-post layer", title=f"Live Qwen causal heatmap (α={float(patch_strength.value):.2f})")
            _lab_ax.set_xticks(range(len(_lab_tokens)), [f"{i} · {token}" for i, token in enumerate(_lab_tokens)], rotation=55, ha="right")
            _lab_ax.set_yticks(range(len(_lab_layer_indices)), _lab_layer_indices)
            _lab_colorbar = _lab_fig.colorbar(_lab_image, ax=_lab_ax, pad=0.02)
            _lab_colorbar.set_label("Relevant patch − answer-matched control (logit Δ)")
            _lab_fig.tight_layout()
            _lab_name = torch.cuda.get_device_name(0) if _lab_device.type == "cuda" else "CPU fallback (every fourth layer)"
            _lab_total_seconds = time.time() - _lab_started
            lab_output = mo.vstack([
                mo.callout(
                    f"Device: {_lab_name}. Clean margin: {_lab_clean_margin:+.2f}; corrupted margin: {_lab_corrupt_margin:+.2f}. Strongest specific effect: {_lab_difference[_lab_best_row, _lab_best_token]:+.2f} logits at layer {_lab_best_layer}, token {_lab_best_token} (`{_lab_tokens[_lab_best_token]}`). Scan: {_lab_scan_seconds:.2f} s; total: {_lab_total_seconds:.2f} s.",
                    kind="success",
                ),
                _lab_fig,
            ])
            del _lab_model
            if _lab_device.type == "cuda":
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError, ValueError, OSError) as _lab_error:
            lab_output = mo.callout(f"The live scan could not run: {_lab_error}. The validated frozen heatmap above remains available.", kind="warn")
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
