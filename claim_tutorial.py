# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
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
        {"Experiment": "Toy GPU lab validation", "Code": "[Bounded lab branch](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/molab-toy-gpu-lab-validation)", "Verdict": "Teaching-only validated", "Compute": "1× RTX PRO 6000, 0.32 s code runtime"},
    ])
    mo.vstack([
        mo.md("""
        ## 5. Limitations and provenance

        - [Zero-shot control code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2) contains the runner, fixed baseline configuration, Blackwell-compatible manifest, and novelty evaluation.
        - [Confirmatory reproduction code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching) contains LoRA training, the fixed synthetic mappings, full 24×24 scan, random-position control, and exact-token scorer.
        - [Toy GPU lab validation code](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/molab-toy-gpu-lab-validation) contains the bounded synthetic intervention and shuffled-source control used below.

        The reproduction used 0.0759 measured GPU-hours. This was a user-owned Kubernetes cluster, so no provider price or monetary charge was exposed; dollar cost is therefore **not available**, rather than assumed to be zero.

        Major substitutions are Qwen-2.5-0.5B for 1.5B, 24 synthetic chains for 1,000 STaRK-Prime facts, 12 patch scans for 1,000 evaluations, and teacher-forced exact-token scoring for reported generated exact match. The released anonymous archive omitted its datasets, configuration files, and trained checkpoints, preventing an exact archival rerun.
        """),
        mo.ui.table(provenance, selection=None, pagination=False),
    ])
    return (provenance,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Optional interactive GPU lab — synthetic teaching experiment

    This toy model makes the routing idea visible without downloading an LLM. A binary fact is strongly represented in a **storage state** but weakly represented at a **computation state**. The intervention adds the relevant storage state into computation. A shuffled-source control adds equally sized but irrelevant information.

    Change the stored signal, then press **Run toy intervention**. It uses CUDA when available (including molab's attached RTX PRO 6000), otherwise CPU. Work is capped at 4,096 scalar samples and 11 intervention strengths; the validated GPU path took 0.32 seconds including setup around the function.
    """)
    return


@app.cell
def _(mo):
    signal_strength = mo.ui.slider(0.25, 2.5, step=0.25, value=1.5, label="Stored signal strength")
    run_lab = mo.ui.run_button(label="Run toy intervention")
    mo.hstack([signal_strength, run_lab], justify="start", gap=2)
    return run_lab, signal_strength


@app.cell
def _(mo, np, plt, run_lab, signal_strength):
    if not run_lab.value:
        lab_output = mo.callout("The lab is idle. Expensive work never starts automatically; press the button when ready.", kind="neutral")
    else:
        try:
            import torch
            toy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = torch.Generator(device=toy_device).manual_seed(7)
            toy_n = 4096
            toy_labels = torch.randint(0, 2, (toy_n,), generator=generator, device=toy_device) * 2 - 1
            toy_source = torch.randn(toy_n, generator=generator, device=toy_device) + float(signal_strength.value) * toy_labels
            toy_stranded = torch.randn(toy_n, generator=generator, device=toy_device) + 0.10 * toy_labels
            toy_shuffled = toy_source[torch.randperm(toy_n, generator=generator, device=toy_device)]
            toy_strengths = torch.linspace(0, 1, 11, device=toy_device)
            toy_patch = [((toy_stranded + a * toy_source).sign() == toy_labels).float().mean().item() for a in toy_strengths]
            toy_control = [((toy_stranded + a * toy_shuffled).sign() == toy_labels).float().mean().item() for a in toy_strengths]
            toy_x = toy_strengths.cpu().numpy()
            toy_fig, toy_ax = plt.subplots(figsize=(7.4, 4.0))
            toy_ax.plot(toy_x, toy_patch, marker="o", label="Relevant storage patch", color="#2563eb")
            toy_ax.plot(toy_x, toy_control, marker="o", label="Shuffled-source control", color="#f59e0b")
            toy_ax.axhline(0.5, linestyle="--", color="#94a3b8")
            toy_ax.set(xlabel="Intervention strength", ylabel="Toy answer accuracy", ylim=(0.4, 1.02), title="Only relevant routed information improves use")
            toy_ax.legend(); toy_ax.grid(alpha=0.18); toy_fig.tight_layout()
            toy_name = torch.cuda.get_device_name(0) if toy_device.type == "cuda" else "CPU fallback"
            lab_output = mo.vstack([mo.callout(f"Device: {toy_name}. Synthetic samples: {toy_n:,}. This plot is teaching-only.", kind="success"), toy_fig])
        except ImportError:
            lab_output = mo.callout("PyTorch is unavailable in this environment. Attach a molab GPU runtime or install a standard PyTorch build; frozen reproduction evidence above remains available.", kind="warn")
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
