# When memorized facts fail to become usable knowledge

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rehaanahmad2013/towards-mechanistically-understanding/blob/main/claim_tutorial.py)

This repository contains a tutorial-style, independently runnable marimo notebook for a downscaled reproduction of the main illustrative claim in [*Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Fine-tuning*](https://arxiv.org/abs/2607.08393).

The result is **partially reproduced**: a LoRA-tuned Qwen-2.5-0.5B model recalled all 48 injected facts but solved none of 24 held-out two-hop questions. Oracle entity self-patching recovered 3 of 12 scanned failures (25.0%), compared with 1 of 12 (8.3%) for a random-position oracle. The paper's middle-layer localization did not appear in this small synthetic setting.

Open `claim_tutorial.py` in molab to see the frozen evidence, provenance, controls, limitations, and an optional real-Qwen activation-patching heatmap on the attached GPU. Formal reproduction code is preserved on immutable `orx/*` experiment branches linked inside the notebook.

## Experiment log

Each row links to an immutable experiment branch containing the runner, fixed configuration, Kubernetes manifest, and evaluation method used at that stage.

| Branch | What it tried | Outcome and compute |
|---|---|---|
| [Baseline](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/baseline) | Established the original runner and Kubernetes job contract. | Failed before cloning because the injected shell script was expanded incorrectly; no research result was produced. |
| [Compatible Kubernetes baseline](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/compatible-k8s-baseline) | Corrected script evaluation without changing the inherited run command. | Reached model inference, then revealed that PyTorch 2.6/CUDA 12.4 lacked kernels for the cluster's Blackwell GPU; no research result was produced. |
| [Blackwell zero-shot control](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2) | Moved the image to PyTorch 2.7.1/CUDA 12.8 and measured the untuned Qwen2.5-0.5B-Instruct control on 24 synthetic fact chains. | **0/48 direct facts and 0/24 two-hop questions**, confirming the facts were novel. RTX PRO 6000, 7.66 s (about 0.0021 GPU-hours). |
| [Synthetic entity self-patching](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/synthetic-entity-self-patching) | Trained the LoRA adapter and attempted the first entity-position activation scan. | Training converged, but analysis stopped when isolated and in-context tokenization used different entity anchors. This motivated a context-aware anchor rather than yielding a patching metric. |
| [Context-aware entity self-patching](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching) | Recomputed entity anchors in context and ran the full 24×24 residual-stream patch scan with a random-position oracle control. | **Partially reproduced:** 48/48 fact recall, 0/24 natural two-hop answers, and 3/12 patched recoveries versus 1/12 for the control. The paper's middle-layer localization did not reproduce. RTX PRO 6000, 273.38 s (about 0.0759 GPU-hours). |
| [Answer-matched activation heatmap](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/answer-matched-activation-heatmap) | Loaded Qwen2.5-0.5B and patched every residual layer × prompt-token position, with a route-changed but answer-matched control. | Teaching-only—not reproduction evidence. The relevant patch peaked at +22.75 France-vs-Germany logits versus −0.25 for the control at the same cell. The paired 24×15 scan took 0.68 s on an RTX PRO 6000 (6.09 s including model load). |
