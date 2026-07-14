# When memorized facts fail to become usable knowledge

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rehaanahmad2013/towards-mechanistically-understanding/blob/main/claim_tutorial.py)

This repository contains a tutorial-style, independently runnable marimo notebook for a downscaled reproduction of the main illustrative claim in [*Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Fine-tuning*](https://arxiv.org/abs/2607.08393).

The result is **partially reproduced**: a LoRA-tuned Qwen-2.5-0.5B model recalled all 48 injected facts but solved none of 24 held-out two-hop questions. Oracle entity self-patching recovered 3 of 12 scanned failures (25.0%), compared with 1 of 12 (8.3%) for a random-position oracle. The paper's middle-layer localization did not appear in this small synthetic setting.

Open `claim_tutorial.py` in molab to see the frozen evidence, provenance, controls, limitations, and an optional sub-second synthetic GPU lab. Formal reproduction code is preserved on immutable `orx/*` experiment branches linked inside the notebook.

## Experiment log

Each row links to an immutable experiment branch containing the runner, fixed configuration, Kubernetes manifest, and evaluation method used at that stage.

| Branch | What it tried | Outcome and compute |
|---|---|---|
| [Baseline](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/baseline) | Established the original runner and Kubernetes job contract. | Failed before cloning because the injected shell script was expanded incorrectly; no research result was produced. |
| [Compatible Kubernetes baseline](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/compatible-k8s-baseline) | Corrected script evaluation without changing the inherited run command. | Reached model inference, then revealed that PyTorch 2.6/CUDA 12.4 lacked kernels for the cluster's Blackwell GPU; no research result was produced. |
| [Blackwell zero-shot control](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/blackwell-zero-shot-control-2) | Moved the image to PyTorch 2.7.1/CUDA 12.8 and measured the untuned Qwen2.5-0.5B-Instruct control on 24 synthetic fact chains. | **0/48 direct facts and 0/24 two-hop questions**, confirming the facts were novel. RTX PRO 6000, 7.66 s (about 0.0021 GPU-hours). |
| [Synthetic entity self-patching](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/synthetic-entity-self-patching) | Trained the LoRA adapter and attempted the first entity-position activation scan. | Training converged, but analysis stopped when isolated and in-context tokenization used different entity anchors. This motivated a context-aware anchor rather than yielding a patching metric. |
| [Context-aware entity self-patching](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/context-aware-entity-self-patching) | Recomputed entity anchors in context and ran the full 24×24 residual-stream patch scan with a random-position oracle control. | **Partially reproduced:** 48/48 fact recall, 0/24 natural two-hop answers, and 3/12 patched recoveries versus 1/12 for the control. The paper's middle-layer localization did not reproduce. RTX PRO 6000, 273.38 s (about 0.0759 GPU-hours). |
| [Molab toy GPU lab](https://github.com/rehaanahmad2013/towards-mechanistically-understanding/tree/orx/molab-toy-gpu-lab-validation) | Validated the notebook's bounded synthetic intervention demo on the target GPU class. | Teaching-only—not reproduction evidence. The relevant intervention raised accuracy from 0.5322 to 0.8660 while the shuffled control ended at 0.5200. RTX PRO 6000, 0.323 s. |

<details>
<summary>Internal run references</summary>

These IDs identify the corresponding managed `orx` runs: baseline `fd5fb806-279a-46ce-b1fd-d1655ebfe8df`; Kubernetes compatibility `4e71c53d-028c-4379-a3f3-419f570f15fa`; zero-shot control `21a46750-e345-4f51-b09d-6168d514b5ac`; first entity scan `7ccdf903-44ed-4ad6-85d7-e6a8cd7313ae`; confirmatory reproduction `b77b5ff4-0c04-421b-aff6-a14607a64446`; GPU lab validation `2c549be7-904a-46e2-9cab-c1ec372ec894`.

</details>
