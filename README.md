# When memorized facts fail to become usable knowledge

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rehaanahmad2013/towards-mechanistically-understanding/blob/main/claim_tutorial.py)

This repository contains a tutorial-style, independently runnable marimo notebook for a downscaled reproduction of the main illustrative claim in [*Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Fine-tuning*](https://arxiv.org/abs/2607.08393).

The result is **partially reproduced**: a LoRA-tuned Qwen-2.5-0.5B model recalled all 48 injected facts but solved none of 24 held-out two-hop questions. Oracle entity self-patching recovered 3 of 12 scanned failures (25.0%), compared with 1 of 12 (8.3%) for a random-position oracle. The paper's middle-layer localization did not appear in this small synthetic setting.

Open `claim_tutorial.py` in molab to see the frozen evidence, provenance, controls, limitations, and an optional sub-second synthetic GPU lab. Formal reproduction code is preserved on immutable `orx/*` experiment branches linked inside the notebook.
