#!/usr/bin/env python3
"""Downscaled reproduction of the knowing-using gap and self-patching claim.

The immutable baseline runs the pretrained model. Child experiments switch the
committed config to ``reproduction`` and fine-tune on the same fixed facts.
Everything needed for analysis is printed to stdout as a final JSON record.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


REQUIRED = {"torch": "torch", "transformers": "transformers==4.51.3", "peft": "peft==0.15.2"}


def ensure_dependencies() -> None:
    missing = [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_dependencies()

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


NAMES = [
    "Aldren", "Bexley", "Corvin", "Damaris", "Elowen", "Fenric", "Galen", "Hesper",
    "Ilyra", "Jorren", "Kaelis", "Luneth", "Mirel", "Nerys", "Orlan", "Perrin",
    "Quenby", "Rhosyn", "Sable", "Tavian", "Ulric", "Vesper", "Wystan", "Xanthe",
    "Ysoria", "Zephan", "Averil", "Briony", "Caspian", "Delwyn", "Eirwen", "Fintan",
    "Grisel", "Hadrian", "Isolde", "Jasper", "Kerensa", "Leoric", "Morwen", "Nolan",
    "Oriana", "Pascal", "Roslin", "Sylvan", "Theron", "Una", "Valen", "Willow",
    "Xenia", "Yestin", "Zorion", "Anwen", "Bram", "Cerys", "Dorian", "Eleri",
    "Faolan", "Gwenna", "Hywel", "Ivo", "Jessamy", "Kellan", "Lowri", "Merrick",
    "Nyssa", "Osric", "Priya", "Riven", "Senna", "Torin", "Urien", "Verity"
]


def build_chains(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    names = NAMES.copy()
    rng.shuffle(names)
    if 3 * n > len(names):
        raise ValueError("Not enough fixed names")
    return [dict(head=names[3*i], bridge=names[3*i+1], answer=names[3*i+2]) for i in range(n)]


def messages(prompt: str, answer: str | None = None) -> list[dict[str, str]]:
    out = [{"role": "system", "content": "Answer with only the requested name."},
           {"role": "user", "content": prompt}]
    if answer is not None:
        out.append({"role": "assistant", "content": answer})
    return out


def fact_prompts(chain: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (f"In the newly defined Ravel index, which name immediately follows {chain['head']}?", chain["bridge"]),
        (f"In the newly defined Sable index, which value is assigned to {chain['bridge']}?", chain["answer"]),
    ]


def chain_prompt(chain: dict[str, str]) -> str:
    return ("Use these rules: first find the name immediately following the query name in the newly "
            "defined Ravel index; then return the value assigned to that intermediate name in the "
            f"newly defined Sable index. Starting from {chain['head']}, what final value do you get?")


class FactDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128):
        self.items = []
        for prompt, answer in rows:
            prompt_text = tokenizer.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages(prompt, answer), tokenize=False, add_generation_prompt=False)
            full = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
            labels = full["input_ids"].clone()
            labels[:, :len(prompt_ids)] = -100
            self.items.append({"input_ids": full["input_ids"][0], "attention_mask": full["attention_mask"][0], "labels": labels[0]})

    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def collate(items, pad_id):
    width = max(len(x["input_ids"]) for x in items)
    batch = {}
    for key in ("input_ids", "attention_mask", "labels"):
        fill = -100 if key == "labels" else (0 if key == "attention_mask" else pad_id)
        batch[key] = torch.stack([torch.nn.functional.pad(x[key], (0, width-len(x[key])), value=fill) for x in items])
    return batch


def normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip().splitlines()[0] if ch.isalnum())


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, device, max_new_tokens=12) -> str:
    text = tokenizer.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def find_subsequence(sequence: list[int], subsequence: list[int]) -> list[int]:
    hits = []
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i+len(subsequence)] == subsequence:
            hits.extend(range(i, i+len(subsequence)))
    if not hits:
        raise ValueError(f"Anchor token sequence {subsequence} not found")
    return hits


def prompt_and_answer(tokenizer, prompt: str, answer: str):
    prompt_text = tokenizer.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + answer
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    answer_start = prompt_ids.shape[1]
    return prompt_text, prompt_ids, full_ids, answer_start


@torch.inference_mode()
def cache_residuals(model, input_ids, layers):
    cache = {}
    handles = []
    for idx, layer in enumerate(layers):
        def hook(_module, _inputs, output, idx=idx):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[idx] = hidden.detach().clone()
        handles.append(layer.register_forward_hook(hook))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles: handle.remove()
    return cache


@torch.inference_mode()
def token_exact_with_patch(model, full_ids, answer_start, source, target_layer, positions, layers):
    def patch(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        valid = [p for p in positions if p < patched.shape[1]]
        patched[:, valid, :] = source[:, valid, :].to(patched.dtype)
        return (patched, *output[1:]) if isinstance(output, tuple) else patched
    handle = layers[target_layer].register_forward_hook(patch)
    try:
        logits = model(input_ids=full_ids, use_cache=False).logits
    finally:
        handle.remove()
    predicted = logits[:, answer_start-1:full_ids.shape[1]-1].argmax(-1)
    gold = full_ids[:, answer_start:]
    return bool(torch.equal(predicted, gold))


def evaluate(model, tokenizer, chains, device, scan_samples, do_scan):
    direct = []
    chaining = []
    examples = []
    for chain in chains:
        for prompt, answer in fact_prompts(chain):
            pred = generate_answer(model, tokenizer, prompt, device)
            direct.append(normalize(pred) == normalize(answer))
        prompt = chain_prompt(chain)
        pred = generate_answer(model, tokenizer, prompt, device)
        chaining.append(normalize(pred) == normalize(chain["answer"]))
        if len(examples) < 4:
            examples.append({"head": chain["head"], "answer": chain["answer"], "prediction": pred})
    result = {"memorization_accuracy": sum(direct)/len(direct), "chaining_accuracy": sum(chaining)/len(chaining),
              "memorization_n": len(direct), "chaining_n": len(chaining), "examples": examples}
    print("PRE_SCAN_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if not do_scan:
        return result

    layers = model.base_model.model.model.layers
    n_layers = len(layers)
    entity_success = random_success = unpatched_tf = 0
    best_pairs = []
    for chain in chains[:scan_samples]:
        prompt = chain_prompt(chain)
        _text, _prompt_ids, full_ids, answer_start = prompt_and_answer(tokenizer, prompt, chain["answer"])
        full_ids = full_ids.to(device)
        cache = cache_residuals(model, full_ids, layers)
        ids = full_ids[0].tolist()
        anchor_candidates = [
            tokenizer(chain["head"], add_special_tokens=False)["input_ids"],
            tokenizer(" " + chain["head"], add_special_tokens=False)["input_ids"],
        ]
        entity_positions = None
        for anchor_ids in anchor_candidates:
            try:
                entity_positions = find_subsequence(ids[:answer_start], anchor_ids)
                break
            except ValueError:
                pass
        if entity_positions is None:
            raise ValueError(f"Context-aware anchor match failed for {chain['head']}")
        random_positions = [1]
        base_ok = token_exact_with_patch(model, full_ids, answer_start, cache[0], 0, entity_positions, layers)
        unpatched_tf += int(base_ok)
        found_entity = found_random = False
        first_pair = None
        for source_layer in range(n_layers):
            for target_layer in range(n_layers):
                if not found_entity and token_exact_with_patch(model, full_ids, answer_start, cache[source_layer], target_layer, entity_positions, layers):
                    found_entity, first_pair = True, [source_layer, target_layer]
                if not found_random and token_exact_with_patch(model, full_ids, answer_start, cache[source_layer], target_layer, random_positions, layers):
                    found_random = True
                if found_entity and found_random:
                    break
            if found_entity and found_random:
                break
        entity_success += int(found_entity)
        random_success += int(found_random)
        best_pairs.append(first_pair)
        print(json.dumps({"scan_item": chain["head"], "unpatched": base_ok, "entity_oracle": found_entity,
                          "random_oracle": found_random, "first_entity_pair": first_pair}), flush=True)
    result.update({
        "teacher_forced_unpatched_accuracy": unpatched_tf/scan_samples,
        "entity_oracle_self_patch_accuracy": entity_success/scan_samples,
        "random_position_oracle_accuracy": random_success/scan_samples,
        "scan_n": scan_samples,
        "layers": n_layers,
        "first_successful_pairs": best_pairs,
        "scoring": "exact match of every teacher-forced answer token; oracle scans all residual-post source/target layer pairs"
    })
    return result


def run_toy_lab(device, signal_strength=1.5, samples=4096, seed=7):
    """Bounded synthetic analogue used by the notebook's optional GPU lab."""
    generator = torch.Generator(device=device).manual_seed(seed)
    labels = torch.randint(0, 2, (samples,), generator=generator, device=device) * 2 - 1
    source = torch.randn(samples, generator=generator, device=device) + signal_strength * labels
    stranded = torch.randn(samples, generator=generator, device=device) + 0.10 * labels
    shuffled = source[torch.randperm(samples, generator=generator, device=device)]
    strengths = torch.linspace(0, 1, 11, device=device)
    patched_accuracy, control_accuracy = [], []
    for alpha in strengths:
        patched_accuracy.append(((stranded + alpha * source).sign() == labels).float().mean().item())
        control_accuracy.append(((stranded + alpha * shuffled).sign() == labels).float().mean().item())
    return {"strengths": strengths.cpu().tolist(), "patched_accuracy": patched_accuracy,
            "shuffled_control_accuracy": control_accuracy, "samples": samples,
            "signal_strength": signal_strength, "seed": seed}


def main():
    started = time.time()
    config = json.loads(Path("experiment_config.json").read_text())
    random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "environment", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "torch": torch.__version__, "config": config}), flush=True)
    if config["mode"] == "gpu_lab_validation":
        result = run_toy_lab(device)
        result.update({"event": "final_result", "mode": config["mode"], "device": str(device),
                       "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                       "runtime_seconds": time.time()-started})
        print("FINAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                                 trust_remote_code=True).to(device)
    chains = build_chains(config["num_chains"], config["seed"])
    if config["mode"] == "reproduction":
        lora = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        rows = [row for chain in chains for row in fact_prompts(chain)]
        data = FactDataset(rows, tokenizer)
        loader = DataLoader(data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(config["seed"]),
                            collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
        model.train()
        for epoch in range(config["epochs"]):
            losses = []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            print(json.dumps({"epoch": epoch+1, "loss": sum(losses)/len(losses)}), flush=True)
        model.eval()
    result = evaluate(model, tokenizer, chains, device, config["scan_samples"], config["mode"] == "reproduction")
    result.update({"event": "final_result", "mode": config["mode"], "model": config["model"], "seed": config["seed"],
                   "runtime_seconds": time.time()-started, "device": str(device),
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    print("FINAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
