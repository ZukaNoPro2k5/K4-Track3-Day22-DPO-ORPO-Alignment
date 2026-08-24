#!/usr/bin/env python3
"""CLI wrapper for NB5 logic — merge adapter + quantize to GGUF.

Usage:
    python scripts/merge_and_gguf.py
    python scripts/merge_and_gguf.py --quant q5_k_m
    python scripts/merge_and_gguf.py --quant q4_k_m --quant q5_k_m --quant q8_0

Mirrors `notebooks/05_merge_deploy_gguf.py` cells 1-3. Used if you want to add
extra GGUF tiers (the +3 'GGUF release published' rigor add-on).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_gguf_files(*roots: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                found[str(candidate.resolve())] = candidate
    return sorted(found.values())


def normalized_quant(path: Path) -> str:
    return "".join(ch for ch in path.name.lower() if ch.isalnum())


def normalize_gguf_files(destination: Path, *roots: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    for source in find_gguf_files(destination, *roots):
        if source.parent.resolve() == destination.resolve():
            continue
        target = destination / source.name
        if not target.exists():
            shutil.move(str(source), str(target))
            print(f"Moved GGUF output into {target}")
    return find_gguf_files(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--dpo-path", default=str(REPO / "adapters" / "dpo"))
    parser.add_argument("--merged-output", default=str(REPO / "adapters" / "merged-fp16"))
    parser.add_argument("--gguf-output", default=str(REPO / "gguf"))
    parser.add_argument("--quant", action="append", default=None,
                        help="Quantization tier(s). Repeat for multiple. Default: q4_k_m")
    args = parser.parse_args()

    quants = args.quant or ["q4_k_m"]

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    base = (
        "unsloth/Qwen2.5-3B-bnb-4bit" if tier == "T4"
        else "unsloth/Qwen2.5-7B-bnb-4bit"
    )
    merge_base = (
        "Qwen/Qwen2.5-3B" if tier == "T4"
        else "Qwen/Qwen2.5-7B"
    )
    max_len = 512 if tier == "T4" else 1024

    Path(args.merged_output).mkdir(parents=True, exist_ok=True)
    Path(args.gguf_output).mkdir(parents=True, exist_ok=True)

    print(f"Tier: {tier}  base: {base}  quants: {quants}")

    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unsloth import FastLanguageModel

    merged_output = Path(args.merged_output)
    merged_ready = (
        (merged_output / "config.json").exists()
        and any(merged_output.glob("*.safetensors"))
    )
    if merged_ready:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(str(merged_output))
        print(f"Reusing completed merged FP16 checkpoint at {merged_output}")
    else:
        # Step 1: the DPO adapter is the SFT LoRA continued with the DPO objective.
        # Merge against the unquantized base. Merging into bitsandbytes Linear4bit
        # modules is version-sensitive and currently breaks on Colab (`base_layer`).
        model = AutoModelForCausalLM.from_pretrained(
            merge_base,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(merge_base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(model, args.dpo_path, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
        print(f"Loaded and merged combined SFT+DPO adapter into {merge_base}")

    # Step 2: save merged FP16
    if not merged_ready:
        merged_tmp = merged_output.with_name(f".{merged_output.name}.tmp")
        if merged_tmp.exists():
            shutil.rmtree(merged_tmp)
        model.save_pretrained(
            str(merged_tmp), safe_serialization=True, max_shard_size="2GB",
        )
        tokenizer.save_pretrained(str(merged_tmp))
        if merged_output.exists():
            shutil.rmtree(merged_output)
        merged_tmp.replace(merged_output)
        print(f"Saved merged FP16 to {merged_output}")

    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: recover outputs from Unsloth's alternate layout, then only run
    # quantizations that are genuinely missing.
    gguf_output = Path(args.gguf_output)
    existing = normalize_gguf_files(gguf_output, REPO)
    missing_quants = [
        q for q in quants
        if not any(
            "".join(ch for ch in q.lower() if ch.isalnum()) in normalized_quant(path)
            for path in existing
        )
    ]
    if missing_quants:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(merged_output),
            max_seq_length=max_len, dtype=None, load_in_4bit=False,
        )
        for q in missing_quants:
            print(f"Quantizing to GGUF {q}...")
            result = model.save_pretrained_gguf(
                args.gguf_output, tokenizer, quantization_method=q,
            )
            print(f"Unsloth GGUF result: {result!r}")
        normalize_gguf_files(gguf_output, REPO)
    else:
        model = None
        print(f"All requested GGUF tiers already exist: {quants}")

    print(f"\nGGUF files in {args.gguf_output}:")
    for p in sorted(Path(args.gguf_output).iterdir()):
        if p.suffix == ".gguf":
            print(f"  {p.name:50s}  {p.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
