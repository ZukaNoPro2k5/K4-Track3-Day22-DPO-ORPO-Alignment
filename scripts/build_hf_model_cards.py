#!/usr/bin/env python3
"""Create Hub-ready README model cards from the artifacts produced by the lab."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    dpo_dir = REPO / "adapters/dpo"
    gguf_dir = REPO / "gguf"
    if not dpo_dir.exists():
        raise SystemExit("adapters/dpo is missing; run NB3 first")

    metrics = read_json(dpo_dir / "dpo_metrics.json")
    benchmark = read_json(REPO / "data/eval/benchmark_results.json")
    base = metrics.get("base_model", "unsloth/Qwen2.5-3B-bnb-4bit")
    model_card = f"""---
base_model: {base}
library_name: peft
pipeline_tag: text-generation
language:
- vi
- en
tags:
- dpo
- alignment
- qwen2.5
license: apache-2.0
---

# Lab 22 Vietnamese DPO adapter

LoRA adapter produced for the VinUni AICB Track 3 Day 22 DPO/ORPO Alignment lab.

## Training

- Base model: `{base}`
- SFT dataset: `{os.environ.get('SFT_DATASET', 'bkai-foundation-models/vi-alpaca')}`
- Preference dataset: `{os.environ.get('PREF_DATASET', 'argilla/ultrafeedback-binarized-preferences-cleaned')}`
- Method: SFT followed by DPO
- DPO beta: `{metrics.get('beta', 0.1)}`
- Learning rate: `{metrics.get('lr', '5e-7')}`
- Epochs: `{metrics.get('epochs', 1)}`
- LoRA: r=16, alpha=32

## Evaluation

- Final DPO train loss: `{metrics.get('final_train_loss', 'run pending')}`
- End chosen-minus-rejected reward gap: `{metrics.get('end_reward_gap', 'run pending')}`
- Benchmark output: `data/eval/benchmark_results.json` in the companion GitHub repository.
- Recorded benchmark metrics: `{json.dumps(benchmark.get('metrics', {}), ensure_ascii=False)}`

## Usage

Load `{base}` in 4-bit mode and attach this repository with `PeftModel.from_pretrained`.

## Limitations

This is a small educational experiment, not a production safety model. Outputs can be
incorrect or unsafe and require human review. The training and evaluations are limited
in size and do not establish broad Vietnamese-language robustness.
"""
    (dpo_dir / "README.md").write_text(model_card)
    print(f"Saved {dpo_dir / 'README.md'}")

    if gguf_dir.exists() and list(gguf_dir.glob("*.gguf")):
        quantizations = ", ".join(sorted(path.stem for path in gguf_dir.glob("*.gguf")))
        gguf_card = f"""---
base_model: {base}
library_name: gguf
pipeline_tag: text-generation
language:
- vi
- en
tags:
- gguf
- dpo
- qwen2.5
---

# Lab 22 Vietnamese DPO — GGUF release

Merged deployment build of the Lab 22 SFT+DPO adapter. Available artifacts:
{quantizations}.

At minimum the grading release contains Q4_K_M and Q5_K_M. This educational model
inherits the limitations and license requirements of its base model and training data.
"""
        (gguf_dir / "README.md").write_text(gguf_card)
        print(f"Saved {gguf_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
