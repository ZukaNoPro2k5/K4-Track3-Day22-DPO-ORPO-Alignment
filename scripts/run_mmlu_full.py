#!/usr/bin/env python3
"""Run the rubric's full MMLU add-on and compare it with the sampled-500 run.

Each model result is cached independently, so rerunning after a Colab disconnect
keeps a completed SFT or DPO half. The active lm-eval process itself cannot be
resumed mid-model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_OUT = REPO / "data/eval"
LIMIT_MMLU = 14000


def extract_score(results: dict) -> float:
    for metrics in results.values():
        if "acc,none" in metrics:
            return float(metrics["acc,none"])
    values = [
        value for metrics in results.values() for key, value in metrics.items()
        if key.startswith("acc") and isinstance(value, (int, float))
    ]
    return sum(values) / len(values) if values else float("nan")


def run_model(label: str, adapter: Path) -> tuple[dict, Path]:
    cache = EVAL_OUT / f"mmlu_full_{label}.json"
    drive_out = os.environ.get("DRIVE_OUT")
    drive_cache = Path(drive_out) / "data/eval" / cache.name if drive_out else None
    if not cache.exists() and drive_cache and drive_cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_cache, cache)
        print(f"Restored {label} MMLU cache from Drive: {drive_cache}")
    if cache.exists():
        print(f"Reusing completed {label} result: {cache}")
        return json.loads(cache.read_text()), cache

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    base = "unsloth/Qwen2.5-3B-bnb-4bit" if tier == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    output_dir = EVAL_OUT / f"lm-{label}-mmlu-full"
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={base},peft={adapter},load_in_4bit=True",
        "--tasks", "mmlu", "--num_fewshot", "5",
        "--limit", str(LIMIT_MMLU), "--batch_size", "1",
        "--device", "cuda:0", "--output_path", str(output_dir),
    ]
    print(f"Running full MMLU for {label}; this is the long T4 stage...")
    process = subprocess.run(cmd, text=True, timeout=8 * 60 * 60)
    if process.returncode:
        raise RuntimeError(f"lm-eval failed for {label} with exit code {process.returncode}")
    candidates = sorted(output_dir.glob("**/results*.json"))
    if not candidates:
        raise RuntimeError(f"No lm-eval result JSON found under {output_dir}")
    raw = json.loads(candidates[-1].read_text())
    result = {
        "label": label,
        "limit_mmlu": LIMIT_MMLU,
        "score": extract_score(raw["results"]),
        "source_result": str(candidates[-1].relative_to(REPO)),
        "results": raw["results"],
    }
    cache.write_text(json.dumps(result, indent=2))
    if drive_cache:
        drive_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache, drive_cache)
        print(f"Backed up completed {label} MMLU result to Drive")
    return result, cache


def main() -> int:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    sft, _ = run_model("sft", REPO / "adapters/sft-mini")
    dpo, _ = run_model("dpo", REPO / "adapters/dpo")
    sampled_file = EVAL_OUT / "benchmark_results.json"
    sampled = json.loads(sampled_file.read_text()) if sampled_file.exists() else None
    sampled_scores = sampled.get("metrics", {}).get("MMLU") if sampled else None
    report = {
        "limit_mmlu": LIMIT_MMLU,
        "full": {"sft": sft["score"], "dpo": dpo["score"]},
        "full_delta": dpo["score"] - sft["score"],
        "sampled_500": sampled_scores,
    }
    output = EVAL_OUT / "benchmark_results_mmlu_full.json"
    output.write_text(json.dumps(report, indent=2))

    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["sampled-500", "full-14k"] if sampled_scores else ["full-14k"]
    sft_scores = ([sampled_scores["sft"]] if sampled_scores else []) + [sft["score"]]
    dpo_scores = ([sampled_scores["dpo"]] if sampled_scores else []) + [dpo["score"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, sft_scores, width, label="SFT-only")
    ax.bar(x + width / 2, dpo_scores, width, label="SFT+DPO")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("MMLU accuracy")
    ax.set_title("MMLU sampled vs full coverage")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    screenshot = REPO / "submission/screenshots/bonus-mmlu-full.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(screenshot, dpi=140, bbox_inches="tight")
    print(json.dumps(report, indent=2))
    print(f"Saved {output} and {screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
