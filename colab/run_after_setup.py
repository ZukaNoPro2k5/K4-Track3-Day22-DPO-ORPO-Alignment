#!/usr/bin/env python3
"""Resume the full Lab 22 pipeline after cells 0a–0g have completed.

Use this from an already-prepared Colab runtime to avoid re-cloning and
re-installing dependencies:
    python colab/run_after_setup.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/content/lab22") if Path("/content/lab22").exists() else Path(__file__).resolve().parent.parent


def run(label: str, *args: str, check: bool = True) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, *args], cwd=WORK, env=os.environ.copy(), check=check)


def upload_to_hub() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN absent — skipping Hub upload")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    adapter_repo = os.environ.get("HF_REPO") or f"{user}/lab22-dpo-qwen3b-vn"
    api.create_repo(adapter_repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=adapter_repo, repo_type="model", folder_path=str(WORK / "adapters/dpo"))
    print(f"Adapter + model card: https://huggingface.co/{adapter_repo}")

    if list((WORK / "gguf").glob("*.gguf")):
        gguf_repo = os.environ.get("HF_GGUF_REPO") or f"{user}/lab22-dpo-qwen3b-vn-gguf"
        api.create_repo(gguf_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=gguf_repo, repo_type="model", folder_path=str(WORK / "gguf"))
        print(f"GGUF release: https://huggingface.co/{gguf_repo}")


def backup_to_drive() -> None:
    drive_out = os.environ.get("DRIVE_OUT")
    if not drive_out or not Path(drive_out).exists():
        print("Drive not mounted — skipping backup")
        return
    targets = ["submission", "data/eval", "data/pref", "adapters/sft-mini", "adapters/dpo",
               "adapters/dpo-b0.05", "adapters/dpo-b0.10", "adapters/dpo-b0.50",
               "bonus/adapters/dpo-bonus"]
    if os.environ.get("BACKUP_GGUF_TO_DRIVE", "0").lower() in {"1", "true", "yes"}:
        targets.append("gguf")
    destination = Path(drive_out)
    for target in targets:
        source = WORK / target
        if not source.exists():
            continue
        dest = destination / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
    print(f"Backed artifacts up to {destination}")


def main() -> None:
    os.environ.setdefault("COMPUTE_TIER", "T4")
    os.environ.setdefault("SFT_DATASET", "bkai-foundation-models/vi-alpaca")
    os.environ.setdefault("PREF_DATASET", "argilla/ultrafeedback-binarized-preferences-cleaned")
    os.environ.setdefault("WANDB_PROJECT", "lab22-dpo")
    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "online")

    run("NB1 — SFT mini", "notebooks/01_sft_mini.py")
    run("NB2 — preference data", "notebooks/02_preference_data.py")
    run("NB3 — DPO training", "notebooks/03_dpo_train.py")
    run("NB4 — side-by-side eval", "notebooks/04_compare_and_eval.py")
    run("NB5 — Q4 GGUF", "notebooks/05_merge_deploy_gguf.py")
    run("NB5 extra — Q5 + Q8 GGUF", "scripts/merge_and_gguf.py", "--quant", "q5_k_m", "--quant", "q8_0")
    run("NB6 — benchmark suite", "notebooks/06_benchmark.py")
    if os.environ.get("RUN_EXHAUSTIVE_BONUS", "1").lower() not in {"0", "false", "no"}:
        run("MMLU full 14k", "scripts/run_mmlu_full.py")
    run("beta=0.05", "scripts/train_dpo.py", "--beta", "0.05", "--output-dir", "adapters/dpo-b0.05")
    run("beta=0.10", "scripts/train_dpo.py", "--beta", "0.1", "--output-dir", "adapters/dpo-b0.10")
    run("beta=0.50", "scripts/train_dpo.py", "--beta", "0.5", "--output-dir", "adapters/dpo-b0.50")
    run("beta sweep eval + plot", "scripts/eval_beta_sweep.py", "--sweep-dir", "adapters")
    run("mental-health bonus", "bonus/train.py")
    run("build Hub model cards", "scripts/build_hf_model_cards.py")
    upload_to_hub()
    run("submission verify", "scripts/verify.py", check=False)
    backup_to_drive()


if __name__ == "__main__":
    main()
