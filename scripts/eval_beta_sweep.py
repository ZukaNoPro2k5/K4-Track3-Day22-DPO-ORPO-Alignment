#!/usr/bin/env python3
"""Evaluate fixed prompts for every beta adapter and plot gap + judged win-rate.

This is intentionally separate from training so a disconnected Colab session can
resume from the already-saved ``adapters/dpo-b*`` directories.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def generate(adapter_path: Path, prompts: list[dict]) -> list[str]:
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    base = "unsloth/Qwen2.5-3B-bnb-4bit" if tier == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    max_len = 512 if tier == "T4" else 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_len, dtype=None, load_in_4bit=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, str(adapter_path))
    FastLanguageModel.for_inference(model)

    outputs = []
    for item in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def judge(prompt: str, a: str, b: str) -> dict:
    from openai import OpenAI

    message = f"""Judge two Vietnamese assistant answers for helpfulness, correctness, safety, and appropriate length.

User prompt: {prompt}

Response A: {a}

Response B: {b}

Return JSON only: {{"winner":"A"|"B"|"tie","reason":"one short sentence"}}"""
    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.6-terra"),
        messages=[{"role": "user", "content": message}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"winner": "tie", "reason": "judge response could not be parsed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(REPO / "adapters"))
    parser.add_argument("--output", default=str(REPO / "submission/screenshots/bonus-beta-sweep.png"))
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to measure beta-sweep win-rate")

    prompt_file = REPO / "data/eval/prompts.json"
    if not prompt_file.exists():
        raise SystemExit("Run NB4 first so data/eval/prompts.json exists")
    prompts = json.loads(prompt_file.read_text())
    sweep_dirs = sorted(Path(args.sweep_dir).glob("dpo-b*"))
    if not sweep_dirs:
        raise SystemExit("No adapters/dpo-b* directories found; run the beta training cells first")

    cache_dir = REPO / "data/eval/beta_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sft_cache = cache_dir / "sft.json"
    if sft_cache.exists():
        sft_outputs = json.loads(sft_cache.read_text())
    else:
        sft_outputs = generate(REPO / "adapters/sft-mini", prompts)
        sft_cache.write_text(json.dumps(sft_outputs, ensure_ascii=False, indent=2))

    rows = []
    for adapter in sweep_dirs:
        metrics_path = adapter / "dpo_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        beta = float(metrics["beta"])
        result_path = cache_dir / f"beta-{beta:g}-judgments.json"
        output_path = cache_dir / f"beta-{beta:g}-outputs.json"
        if result_path.exists() and output_path.exists():
            judgments = json.loads(result_path.read_text())
            beta_outputs = json.loads(output_path.read_text())
        else:
            beta_outputs = generate(adapter, prompts)
            output_path.write_text(json.dumps(beta_outputs, ensure_ascii=False, indent=2))
            judgments = []
            for index, (item, sft, tuned) in enumerate(zip(prompts, sft_outputs, beta_outputs)):
                flip = index % 2 == 1
                a, b = (tuned, sft) if flip else (sft, tuned)
                result = judge(item["prompt"], a, b)
                winner = result.get("winner")
                if winner in {"A", "B"}:
                    winner_model = ("beta" if winner == "A" else "sft") if flip else ("sft" if winner == "A" else "beta")
                else:
                    winner_model = "tie"
                result.update({"id": item["id"], "winner_model": winner_model})
                judgments.append(result)
            result_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2))

        wins = sum(j.get("winner_model") == "beta" for j in judgments)
        ties = sum(j.get("winner_model") == "tie" for j in judgments)
        total = len(judgments)
        win_rate = (wins + 0.5 * ties) / total if total else 0.0
        mean_length = sum(len(text.split()) for text in beta_outputs) / len(beta_outputs)
        metrics.update({"win_rate_vs_sft": win_rate, "mean_output_words": mean_length})
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        rows.append({
            "beta": beta,
            "reward_gap": metrics.get("end_reward_gap"),
            "win_rate_vs_sft": win_rate,
            "mean_output_words": mean_length,
        })

    rows.sort(key=lambda row: row["beta"])
    result_file = REPO / "data/eval/beta_sweep_results.json"
    result_file.write_text(json.dumps({"n_prompts": len(prompts), "results": rows}, indent=2))

    import matplotlib.pyplot as plt

    betas = [row["beta"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(betas, [row["reward_gap"] for row in rows], marker="o", linewidth=2)
    axes[0].set_title("Reward gap vs beta")
    axes[0].set_ylabel("chosen - rejected")
    axes[1].plot(betas, [row["win_rate_vs_sft"] for row in rows], marker="o", linewidth=2, color="#c83538")
    axes[1].set_title(f"Judge win-rate vs beta ({len(prompts)} prompts)")
    axes[1].set_ylabel("win-rate vs SFT (ties = 0.5)")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("beta")
        axis.set_xscale("log")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(json.dumps(rows, indent=2))
    print(f"Saved {result_file} and {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
