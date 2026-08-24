"""CPU-only smoke tests — run without a GPU (no torch/unsloth/trl import).

These guard the lab source against the most common breakages so `make test`
is a real gate, not a no-op:
- every notebook/script file exists and is valid Python (catches syntax errors)
- the TRL trainer calls use `processing_class=` (TRL >= 0.13), NOT the removed
  `tokenizer=` arg — the regression that broke NB1/NB3 on the resolved trl 0.19.x

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_sft_mini", "02_preference_data", "03_dpo_train",
    "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
]


def test_notebooks_exist_and_parse():
    for nb in NOTEBOOKS:
        p = REPO / "notebooks" / f"{nb}.py"
        assert p.exists(), f"missing notebook {p}"
        ast.parse(p.read_text(encoding="utf-8"))  # SyntaxError if broken


def test_scripts_parse():
    for p in (REPO / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))


def test_colab_notebooks_are_valid_json():
    for p in (REPO / "colab").glob("*.ipynb"):
        json.loads(p.read_text(encoding="utf-8"))  # ValueError if corrupt


def test_full_pipeline_cells_parse_and_fail_fast():
    p = REPO / "colab" / "Lab22_FULL_PIPELINE.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"{p}:code-cell-{index}")
    joined = "\n".join(code_cells)
    assert "def run_step(" in joined
    assert "!cd /content/lab22" not in joined
    assert "os.environ['DRIVE_OUT'] = DRIVE_OUT" in joined


def test_gguf_export_uses_combined_dpo_adapter():
    notebook = (REPO / "notebooks" / "05_merge_deploy_gguf.py").read_text(encoding="utf-8")
    script = (REPO / "scripts" / "merge_and_gguf.py").read_text(encoding="utf-8")
    assert "PeftModel.from_pretrained(model, str(DPO_PATH))" in notebook
    assert "PeftModel.from_pretrained(model, args.dpo_path)" in script


def test_trainer_uses_processing_class_not_tokenizer():
    # TRL >= 0.13 removed the `tokenizer=` arg in favour of `processing_class=`.
    # With the requirements pin `trl>=0.12,<0.20` a fresh install resolves to
    # 0.19.x, where `DPOTrainer/SFTTrainer(tokenizer=...)` raises TypeError.
    targets = [
        "notebooks/01_sft_mini.py",
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "colab/Lab22_DPO_T4.ipynb",
        "colab/Lab22_DPO_BigGPU.ipynb",
    ]
    offenders = [t for t in targets if "tokenizer=tokenizer" in (REPO / t).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still pass tokenizer=tokenizer to a TRL trainer; "
        f"use processing_class=tokenizer (tokenizer= removed in trl>=0.13)."
    )
