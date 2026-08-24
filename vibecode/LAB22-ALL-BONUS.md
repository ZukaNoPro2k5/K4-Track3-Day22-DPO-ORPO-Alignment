# LAB22-ALL-BONUS — delivery record

## VISION / scope

Run the existing T4 Lab 22 core pipeline and produce evidence for every rigor
add-on listed in `rubric.md` (+37 available, grading cap +20). Preserve the
existing core output contracts. Runtime-generated scores and personal reflection
must remain real rather than pre-filled.

Workflow depth: **medium**, because the change crosses training, evaluation,
publishing, Colab orchestration, and submission evidence.

Roles: learner = Homeowner/Accountable Engineer/release owner; Codex = Global
Contractor + AI Builder. The learner retains final responsibility for API spend,
public links, reflection, and submission.

## BLUEPRINT / decisions

- Keep GPT-5.6 Terra as the primary judge requested by the learner.
- Also run the rubric's exact `gpt-4o-mini` + `claude-haiku-4-5` pair and report
  disagreement for NB4 and AlpacaEval-lite.
- Keep sampled NB6 and isolate full MMLU 14k as a cached long-running stage.
- Evaluate all beta adapters on the same eight prompts and plot reward gap plus
  judge win-rate.
- Generate Hub README model cards from actual run metrics before upload.
- Capture W&B run URLs in a durable JSON artifact.
- Fail before training when exhaustive mode lacks a required service key.

## TIP / acceptance criteria

1. Existing core NB1–NB4 behavior and filenames remain compatible.
2. Q4_K_M, Q5_K_M, and Q8_0 export/upload paths remain present.
3. Sampled NB6 still writes the four-benchmark comparison.
4. Beta sweep writes measured reward-gap and win-rate data plus a plot.
5. Full MMLU uses `LIMIT_MMLU=14000`, caches per model, and compares sampled/full.
6. NB4 and AlpacaEval-lite write cross-judge disagreement reports.
7. Adapter and GGUF Hub uploads include README model cards.
8. W&B run URLs are saved for submission.
9. The generated Colab notebook exposes exhaustive mode and fails fast on missing keys.

Non-goals: fabricate scores/reflection, bypass Colab authentication, or claim a
GPU run passed without runtime evidence.

## AI completion report

Implemented all nine criteria across the notebook sources, new resumable bonus
scripts, Hub publishing, W&B capture, reflection template, Make targets, and the
generated 37-cell Colab notebook. The unrelated untracked `colab_diag.py` was not
modified.

## Engineer verification

- `python3 -m py_compile ...`: passed for every changed/new Python source.
- `python3 -m json.tool colab/Lab22_FULL_PIPELINE.ipynb`: passed.
- Generated-notebook exhaustive cell assertions: passed.
- `python3 scripts/test_smoke.py`: passed.
- `git diff --check`: passed.
- GPU execution: pending; VS Code Colab keep-alive is currently timing out, so no
  runtime-success claim is made.

Requirement coverage before GPU execution: **9/9 implemented, 0/9 empirically
executed in the remote T4 runtime**. The latter advances only after reconnecting
the Colab kernel and running the notebook.
