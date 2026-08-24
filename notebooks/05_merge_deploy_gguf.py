# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF  (OPTIONAL / BONUS)
#
# > **Optional (bonus).** Core lab = NB1--NB4. GGUF export builds llama.cpp at
# > runtime and is the most fragile step --- skip on free Colab T4 if short on time.
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import shutil
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MERGE_BASE_MODEL = (
    "Qwen/Qwen2.5-3B" if COMPUTE_TIER == "T4"
    else "Qwen/Qwen2.5-7B"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
MERGED_TMP_PATH = REPO_ROOT / "adapters" / ".merged-fp16.tmp"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"


def find_gguf_files(*roots: Path) -> list[Path]:
    """Find GGUF outputs across Unsloth's old and new output layouts."""
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                found[str(candidate.resolve())] = candidate
    return sorted(found.values())


def is_q4_k_m(path: Path) -> bool:
    normalized = "".join(ch for ch in path.name.lower() if ch.isalnum())
    return "q4km" in normalized

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load the FP16 base + merge the DPO adapter

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

merged_ready = (
    (MERGED_PATH / "config.json").exists()
    and any(MERGED_PATH.glob("*.safetensors"))
)
if merged_ready:
    model = None
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_PATH))
    print(f"Reusing completed merged FP16 checkpoint at {MERGED_PATH}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MERGE_BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The DPO adapter continues the SFT LoRA, so it already contains both stages.
    model = PeftModel.from_pretrained(model, str(DPO_PATH), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    print(f"Loaded combined SFT+DPO adapter from {DPO_PATH}")
    print(f"Merged it into full-precision base {MERGE_BASE_MODEL}")

# %% [markdown]
# > **Note:** NB3 continues training the SFT LoRA with the DPO objective. The
# > resulting `adapters/dpo` directory is therefore the combined SFT+DPO adapter
# > and is the only adapter that should be merged here.

# %% [markdown]
# ## 2. Save merged FP16 weights
#
# Standard PEFT merge is intentionally used here instead of merging a bitsandbytes
# `Linear4bit` model.  The latter is version-sensitive and can fail with
# `Linear4bit has no attribute base_layer` on Colab's current Unsloth/PEFT stack.

# %%
# Write to a temporary sibling first so a failed rerun never leaves a half-written
# directory that looks like a valid merged model.
if not merged_ready:
    if MERGED_TMP_PATH.exists():
        shutil.rmtree(MERGED_TMP_PATH)
    model.save_pretrained(
        str(MERGED_TMP_PATH),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(str(MERGED_TMP_PATH))
    if MERGED_PATH.exists():
        shutil.rmtree(MERGED_PATH)
    MERGED_TMP_PATH.replace(MERGED_PATH)
    print(f"Saved merged FP16 to {MERGED_PATH}")

# Free GPU memory before GGUF conversion (which spawns a subprocess that needs RAM)
import gc

if model is not None:
    del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Quantize to GGUF Q4_K_M
#
# Q4_K_M is the sweet spot: ~4× compression vs FP16, minimal quality loss.
# Unsloth wraps llama.cpp's `quantize` binary — first run downloads + compiles
# llama.cpp (~3 min) then quantizes (~30 s).

# %%
# Reload the merged model — Unsloth's GGUF saver expects a live model handle.
from unsloth import FastLanguageModel as FLM

model, tokenizer = FLM.from_pretrained(
    model_name=str(MERGED_PATH),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=False,    # already merged; load full precision
)

# %%
# Save GGUF in 1 quantization tier (Q4_K_M). Add more tiers below if you want the
# +3 "GGUF release published" rigor add-on.
gguf_result = model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"Unsloth GGUF result: {gguf_result!r}")
print(f"Saved GGUF Q4_K_M to {GGUF_DIR}")

# Unsloth versions differ in whether they place the result inside `GGUF_DIR`
# or next to it with the directory name as a filename prefix. Normalize both.
all_ggufs = find_gguf_files(GGUF_DIR, REPO_ROOT)
for source in all_ggufs:
    if source.parent.resolve() == GGUF_DIR.resolve():
        continue
    destination = GGUF_DIR / source.name
    if not destination.exists():
        shutil.move(str(source), str(destination))
        print(f"Moved GGUF output into {destination}")

# %% [markdown]
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = [p for p in find_gguf_files(GGUF_DIR) if is_q4_k_m(p)]
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merge_base_model": MERGE_BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
