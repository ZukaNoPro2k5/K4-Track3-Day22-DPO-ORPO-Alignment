#!/usr/bin/env python3
"""
Script tạo Lab22_FULL_PIPELINE.ipynb - Notebook Colab chạy 1 lèo từ đầu đến cuối.
Chạy: python colab/build_full_pipeline.py
Output: colab/Lab22_FULL_PIPELINE.ipynb
"""
import json
from pathlib import Path

CELLS = []

def md(*lines):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": list(lines)})

def code(*lines, tags=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": list(lines),
    }
    CELLS.append(cell)

# ─── HEADER ─────────────────────────────────────────────────────────────────
md("# 🚀 Lab 22 — DPO/ORPO Alignment: FULL PIPELINE (Run All)\n",
   "\n",
   "**Track 3 · Day 22 · VinUni AICB program**\n",
   "\n",
   "Notebook này chạy **toàn bộ** Lab 22 từ đầu đến cuối trong 1 lần bấm Run All:\n",
   "1. ⚙️ Setup & Install deps\n",
   "2. 🤖 NB1: SFT-mini build\n",
   "3. 📊 NB2: Preference data prep\n",
   "4. 🎯 NB3: DPO training + reward curves\n",
   "5. 📋 NB4: Side-by-side eval + judge\n",
   "6. 📦 NB5 (Bonus +6): Merge + GGUF export\n",
   "7. 📈 NB6 (Bonus +8): IFEval/GSM8K/MMLU/AlpacaEval benchmark\n",
   "8. 🔬 β-sweep (Bonus +6): β ∈ {0.05, 0.1, 0.5}\n",
   "9. 🧠 Bonus: Mental Health VN Domain DPO\n",
   "10. 🤗 HuggingFace Hub Push (Bonus +5 Option B)\n",
   "11. ✅ make verify (Submission gatekeeper)\n",
   "12. 💾 Sync to Google Drive\n",
   "\n",
   "> **Trước khi chạy:**\n",
   "> 1. Runtime → Change runtime type → **T4 GPU**\n",
   "> 2. Trong Colab/VS Code: thêm `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY` vào **Colab Secrets**. Khi chạy local, copy `.env.example` thành `.env`.\n",
   "> 3. Runtime → Run all (`Ctrl+F9`)\n",
)

# ─── SECTION 0: SECRETS & ENV ──────────────────────────────────────────────
md("---\n", "## ⚙️ Section 0: Secrets, Environment & Google Drive\n")

code(
    "# 0a. Load secrets/config từ Colab Secrets; không bao giờ in giá trị key\n",
    "import os\n",
    "SECRET_KEYS = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY']\n",
    "CONFIG_KEYS = ['JUDGE_MODEL', 'HF_REPO', 'HF_GGUF_REPO', 'WANDB_PROJECT',\n",
    "               'GITHUB_USER', 'GITHUB_REPO', 'DRIVE_OUT', 'BACKUP_GGUF_TO_DRIVE']\n",
    "try:\n",
    "    from google.colab import userdata\n",
    "    for key in SECRET_KEYS + CONFIG_KEYS:\n",
    "        try:\n",
    "            value = userdata.get(key)\n",
    "        except Exception:\n",
    "            value = None\n",
    "        if value:\n",
    "            os.environ[key] = value\n",
    "    print('Colab Secrets loaded: ' + ', '.join(k for k in SECRET_KEYS if os.environ.get(k)))\n",
    "except ImportError:\n",
    "    print('Not on Colab — will load local .env after the repo is available')\n",
)

code(
    "# 0b. Mount Google Drive để backup artifacts\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    DRIVE_OUT = os.environ.get('DRIVE_OUT', '/content/drive/MyDrive/Lab22_DPO_artifacts')\n",
    "    os.makedirs(DRIVE_OUT, exist_ok=True)\n",
    "    os.environ['DRIVE_OUT'] = DRIVE_OUT\n",
    "    print(f'✓ Google Drive mounted → {DRIVE_OUT}')\n",
    "except Exception as e:\n",
    "    print(f'Drive mount skipped: {e}')\n",
    "    DRIVE_OUT = None\n",
    "    os.environ.pop('DRIVE_OUT', None)\n",
)

code(
    "# 0c. Set COMPUTE_TIER, probe GPU\n",
    "import os, torch\n",
    "os.environ.setdefault('COMPUTE_TIER', 'T4')\n",
    "# Lab coach recommended dataset (higher quality native VN, same Alpaca format)\n",
    "os.environ.setdefault('SFT_DATASET', 'bkai-foundation-models/vi-alpaca')\n",
    "assert torch.cuda.is_available(), 'Enable GPU: Runtime → Change runtime type → T4 GPU'\n",
    "gpu = torch.cuda.get_device_properties(0)\n",
    "VRAM_GB = gpu.total_memory / 1e9\n",
    "print(f'GPU: {gpu.name}  ({VRAM_GB:.1f} GB)')\n",
    "if VRAM_GB >= 35:\n",
    "    os.environ['COMPUTE_TIER'] = 'BIGGPU'\n",
    "    print('→ Auto-detected BigGPU tier (A100/L4)')\n",
    "else:\n",
    "    print('→ T4 tier confirmed')\n",
    "TIER = os.environ['COMPUTE_TIER']\n",
)

code(
    "# 0d. Screenshot: GPU info (capture output as 01-setup-gpu.png)\n",
    "import os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "print('Ready to clone repo')\n",
)

# ─── SECTION 0e: CLONE REPO & INSTALL ──────────────────────────────────────
code(
    "# 0e. Clone đúng repo đã cấu hình\n",
    "GITHUB_USER = os.environ.get('GITHUB_USER', 'ZukaNoPro2k5')\n",
    "REPO_NAME = os.environ.get('GITHUB_REPO', 'K4-Track3-Day22-DPO-ORPO-Alignment')\n",
    "import subprocess, os, shutil\n",
    "if WORK.exists():\n",
    "    shutil.rmtree(WORK)\n",
    "result = subprocess.run(\n",
    "    ['git', 'clone', f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git', str(WORK)],\n",
    "    capture_output=True, text=True\n",
    ")\n",
    "if result.returncode == 0:\n",
    "    print(f'✓ Cloned {GITHUB_USER}/{REPO_NAME} to {WORK}')\n",
    "else:\n",
    "    raise RuntimeError(f'Clone failed: {result.stderr}')\n",
    "for d in ['submission/screenshots', 'adapters/sft-mini', 'adapters/dpo', 'gguf']:\n",
    "    (WORK / d).mkdir(parents=True, exist_ok=True)\n",
    "os.chdir(WORK / 'notebooks')\n",
)

code(
    "# 0f. Install dependencies từ source-of-truth requirements.txt (3-8 min)\n",
    "import subprocess, sys\n",
    "print('Installing dependencies... (~3-5 min)')\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',\n",
    "                str(WORK / 'requirements.txt')], check=True)\n",
    "print('✓ All packages installed')\n",
)

code(
    "# 0g. Load optional repo .env, validate config, and define fail-fast runner\n",
    "import os, subprocess, sys\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv(WORK / '.env', override=False)  # local/server-mounted file only\n",
    "os.environ.setdefault('SFT_DATASET', 'bkai-foundation-models/vi-alpaca')\n",
    "os.environ.setdefault('PREF_DATASET', 'argilla/ultrafeedback-binarized-preferences-cleaned')\n",
    "os.environ.setdefault('WANDB_PROJECT', 'lab22-dpo')\n",
    "if os.environ.get('WANDB_API_KEY'):\n",
    "    os.environ.setdefault('WANDB_MODE', 'online')\n",
    "\n",
    "def run_step(label, *args, check=True):\n",
    "    print(f'\\n=== {label} ===', flush=True)\n",
    "    result = subprocess.run([sys.executable, *args], cwd=WORK, env=os.environ.copy())\n",
    "    if check and result.returncode != 0:\n",
    "        raise RuntimeError(f'{label} failed with exit code {result.returncode}')\n",
    "    return result.returncode\n",
    "\n",
    "print(f\"SFT_DATASET={os.environ['SFT_DATASET']}\")\n",
    "print(f\"PREF_DATASET={os.environ['PREF_DATASET']}\")\n",
    "print('Enabled integrations: ' + (', '.join(k for k in SECRET_KEYS if os.environ.get(k)) or 'none'))\n",
    "\n",
    "# Fail early on an incompatible environment instead of after training starts.\n",
    "import shutil\n",
    "free_gb = shutil.disk_usage('/content').free / 1024**3\n",
    "assert free_gb >= 20, f'Need at least 20 GB free disk; only {free_gb:.1f} GB available'\n",
    "preflight_code = (\n",
    "    \"import unsloth, torch, trl, peft, datasets, bitsandbytes; \"\n",
    "    \"assert torch.cuda.is_available(), 'CUDA unavailable'; \"\n",
    "    \"print(f'Preflight OK: torch={torch.__version__}, trl={trl.__version__}, peft={peft.__version__}')\"\n",
    ")\n",
    "subprocess.run([sys.executable, '-c', preflight_code], cwd=WORK, check=True, env=os.environ.copy())\n",
    "print(f'Free disk: {free_gb:.1f} GB')\n",
)

# ─── NB1 ────────────────────────────────────────────────────────────────────
md("---\n", "## 🤖 NB1 — SFT-mini Build (~10 min T4)\n")

code(
    "# NB1: Run notebook 1 — SFT mini checkpoint\n",
    "import os\n",
    "os.environ.setdefault('COMPUTE_TIER', 'T4')\n",
    "os.environ.setdefault('SFT_DATASET', 'bkai-foundation-models/vi-alpaca')\n",
    "\n",
    "run_step('NB1 — SFT mini', 'notebooks/01_sft_mini.py')\n",
)

code(
    "# NB1 sanity check: verify adapter exists\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "adapter_cfg = WORK / 'adapters/sft-mini/adapter_config.json'\n",
    "assert adapter_cfg.exists(), f'Missing {adapter_cfg}'\n",
    "import json\n",
    "cfg = json.loads(adapter_cfg.read_text())\n",
    "assert cfg.get('r') == 16, f'r={cfg.get(\"r\")} != 16'\n",
    "assert cfg.get('lora_alpha') == 32, f'lora_alpha={cfg.get(\"lora_alpha\")} != 32'\n",
    "print(f'✓ NB1 check: adapter_config.json verified (r={cfg[\"r\"]}, lora_alpha={cfg[\"lora_alpha\"]})')\n",
)

# ─── NB2 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📊 NB2 — Preference Data Prep (~2 min)\n")

code(
    "# NB2: Run notebook 2 — preference data\n",
    "run_step('NB2 — preference data', 'notebooks/02_preference_data.py')\n",
)

# ─── NB3 ────────────────────────────────────────────────────────────────────
md("---\n", "## 🎯 NB3 — DPO Training (~15-20 min T4)\n")

code(
    "# NB3: Run notebook 3 — DPO training\n",
    "run_step('NB3 — DPO training', 'notebooks/03_dpo_train.py')\n",
    "\n",
    "import json\n",
    "try:\n",
    "    metrics = json.loads(open('/content/lab22/adapters/dpo/dpo_metrics.json').read())\n",
    "    print(f'✓ NB3 DONE  loss={metrics[\"final_train_loss\"]:.4f}  gap={metrics.get(\"end_reward_gap\",\"N/A\")}')\n",
    "except Exception as e:\n",
    "    print('⚠ Could not read DPO metrics, maybe training failed?')\n",
)

# ─── NB4 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📋 NB4 — Side-by-Side Eval + Judge (~3-5 min)\n")

code(
    "# NB4: Run notebook 4 — compare and eval\n",
    "run_step('NB4 — compare and judge', 'notebooks/04_compare_and_eval.py')\n",
    "\n",
    "import json\n",
    "from collections import Counter\n",
    "from pathlib import Path\n",
    "judge_f = Path('/content/lab22/data/eval/judge_results.json')\n",
    "if judge_f.exists():\n",
    "    wins = Counter(x.get('winner') for x in json.loads(judge_f.read_text()))\n",
    "    print(f'✓ NB4 DONE  SFT={wins.get(\"A\",0)}/8  DPO={wins.get(\"B\",0)}/8  tie={wins.get(\"tie\",0)}/8')\n",
)

# ─── NB5 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📦 NB5 — Merge + GGUF Export (Bonus +6)\n")

code(
    "# NB5: Merge adapter + export GGUF Q4_K_M\n",
    "run_step('NB5 — merge and Q4 GGUF', 'notebooks/05_merge_deploy_gguf.py')\n",
    "\n",
    "from pathlib import Path\n",
    "for f in Path('/content/lab22/gguf').glob('*.gguf'):\n",
    "    print(f'✓ NB5 DONE: {f.name}  ({f.stat().st_size/1e6:.0f} MB)')\n",
)

code(
    "# NB5 extra: Export Q5_K_M + Q8_0 (Bonus GGUF release +3 pts)\n",
    "run_step('NB5 extra — Q5 and Q8 GGUF', 'scripts/merge_and_gguf.py',\n",
    "         '--quant', 'q5_k_m', '--quant', 'q8_0')\n",
)

# ─── NB6 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📈 NB6 — Benchmark Suite (Bonus +8, ~30 min T4)\n")

code(
    "# NB6: IFEval / GSM8K / MMLU / AlpacaEval-lite benchmark\n",
    "import os\n",
    "os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'\n",
    "run_step('NB6 — benchmark suite', 'notebooks/06_benchmark.py')\n",
    "\n",
    "import json\n",
    "from pathlib import Path\n",
    "bench_json = Path('/content/lab22/data/eval/benchmark_results.json')\n",
    "if bench_json.exists():\n",
    "    b = json.loads(bench_json.read_text())\n",
    "    print('✓ NB6 DONE — Benchmark results:')\n",
    "    for name, scores in b.get('metrics', {}).items():\n",
    "        d = b.get('deltas', {}).get(name, 'N/A')\n",
    "        arrow = '↑' if isinstance(d, float) and d > 0 else '↓' if isinstance(d, float) and d < 0 else '—'\n",
    "        print(f'   {name:18s}  SFT={scores[\"sft\"]:.3f}  DPO={scores[\"dpo\"]:.3f}  Δ={d} {arrow}')\n",
)

# ─── BETA SWEEP ─────────────────────────────────────────────────────────────
md("---\n", "## 🔬 β-sweep Mini-Experiment (Bonus +6)\n")

code(
    "# Beta-sweep: train DPO 3 lần với beta ∈ {0.05, 0.1, 0.5}\n",
    "run_step('beta=0.05', 'scripts/train_dpo.py', '--beta', '0.05', '--output-dir', 'adapters/dpo-b0.05')\n",
    "run_step('beta=0.10', 'scripts/train_dpo.py', '--beta', '0.1', '--output-dir', 'adapters/dpo-b0.10')\n",
    "run_step('beta=0.50', 'scripts/train_dpo.py', '--beta', '0.5', '--output-dir', 'adapters/dpo-b0.50')\n",
    "\n",
    "# Plot beta sweep\n",
    "run_step('beta sweep plot', 'scripts/eval_judge.py', '--sweep-dir', 'adapters',\n",
    "         '--output', 'submission/screenshots/bonus-beta-sweep.png')\n",
)

# ─── BONUS DOMAIN ────────────────────────────────────────────────────────────
md("---\n", "## 🧠 Bonus: Mental Health VN Domain DPO (~10 min)\n")

code(
    "# Bonus domain: train DPO trên 200 preference pairs tiếng Việt\n",
    "run_step('bonus mental-health DPO', 'bonus/train.py')\n",
)

code(
    "# Bonus: Take screenshot của demo Gradio\n",
    "# (Gradio demo sẽ tạo public URL — note URL xuống, chụp màn hình)\n",
    "import threading, time\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "print('Launching Gradio demo (will get public URL)...')\n",
    "print('Copy the public URL và chụp màn hình → lưu vào submission/screenshots/bonus-creative-challenge.png')\n",
    "print()\n",
    "print('Demo sẽ chạy 60 giây rồi tự tắt. Trong thời gian đó:')\n",
    "print('1. Click vào public URL bên dưới')\n",
    "print('2. Test vài prompt tiếng Việt')\n",
    "print('3. Chụp màn hình')\n",
    "\n",
    "import subprocess, os\n",
    "proc = subprocess.Popen(\n",
    "    ['python', str(WORK / 'bonus/demo/serve.py')],\n",
    "    cwd=WORK, env={**os.environ, 'PORT': '7860'}\n",
    ")\n",
    "try:\n",
    "    time.sleep(60)\n",
    "finally:\n",
    "    proc.terminate()\n",
    "    try:\n",
    "        proc.wait(timeout=10)\n",
    "    except subprocess.TimeoutExpired:\n",
    "        proc.kill()\n",
    "    print('Demo stopped.')\n",
)

# ─── HUGGINGFACE PUSH ────────────────────────────────────────────────────────
md("---\n", "## 🤗 HuggingFace Hub Push (Bonus Option B +5 pts)\n")

code(
    "# HuggingFace Hub Push (nếu HF_TOKEN có); token không đi qua command line\n",
    "import os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "HF_TOKEN = os.environ.get('HF_TOKEN')\n",
    "if not HF_TOKEN:\n",
    "    print('⚠ HF_TOKEN not set — skipping HuggingFace push (Bonus +5). Add to Colab Secrets.')\n",
    "else:\n",
    "    from huggingface_hub import HfApi\n",
    "    try:\n",
    "        api = HfApi(token=HF_TOKEN)\n",
    "        user = api.whoami()['name']\n",
    "        print(f'Logged in as: {user}')\n",
    "        repo_dpo = os.environ.get('HF_REPO') or f'{user}/lab22-dpo-qwen3b-vn'\n",
    "        api.create_repo(repo_dpo, repo_type='model', exist_ok=True)\n",
    "        api.upload_folder(repo_id=repo_dpo, repo_type='model',\n",
    "                          folder_path=str(WORK / 'adapters/dpo'))\n",
    "        print(f'✓ DPO adapter pushed to https://huggingface.co/{repo_dpo}')\n",
    "\n",
    "        gguf_files = list((WORK / 'gguf').glob('*.gguf'))\n",
    "        if gguf_files:\n",
    "            repo_gguf = os.environ.get('HF_GGUF_REPO') or f'{user}/lab22-dpo-qwen3b-vn-gguf'\n",
    "            api.create_repo(repo_gguf, repo_type='model', exist_ok=True)\n",
    "            api.upload_folder(repo_id=repo_gguf, repo_type='model',\n",
    "                              folder_path=str(WORK / 'gguf'))\n",
    "            print(f'✓ GGUF files pushed to https://huggingface.co/{repo_gguf}')\n",
    "        print('Add these links to submission/REFLECTION.md!')\n",
    "    except Exception as exc:\n",
    "        print(f'⚠ Hugging Face push failed; Drive backup will still run: {exc}')\n",
)

# ─── MAKE VERIFY ─────────────────────────────────────────────────────────────
md("---\n", "## ✅ Submission Verify (make verify)\n")

code(
    "# Chạy pre-submission gatekeeper\n",
    "verify_rc = run_step('submission verify', 'scripts/verify.py', check=False)\n",
    "if verify_rc:\n",
    "    print('⚠ Verify chưa pass hoàn toàn; điều này bình thường nếu REFLECTION/screenshots chưa điền.')\n",
)

# ─── SYNC TO DRIVE ────────────────────────────────────────────────────────────
md("---\n", "## 💾 Sync All Artifacts to Google Drive\n")

code(
    "# Copy toàn bộ artifacts quan trọng sang Google Drive để không mất khi Colab reset\n",
    "import shutil, os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "DRIVE_OUT = os.environ.get('DRIVE_OUT', None)\n",
    "if DRIVE_OUT and Path(DRIVE_OUT).exists():\n",
    "    to_backup = [\n",
    "        'submission',\n",
    "        'data/eval',\n",
    "        'data/pref',\n",
    "        'adapters/sft-mini',\n",
    "        'adapters/dpo',\n",
    "        'adapters/dpo-b0.05',\n",
    "        'adapters/dpo-b0.10',\n",
    "        'adapters/dpo-b0.50',\n",
    "        'bonus/adapters/dpo-bonus',\n",
    "        'notebooks/01_sft_mini.ipynb',\n",
    "        'notebooks/02_preference_data.ipynb',\n",
    "        'notebooks/03_dpo_train.ipynb',\n",
    "        'notebooks/04_compare_and_eval.ipynb',\n",
    "        'notebooks/05_merge_deploy_gguf.ipynb',\n",
    "        'notebooks/06_benchmark.ipynb',\n",
    "        'bonus/demo/5-samples.md',\n",
    "        'bonus/MODEL-CARD.md',\n",
    "        'bonus/README.md',\n",
    "    ]\n",
    "    if os.environ.get('BACKUP_GGUF_TO_DRIVE', '0').lower() in {'1', 'true', 'yes'}:\n",
    "        to_backup.append('gguf')\n",
    "    drive_dir = Path(DRIVE_OUT)\n",
    "    for item in to_backup:\n",
    "        src = WORK / item\n",
    "        dst = drive_dir / item\n",
    "        if src.exists():\n",
    "            dst.parent.mkdir(parents=True, exist_ok=True)\n",
    "            if src.is_dir():\n",
    "                shutil.copytree(src, dst, dirs_exist_ok=True)\n",
    "            else:\n",
    "                shutil.copy2(src, dst)\n",
    "            print(f'✓ Backed up: {item}')\n",
    "        else:\n",
    "            print(f'⚠ Not found (skipping): {item}')\n",
    "    print(f'\\n✓ All artifacts synced to Google Drive: {DRIVE_OUT}')\n",
    "else:\n",
    "    print('Drive not mounted — skipping backup. Mount Google Drive in cell 0b.')\n",
)

# ─── NEXT STEPS ─────────────────────────────────────────────────────────────
md(
    "---\n",
    "## 📋 Bước cuối: Điền REFLECTION.md và nộp bài\n",
    "\n",
    "1. **Tải về máy** toàn bộ `submission/screenshots/` từ Google Drive (10 ảnh).\n",
    "2. **Mở** `submission/REFLECTION.md` — điền vào các ô `_<điền sau khi chạy>_` với số liệu thực tế.\n",
    "3. **Commit và push** lên GitHub repo public của bạn:\n",
    "   ```bash\n",
    "   git add -A\n",
    "   git commit -m 'Lab 22 submission — <Họ Tên>'\n",
    "   git push origin main\n",
    "   ```\n",
    "4. **Copy link GitHub** nộp vào ô submission Day 22 trên VinUni LMS.\n",
    "\n",
    "> **⚠️ Nhớ:** Repo phải **PUBLIC** đến khi điểm được công bố.\n",
)

# ─── BUILD NOTEBOOK ──────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": CELLS,
}

out = Path(__file__).parent / "Lab22_FULL_PIPELINE.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"✓ Created {out}")
print(f"  Total cells: {len(CELLS)}")
