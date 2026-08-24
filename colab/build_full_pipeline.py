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
   "> 2. Thêm Secrets (biểu tượng 🔑 bên trái): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`\n",
   "> 3. Runtime → Run all (`Ctrl+F9`)\n",
)

# ─── SECTION 0: SECRETS & ENV ──────────────────────────────────────────────
md("---\n", "## ⚙️ Section 0: Secrets, Environment & Google Drive\n")

code(
    "# 0a. Load secrets từ Colab Secrets (nếu có)\n",
    "import os\n",
    "try:\n",
    "    from google.colab import userdata\n",
    "    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY']:\n",
    "        try:\n",
    "            os.environ[key] = userdata.get(key)\n",
    "            print(f'✓ {key} loaded')\n",
    "        except:\n",
    "            print(f'⚠ {key} not found in secrets — some bonus features will be skipped')\n",
    "except ImportError:\n",
    "    print('Not on Colab — reading from environment')\n",
)

code(
    "# 0b. Mount Google Drive để backup artifacts\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    DRIVE_OUT = '/content/drive/MyDrive/Lab22_DPO_artifacts'\n",
    "    os.makedirs(DRIVE_OUT, exist_ok=True)\n",
    "    print(f'✓ Google Drive mounted → {DRIVE_OUT}')\n",
    "except Exception as e:\n",
    "    print(f'Drive mount skipped: {e}')\n",
    "    DRIVE_OUT = None\n",
)

code(
    "# 0c. Set COMPUTE_TIER, probe GPU\n",
    "import os, torch\n",
    "os.environ['COMPUTE_TIER'] = 'T4'\n",
    "# Lab coach recommended dataset (higher quality native VN, same Alpaca format)\n",
    "os.environ['SFT_DATASET'] = 'bkai-foundation-models/vi-alpaca'\n",
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
    "import subprocess, shutil\n",
    "from pathlib import Path\n",
    "\n",
    "WORK = Path('/content/lab22')\n",
    "for d in ['notebooks','data/pref','data/eval','adapters/sft-mini','adapters/dpo',\n",
    "          'adapters/merged-fp16','gguf','submission/screenshots','bonus/data','bonus/adapters/dpo-bonus','bonus/demo']:\n",
    "    (WORK / d).mkdir(parents=True, exist_ok=True)\n",
    "os.chdir(WORK / 'notebooks')\n",
    "print(f'Working dir: {Path.cwd()}')\n",
    "print(f'Directories created under {WORK}')\n",
)

# ─── SECTION 0e: CLONE REPO & INSTALL ──────────────────────────────────────
code(
    "# 0e. Clone repo (replace <your-username> với GitHub username của bạn)\n",
    "GITHUB_USER = 'ZukaNoPro2k5'  # ← đổi thành username của bạn\n",
    "REPO_NAME = 'K4-Track3-Day22-DPO-ORPO-Alignment'\n",
    "import subprocess\n",
    "result = subprocess.run(\n",
    "    ['git', 'clone', f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git', str(WORK)],\n",
    "    capture_output=True, text=True\n",
    ")\n",
    "if result.returncode == 0:\n",
    "    print(f'✓ Cloned {GITHUB_USER}/{REPO_NAME} to {WORK}')\n",
    "else:\n",
    "    print('Repo already exists or clone failed — continuing with existing files')\n",
    "    print(result.stderr[:500])\n",
    "os.chdir(WORK / 'notebooks')\n",
)

code(
    "# 0f. Install all dependencies (3-5 min)\n",
    "import subprocess\n",
    "print('Installing dependencies... (~3-5 min)')\n",
    "subprocess.run([\n",
    "    'pip', 'install', '-q',\n",
    "    'unsloth>=2025.10,<2026.5', 'trl>=0.12,<0.20', 'peft>=0.13,<1.0',\n",
    "    'bitsandbytes>=0.44,<1.0', 'datasets>=3.1,<4.0', 'accelerate>=1.1,<2.0',\n",
    "    'jupytext>=1.16,<2.0',\n",
    "    'llama-cpp-python>=0.3,<1.0', 'lm-eval[ifeval,math]>=0.4.5,<1.0',\n",
    "    'matplotlib>=3.9,<4.0', 'pandas>=2.2,<3.0', 'pyarrow>=17,<22',\n",
    "    'openai>=1.55,<2.0', 'anthropic>=0.40,<1.0',\n",
    "    'gradio>=4.0,<5.0', 'huggingface_hub>=0.24',\n",
    "], check=True)\n",
    "print('✓ All packages installed')\n",
)

# ─── NB1 ────────────────────────────────────────────────────────────────────
md("---\n", "## 🤖 NB1 — SFT-mini Build (~10 min T4)\n")

code(
    "# NB1: Run notebook 1 — SFT mini checkpoint\n",
    "import subprocess, os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "os.environ['COMPUTE_TIER'] = os.environ.get('COMPUTE_TIER', 'T4')\n",
    "\n",
    "# Convert jupytext → ipynb\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/01_sft_mini.py')], check=True)\n",
    "\n",
    "# Execute notebook\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=900',\n",
    "    str(WORK / 'notebooks/01_sft_mini.ipynb')\n",
    "], capture_output=True, text=True)\n",
    "\n",
    "if r.returncode == 0:\n",
    "    print('✓ NB1 SFT-mini DONE')\n",
    "    print(f'   Adapter saved: {WORK}/adapters/sft-mini/')\n",
    "    print(f'   Screenshot: {WORK}/submission/screenshots/02-sft-loss.png')\n",
    "else:\n",
    "    print('✗ NB1 failed:')\n",
    "    print(r.stdout[-2000:])\n",
    "    print(r.stderr[-1000:])\n",
    "    raise RuntimeError('NB1 failed. See output above.')\n",
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
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/02_preference_data.py')], check=True)\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=300',\n",
    "    str(WORK / 'notebooks/02_preference_data.ipynb')\n",
    "], capture_output=True, text=True)\n",
    "\n",
    "if r.returncode == 0:\n",
    "    parquet = WORK / 'data/pref/train.parquet'\n",
    "    print(f'✓ NB2 DONE — {parquet} ({parquet.stat().st_size/1e6:.1f} MB)')\n",
    "else:\n",
    "    print('✗ NB2 failed:'); print(r.stderr[-1000:])\n",
    "    raise RuntimeError('NB2 failed')\n",
)

# ─── NB3 ────────────────────────────────────────────────────────────────────
md("---\n", "## 🎯 NB3 — DPO Training (~15-20 min T4)\n")

code(
    "# NB3: Run notebook 3 — DPO training\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/03_dpo_train.py')], check=True)\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=2400',\n",
    "    str(WORK / 'notebooks/03_dpo_train.ipynb')\n",
    "], capture_output=True, text=True)\n",
    "\n",
    "if r.returncode == 0:\n",
    "    import json\n",
    "    metrics = json.loads((WORK / 'adapters/dpo/dpo_metrics.json').read_text())\n",
    "    print(f'✓ NB3 DONE')\n",
    "    print(f'   Final DPO loss:   {metrics[\"final_train_loss\"]:.4f}')\n",
    "    print(f'   End reward gap:   {metrics.get(\"end_reward_gap\", \"N/A\")}')\n",
    "    print(f'   Chosen reward:    {metrics.get(\"end_chosen_reward\", \"N/A\")}')\n",
    "    print(f'   Rejected reward:  {metrics.get(\"end_rejected_reward\", \"N/A\")}')\n",
    "    print(f'   Screenshot: submission/screenshots/03-dpo-reward-curves.png')\n",
    "else:\n",
    "    print('✗ NB3 failed:'); print(r.stderr[-1000:])\n",
    "    raise RuntimeError('NB3 failed')\n",
)

# ─── NB4 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📋 NB4 — Side-by-Side Eval + Judge (~3-5 min)\n")

code(
    "# NB4: Run notebook 4 — compare and eval\n",
    "import subprocess, os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "# Pass API keys nếu có\n",
    "env = os.environ.copy()\n",
    "\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/04_compare_and_eval.py')], check=True)\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=600',\n",
    "    str(WORK / 'notebooks/04_compare_and_eval.ipynb')\n",
    "], capture_output=True, text=True, env=env)\n",
    "\n",
    "if r.returncode == 0:\n",
    "    import json\n",
    "    judge = json.loads((WORK / 'data/eval/judge_results.json').read_text())\n",
    "    from collections import Counter\n",
    "    wins = Counter(r.get('winner') for r in judge)\n",
    "    print(f'✓ NB4 DONE')\n",
    "    print(f'   SFT-only wins: {wins.get(\"A\",0)}/8')\n",
    "    print(f'   SFT+DPO wins:  {wins.get(\"B\",0)}/8')\n",
    "    print(f'   Ties:          {wins.get(\"tie\",0)}/8')\n",
    "    print(f'   Screenshots saved to submission/screenshots/')\n",
    "else:\n",
    "    print('✗ NB4 failed:'); print(r.stderr[-1000:])\n",
    "    raise RuntimeError('NB4 failed')\n",
)

# ─── NB5 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📦 NB5 — Merge + GGUF Export (Bonus +6)\n")

code(
    "# NB5: Merge adapter + export GGUF Q4_K_M\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/05_merge_deploy_gguf.py')], check=True)\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=1800',\n",
    "    str(WORK / 'notebooks/05_merge_deploy_gguf.ipynb')\n",
    "], capture_output=True, text=True)\n",
    "\n",
    "gguf_files = list((WORK / 'gguf').glob('*.gguf'))\n",
    "if r.returncode == 0 and gguf_files:\n",
    "    for f in gguf_files:\n",
    "        print(f'✓ NB5 DONE: {f.name}  ({f.stat().st_size/1e6:.0f} MB)')\n",
    "    print(f'   Screenshot: submission/screenshots/06-gguf-smoke.png')\n",
    "else:\n",
    "    print('⚠ NB5 may have failed or GGUF not found (check if llama-cpp-python compiled correctly)')\n",
    "    print(r.stderr[-500:])\n",
)

code(
    "# NB5 extra: Export Q5_K_M + Q8_0 (Bonus GGUF release +3 pts)\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "r = subprocess.run([\n",
    "    'python', str(WORK / 'scripts/merge_and_gguf.py'),\n",
    "    '--quant', 'q5_k_m',\n",
    "    '--quant', 'q8_0',\n",
    "], capture_output=True, text=True, cwd=WORK)\n",
    "\n",
    "for f in sorted((WORK / 'gguf').glob('*.gguf')):\n",
    "    print(f'  GGUF: {f.name}  ({f.stat().st_size/1e6:.0f} MB)')\n",
)

# ─── NB6 ────────────────────────────────────────────────────────────────────
md("---\n", "## 📈 NB6 — Benchmark Suite (Bonus +8, ~30 min T4)\n")

code(
    "# NB6: IFEval / GSM8K / MMLU / AlpacaEval-lite benchmark\n",
    "import subprocess, os\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'\n",
    "\n",
    "subprocess.run(['jupytext', '--to', 'notebook', '--update', str(WORK / 'notebooks/06_benchmark.py')], check=True)\n",
    "r = subprocess.run([\n",
    "    'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',\n",
    "    '--ExecutePreprocessor.timeout=5400',\n",
    "    str(WORK / 'notebooks/06_benchmark.ipynb')\n",
    "], capture_output=True, text=True, env={**os.environ})\n",
    "\n",
    "bench_json = WORK / 'data/eval/benchmark_results.json'\n",
    "if bench_json.exists():\n",
    "    import json\n",
    "    b = json.loads(bench_json.read_text())\n",
    "    print('✓ NB6 DONE — Benchmark results:')\n",
    "    for name, scores in b.get('metrics', {}).items():\n",
    "        d = b.get('deltas', {}).get(name, 'N/A')\n",
    "        arrow = '↑' if isinstance(d, float) and d > 0 else '↓' if isinstance(d, float) and d < 0 else '—'\n",
    "        print(f'   {name:18s}  SFT={scores[\"sft\"]:.3f}  DPO={scores[\"dpo\"]:.3f}  Δ={d} {arrow}')\n",
    "else:\n",
    "    print('⚠ NB6 benchmark_results.json not found. See error:')\n",
    "    print(r.stderr[-1000:])\n",
)

# ─── BETA SWEEP ─────────────────────────────────────────────────────────────
md("---\n", "## 🔬 β-sweep Mini-Experiment (Bonus +6)\n")

code(
    "# Beta-sweep: train DPO 3 lần với beta ∈ {0.05, 0.1, 0.5}\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "for beta, label in [('0.05', 'dpo-b0.05'), ('0.1', 'dpo-b0.10'), ('0.5', 'dpo-b0.50')]:\n",
    "    out_dir = WORK / 'adapters' / label\n",
    "    out_dir.mkdir(parents=True, exist_ok=True)\n",
    "    print(f'\\n=== Training β={beta} → {label} ===')\n",
    "    r = subprocess.run([\n",
    "        'python', str(WORK / 'scripts/train_dpo.py'),\n",
    "        '--beta', beta,\n",
    "        '--output-dir', str(out_dir),\n",
    "    ], capture_output=True, text=True, cwd=WORK)\n",
    "    if r.returncode == 0:\n",
    "        import json\n",
    "        m = json.loads((out_dir / 'dpo_metrics.json').read_text())\n",
    "        print(f'   Loss={m[\"final_train_loss\"]:.4f}  gap={m.get(\"end_reward_gap\",\"N/A\")}')\n",
    "    else:\n",
    "        print(f'   ⚠ Failed: {r.stderr[-300:]}')\n",
    "\n",
    "# Plot beta sweep\n",
    "r = subprocess.run([\n",
    "    'python', str(WORK / 'scripts/eval_judge.py'),\n",
    "    '--sweep-dir', str(WORK / 'adapters'),\n",
    "    '--output', str(WORK / 'submission/screenshots/bonus-beta-sweep.png'),\n",
    "], capture_output=True, text=True, cwd=WORK)\n",
    "print('\\n✓ β-sweep plot saved to submission/screenshots/bonus-beta-sweep.png')\n",
)

# ─── BONUS DOMAIN ────────────────────────────────────────────────────────────
md("---\n", "## 🧠 Bonus: Mental Health VN Domain DPO (~10 min)\n")

code(
    "# Bonus domain: train DPO trên 200 preference pairs tiếng Việt\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "r = subprocess.run(\n",
    "    ['python', str(WORK / 'bonus/train.py')],\n",
    "    capture_output=True, text=True, cwd=WORK\n",
    ")\n",
    "if r.returncode == 0:\n",
    "    print('✓ Bonus domain training DONE')\n",
    "    print(r.stdout[-500:])\n",
    "else:\n",
    "    print('⚠ Bonus training failed:')\n",
    "    print(r.stderr[-500:])\n",
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
    "import subprocess, signal, os\n",
    "proc = subprocess.Popen(\n",
    "    ['python', str(WORK / 'bonus/demo/serve.py')],\n",
    "    cwd=WORK, env={**os.environ, 'PORT': '7860'}\n",
    ")\n",
    "time.sleep(60)\n",
    "proc.terminate()\n",
    "print('Demo stopped.')\n",
)

# ─── HUGGINGFACE PUSH ────────────────────────────────────────────────────────
md("---\n", "## 🤗 HuggingFace Hub Push (Bonus Option B +5 pts)\n")

code(
    "# HuggingFace Hub Push (nếu HF_TOKEN có)\n",
    "import os, subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "HF_TOKEN = os.environ.get('HF_TOKEN')\n",
    "if not HF_TOKEN:\n",
    "    print('⚠ HF_TOKEN not set — skipping HuggingFace push (Bonus +5). Add to Colab Secrets.')\n",
    "else:\n",
    "    # Login\n",
    "    subprocess.run(['huggingface-cli', 'login', '--token', HF_TOKEN], check=True)\n",
    "\n",
    "    # Lấy HF username\n",
    "    from huggingface_hub import whoami\n",
    "    user = whoami()['name']\n",
    "    print(f'Logged in as: {user}')\n",
    "\n",
    "    # Push DPO adapter\n",
    "    REPO_DPO = f'{user}/lab22-dpo-qwen3b-vn'\n",
    "    subprocess.run([\n",
    "        'huggingface-cli', 'upload', REPO_DPO,\n",
    "        str(WORK / 'adapters/dpo'), './',\n",
    "        '--repo-type', 'model'\n",
    "    ], check=True)\n",
    "    print(f'✓ DPO adapter pushed to https://huggingface.co/{REPO_DPO}')\n",
    "\n",
    "    # Push GGUF (nếu có)\n",
    "    gguf_files = list((WORK / 'gguf').glob('*.gguf'))\n",
    "    if gguf_files:\n",
    "        REPO_GGUF = f'{user}/lab22-dpo-qwen3b-vn-gguf'\n",
    "        subprocess.run([\n",
    "            'huggingface-cli', 'upload', REPO_GGUF,\n",
    "            str(WORK / 'gguf'), './',\n",
    "            '--repo-type', 'model'\n",
    "        ], check=True)\n",
    "        print(f'✓ GGUF files pushed to https://huggingface.co/{REPO_GGUF}')\n",
    "\n",
    "    print(f'\\nAdd these links to submission/REFLECTION.md!')\n",
)

# ─── MAKE VERIFY ─────────────────────────────────────────────────────────────
md("---\n", "## ✅ Submission Verify (make verify)\n")

code(
    "# Chạy pre-submission gatekeeper\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "WORK = Path('/content/lab22')\n",
    "\n",
    "r = subprocess.run(\n",
    "    ['python', str(WORK / 'scripts/verify.py')],\n",
    "    capture_output=True, text=True, cwd=WORK\n",
    ")\n",
    "print(r.stdout)\n",
    "if r.returncode == 0:\n",
    "    print('✓ VERIFY PASSED — Ready to submit!')\n",
    "else:\n",
    "    print('✗ VERIFY FAILED — Fix the items above before submitting.')\n",
    "    print('Note: REFLECTION.md must be filled in (no placeholder <...> text)')\n",
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
    "        'adapters/dpo/dpo_metrics.json',\n",
    "        'adapters/sft-mini/adapter_config.json',\n",
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
