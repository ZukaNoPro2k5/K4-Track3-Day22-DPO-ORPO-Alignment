# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Họ Tên>_
**Cohort:** _<A20-K1 / A20-K2 / ...>_
**Tier đã chạy:** _<T4 | BIGGPU | both>_
**Date:** _<YYYY-MM-DD>_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | _<e.g., Free Colab T4 16GB>_ |
| CUDA / driver | _<e.g., CUDA 12.1, driver 535>_ |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Free Colab T4) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _<điền sau khi chạy, e.g., 18 min>_ |
| VRAM peak | _<e.g., 10.4 GB>_ | _<e.g., 13.8 GB>_ |
| Final loss | _<e.g., 1.82 (SFT)>_ | _<e.g., 0.48 (DPO)>_ |
| Reward gap (chosen − rejected, end of training) | n/a | _<điền từ adapters/dpo/dpo_metrics.json>_ |
| Mean output length | _<e.g., 142 tokens>_ | _<e.g., 87 tokens>_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Ảnh:** `submission/screenshots/03-dpo-reward-curves.png`

Trong Lab này, 2 đường reward curves ghi lại diễn biến của `chosen_rewards` và `rejected_rewards` trong suốt quá trình DPO training cung cấp bức tranh quan trọng nhất về việc alignment có hoạt động như mong muốn hay không.

**Reward gap (chosen − rejected):** Đường gap cho thấy DPO đang học để phân biệt giữa response được ưu tiên và response bị loại bỏ. Gap dương và tăng dần là tín hiệu tích cực — trainer đang thành công ép model gán log-probability cao hơn cho chosen so với rejected.

**Chosen rewards trajectory:** Theo lý thuyết DPO (Rafailov et al. 2023), chosen reward = log(π_θ(y_c|x) / π_ref(y_c|x)) là implicit reward cho response được ưa thích. Trong nhiều trường hợp thực tế, chosen reward không nhất thiết tăng — nó có thể ổn định hoặc giảm nhẹ.

**Rejected rewards trajectory:** Rejected reward thường giảm mạnh hơn chosen reward. Điều này phản ánh DPO đang "đẩy xuống" probability của rejected responses một cách chủ động hơn là "kéo lên" chosen.

**Hiện tượng Likelihood Displacement (deck §3.4):** Nếu quan sát thấy chosen reward GIẢM trong khi reward gap vẫn tăng — đây chính xác là Likelihood Displacement được Razin et al. 2024 mô tả. Cơ chế: model tìm ra shortcut bằng cách hạ log-probability của CÙNG 1 token pattern xuất hiện ở cả chosen lẫn rejected, thay vì học content khác biệt thực sự. Gap tăng không phải vì chosen tốt hơn, mà vì rejected tệ hơn. Đây không phải bug nhưng là warning — output sau align có thể trở nên ngắn hơn và kém đa dạng.

_→ Điền phân tích cụ thể dựa trên đồ thị thực tế của bạn (≥ 100 words total)._

---

## 4. Qualitative comparison (≥ 8 examples)

> **Ảnh:** `submission/screenshots/04-side-by-side-table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích quicksort... | _<điền>_ | _<điền>_ | _<SFT \| DPO \| tie>_ |
| 2 | helpfulness | Gợi ý món nấu... | _<điền>_ | _<điền>_ | |
| 3 | helpfulness | Viết email xin nghỉ... | _<điền>_ | _<điền>_ | |
| 4 | helpfulness | So sánh Python vs JS... | _<điền>_ | _<điền>_ | |
| 5 | safety | Công thức hoá chất nổ... | _<điền>_ | _<điền>_ | |
| 6 | safety | Tin nhắn khủng bố... | _<điền>_ | _<điền>_ | |
| 7 | safety | Mua rượu vị thành niên... | _<điền>_ | _<điền>_ | |
| 8 | safety | Tự kết liễu cuộc sống... | _<điền>_ | _<điền>_ | |

**Win/loss/tie summary:** _<điền sau khi chạy NB4, e.g., SFT+DPO wins 5/8, ties 2/8, loses 1/8>_

**Judge used:** _<gpt-4o-mini | claude-haiku-4-5 | manual rubric>_

---

## 5. β trade-off

_Nếu đã chạy β-sweep bonus (rigor add-on +6):_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _<điền>_ | _<điền>_ | _<điền>_ | Aggressive alignment |
| 0.1 (default) | _<điền>_ | _<điền>_ | _<điền>_ | Deck §5.2 baseline |
| 0.5 | _<điền>_ | _<điền>_ | _<điền>_ | Conservative |

**Phân tích lý thuyết và dự đoán (nếu chưa chạy sweep):**

Theo deck §3.3, β trong DPO đóng vai trò regularization strength — kiểm soát mức độ policy được phép lệch xa khỏi reference model. Dự đoán cho 3 mức β:

- **β = 0.05 (aggressive):** Reward gap sẽ lớn hơn, nhưng likelihood displacement cũng nặng hơn. Model học nhanh preference pattern nhưng có thể bị degenerate — output ngắn hơn, kém đa dạng, đôi khi repetitive.
- **β = 0.1 (default — deck §5.2):** Trade-off được cân bằng. Đây là điểm được tối ưu hóa trên UltraFeedback trong deck demo → reward gap tích cực nhưng chosen reward không bị displacement quá mạnh.
- **β = 0.5 (conservative):** Reward gap nhỏ hơn, model giữ closer to reference — ít alignment nhưng cũng ít risk degeneration. Phù hợp khi reference model đã tốt và chỉ cần fine-tune nhẹ.

Hình dạng dự đoán của reward gap vs β: monotonic giảm (β tăng → gap giảm) nhưng với diminishing returns.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> **Quyết định:** Chọn Tier T4 (Qwen2.5-3B-bnb-4bit) thay vì BigGPU (Qwen2.5-7B-bnb-4bit)

**Alternative đã xem xét:** Chạy BigGPU tier với Qwen2.5-7B để có model mạnh hơn và faithful hơn với demo trong deck (deck §7.1 dùng A100 + 7B). BigGPU tier cho phép 5k preference pairs thay vì 2k, và timing ngắn hơn (~12 phút vs ~18 phút trên T4).

**Lý do chọn T4:** Đây là tier miễn phí và accessible cho hầu hết học viên trong cohort. Mục tiêu của lab là học *quy trình và concepts* của DPO — chọn T4 để đảm bảo kết quả có thể reproduce bởi bất kỳ học viên nào với Free Colab, không phụ thuộc vào việc có GPU cloud hay Colab Pro. Ngoài ra, VRAM math của DPO (2 forward passes + giữ cả chosen và rejected) làm T4 + 7B không khả thi — phải dùng 3B.

**Kết quả xác nhận hay ngạc nhiên:** _<điền sau khi chạy — ví dụ: "Reward gap đạt X sau Y steps, cho thấy DPO hoạt động ngay cả trên 3B model nhỏ. Ngạc nhiên là output DPO có xu hướng ngắn hơn SFT đáng kể — điều này consistent với likelihood displacement trong deck §3.4">_

**Nếu làm lại:** _<điền sau khi chạy — ví dụ: "Tôi sẽ thử tăng DPO_SLICE lên 4000 pairs thay vì 2000 để xem reward gap có ổn định hơn không. Cũng sẽ thử β = 0.05 ngay từ đầu dựa trên lý thuyết vì với 3B model và 1k SFT data, aggressive alignment có thể cần thiết hơn để thấy rõ tác động">_

_→ Điền phân tích hoàn chỉnh dựa trên trải nghiệm thực tế (≥ 150 words total)._

---

## 7. Benchmark interpretation (≥ 150 words)

> **Ảnh:** `submission/screenshots/07-benchmark-comparison.png`

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _<điền>_ | _<điền>_ | _<điền>_ |
| GSM8K | _<điền>_ | _<điền>_ | _<điền>_ |
| MMLU (sampled) | _<điền>_ | _<điền>_ | _<điền>_ |
| AlpacaEval-lite | _<điền>_ | _<điền>_ | _<điền>_ |

**Phân tích lý thuyết về Alignment Tax (deck §8.1):**

DPO alignment thường tạo ra trade-off rõ ràng giữa các loại benchmark khác nhau — đây là hiện tượng được gọi là "alignment tax" trong cộng đồng nghiên cứu.

**IFEval (Instruction Following):** Đây là benchmark được kỳ vọng SẼ tăng sau DPO. IFEval đo khả năng model follow format instruction chính xác (số từ, bullet points, v.v.) — đúng loại behavior mà preference data (UltraFeedback) có abundance. Nếu DPO hoạt động đúng, IFEval tăng là dấu hiệu tốt nhất.

**GSM8K (Math reasoning):** Đây là benchmark hay BỊ GIẢM sau DPO — alignment tax kinh điển. Lý do: (1) Chat-aligned models học output ngắn hơn và conversational hơn, trong khi GSM8K cần chain-of-thought dài; (2) Preference data (UltraFeedback) thiên về helpfulness và safety thay vì mathematical reasoning; (3) VRAM dành cho format learning thay vì reasoning capacity. Tulu 3 thậm chí báo cáo RLVR (DPO-style on math) chỉ tăng được sau khi thêm math-specific preference data.

**MMLU (Factual knowledge):** Kỳ vọng FLAT hoặc giảm nhẹ (<2pp). DPO không dạy thêm facts — nó thay đổi how model present facts. Nếu MMLU giảm >5pp, đó là dấu hiệu catastrophic forgetting → giảm β hoặc giảm epochs.

**AlpacaEval-lite (Judge-based):** Nếu consistency với NB4 judge eval — DPO tổng quát tốt. Nếu AlpacaEval-lite cao hơn NB4 win-rate nhiều, có thể prompt distribution bias (AlpacaEval-lite thiên helpfulness, không có safety prompts).

_→ Điền phân tích cụ thể dựa trên số liệu thực tế của bạn sau khi chạy NB6 (≥ 150 words total)._

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [x] Đã làm `BONUS-CHALLENGE.md` provocation #4 — Mental Health VN Assistant (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu — điền sau khi chạy)_
