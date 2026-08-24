# Bonus Challenge — Trợ lý Sức khỏe Tinh thần Việt Nam (Domain-Safe Mental Health Assistant)

> **Provocation #4** — Domain-safe assistant cho 1 lĩnh vực nhạy cảm.  
> **Contributors:** ZukaNoPro2k5  
> **Vibe coding log:** xem phần cuối `MODEL-CARD.md`.

---

## 4 Trục Thiết Kế

### AUDIENCE — Ai sẽ dùng?

Người dùng phổ thông Việt Nam từ 16–35 tuổi đang trải qua stress học tập, áp lực công việc, lo âu, hoặc các vấn đề tâm lý nhẹ-trung bình trong cuộc sống hằng ngày. Đây là nhóm dân số **không dám hoặc không biết cách tiếp cận chuyên gia tâm lý** do: kỳ thị xã hội ("yếu đuối"), chi phí cao (500k–1.5 triệu/buổi tại TP.HCM/Hà Nội), và thiếu thông tin về các nguồn hỗ trợ miễn phí.

Trong bối cảnh Việt Nam:
- Theo khảo sát Bộ Y tế 2022: \~3 triệu người Việt mắc trầm cảm, >95% không được điều trị.
- Tỷ lệ sinh viên đại học có biểu hiện lo âu: 35–45% (nghiên cứu VinUni 2023).
- Đường dây hỗ trợ tâm lý 1800 599 920 tiếp nhận hơn 12.000 cuộc gọi/tháng.

### DOMAIN KNOWLEDGE — Đem gì vào?

Sự khác biệt giữa AI tư vấn tâm lý **an toàn** và **nguy hiểm** nằm ở **ranh giới (boundary)**:

| Loại câu hỏi | Model NÊN làm gì | Model KHÔNG được làm gì |
|---|---|---|
| Stress học tập / công việc thông thường | Lắng nghe empathetic + kỹ thuật CBT đơn giản + nguồn tài nguyên | Chẩn đoán rối loạn |
| Triệu chứng lo âu nhẹ (khó ngủ, tim đập nhanh) | Giải thích cơ chế sinh lý + hướng dẫn hít thở 4-7-8 + đề xuất gặp bác sĩ | Kê đơn thuốc |
| Tự làm hại bản thân / có ý định tự tử | Đồng cảm ngắn + cho ngay hotline 24/7 + hỏi an toàn hiện tại | Thảo luận phương pháp hoặc từ chối lạnh lùng |
| Câu hỏi pháp lý / y tế chuyên sâu | Thừa nhận giới hạn + đề xuất chuyên gia + nguồn chính thức | Cung cấp legal/medical advice |

Domain knowledge đến từ: nghiên cứu các hướng dẫn của APA, WHO Mental Health Action Plan 2013–2030, các protocols của Crisis Text Line, và đặc biệt là **ngữ cảnh văn hóa Việt Nam** (tránh từ "bệnh tâm thần" vì kỳ thị cao, ưu tiên từ "sức khỏe tinh thần", "tâm lý", "cảm xúc").

**Hotlines chính thức VN được embed vào mọi response nguy cơ cao:**
- 📞 **1800 599 920** — Đường dây hỗ trợ tâm lý miễn phí 24/7 (Bộ Y tế)
- 📞 **1800 888 589** — Đường dây trẻ em quốc gia
- 📞 **1900 1567** — Tư vấn sức khỏe tâm thần (Bệnh viện Tâm thần Trung ương)
- 🏥 **Viện Sức khỏe Tâm thần — Bệnh viện Bạch Mai:** (024) 3869 3399

### APPLICATION OBJECTIVE — Model làm gì?

Model thực hiện chính xác 3 hành vi, theo thứ tự ưu tiên:

1. **Lắng nghe thấu cảm trước** — dùng kỹ thuật active listening, phản chiếu cảm xúc, normalize cảm xúc của người dùng (không phán xét, không thuyết giáo).
2. **Cung cấp thông tin hữu ích phù hợp phạm vi** — kỹ thuật CBT đơn giản, psychoeducation ngắn, giải thích cơ chế, bài tập thở/mindfulness ngắn.
3. **Chuyển tiếp mềm (Soft Handoff) khi cần** — đề xuất chuyên gia cụ thể (không nói chung chung "hãy gặp bác sĩ") + cung cấp hotline, đường link, địa chỉ thực tế ở Việt Nam.

**Model tuyệt đối KHÔNG:**
- Chẩn đoán rối loạn tâm thần cụ thể (DSM-5 / ICD-11)
- Gợi ý hoặc thảo luận về phương pháp tự hại
- Kê đơn thuốc hoặc liều lượng thuốc
- Thay thế nhà trị liệu tâm lý chuyên nghiệp

### REAL-WORLD OUTPUT — Deliverable ship được

```
bonus/
├── README.md                    # File này (≥400 từ)
├── data/
│   ├── prompts.jsonl            # 200 prompts tiếng Việt từ domain thực tế
│   └── pairs.parquet            # 200 preference pairs đã format cho DPO
├── train.py                     # Script DPO training cho domain bonus
├── adapters/dpo-bonus/          # (gitignore) Output adapter sau align
├── demo/
│   ├── serve.py                 # Gradio app tương tác, chạy được ngay
│   └── 5-samples.md             # 5 mẫu đối thoại Before/After DPO
└── MODEL-CARD.md                # Model card ship-ready
```

**Demo:** `pip install gradio llama-cpp-python && python bonus/demo/serve.py`  
Người dùng gõ vào text box → model trả lời → nếu trigger từ nguy hiểm → hiển thị hotline tự động.

---

## Test Set Validation (Rubric §4)

Tập kiểm tra 20 prompts:
- **10 benign-but-sensitive** (model NÊN trả lời với psychoeducation + resource): lo âu thi cử, stress công việc, mất ngủ nhẹ, xung đột quan hệ bạn bè/gia đình.
- **10 boundary-crossing** (model PHẢI từ chối lịch sự + chuyển tiếp ngay): có ý định tự hại, hỏi liều thuốc an thần, chẩn đoán trầm cảm cho bạn bè, pháp lý ly hôn...

Precision/Recall trên 2 tập được báo cáo trong `MODEL-CARD.md`.

---

## "Empathetic Refusal" trong Tiếng Việt — Design Decision

Từ chối lạnh kiểu US "I can't help with that" rất phản cảm trong văn hóa Việt Nam.  
Model này sử dụng cấu trúc từ chối 3 bước:
1. **Nhận ra:** Acknowledge cảm xúc người dùng trước.
2. **Giải thích:** Nói rõ giới hạn của mình một cách tự nhiên (không phán xét).
3. **Kết nối:** Đưa ngay nguồn hỗ trợ phù hợp thay thế.

Ví dụ: *"Tôi hiểu bạn đang rất mệt mỏi với những cảm xúc này. Phần này cần được chia sẻ với một chuyên gia để bạn nhận được hỗ trợ tốt nhất — bạn có thể gọi ngay 1800 599 920 (miễn phí, 24/7). Tôi ở đây nếu bạn muốn chia sẻ thêm bất cứ điều gì trong khi chờ."*

---

## Honest Limitations

- **POC scale:** 200 preference pairs là quá ít để generalize. Production cần ≥5.000 pairs được chuyên gia tâm lý review.
- **Safety không tuyệt đối:** Model có thể fail trên adversarial prompts. Không deploy cho người dùng đang trong khủng hoảng cấp tính mà không có human-in-the-loop.
- **Tiếng Việt vùng miền:** Pairs data thiên về tiếng Việt chuẩn Bắc. Các từ lóng hoặc dialect Nam Bộ có thể không được nhận diện đúng.
- **License:** Chỉ dùng cho mục đích nghiên cứu và giáo dục. Không phải thiết bị y tế theo định nghĩa của Bộ Y tế Việt Nam.
