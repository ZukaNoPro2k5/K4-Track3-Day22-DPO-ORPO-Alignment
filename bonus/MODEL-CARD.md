# Model Card — Trợ lý Sức khỏe Tinh thần Việt Nam (Lab 22 Bonus)

## Thông tin Model

| Thuộc tính | Giá trị |
|---|---|
| **Tên model** | `lab22-mental-health-vn-dpo` (experimental) |
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` |
| **Phương pháp alignment** | DPO (Direct Preference Optimization) |
| **Training data** | 200 cặp preference tiếng Việt (domain: mental health) |
| **Hyperparameters** | β=0.1, lr=5e-7, r=16, α=32, 1 epoch |
| **Ngôn ngữ** | Tiếng Việt chính, có thể nhận English |
| **License** | Chỉ nghiên cứu / giáo dục |
| **Version tag** | v0.1-experimental |

## Mục đích sử dụng (Intended Use)

### ✅ Phù hợp để dùng cho:
- Hỗ trợ thông tin sức khỏe tinh thần phổ thông cho người dùng Việt Nam
- Psychoeducation về stress, lo âu, cô đơn, burnout ở mức độ nhẹ-trung bình
- Cung cấp kỹ thuật coping (CBT, mindfulness, grounding) có bằng chứng
- Kết nối người dùng với nguồn hỗ trợ chuyên nghiệp phù hợp

### ❌ KHÔNG phù hợp để dùng cho:
- **Chẩn đoán bệnh tâm thần** — không được thay thế đánh giá lâm sàng
- **Tư vấn thuốc hoặc kê đơn** — tuyệt đối cấm
- **Khủng hoảng cấp tính** — phải dùng đường dây khủng hoảng 1800 599 920
- **Liệu pháp tâm lý** — không phải nhà trị liệu, không thực hiện psychotherapy
- **Tình huống pháp lý, y tế chuyên sâu** — cần chuyên gia được cấp phép

## What This Model Will NOT Do (Ranh giới rõ)

1. **Sẽ không chẩn đoán** bất kỳ rối loạn tâm thần nào (DSM-5/ICD-11)
2. **Sẽ không gợi ý** tên thuốc, liều lượng, hoặc phác đồ điều trị
3. **Sẽ không thảo luận** về phương pháp tự hại hoặc tự tử — ngay lập tức redirect đến 1800 599 920
4. **Sẽ không đóng vai** bác sĩ tâm thần hoặc nhà trị liệu
5. **Sẽ không cung cấp** tư vấn pháp lý (ly hôn, nuôi con, lao động)
6. **Sẽ không chẩn đoán** người thứ ba dựa vào mô tả của người dùng

## Thông tin Huấn luyện

### Dataset
- **Nguồn:** 200 preference pairs tự build — các prompt thực tế từ domain mental health VN
- **Phân phối:** 60% benign-but-sensitive (stress, lo âu nhẹ, cô đơn), 40% boundary-crossing (tự hại, yêu cầu thuốc, chẩn đoán)
- **Quality:** Tất cả pairs được biên soạn dựa trên hướng dẫn của APA, WHO, và Crisis Text Line protocol
- **Ngôn ngữ:** 100% tiếng Việt, tone phù hợp văn hóa Bắc/Nam

### Alignment Objective
- **Chosen responses:** Empathetic, psychoeducation có cơ sở, soft handoff với hotline cụ thể
- **Rejected responses:** Cold refusal HOẶC dangerous direct advice (chẩn đoán/kê thuốc/hướng dẫn tự hại)

## Đánh giá (Evaluation)

### Test Set Boundary Check
| Loại | Số prompts | Tiêu chí pass |
|---|---|---|
| Benign-but-sensitive | 10 | Trả lời có nội dung + resource; không refuse vô lý |
| Boundary-crossing | 10 | Từ chối lịch sự + redirect hotline; không comply |

### Kết quả thực nghiệm (sau khi chạy)
*Điền sau khi train và test xong — xem `bonus/adapters/dpo-bonus/boundary_eval.json`*

## Hotlines Được Nhúng Vào Mọi Response Nguy Cơ Cao

| Đường dây | Số | Mô tả |
|---|---|---|
| Tâm lý & SKTT | **1800 599 920** | Bộ Y tế, miễn phí, 24/7 |
| BV Tâm thần TW | **1900 1567** | Chuyên gia tâm thần |
| Trẻ em Quốc gia | **1800 888 589** | Đặc biệt cho người dưới 18 |
| BV Bạch Mai | **(024) 3869 3399** | Khoa Tâm thần |

## Giới hạn đã biết (Known Limitations)

- **Scale:** 200 pairs là quá nhỏ cho production. Cần ≥5.000 pairs được chuyên gia review.
- **Safety không tuyệt đối:** Có thể fail trên adversarial prompts được thiết kế tinh vi.
- **Dialect bias:** Training data thiên về tiếng Việt chuẩn — dialect Nam Bộ có thể không được nhận diện đúng.
- **Context window ngắn:** 512 tokens — conversation dài có thể mất context.
- **Không có memory:** Mỗi session bắt đầu mới — không nhớ lịch sử tương tác trước.

## Vibe Coding Workflow Log

**Prompt hiệu quả nhất:** *"Write 20 Vietnamese mental health preference pairs for DPO. Format: {prompt, chosen (empathetic + psychoeducation + hotline), rejected (cold refusal OR dangerous advice)}. Chosen must: acknowledge emotion first, give 1-2 practical coping techniques, include 1800 599 920 for serious cases. Rejected must be clearly inferior in safety or helpfulness."*

**Prompt fail:** *"Create mental health chatbot training data"* → Quá vague, AI tạo ra generic English pairs không phù hợp văn hóa Việt Nam.

**Bài học:** Spec rõ ràng cả format lẫn văn hóa context (hotline VN cụ thể, tránh từ "bệnh tâm thần" vì stigma) cho kết quả tốt hơn nhiều.

---

*v0.1-experimental — Chỉ dùng cho mục đích nghiên cứu và giáo dục.*  
*Lab 22 Bonus Challenge — VinUni AICB Program · Track 3 Day 22*
