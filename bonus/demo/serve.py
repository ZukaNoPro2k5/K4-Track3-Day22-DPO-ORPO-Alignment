#!/usr/bin/env python3
"""
Demo Gradio: Domain-Safe Mental Health Assistant (Tiếng Việt)
Chạy: python bonus/demo/serve.py

Cần: pip install gradio llama-cpp-python
Hoặc dùng GGUF nếu đã chạy NB5: tự động detect.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BONUS_DIR = Path(__file__).resolve().parent.parent
ADAPTER_PATH = BONUS_DIR / "adapters" / "dpo-bonus"
GGUF_DIR = REPO_ROOT / "gguf"

# Hotlines nhúng vào mọi response nguy hiểm
SAFETY_HOTLINES = """
---
📞 **Đường dây hỗ trợ khủng hoảng (miễn phí, 24/7):**
- **1800 599 920** — Tâm lý & Sức khỏe Tâm thần (Bộ Y tế)
- **1900 1567** — Bệnh viện Tâm thần Trung ương
- **1800 888 589** — Đường dây Trẻ em Quốc gia
"""

CRISIS_KEYWORDS = [
    "tự tử", "tự làm hại", "kết thúc cuộc sống", "muốn chết",
    "không muốn sống", "tự vẫn", "kết liễu", "chấm dứt tất cả",
    "làm đau bản thân", "tự làm đau"
]

BOUNDARY_KEYWORDS = [
    "kê đơn thuốc", "liều lượng thuốc", "mua thuốc", "chẩn đoán",
    "đóng vai bác sĩ", "tư vấn pháp lý", "ly hôn pháp lý"
]

SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ sức khỏe tinh thần tiếng Việt. Bạn được thiết kế để:

1. LUÔN lắng nghe thấu cảm trước — phản chiếu cảm xúc, normalize trải nghiệm
2. Cung cấp psychoeducation ngắn gọn và kỹ thuật coping thực tế (CBT, mindfulness)
3. Chuyển tiếp mềm (soft handoff) đến chuyên gia khi cần — luôn kèm hotline cụ thể

Bạn KHÔNG:
- Chẩn đoán rối loạn tâm thần
- Kê đơn thuốc hoặc gợi ý liều lượng
- Thảo luận về phương pháp tự hại
- Đóng vai nhà trị liệu hay bác sĩ

Khi gặp nguy cơ tự hại: phản hồi ngắn gọn, empathetic, cho ngay 1800 599 920.
"""


def is_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


def is_boundary(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in BOUNDARY_KEYWORDS)


def load_model():
    """Load model: ưu tiên GGUF nếu có, fallback về HF adapter."""
    # 1. Thử load GGUF
    gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
    if gguf_files:
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=str(gguf_files[0]), n_ctx=512, n_gpu_layers=-1, verbose=False)
            print(f"✓ Loaded GGUF: {gguf_files[0].name}")
            return "gguf", llm
        except Exception as e:
            print(f"GGUF load failed ({e}), trying HF adapter...")

    # 2. Fallback về HF adapter
    if ADAPTER_PATH.exists():
        import torch
        from peft import PeftModel
        from unsloth import FastLanguageModel

        COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
        BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=512, dtype=None, load_in_4bit=True
        )
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        FastLanguageModel.for_inference(model)
        print(f"✓ Loaded HF adapter: {ADAPTER_PATH}")
        return "hf", (model, tokenizer)

    # 3. Fallback về base model (no alignment)
    print("⚠ No trained adapter found. Using base model — train bonus/train.py first for aligned responses.")
    import torch
    from unsloth import FastLanguageModel
    COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=512, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return "hf_base", (model, tokenizer)


MODEL_TYPE = None
MODEL = None


def chat(message: str, history: list) -> str:
    global MODEL_TYPE, MODEL

    if MODEL is None:
        MODEL_TYPE, MODEL = load_model()

    # Crisis override — không cần gọi model
    if is_crisis(message):
        return (
            "Tôi nghe thấy bạn và tôi quan tâm đến sự an toàn của bạn.\n\n"
            "**Vui lòng gọi ngay: 📞 1800 599 920** (miễn phí, 24/7)\n\n"
            "Bạn có đang an toàn ngay lúc này không?\n"
            + SAFETY_HOTLINES
        )

    # Boundary override
    if is_boundary(message):
        return (
            "Tôi hiểu bạn đang tìm kiếm thông tin trong lĩnh vực này, nhưng đây là vùng cần chuyên gia được đào tạo.\n\n"
            "Tôi không thể cung cấp tư vấn y tế, pháp lý, hoặc chẩn đoán — không phải vì tôi không quan tâm, "
            "mà vì làm đúng điều đó đòi hỏi thông tin đầy đủ và trách nhiệm nghề nghiệp.\n\n"
            "Bạn có thể liên hệ:\n" + SAFETY_HOTLINES
        )

    # Gọi model bình thường
    if MODEL_TYPE == "gguf":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-3:]:  # Chỉ giữ 3 turns gần nhất
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})

        resp = MODEL.create_chat_completion(messages=messages, max_tokens=400, temperature=0.1)
        return resp["choices"][0]["message"]["content"]

    else:  # hf or hf_base
        model, tokenizer = MODEL
        import torch

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-3:]:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=400, do_sample=True, temperature=0.3)
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def build_app():
    import gradio as gr

    with gr.Blocks(
        title="Trợ lý Sức khỏe Tinh thần Việt Nam",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0;">
            <h1>🧠 Trợ lý Sức khỏe Tinh thần Việt Nam</h1>
            <p style="color: #666; font-size: 14px;">
                Lắng nghe thấu cảm · Psychoeducation · Kết nối chuyên gia
            </p>
        </div>
        """)

        gr.HTML("""
        <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <strong>⚠️ Lưu ý quan trọng:</strong> Đây là công cụ hỗ trợ thông tin, 
            <strong>không phải</strong> thay thế cho bác sĩ hoặc nhà trị liệu tâm lý. 
            Trong trường hợp khẩn cấp, hãy gọi ngay <strong>1800 599 920</strong> (miễn phí, 24/7).
        </div>
        """)

        chatbot = gr.Chatbot(
            height=450,
            label="Trò chuyện",
            bubble_full_width=False,
        )
        msg = gr.Textbox(
            label="Chia sẻ với tôi...",
            placeholder="Tôi đang cảm thấy...",
            lines=2,
        )

        with gr.Row():
            submit = gr.Button("Gửi", variant="primary")
            clear = gr.Button("Làm mới")

        gr.HTML(f"""
        <div style="background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; margin: 10px 0; border-radius: 4px; font-size: 13px;">
            <strong>📞 Đường dây hỗ trợ 24/7 (miễn phí):</strong><br>
            • <strong>1800 599 920</strong> — Tâm lý & Sức khỏe Tâm thần (Bộ Y tế)<br>
            • <strong>1900 1567</strong> — BV Tâm thần Trung ương<br>
            • <strong>1800 888 589</strong> — Đường dây Trẻ em Quốc gia
        </div>
        """)

        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            bot_message = chat(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

        gr.HTML("""
        <div style="text-align:center; color: #aaa; font-size: 12px; padding: 10px;">
            Lab 22 Bonus — VinUni AICB · Model: Qwen2.5-3B DPO-aligned (Mental Health VN) · 
            Chỉ dùng cho mục đích nghiên cứu và giáo dục
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("Starting Mental Health Assistant demo...")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,  # Tạo public URL cho Colab
    )
