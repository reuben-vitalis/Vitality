"""Vitality — detection system frontend (Gradio).

Run locally:  python app.py   ->  http://127.0.0.1:7860
"""
import os
import tempfile
from pathlib import Path

import gradio as gr

from detectors import ai_image, spam, watermark

APP_TITLE = "Vitality — Detection System"

if os.getenv("SPACE_ID"):               # on Hugging Face: download/load the AI-image model at startup
    try:
        ai_image._classifier()
    except Exception as exc:
        print(f"[app] could not preload AI-image model: {exc}")


# ---------- helpers ----------
def _bar(p: float) -> str:
    """Simple text gauge, e.g. ███████░░░ 70%"""
    filled = round(p * 10)
    return "█" * filled + "░" * (10 - filled) + f" {p:.0%}"


def _error(msg: str) -> str:
    return f"### ⚠️ {msg}"


# ---------- Watermark tab ----------
def verify_watermark(path):
    if not path:
        return _error("Please upload an image first.")
    try:
        r = watermark.verify(path)
    except Exception as exc:
        return _error(str(exc))
    icon = {"authentic": "✅", "forged": "🚫", "none": "➖"}[r["status"]]
    lines = [f"## {icon} {r['message']}"]
    if r.get("ai_watermark"):
        lines.append(f"**🤖 Hidden AI watermark found:** {r['ai_watermark']}. This image was made by an AI generator.")
    if r.get("sha256"):
        lines.append(f"**SHA-256 fingerprint:** `{r['sha256']}`  \nIdentical files always share this fingerprint; any change to the file changes it.")
    return "\n\n".join(lines)


def add_watermark(path, owner):
    if not path:
        return _error("Please upload an image first."), None, None
    try:
        marked, info = watermark.embed(path, owner)
    except Exception as exc:
        return _error(str(exc)), None, None
    out = Path(tempfile.mkdtemp()) / f"{Path(path).stem}_vitality.png"
    marked.save(out, format="PNG")
    text = (f"## ✅ Watermark added\n\n"
            f"**Owner:** {info['owner']} · **Date:** {info['date']} · "
            f"**Image quality:** {info['quality_psnr_db']} dB PSNR (above 30 dB is invisible to the eye)\n\n"
            f"**SHA-256 fingerprint of the new file:** `{info['sha256']}`\n\n"
            "Download the PNG below. Sharing it as JPEG or resizing it is fine; cropping it is not.")
    return text, str(out), str(out)


# ---------- AI image tab ----------
def check_ai_image(path):
    if not path:
        return _error("Please upload an image first."), None
    try:
        r = ai_image.analyse(path)
    except Exception as exc:
        return _error(str(exc)), None
    icon = {"Likely AI-generated": "🤖", "Likely a real photo": "📷"}.get(r["verdict"], "❔")
    lines = [f"## {icon} {r['verdict']}",
             f"**AI likelihood:** `{_bar(r['ai_probability'])}`  \n*Based on: {r['basis']}*"]
    if r["evidence"]:
        lines.append("**Evidence found in the file:**\n" + "\n".join(f"- {e}" for e in r["evidence"]))
    details = [f"- Pretrained model: {'unavailable' if r['model_probability'] is None else f'{r['model_probability']:.0%} AI'}",
               f"- Frequency (FFT) artefact score: {r['fft_score']:.0%}",
               f"- Camera metadata: {r['camera'] or 'none (common after messaging apps strip it)'}"]
    lines.append("**Signal details:**\n" + "\n".join(details))
    lines.append("> This is a probability, not proof. Real photos that were heavily compressed or edited "
                 "can be mistaken for AI images.")
    return "\n\n".join(lines), r["spectrum"]


# ---------- Spam tab ----------
def check_spam(text):
    try:
        r = spam.predict(text)
    except Exception as exc:
        return _error(str(exc))
    is_spam = r["verdict"].startswith("Spam")
    lines = [f"## {'🚨' if is_spam else '✅'} {r['verdict']}",
             f"**Spam / phishing likelihood:** `{_bar(r['spam_probability'])}`"]
    if r["top_words"]:
        lines.append("**Words that triggered it:** " + ", ".join(f"`{w}`" for w in r["top_words"]))
    if r["red_flags"]:
        lines.append("**Red flags:**\n" + "\n".join(
            f"- **{name}:** {', '.join(hits)}" for name, hits in r["red_flags"].items()))
    elif not is_spam:
        lines.append("No red flags found.")
    if is_spam:
        lines.append("> Do not click links, share PINs or passwords, or send money. "
                     "Contact the organisation through its official number or website.")
    return "\n\n".join(lines)


SPAM_EXAMPLES = [
    "URGENT: Your M-Pesa account has been locked. Send your PIN within 10 minutes to avoid suspension.",
    "Congratulations! You have won KES 150,000. Click here to claim your prize before it expires today.",
    "Hi, the team meeting has moved to 3pm tomorrow in the main hall.",
    "Kesho tuko na presentation, tukutane mapema kupitia slides.",
]

ABOUT = """
## About Vitality

Vitality checks digital content with three detectors:

| Detector | How it works |
|---|---|
| **Watermark** | Hides a 20-byte message in the image's frequency patterns (DWT-DCT-SVD). The message carries an owner tag and date, **signed with HMAC-SHA256**, and protected by **Reed–Solomon error correction** so it survives JPEG compression and resizing. Also detects the public watermark Stable Diffusion adds to its images. |
| **AI image** | Combines evidence in the file (AI-tool tags, generation prompts, C2PA credentials, Stable Diffusion watermark), a pretrained Swin transformer (`Organika/sdxl-detector`) and an FFT frequency check. |
| **Spam & Phishing** | Logistic Regression on word and character patterns, trained on 644 labelled messages (English, Swahili and Sheng; 92.4% cross-validated accuracy), plus red-flag rules for urgency, authority and manipulation. |

**Limitations:** results are probabilities, not proof. The AI-image model can mistake heavily compressed real photos for AI images, and watermarks do not survive cropping.

**Privacy:** uploaded content is processed in memory to produce the result and is not stored by the app.

Built by **Reuben Vitalis** as the capstone project of the **Wecncode AI in Cybersecurity Bootcamp** (2026).
The AI-image model is licensed CC-BY-NC-3.0 (non-commercial, educational use).
"""

# ---------- Layout ----------
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown("# 🛡️ Vitality\n### Check images for watermarks and AI generation, and messages for spam and phishing.")

    with gr.Tab("💧 Watermark"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Verify an image")
                v_img = gr.Image(type="filepath", label="Image to check", sources=["upload", "clipboard"])
                v_btn = gr.Button("Check watermark", variant="primary")
            with gr.Column():
                v_out = gr.Markdown()
        v_btn.click(verify_watermark, v_img, v_out)

        gr.Markdown("---\n### Add a watermark to your own image")
        with gr.Row():
            with gr.Column():
                a_img = gr.Image(type="filepath", label="Your image (at least 256 × 256 pixels)", sources=["upload", "clipboard"])
                a_owner = gr.Textbox(label="Owner tag (1–3 letters or digits)", value="RVT", max_length=3)
                a_btn = gr.Button("Add watermark", variant="primary")
            with gr.Column():
                a_out = gr.Markdown()
                a_preview = gr.Image(label="Watermarked image", interactive=False)
                a_file = gr.File(label="Download watermarked PNG")
        a_btn.click(add_watermark, [a_img, a_owner], [a_out, a_preview, a_file])

    with gr.Tab("🤖 AI Image"):
        with gr.Row():
            with gr.Column():
                ai_img = gr.Image(type="filepath", label="Image to check", sources=["upload", "clipboard"])
                ai_btn = gr.Button("Check image", variant="primary")
                gr.Markdown("*The first check takes longer while the model loads.*")
            with gr.Column():
                ai_out = gr.Markdown()
                ai_spec = gr.Image(label="Frequency spectrum (FFT)", interactive=False)
        ai_btn.click(check_ai_image, ai_img, [ai_out, ai_spec])

    with gr.Tab("📩 Spam & Phishing"):
        with gr.Row():
            with gr.Column():
                s_text = gr.Textbox(label="Paste a message (SMS, email or chat)", lines=6,
                                    placeholder="e.g. Your account has been suspended. Click here to verify…")
                s_btn = gr.Button("Check message", variant="primary")
                gr.Examples(SPAM_EXAMPLES, s_text, label="Try an example")
            with gr.Column():
                s_out = gr.Markdown()
        s_btn.click(check_spam, s_text, s_out)

    with gr.Tab("ℹ️ About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(theme="Maani/MonoNeo")
