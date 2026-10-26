# Vitality — Gradio frontend. Placeholder: the real app is built in Phase 4.
import gradio as gr

with gr.Blocks(title="Vitality") as demo:
    gr.Markdown("# Vitality\nDetection system — setup check. If you can see this page, Phase 0 is complete.")

if __name__ == "__main__":
    demo.launch()