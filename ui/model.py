from modules.constants import models
import gradio as gr

model = gr.Dropdown(interactive=True, value="anything-v4.5-pruned.ckpt [65745d25]", show_label=True,
                    label="Stable Diffusion Checkpoint", choices=models, elem_id="model_dd")