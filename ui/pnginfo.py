import gradio as gr
from modules.utils import get_exif_data
from modules.constants import css

with gr.Blocks(css=css) as pnginfo_tab:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil")

        with gr.Column():
            exif_output = gr.HTML(label="EXIF Data")
            #send_to_txt2img_btn = gr.Button("Send to txt2img") TODO: add send_to_txt2img_btn and its functional

    image_input.upload(get_exif_data, inputs=[image_input], outputs=exif_output)