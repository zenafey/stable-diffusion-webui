import gradio as gr
from modules.constants import css
from modules.utils import update_btn_start, update_btn_end
from modules import inference


with gr.Blocks(css=css) as extras_tab:

    with gr.Row():
        with gr.Tab("Single Image"):
            with gr.Column():
                upscale_image_input = gr.Image(type="pil")
                upscale_btn = gr.Button("Generate", variant="primary")
                upscale_stop_btn = gr.Button("Stop", variant="stop", visible=False)
                with gr.Tab("Scale by"):
                    upscale_scale = gr.Radio([2, 4], value=2, label="Resize")

        upscale_output = gr.Image()

    upscale_event_start = upscale_btn.click(
        fn=update_btn_start,
        outputs=[upscale_btn, upscale_stop_btn],
        queue=False
    )
    upscale_event = upscale_event_start.then(
        fn=inference.upscale,
        inputs=[upscale_image_input, upscale_scale],
        outputs=[upscale_output]
    )
    upscale_event_end = upscale_event.then(
        fn=update_btn_end,
        outputs=[upscale_btn, upscale_stop_btn],
        queue=False
    )

    upscale_stop_btn.click(fn=update_btn_end, outputs=[upscale_btn, upscale_stop_btn], cancels=[upscale_event],
                           queue=False)
