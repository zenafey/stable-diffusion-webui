import gradio as gr
from modules.constants import samplers, loras, controlnet_modules, controlnet_models
from modules.utils import place_lora




def create_default(settings=False):
    with gr.Blocks():
        with gr.Row():
            with gr.Column(scale=6, min_width=600):
                prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k",
                                    placeholder="Prompt", show_label=False, lines=3)
                negative_prompt = gr.Textbox(placeholder="Negative Prompt", show_label=False, lines=3,
                                             value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant='primary', elem_id="generate")

                stop_btn = gr.Button("Cancel", variant="stop", elem_id="generate", visible=False)

        with gr.Row():
            with gr.Column():
                with gr.Tab("Generation"):
                    if settings:
                        image_input = gr.Image(type="pil", tool="sketch")
                    else:
                        image_input = None

                    with gr.Row():
                        with gr.Column(scale=1):
                            sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Sampling Method",
                                                  choices=samplers)

                        with gr.Column(scale=1):
                            steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=50, value=25, step=1)

                    with gr.Row():
                        with gr.Column(scale=8):
                            width = gr.Slider(label="Width", maximum=1024, value=512, step=8)
                            height = gr.Slider(label="Height", maximum=1024, value=512, step=8)

                        with gr.Column(scale=1):
                            batch_size = gr.Slider(label="Batch Size", maximum=1, value=1)
                            batch_count = gr.Slider(label="Batch Count", minimum=1, maximum=4, value=1, step=1)

                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                    seed = gr.Number(label="Seed", value=-1)

                    if settings:
                        with gr.Accordion("Advanced image2image settings", open=False):
                            i2i_type = gr.Dropdown(value="img2img", label="Image2image type",
                                                   choices=['img2img', 'inpainting', 'controlnet'], interactive=True)

                            with gr.Tab("img2img"):
                                denoising = gr.Slider(label="Denoising Strength", minimum=0, maximum=1, value=0.7,
                                                      step=0.1)

                            with gr.Tab("inpainting"):
                                mask_blur = gr.Number(label="mask_blur", value=4)
                                inpainting_fill = gr.Slider(label="inpainting_fill", minimum=0, maximum=3, value=1,
                                                            step=1)
                                inpainting_mask_invert = gr.Slider(label="inpainting_mask_invert", minimum=0, maximum=1,
                                                                   step=1, value=1)
                                inpainting_full_res = gr.Checkbox(label="inpainting_full_res", value=True)

                            with gr.Tab("controlnet"):
                                controlnet_model = gr.Dropdown(value="control_v11p_sd15_canny [d14c016b]", label="controlnet_model", choices=controlnet_models)
                                controlnet_module = gr.Dropdown(value="none", label="controlnet_module", choices=controlnet_modules)
                                controlnet_mode = gr.Slider(label="controlnet_mode", minimum=0, maximum=2, value=1, step=1)
                                threshold_a = gr.Number(label="threshold_a", value=100)
                                threshold_b = gr.Number(label="threshold_b", value=200)
                                resize_mode = gr.Slider(label="resize_mode", minimum=0, maximum=2, value=1, step=1)

                    else:
                        (denoising, mask_blur, inpainting_fill, inpainting_mask_invert, inpainting_full_res,
                         controlnet_model, controlnet_module, controlnet_mode, threshold_a, threshold_b,
                         resize_mode, i2i_type) = None, None, None, None, None, None, None, None, None, None, None, None


                with gr.Tab("Lora") as loratab:
                    with gr.Row():
                        for lora in loras:
                            lora_btn = gr.Button(lora, size="sm")
                            lora_btn.click(place_lora, inputs=[prompt, lora_btn], outputs=prompt)

            with gr.Column():
                image_output = gr.Gallery(columns=3, value=[
                    "https://images.prodia.xyz/8ede1a7c-c0ee-4ded-987d-6ffed35fc477.png"])

    return (prompt, negative_prompt, generate_btn, stop_btn, sampler, steps, width, height, batch_size,
            batch_count, cfg_scale, seed, loratab, image_output, image_input, denoising, mask_blur, inpainting_fill,
            inpainting_mask_invert, inpainting_full_res, controlnet_model, controlnet_module, controlnet_mode,
            threshold_a, threshold_b, resize_mode, i2i_type)


