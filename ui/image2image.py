import gradio as gr
from ui.model import model
from modules.constants import css
from modules.utils import update_btn_start, update_btn_end
from modules import inference
from ui.templates import create_default

with gr.Blocks(css=css) as i2i_tab:
        (prompt, negative_prompt, generate_btn, stop_btn, sampler, steps, width, height, batch_size, batch_count,
         cfg_scale, seed, loratab, image_output, image_input, denoising, mask_blur, inpainting_fill, inpainting_mask_invert,
         inpainting_full_res, controlnet_model, controlnet_module, controlnet_mode, threshold_a, threshold_b,
         resize_mode, type) = create_default(True)

        if type == "img2img":
            fn = inference.img2img
            inputs = [image_input, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height,
                      seed, batch_count]
        elif type == "inpainting":
            fn = inference.inpainting
            inputs = [image_input, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height,
                      seed, batch_count, mask_blur, inpainting_fill, inpainting_mask_invert, inpainting_full_res]
        elif type == "controlnet":
            fn = inference.controlnet
            inputs = [image_input, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed,
                      batch_count, controlnet_model, controlnet_module, controlnet_mode, threshold_a,
                      threshold_b, resize_mode]
        else:
            fn = inference.img2img
            inputs = [image_input, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height,
                      seed, batch_count]

        event_start = generate_btn.click(
            update_btn_start,
            outputs=[generate_btn, stop_btn],
            queue=False
        )
        event = event_start.then(fn,
                                 inputs=inputs,
                                 outputs=[image_output])
        event_end = event.then(
            update_btn_end,
            outputs=[generate_btn, stop_btn],
            queue=False
        )

        stop_btn.click(fn=update_btn_end, outputs=[generate_btn, stop_btn], cancels=[event],
                           queue=False)