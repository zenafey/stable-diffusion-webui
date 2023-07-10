import gradio as gr
from fetch import get_values
from dotenv import load_dotenv
load_dotenv()
import os
import prodia
import requests
import random
from datetime import datetime
import os

prodia_key = os.getenv('PRODIA_X_KEY', None)
if prodia_key is None:
    print("Please set PRODIA_X_KEY in .env, closing...")
    exit()
client = prodia.Client(api_key=prodia_key)

def process_input_text2img(prompt, negative_prompt, steps, cfg_scale, number, seed, model, sampler, aspect_ratio, upscale, save):
    images = []
    for image in range(number):
        result = client.txt2img(prompt=prompt, negative_prompt=negative_prompt, model=model, sampler=sampler,
                                steps=steps, cfg_scale=cfg_scale, seed=seed, aspect_ratio=aspect_ratio, upscale=upscale)
        images.append(result.url)
        if save:
            date = datetime.now()
            if not os.path.isdir(f'./outputs/{date.year}-{date.month}-{date.day}'):
                os.mkdir(f'./outputs/{date.year}-{date.month}-{date.day}')
            img_data = requests.get(result.url).content
            with open(f"./outputs/{date.year}-{date.month}-{date.day}/{random.randint(1, 10000000000000)}_{result.seed}.png", "wb") as f:
                f.write(img_data)
    return images

def process_input_img2img(init, prompt, negative_prompt, steps, cfg_scale, number, seed, model, sampler, ds, upscale, save):
    images = []
    for image in range(number):
        result = client.img2img(imageUrl=init, prompt=prompt, negative_prompt=negative_prompt, model=model, sampler=sampler,
                                steps=steps, cfg_scale=cfg_scale, seed=seed, denoising_strength=ds, upscale=upscale)
        images.append(result.url)
        if save:
            date = datetime.now()
            if not os.path.isdir(f'./outputs/{date.year}-{date.month}-{date.day}'):
                os.mkdir(f'./outputs/{date.year}-{date.month}-{date.day}')
            img_data = requests.get(result.url).content
            with open(f"./outputs/{date.year}-{date.month}-{date.day}/{random.randint(1, 10000000000000)}_{result.seed}.png", "wb") as f:
                f.write(img_data)
    return images

"""
def process_input_control(init, prompt, negative_prompt, steps, cfg_scale, number, seed, model, control_model, sampler):
    images = []
    for image in range(number):
        result = client.controlnet(imageUrl=init, prompt=prompt, negative_prompt=negative_prompt, model=model, sampler=sampler,
                                steps=steps, cfg_scale=cfg_scale, seed=seed, controlnet_model=control_model)
        images.append(result.url)
    return images
"""

with gr.Blocks() as demo:
    gr.Markdown("""
    # Prodia API web-ui by @zenafey
    
    This is simple web-gui for using Prodia API easily, build on Python, gradio, prodiapy
    """)
    with gr.Tab(label="text2img"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=2)
                negative = gr.Textbox(label="Negative Prompt", lines=3, placeholder="badly drawn")

                with gr.Row():
                    steps = gr.Slider(label="Steps", value=30, step=1, maximum=50, minimum=1, interactive=True)
                    cfg = gr.Slider(label="CFG Scale", maximum=20, minimum=1, value=7, interactive=True)

                with gr.Row():
                    num = gr.Slider(label="Number of images", value=1, step=1, minimum=1, interactive=True)
                    seed = gr.Slider(label="Seed", value=-1, minimum=-1, maximum=4294967295, interactive=True)

                with gr.Row():
                    model = gr.Dropdown(label="Model", choices=get_values()[0], value="v1-5-pruned-emaonly.ckpt [81761151]", interactive=True)
                    sampler = gr.Dropdown(label="Sampler", choices=get_values()[1], value="DDIM", interactive=True)

                with gr.Row():
                    ar = gr.Radio(label="Aspect Ratio", choices=["square", "portrait", "landscape"], value="square", interactive=True)
                    with gr.Column():
                        upscale = gr.Checkbox(label="upscale", interactive=True)
                        save = gr.Checkbox(label="auto save", interactive=True)

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                result_image = gr.Gallery(label="Result Image(s)")
        run_btn.click(
            process_input_text2img,
            inputs=[
                prompt,
                negative,
                steps,
                cfg,
                num,
                seed,
                model,
                sampler,
                ar,
                upscale,
                save
            ],
            outputs=[result_image],
        )

    with gr.Tab(label="img2img"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=2)

                with gr.Row():
                    negative = gr.Textbox(label="Negative Prompt", lines=3, placeholder="badly drawn")
                    init_image = gr.Textbox(label="Init Image Url", lines=2, placeholder="https://cdn.openai.com/API/images/guides/image_generation_simple.webp")


                with gr.Row():
                    steps = gr.Slider(label="Steps", value=30, step=1, maximum=50, minimum=1, interactive=True)
                    cfg = gr.Slider(label="CFG Scale", maximum=20, minimum=1, value=7, interactive=True)

                with gr.Row():
                    num = gr.Slider(label="Number of images", value=1, step=1, minimum=1, interactive=True)
                    seed = gr.Slider(label="Seed", value=-1, minimum=-1, maximum=4294967295, interactive=True)

                with gr.Row():
                    model = gr.Dropdown(label="Model", choices=get_values()[0], value="v1-5-pruned-emaonly.ckpt [81761151]", interactive=True)
                    sampler = gr.Dropdown(label="Sampler", choices=get_values()[1], value="DDIM", interactive=True)

                with gr.Row():
                    ds = gr.Slider(label="Denoising strength", maximum=0.9, minimum=0.1, value=0.5, interactive=True)
                    with gr.Column():
                        upscale = gr.Checkbox(label="upscale", interactive=True)
                        save = gr.Checkbox(label="auto save", interactive=True)

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                result_image = gr.Gallery(label="Result Image(s)")
        run_btn.click(
            process_input_img2img,
            inputs=[
                init_image,
                prompt,
                negative,
                steps,
                cfg,
                num,
                seed,
                model,
                sampler,
                ds,
                upscale,
                save
            ],
            outputs=[result_image],
        )

    with gr.Tab(label="controlnet(coming soon)"):
        gr.Button(label="lol")


if __name__ == "__main__":
    demo.launch(show_api=True)

