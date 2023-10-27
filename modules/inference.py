import os
import gradio as gr
from threading import Thread
from io import BytesIO
import base64

from prodiapy import Custom
from dotenv import load_dotenv

project_dir = os.path.join(os.path.dirname(__file__), '..', '..')
dotenv_path = os.path.join(project_dir, '.env')

load_dotenv()
pipe = Custom(os.getenv("PRODIA_API_KEY"))


def image_to_base64(image):
    # Convert the image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change format to PNG if needed

    # Encode the bytes to base64
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')  # Convert bytes to string


def txt2img(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed, batch_count):
    total_images = []
    threads = []

    def generate_one_image():
        result = pipe.create(
            "/sd/generate",
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            width=width,
            height=height,
            seed=seed
        )
        job = pipe.wait_for(result)
        total_images.append(job['imageUrl'])

    for x in range(batch_count):
        t = Thread(target=generate_one_image)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return gr.update(value=total_images, preview=False)


def img2img(input_image, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed,
            batch_count):
    if input_image is None:
        return

    total_images = []
    threads = []

    def generate_one_image():
        result = pipe.create(
            "/sd/transform",
            imageData=image_to_base64(input_image),
            denoising_strength=denoising,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            width=width,
            height=height,
            seed=seed

        )
        job = pipe.wait_for(result)
        total_images.append(job['imageUrl'])

    for x in range(batch_count):
        t = Thread(target=generate_one_image)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return gr.update(value=total_images, preview=False)


def inpainting(input_images, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed,
            batch_count, mask_blur, inpainting_fill, inpainting_mask_invert, inpainting_full_res):

    if input_images is None:
        return

    total_images = []
    threads = []

    def generate_one_image():
        result = pipe.create(
            "/sd/inpainting",
            imageData=image_to_base64(input_images['image']),
            maskData=image_to_base64(input_images['mask']),
            denoising_strength=denoising,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            width=width,
            height=height,
            seed=seed,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            inpainting_mask_invert=inpainting_mask_invert,
            inpainting_full_res=inpainting_full_res
        )
        job = pipe.wait_for(result)
        total_images.append(job['imageUrl'])

    for x in range(batch_count):
        t = Thread(target=generate_one_image)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return gr.update(value=total_images, preview=False)


def controlnet(input_images, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed,
            batch_count, controlnet_model, controlnet_module, controlnet_mode, threshold_a, threshold_b, resize_mode):

    if input_images is None:
        return

    total_images = []
    threads = []

    def generate_one_image():
        result = pipe.create(
            "/sd/inpainting",
            imageData=image_to_base64(input_images['image']),
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            width=width,
            height=height,
            seed=seed,
            controlnet_model=controlnet_model,
            controlnet_module=controlnet_module,
            controlnet_mode=controlnet_mode,
            threshold_a=threshold_a,
            threshold_b=threshold_b,
            resize_mode=resize_mode
        )
        job = pipe.wait_for(result)
        total_images.append(job['imageUrl'])

    for x in range(batch_count):
        t = Thread(target=generate_one_image)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return gr.update(value=total_images, preview=False)


def upscale(image, scale):
    if image is None:
        return

    job = pipe.create(
        '/upscale',
        imageData=image_to_base64(image),
        resize=scale
    )
    image = pipe.wait_for(job)['imageUrl']

    return image
