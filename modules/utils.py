import re
import html
import json
import gradio as gr
from modules.constants import model_names, samplers


def load_config():
    with open("./config.json") as f:
        data = json.load(f)
    return data


def extract_data(text):
    results = {}
    patterns = {
        'prompt': r'(.*)',
        'negative_prompt': r'Negative prompt: (.*)',
        'steps': r'Steps: (\d+),',
        'seed': r'Seed: (\d+),',
        'sampler': r'Sampler:\s*([^\s,]+(?:\s+[^\s,]+)*)',
        'model': r'Model:\s*([^\s,]+)',
        'cfg_scale': r'CFG scale:\s*([\d\.]+)',
        'size': r'Size:\s*([0-9]+x[0-9]+)'
    }
    for key in ['prompt', 'negative_prompt', 'steps', 'seed', 'sampler', 'model', 'cfg_scale', 'size']:
        match = re.search(patterns[key], text)
        if match:
            results[key] = match.group(1)
        else:
            results[key] = None
    if results['size'] is not None:
        w, h = results['size'].split("x")
        results['w'] = w
        results['h'] = h
    else:
        results['w'] = None
        results['h'] = None
    return results


def place_lora(current_prompt, lora_name):
    pattern = r"<lora:" + lora_name + r":.*?>"

    if re.search(pattern, current_prompt):
        return re.sub(pattern, "", current_prompt)
    else:
        return current_prompt + " <lora:" + lora_name + ":1> "


def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


def get_exif_data(image):
    items = image.info

    info = ''
    for key, text in items.items():
        info += f"""
        <div>
        <p><b>{plaintext_to_html(str(key))}</b></p>
        <p>{plaintext_to_html(str(text))}</p>
        </div>
        """.strip() + "\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return info


def update_btn_start():
    return [
        gr.update(visible=False),
        gr.update(visible=True)
    ]


def update_btn_end():
    return [
        gr.update(visible=True),
        gr.update(visible=False)
    ]


def switch_to_t2i():
    return gr.Tabs.update(selected="t2i")


def model_validate(model_name):
    if model_name in model_names:
        return gr.update(value=model_names[model_name])
    else:
        return gr.update()


def sampler_validate(sampler_name):
    if sampler_name in samplers:
        return gr.update(value=sampler_name)
    else:
        return gr.update()


def send_to_txt2img(image):
    try:
        text = image.info['parameters']
        data = extract_data(text)

        result = [
            gr.update(value=data['prompt']),
            gr.update(value=data['negative_prompt']) if data['negative_prompt'] is not None else gr.update(),
            gr.update(value=int(data['steps'])) if data['steps'] is not None else gr.update(),
            gr.update(value=int(data['seed'])) if data['seed'] is not None else gr.update(),
            gr.update(value=float(data['cfg_scale'])) if data['cfg_scale'] is not None else gr.update(),
            gr.update(value=int(data['w'])) if data['w'] is not None else gr.update(),
            gr.update(value=int(data['h'])) if data['h'] is not None else gr.update(),
            sampler_validate(data['sampler']),
            model_validate(data['model'])
        ]
        return result

    except Exception as e:
        print(e)
        return
