import gradio as gr
import os
from modules.constants import css
from ui.model import model
from ui.text2image import t2i_tab
from ui.image2image import i2i_tab
from ui.extras import extras_tab
from ui.pnginfo import pnginfo_tab
from dotenv import load_dotenv

project_dir = os.path.join(os.path.dirname(__file__), '..', '..')
dotenv_path = os.path.join(project_dir, '.env')

load_dotenv()

with gr.Blocks(css=css, theme=os.getenv('THEME')) as demo:
    model = model.render()

    with gr.Tabs() as tabs:
        with gr.Tab("txt2img", id='t2i'):
            t2i_tab.render()

        with gr.Tab("img2img", id='i2i'):
            i2i_tab.render()

        with gr.Tab("Extras"):
            extras_tab.render()

        with gr.Tab("PNG Info"):
            pnginfo_tab.render()
