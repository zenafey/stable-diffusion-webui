from modules.inference import pipe
import re


def remove_id_and_ext(text):
    text = re.sub(r'\[.*\]$', '', text)
    extension = text[-12:].strip()
    if extension == "safetensors":
        text = text[:-13]
    elif extension == "ckpt":
        text = text[:-4]
    return text


models = pipe.constant("/sd/models")
loras = pipe.constant("/sd/loras")
samplers = pipe.constant("/sd/samplers")
model_names = {}

for model_name in models:
    name_without_ext = remove_id_and_ext(model_name)
    model_names[name_without_ext] = model_name

css = """
:root, .dark{
    --checkbox-label-gap: 0.25em 0.1em;
    --section-header-text-size: 12pt;
    --block-background-fill: transparent;
}
.block.padded:not(.gradio-accordion) {
    padding: 0 !important;
}
div.gradio-container{
    max-width: unset !important;
}
.compact{
    background: transparent !important;
    padding: 0 !important;
}
div.form{
    border-width: 0;
    box-shadow: none;
    background: transparent;
    overflow: visible;
    gap: 0.5em;
}
.block.gradio-dropdown,
.block.gradio-slider,
.block.gradio-checkbox,
.block.gradio-textbox,
.block.gradio-radio,
.block.gradio-checkboxgroup,
.block.gradio-number,
.block.gradio-colorpicker {
    border-width: 0 !important;
    box-shadow: none !important;
}
.gradio-dropdown label span:not(.has-info),
.gradio-textbox label span:not(.has-info),
.gradio-number label span:not(.has-info)
{
    margin-bottom: 0;
}
.gradio-dropdown ul.options{
    z-index: 3000;
    min-width: fit-content;
    max-width: inherit;
    white-space: nowrap;
}
.gradio-dropdown ul.options li.item {
    padding: 0.05em 0;
}
.gradio-dropdown ul.options li.item.selected {
    background-color: var(--neutral-100);
}
.dark .gradio-dropdown ul.options li.item.selected {
    background-color: var(--neutral-900);
}
.gradio-dropdown div.wrap.wrap.wrap.wrap{
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.gradio-dropdown:not(.multiselect) .wrap-inner.wrap-inner.wrap-inner{
    flex-wrap: unset;
}
.gradio-dropdown .single-select{
    white-space: nowrap;
    overflow: hidden;
}
.gradio-dropdown .token-remove.remove-all.remove-all{
    display: none;
}
.gradio-dropdown.multiselect .token-remove.remove-all.remove-all{
    display: flex;
}
.gradio-slider input[type="number"]{
    width: 6em;
}
.block.gradio-checkbox {
    margin: 0.75em 1.5em 0 0;
}
.gradio-html div.wrap{
    height: 100%;
}
div.gradio-html.min{
    min-height: 0;
}
#model_dd {
    width: 16%;
}
"""

controlnet_modules = [
    "none", "canny", "depth", "depth_leres", "depth_leres++", "hed", "hed_safe",
    "mediapipe_face", "mlsd", "normal_map", "openpose", "openpose_hand", "openpose_face",
    "openpose_faceonly", "openpose_full", "clip_vision", "color", "pidinet", "pidinet_safe",
    "pidinet_sketch", "pidinet_scribble", "scribble_xdog", "scribble_hed", "segmentation",
    "threshold", "depth_zoe", "normal_bae", "oneformer_coco", "oneformer_ade20k", "lineart",
    "lineart_coarse", "lineart_anime", "lineart_standard", "shuffle", "tile_resample",
    "invert", "lineart_anime_denoise", "reference_only", "reference_adain",
    "reference_adain+attn", "inpaint", "inpaint_only", "inpaint_only+lama", "tile_colorfix",
    "tile_colorfix+sharp"
]
controlnet_models = [
    "control_v11p_sd15_canny [d14c016b]", "control_v11p_sd15_openpose [cab727d4]", "control_v11p_sd15_softedge [a8575a2a]",
    "control_v11p_sd15_scribble [d4ba51ff]"
]
