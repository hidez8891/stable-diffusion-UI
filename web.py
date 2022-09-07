import gradio as gr

import random
import re
import gc
from io import BytesIO
import base64

from PIL import Image
import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline


pipe_type = ''
pipe = None


def clear_memory():
  gc.collect()
  torch.cuda.empty_cache()


def copy2src(images):
  if images == None or len(images) == 0:
    return [None, None]
  try:
    image_data = re.sub('^data:image/.+;base64,', '', images[0])
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    return [image, image]
  except IndexError:
    print("failed to get image")
  return [None, None]


def txt2img(prompt, width, height, guidance_scale, steps, seed):
  global pipe, pipe_type

  if pipe_type != 'txt2img':
    pipe = None
    clear_memory()

    pipe_type = 'txt2img'
    pipe = StableDiffusionPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      revision="fp16",
      torch_dtype=torch.float16,
      use_auth_token=True
    ).to("cuda")

  seed = random.randint(0, 2**32) if seed == -1 else seed
  generator = torch.Generator("cuda").manual_seed(int(seed))

  pipe.enable_attention_slicing()
  with autocast("cuda"):
    image = pipe(prompt=prompt,
                 height=height, width=width,
                 num_inference_steps=steps, guidance_scale=guidance_scale,
                 generator=generator).images[0]

  return [[image], seed]


def img2img(init_img, prompt, strength, guidance_scale, steps, seed):
  global pipe, pipe_type

  if pipe_type != 'img2img':
    pipe = None
    clear_memory()

    pipe_type = 'img2img'
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      revision="fp16",
      torch_dtype=torch.float16,
      use_auth_token=True
    ).to("cuda")

  init_img = Image.fromarray(init_img)

  seed = random.randint(0, 2**32) if seed == -1 else seed
  generator = torch.Generator("cuda").manual_seed(int(seed))

  pipe.enable_attention_slicing()
  with autocast("cuda"):
    image = pipe(prompt=prompt, init_image=init_img,
                 strength=strength,
                 num_inference_steps=steps, guidance_scale=guidance_scale,
                 generator=generator).images[0]

  return [[image], seed]


def inpaint(inputs, prompt, strength, guidance_scale, steps, seed):
  global pipe, pipe_type

  if pipe_type != 'inpaint':
    pipe = None
    clear_memory()

    pipe_type = 'inpaint'
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      revision="fp16",
      torch_dtype=torch.float16,
      use_auth_token=True
    ).to("cuda")

  init_img = Image.fromarray(inputs['image'])
  mask_img = Image.fromarray(inputs['mask'])

  seed = random.randint(0, 2**32) if seed == -1 else seed
  generator = torch.Generator("cuda").manual_seed(int(seed))

  pipe.enable_attention_slicing()
  with autocast("cuda"):
    image = pipe(prompt=prompt, init_image=init_img, mask_image=mask_img,
                 strength=strength,
                 num_inference_steps=steps, guidance_scale=guidance_scale,
                 generator=generator).images[0]

  return [[image], seed]


with gr.Blocks() as demo:
  prompt = gr.Textbox(label="Prompt")

  with gr.Tab("txt2img"):
    with gr.Row():
      with gr.Column():
        with gr.Row():
          with gr.Column():
            t2i_width  = gr.Slider(label="Width", value=512, step=64, minimum=256, maximum=1024)
            t2i_height = gr.Slider(label="Height", value=512, step=64, minimum=256, maximum=1024)
            t2i_scale  = gr.Slider(label="Scale", value=7.5, step=0.5, minimum=0, maximum=20)
            t2i_steps  = gr.Slider(label="Steps", value=50, step=10, minimum=10, maximum=300)
            t2i_seed   = gr.Number(label="Seed", value=-1)
            t2i_gen    = gr.Button("Generate", variant="primary")

      with gr.Column():
        t2i_gallery  = gr.Gallery(elem_id="gallery").style(height="640px", container=True)
        t2i_seed_res = gr.Number(label="Use Seed", value=-1, interactive=False)
        t2i_copy     = gr.Button("copy to source", variant="secondary")

  t2i_gen.click(fn=txt2img,
                inputs=[prompt, t2i_width, t2i_height, t2i_scale, t2i_steps, t2i_seed],
                outputs=[t2i_gallery, t2i_seed_res])

  with gr.Tab("img2img"):
    with gr.Row():
      with gr.Column():
        with gr.Row():
          with gr.Column():
            i2i_init_img = gr.Image(label="Image", value=None, interactive=True, tool="select")
            i2i_strength = gr.Slider(label="Strength", value=0.6, step=0.05, minimum=0.05, maximum=1)
        with gr.Row():
          with gr.Column():
            i2i_scale = gr.Slider(label="Scale", value=7.5, step=0.5, minimum=0, maximum=20)
            i2i_steps = gr.Slider(label="Steps", value=50, step=10, minimum=10, maximum=300)
            i2i_seed  = gr.Number(label="Seed", value=-1)
            i2i_gen   = gr.Button("Generate", variant="primary")

      with gr.Column():
        i2i_gallery  = gr.Gallery(elem_id="gallery").style(height="640px", container=True)
        i2i_seed_res = gr.Number(label="Use Seed", value=-1, interactive=False)
        i2i_copy     = gr.Button("copy to source", variant="secondary")

  i2i_gen.click(fn=img2img,
                inputs=[i2i_init_img, prompt, i2i_strength, i2i_scale, i2i_steps, i2i_seed],
                outputs=[i2i_gallery, i2i_seed_res])

  with gr.Tab("inpaint"):
    with gr.Row():
      with gr.Column():
        with gr.Row():
          with gr.Column():
            inp_init_img = gr.Image(label="Image", value=None, interactive=True, tool="sketch")
            inp_strength = gr.Slider(label="Strength", value=0.6, step=0.05, minimum=0.05, maximum=1)
        with gr.Row():
          with gr.Column():
            inp_scale = gr.Slider(label="Scale", value=7.5, step=0.5, minimum=0, maximum=20)
            inp_steps = gr.Slider(label="Steps", value=50, step=10, minimum=10, maximum=300)
            inp_seed  = gr.Number(label="Seed", value=-1)
            inp_gen   = gr.Button("Generate", variant="primary")

      with gr.Column():
        inp_gallery  = gr.Gallery(elem_id="gallery").style(height="640px", container=True)
        inp_seed_res = gr.Number(label="Use Seed", value=-1, interactive=False)
        inp_copy     = gr.Button("copy to source", variant="secondary")

  inp_gen.click(fn=inpaint,
                inputs=[inp_init_img, prompt, inp_strength, inp_scale, inp_steps, inp_seed],
                outputs=[inp_gallery, inp_seed_res])

  t2i_copy.click(fn=copy2src,
                 inputs=[t2i_gallery],
                 outputs=[i2i_init_img, inp_init_img])
  i2i_copy.click(fn=copy2src,
                 inputs=[i2i_gallery],
                 outputs=[i2i_init_img, inp_init_img])
  inp_copy.click(fn=copy2src,
                 inputs=[inp_gallery],
                 outputs=[i2i_init_img, inp_init_img])

demo.launch(debug=True)

