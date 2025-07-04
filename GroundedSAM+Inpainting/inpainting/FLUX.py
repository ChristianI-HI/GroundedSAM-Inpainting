# Install Libraries
# !pip install accelerate transformers sentencepiece huggingface_hub openai diffusers pillow

import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from huggingface_hub import login
import os

login(token="") # @TODO: Add your Hugging Face token here

torch.cuda.empty_cache()

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

print("Loading pipeline...")
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

print("Performing inpainting...")
image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"FLUX-output.png")

print("Done!")
