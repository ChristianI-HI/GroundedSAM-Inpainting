# Install Libraries
# !pip install accelerate transformers sentencepiece huggingface_hub openai diffusers pytorch pillow

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

prompt = "United Kingdom Flag on flagpole"

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open("images/building.png")
mask_image = Image.open("images/building_flag.png")

# Convert mask from alpha (inpaint: alpha>0, keep: alpha==0) to black/white (swap: inpaint=black, keep=white)
if mask_image.mode != "RGBA":
  mask_image = mask_image.convert("RGBA")
alpha = mask_image.split()[-1]

# Create a new mask: black (0) where alpha > 0 (inpaint), white (255) where alpha == 0 (keep)
mask_np = np.array(alpha)
binary_mask = np.where(mask_np > 0, 0, 255).astype(np.uint8)
mask_image = Image.fromarray(binary_mask, mode="L")

mask_image.show()

result_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
result_image.save("./SD_output.png")