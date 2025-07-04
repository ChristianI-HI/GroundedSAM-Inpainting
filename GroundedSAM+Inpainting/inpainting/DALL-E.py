# Install Libraries
# !pip install accelerate transformers sentencepiece huggingface_hub openai diffusers pytorch pillow opencv-python
# !pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

from openai import OpenAI
from PIL import Image as PILImage
import os

# Open the original image using PIL
original_image = PILImage.open("images/building.png")
original_mask_image = PILImage.open("images/building_flag.png")

# Convert the image to RGBA format (adds an alpha channel)
rgba_image = original_image.convert("RGBA")
rgba_mask_image = original_mask_image.convert("RGBA")

# Create a new black RGBA background image of size 1024x1024
bg_size = (1024, 1024)
black_bg = PILImage.new("RGBA", bg_size, (0, 0, 0, 255))
black_bg_mask = PILImage.new("RGBA", bg_size, (0, 0, 0, 255))
x = (bg_size[0] - rgba_image.width) // 2
y = (bg_size[1] - rgba_image.height) // 2
black_bg.paste(rgba_image, (x, y), rgba_image)
black_bg_mask.paste(rgba_mask_image, (x, y), rgba_mask_image)
black_bg.save("tmp_rgba.png")
black_bg_mask.save("tmp_mask_rgba.png")

client = OpenAI(
  api_key='', # @TODO: Add your OpenAI API key here
)


# Ask ChatGPT for detailed prompt

with open("tmp_rgba.png", "rb") as image_file, open("tmp_mask_rgba.png", "rb") as mask_file:
    response = client.images.edit(
        model="dall-e-2",
        image=image_file,
        mask=mask_file,
        prompt="United Kingdom Flag on flagpole",
        n=1,
        size="1024x1024"
    )

os.remove("tmp_rgba.png")
os.remove("tmp_mask_rgba.png")

image_url = response.data[0].url
print(image_url)