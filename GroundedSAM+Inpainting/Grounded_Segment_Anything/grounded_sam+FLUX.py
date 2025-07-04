# Environment Setup
# pip install accelerate transformers sentencepiece huggingface_hub openai diffusers pillow
# pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124    # Change the torch version according to your machine
# python -m pip install -e segment_anything
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# pip install --no-build-isolation -e GroundingDINO
# git clone https://github.com/xinyu1205/recognize-anything.git
# pip install -r ./recognize-anything/requirements.txt
# pip install -e ./recognize-anything/

import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline, FluxFillPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
import torch
from PIL import Image as PILImage
import tempfile
import subprocess
import platform
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),   # Reduce the image size to 800px
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def load_pipeline(four_bit=True):
    print("Loading pipeline...")

    orig_pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )

    if four_bit:
        print("Using 4-bit NF4 model...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/FLUX.1-Fill-dev-nf4",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "sayakpaul/FLUX.1-Fill-dev-nf4",
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16
        )
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
    else:
        print("Using full precision model...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            subfolder="transformer",
            revision="refs/pr/4",
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            transformer=transformer,
            torch_dtype=torch.float16
        )

    pipeline.enable_model_cpu_offload()     # Comment out if you have enough GPU memory
    return pipeline.to("cuda")  # Change to "cpu" if you want to run on CPU offloading

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    
    # refined prompt -- cobmine after

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
      show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
      show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
      os.path.join(output_dir, "grounded_sam_output.jpg"),
      bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    # Inpaint over the masked areas one by one
    try:
      # Load FLUX inpainting pipeline
      four_bit = False  # Set to True if you want to use the nf4 model (lower VRAM usage)
      pipe = load_pipeline(four_bit=four_bit)
      ckpt_id = "sayakpaul/FLUX.1-Fill-dev-nf4" if four_bit else "black-forest-labs/FLUX.1-Fill-dev"
      
      # # Load stable diffusion inpainting pipeline
      # pipe = StableDiffusionInpaintPipeline.from_pretrained(
      #      "stabilityai/stable-diffusion-2-inpainting",
      #       torch_dtype=torch.float16,
      # )
      # pipe.to("cuda")

      # Start with the original image
      current_image = image_pil
      # original_size = current_image.size

      for idx, mask in enumerate(masks):
        # Prepare mask for inpainting
        # FLUX expects white (255) for area to change, black (0) for area to keep
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_np = mask_np[0] if mask_np.ndim == 3 else mask_np
        mask_np = (mask_np * 255).astype(np.uint8)  # 1 for mask, 0 for background
        mask_pil = PILImage.fromarray(mask_np)
        # mask_pil = mask_pil.resize(current_image.size)

        # Save each mask
        mask_save_path = os.path.join(output_dir, f"mask_{idx+1}.png")
        mask_pil.save(mask_save_path)
        
        # Show the label for the current mask/box
        label = pred_phrases[idx]
        print(f"\nMask {idx+1} - Label: {label}")
        
        # Visualize the current object to be inpainted
        temp_fig, temp_ax = plt.subplots(figsize=(6, 6))
        temp_ax.imshow(current_image)
        show_mask(mask.cpu().numpy(), temp_ax, random_color=False)
        show_box(boxes_filt[idx].numpy(), temp_ax, label)
        temp_ax.axis('off')
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_fig.savefig(tmpfile.name, bbox_inches="tight", dpi=150, pad_inches=0.0)
            plt.close(temp_fig)
            # Try to open the image using the default image viewer
            try:
              os.startfile(tmpfile.name)
            except Exception as e:
              print("Could not open image preview:", e)

        # Ask user for prompt or skip
        user_input = input(f"Enter inpainting prompt for object '{label}' (or type 'skip' to skip this mask): ")
        if user_input.strip().lower() == "skip":
            print(f"Skipping inpainting for mask {idx+1}.")
            continue
        prompt = user_input

        # Run inpainting
        inpainted = pipe(prompt=prompt, image=current_image, mask_image=mask_pil).images[0]
        
        # Resize back to original size to preserve dimensions
        # inpainted = inpainted.resize(original_size, PILImage.Resampling.LANCZOS)

        # Save inpainted result for this mask
        inpainted.save(os.path.join(output_dir, f"inpainted_output_{idx+1}.jpg"))
        print(f"Inpainting complete for mask {idx+1}. Saved to", os.path.join(output_dir, f"inpainted_output_{idx+1}.jpg"))

        # Use the inpainted image as the base for the next mask
        current_image = inpainted

      # Optionally, save the final result
      current_image.save(os.path.join(output_dir, "inpainted_output_final.jpg"))
      print("Final inpainted image saved to", os.path.join(output_dir, "inpainted_output_final.jpg"))

    except ImportError:
      print("diffusers library not found. Please install diffusers to enable inpainting.")
    except Exception as e:
      print("Inpainting failed:", e)
