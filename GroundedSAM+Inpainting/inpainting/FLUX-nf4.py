# pip install diffusers transformers accelerate fire bitsandbytes

from diffusers import DiffusionPipeline, FluxFillPipeline, FluxTransformer2DModel
import torch
from transformers import T5EncoderModel
from diffusers.utils import load_image
import fire


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

    pipeline.enable_model_cpu_offload()
    return pipeline.to("cuda") 


def load_conditions():
    image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
    mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")
    return image, mask


def main(four_bit: bool = True):
    pipe = load_pipeline(four_bit=four_bit)
    ckpt_id = "sayakpaul/FLUX.1-Fill-dev-nf4" if four_bit else "black-forest-labs/FLUX.1-Fill-dev"

    image, mask = load_conditions()

    result = pipe(
        prompt="a white paper cup",
        image=image,
        mask_image=mask,
        height=512,
        width=512,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0),  # Use GPU for sampling
    ).images[0]

    filename = "output_" + ckpt_id.split("/")[-1].replace(".", "_")
    filename += "_4bit" if four_bit else ""
    result.save(f"{filename}.png")
    print(f"Image saved to: {filename}.png")


if __name__ == "__main__":
    fire.Fire(main)
