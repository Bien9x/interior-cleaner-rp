import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

import config
from modules.controlnet_plus import ControlNetModel_Union, StableDiffusionXLControlNetUnionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
from utils import pil_ensure_rgb
import os

class SDXLControlnetInpaint:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self):
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        controlnet = ControlNetModel_Union.from_pretrained(config.PATH_SDXL_CONTROLNET_UNION,
                                                           torch_dtype=torch.float16, use_safetensors=True)
        self.pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0", controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe=self.pipe.to(self.device)

    def __call__(self,
                 image: Image,
                 mask: Image,
                 prompt: str,
                 neg_prompt: str = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                 guidance_scale: float = 5.0,
                 controlnet_scale: float = 0.9,
                 strength: float = 0.7,
                 num_steps: int = 50,
                 num_images: int = 1,
                 seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
        image = pil_ensure_rgb(image)
        mask = mask.convert('L')
        width, height = image.size
        ratio = np.sqrt(1024. * 1024. / (width * height))
        new_width, new_height = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
        image = image.resize((new_width, new_height))
        mask = mask.resize((new_width, new_height))

        mask_data = np.array(mask)
        controlnet_image = np.array(image)
        controlnet_image[mask_data > 0] = 0
        controlnet_image = Image.fromarray(controlnet_image)
        images = self.pipe(prompt=[prompt] * 1,
                           image=image,
                           mask_image=mask,
                           control_image_list=[0, 0, 0, 0, 0, 0, 0, controlnet_image],
                           negative_prompt=[neg_prompt] * 1,
                           guidance_scale=guidance_scale,
                           controlnet_conditioning_scale=controlnet_scale,
                           strength=strength,
                           num_images_per_prompt=num_images,
                           generator=torch.Generator().manual_seed(seed),
                           width=width,
                           height=height,
                           num_inference_steps=50,
                           union_control=True,
                           union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1]),
                           ).images
        return images
