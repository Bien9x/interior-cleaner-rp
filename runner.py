import os
import time
from typing import List

import torch
from models.inpaint.lama import Lama
from models.diffusion.sdxl import SDXLControlnetInpaint
from models.upscale.upscaler import RealESRGAN
from models.prompting.wd_tagger import TagGenerator
from utils import load_image, convert_to_base64, resize_image
from google.cloud import storage


class Predictor:
    ''' A predictor class that loads the model into memory and runs predictions '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lama = Lama()
        self.diffusion_inpaint = SDXLControlnetInpaint()
        self.upscaler = RealESRGAN(scale=4, device=self.device)
        self.prompter = TagGenerator()
        self.client = storage.Client()
        self.max_inference_resolution = 1024

    def setup(self):
        start_time = time.time()
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.lama.setup()
        self.prompter.setup()
        self.diffusion_inpaint.setup()
        self.upscaler.load_weights()
        end_time = time.time()
        print(f"setup time: {end_time - start_time}")

    @torch.inference_mode()
    def predict(self, image_path, mask_image_path,
                neg_prompt: str = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                guidance_scale: float = 5.0,
                controlnet_scale: float = 0.9,
                strength: float = 0.7,
                num_steps: int = 50,
                num_images: int = 1,
                grow_mask_by: int = 32,
                seed=None):
        """Run a single prediction on the model"""
        start_time = time.time()
        image = load_image(image_path)
        mask = load_image(mask_image_path).convert('L')
        width, height = image.size
        orig_resolution = min(width, height)
        if orig_resolution > self.max_inference_resolution:
            image = resize_image(image, self.max_inference_resolution)
            mask = resize_image(mask, self.max_inference_resolution)

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        image_gan = self.lama(image, mask)
        prompt = "empty, " + self.prompter(image_gan)
        new_resolution = min(image_gan.size)

        diff_images = self.diffusion_inpaint(image_gan, mask, prompt, neg_prompt=neg_prompt,
                                             guidance_scale=guidance_scale, controlnet_scale=controlnet_scale,
                                             strength=strength, num_steps=num_steps, num_images=num_images,
                                             grow_mask_by=grow_mask_by, seed=seed)
        final_images = []
        for image_out in diff_images:
            if new_resolution < orig_resolution:
                image_out = self.upscaler.predict(image_out)
            image_out = image_out.resize((width, height))
            final_images.append(image_out)

        # output_paths = []
        # for i, sample in enumerate(output.images):
        #     if output.nsfw_content_detected and output.nsfw_content_detected[i]:
        #         continue
        #
        #     output_path = f"/tmp/out-{i}.png"
        #     sample.save(output_path)
        #     output_paths.append(output_path)
        #
        # if len(output_paths) == 0:
        #     raise Exception(
        #         "NSFW content detected. Try running it again, or try a different prompt.")
        end_time = time.time()
        print(f"inference took {end_time - start_time} time")
        return final_images
