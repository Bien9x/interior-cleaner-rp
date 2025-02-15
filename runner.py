import os
import time
from typing import List

import torch

from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from models.inpaint.lama import Lama
from models.diffusion.sdxl import SDXLControlnetInpaint
from models.upscale.gan_upscaler import Upscaler
from models.prompting.wd_tagger import TagGenerator


class Predictor:
    ''' A predictor class that loads the model into memory and runs predictions '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lama = Lama()
        self.diffusion_inpaint = SDXLControlnetInpaint()
        self.upscaler = Upscaler()
        self.prompter = TagGenerator()

    def setup(self):
        start_time = time.time()
        self.lama.setup()
        self.prompter.setup()
        self.diffusion_inpaint.setup()
        self.upscaler.setup()
        # """Load the model into memory to make running multiple predictions efficient"""
        # print("Loading pipeline...")
        # safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     SAFETY_MODEL_ID,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # )
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     self.model_id,
        #     # safety_checker=safety_checker,
        #     safety_checker=None,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # ).to(self.device)
        #
        # self.pipe.enable_xformers_memory_efficient_attention()
        # # self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

        end_time = time.time()
        print(f"setup time: {end_time - start_time}")

    @torch.inference_mode()
    def predict(self, image_path, mask_image_path):
        """Run a single prediction on the model"""
        start_time = time.time()

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        # image_gan = self.lama(image)
        #
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
        # end_time = time.time()
        # print(f"inference took {end_time - start_time} time")
        # return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
