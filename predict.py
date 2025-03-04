import time

import torch
from models.inpaint.lama import Lama
from models.diffusion.sdxl import SDXLControlnetInpaint
from models.upscale.upscaler import RealESRGAN
from models.prompting.wd_tagger import TagGenerator
from utils import resize_image
from cog import BasePredictor, Input, Path
from PIL import Image

class Predictor(BasePredictor):
    ''' A predictor class that loads the model into memory and runs predictions '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lama = Lama()
        self.diffusion_inpaint = SDXLControlnetInpaint()
        self.upscaler = RealESRGAN(scale=4, device=self.device)
        self.prompter = TagGenerator()
        # self.client = storage.Client()
        self.max_inference_resolution = 1024

    def setup(self, weights=None):
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
    def predict(self, image_path: Path = Input(
                    description="Input image",
                ), mask_image_path: Path = Input(
                    description="Mask area. White pixels are clean objects and black pixels are preserved",
                ),
                neg_prompt: str = Input(
                    description="Specify things to not see in the output",
                    default="bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth",
                ),
                guidance_scale: float = Input(
                    description="Guidance scale", ge=1, le=20, default=5.0
                ),
                controlnet_scale: float = Input(
                    description="Controlnet scale", ge=0, le=1.0, default=0.9
                ),
                strength: float = Input(
                    description="Denoising strength", ge=0, le=1.0, default=0.7
                ),
                num_steps: int = Input(
                    description="Number of denoising steps", ge=1, le=100, default=50
                ),
                num_images: int = Input(
                    description="Number of images. Higher number of outputs may OOM", ge=1, le=4, default=1
                ),
                grow_mask_by: int = Input(
                    description="Mask expansion", ge=0, le=40, default=33
                ),
                seed=Input(
                    description="Random seed. Leave blank to randomize the seed", default=None
                )):
        """Run a single prediction on the model"""
        start_time = time.time()
        image = Image.open(image_path)
        mask = Image.open(mask_image_path).convert('L')
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
        output_paths = []
        for i, image_out in enumerate(diff_images):
            if new_resolution < orig_resolution:
                image_out = self.upscaler.predict(image_out)
            image_out = image_out.resize((width, height))
            output_path = f"/tmp/out-{i}.png"
            image_out.save(output_path)
            output_paths.append(Path(output_path))

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

        return output_paths
