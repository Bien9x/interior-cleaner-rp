import time
import torch
from models.diffusion.sdxl import SDXLControlnetInpaint
from models.upscale.upscaler import RealESRGAN
from utils import resize_image
from cog import BasePredictor, Input, Path
from PIL import Image


class Predictor(BasePredictor):
    ''' A predictor class that loads the model into memory and runs predictions '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_inpaint = SDXLControlnetInpaint()
        self.upscaler = RealESRGAN(scale=4, device=self.device)
        # self.client = storage.Client()
        self.max_inference_resolution = 1024

    def setup(self, weights=None):
        start_time = time.time()
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.diffusion_inpaint.setup()
        self.upscaler.load_weights()
        end_time = time.time()
        print(f"setup time: {end_time - start_time}")

    @torch.inference_mode()
    def predict(self, image: Path = Input(description="Input image"),
                mask: Path = Input(description="Mask area. White pixels are clean objects and black pixels are preserved"),
                seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None)) -> Path:
        """Run a single prediction on the model"""
        start_time = time.time()
        image = Image.open(image)
        mask = Image.open(mask).convert('L')
        width, height = image.size
        orig_resolution = min(width, height)
        if orig_resolution > self.max_inference_resolution:
            image = resize_image(image, self.max_inference_resolution)
            mask = resize_image(mask, self.max_inference_resolution)

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        new_resolution = min(image.size)

        output_image = self.diffusion_inpaint(image, mask, seed=seed)
        # output_paths = []

        if new_resolution < orig_resolution:
            output_image = self.upscaler.predict(output_image)
        output_image = output_image.resize((width, height))
        output_path = f"/tmp/out.png"
        output_image.save(output_path)
        # output_paths.append(Path(output_path))

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

        return Path(output_path)
