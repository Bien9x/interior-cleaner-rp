from PIL import Image
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur

import config
from utils import pil_ensure_rgb
import os


class SDXLControlnetInpaint:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.base_model_name = 'SG161222/RealVisXL_V5.0'

    def setup(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.pipe = DiffusionPipeline.from_pretrained(
            self.base_model_name,
            custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser.py",
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.dtype,
            cache_dir=config.CACHE_DIR,
            local_files_only=True,
        ).to(device)
        self.pipe = self.pipe.to(self.device)

    def preprocess_image(self, image: Image.Image):
        image = to_tensor(image)
        image = image.unsqueeze_(0).float() * 2 - 1  # [0,1] --> [-1,1]
        if image.shape[1] != 3:
            image = image.expand(-1, 3, -1, -1)
        image = F.interpolate(image, (1024, 1024))
        image = image.to(self.dtype).to(self.device)
        return image

    def preprocess_mask(self, mask: Image.Image):
        mask = to_tensor(mask)
        mask = mask.unsqueeze_(0).float()  # 0 or 1
        mask = F.interpolate(mask, (1024, 1024))
        mask = gaussian_blur(mask, kernel_size=(77, 77))
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(self.dtype).to(self.device)
        return mask

    def __call__(self,
                 image: Image,
                 mask: Image,
                 seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
        prompt = ""  # Set prompt to null
        image = pil_ensure_rgb(image)
        mask = mask.convert('L')
        width, height = image.size
        # ratio = np.sqrt(1024. * 1024. / (width * height))
        # new_width, new_height = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
        # image = image.resize((new_width, new_height))
        # mask = mask.resize((new_width, new_height))
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)
        image = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=1024,
            width=1024,
            AAS=True,  # enable AAS
            strength=0.8,  # inpainting strength
            rm_guidance_scale=9,  # removal guidance scale
            ss_steps=9,  # similarity suppression steps
            ss_scale=0.3,  # similarity suppression scale
            AAS_start_step=0,  # AAS start step
            AAS_start_layer=34,  # AAS start layer
            AAS_end_layer=70,  # AAS end layer
            num_inference_steps=50,  # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
            generator=torch.Generator(device=self.device).manual_seed(seed),
            guidance_scale=1,
        ).images[0]
        return image
