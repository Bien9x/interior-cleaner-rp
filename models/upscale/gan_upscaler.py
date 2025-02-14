import torch
from RealESRGAN import RealESRGAN
from utils import pil_ensure_rgb


class Upscaler:
    def __init__(self, force_cpu=False):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() or not force_cpu else 'cpu'

    def setup(self):
        self.model = RealESRGAN(self.device, scale=4)
        self.model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    def __call__(self, image):
        image = pil_ensure_rgb(image)
        sr_image = self.model.predict(image)
        return sr_image
