from simple_lama_inpainting import SimpleLama
from PIL import Image

from models.inpaint.base import InpaintModel
import cv2
import numpy as np


class Lama(InpaintModel):
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.model = None

    def setup(self):
        self.model = SimpleLama()

    def _forward(self, image: Image, mask: Image):
        result = self.model(image, mask)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)  # bgr image
        return result
