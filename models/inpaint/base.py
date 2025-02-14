from abc import ABC, abstractmethod
import torch
from typing import Optional, List
from PIL import Image
from utils import boxes_from_mask, pad_img_to_modulo
import numpy as np
import cv2
from abc import ABC


class InpaintModel(ABC):
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False

    hd_strategy_crop_trigger_size = 640
    hd_strategy_crop_margin = 128
    sd_keep_unmasked_area = False

    def __init__(self, force_cpu):
        self.device = 'cuda' if torch.cuda.is_available() or not force_cpu else 'cpu'

    def _pad_forward(self, image, mask):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        # logger.info(f"final forward pad size: {pad_image.shape}")

        # image, mask = self.forward_pre_process(image, mask, config)

        result = self._forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        # result, image, mask = self.forward_post_process(result, image, mask, config)

        if self.sd_keep_unmasked_area:
            mask = mask[:, :, np.newaxis]
            result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))
        return result

    def _crop_box(self, image, mask, box):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + self.hd_strategy_crop_margin * 2
        h = box_h + self.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        # logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _run_box(self, image, mask, box):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box)

        return self._pad_forward(crop_img, crop_mask), [l, t, r, b]

    def __call__(self, image: Image, mask: Image):
        image = np.array(image)
        mask = np.array(mask)
        inpaint_result = None
        if max(image.shape) > self.hd_strategy_crop_trigger_size:
            boxes = boxes_from_mask(mask)
            crop_result = []
            for box in boxes:
                crop_image, crop_box = self._run_box(image, mask, box)
                crop_result.append((crop_image, crop_box))

            inpaint_result = image[:, :, ::-1]
            for crop_image, crop_box in crop_result:
                x1, y1, x2, y2 = crop_box
                inpaint_result[y1:y2, x1:x2, :] = crop_image

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask)
        inpaint_result = Image.fromarray(cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB))
        return inpaint_result

    @abstractmethod
    def _forward(self, image, mask):
        pass
