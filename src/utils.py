from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import requests

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def resize_image(input_image: Image.Image | np.ndarray, resolution: int) -> Image.Image:
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
    if len(input_image.shape) == 3:  # RGB
        H, W, C = input_image.shape
    else:  # gray-scale
        H, W = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    img = Image.fromarray(img)
    return img


def convert_to_base64(image: Image.Image) -> str:
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_image(image_source: str, client = None) -> Image.Image:
    """Get image from URL, GCP URL, or base64 string"""
    if not image_source:
        raise ValueError("Image source cannot be empty")

    try:
        if image_source.startswith("http"):
            print('Reading image from url')
            response = requests.get(image_source, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise error for bad status codes
            return Image.open(BytesIO(response.content))
        elif image_source.startswith("gs://"):
            if client is None:
                raise ValueError("GCS client is required")
            # Use storage client to download from GCS
            bucket_name = image_source.split("/")[2]
            blob_path = "/".join(image_source.split("/")[3:])

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            image_bytes = blob.download_as_bytes()
            return Image.open(BytesIO(image_bytes))
        else:
            # Validate base64 string
            try:
                image_data = base64.b64decode(image_source)
            except Exception:
                raise ValueError("Invalid base64 string")
            return Image.open(BytesIO(image_data))
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")
