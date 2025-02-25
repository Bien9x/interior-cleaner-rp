
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
import timm
from diffusers import AutoencoderKL
import torch
from torch.hub import download_url_to_file
from simple_lama_inpainting.models.model import LAMA_MODEL_URL
sys.path.append('.')
import config
from modules.controlnet_plus import StableDiffusionXLControlNetUnionInpaintPipeline

if os.path.exists(config.CACHE_DIR):
    shutil.rmtree(config.CACHE_DIR)
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.PATH_SDXL_CONTROLNET_UNION, exist_ok=True)
# lama
download_url_to_file(LAMA_MODEL_URL,config.PATH_LAMA)
# tagger model
model = timm.create_model("hf-hub:" + config.MODEL_TAGGER_ID, cache_dir=config.CACHE_DIR)
model_weights = timm.models.load_state_dict_from_hf(config.MODEL_TAGGER_ID, cache_dir=config.CACHE_DIR)
csv_path = hf_hub_download(repo_id=config.MODEL_TAGGER_ID, filename="selected_tags.csv", cache_dir=config.CACHE_DIR)

# diffusers
download_url_to_file(config.URL_CONTROLNET_CONFIG,os.path.join(config.PATH_SDXL_CONTROLNET_UNION,'config.json'))
download_url_to_file(config.URL_CONTROLNET_WEIGHT,os.path.join(config.PATH_SDXL_CONTROLNET_UNION,"diffusion_pytorch_model.safetensors"))
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                                    cache_dir=config.CACHE_DIR)
pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant='fp16',
    cache_dir=config.CACHE_DIR
)
