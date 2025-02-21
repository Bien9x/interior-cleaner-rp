from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F
from utils import pil_ensure_rgb, pil_pad_square

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
        probs: Tensor,
        labels: LabelData,
        gen_threshold: float,
        char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ")

    return caption, taglist, rating_labels, char_labels, gen_labels


class TagGenerator:
    def __init__(self, force_cpu=False):
        self.gen_threshold = 0.35
        self.char_threshold = 0.85
        self.model_id = 'vit'
        self.device = 'cuda' if torch.cuda.is_available() or not force_cpu else 'cpu'

        self.model = None
        self.labels = None
        self.transform = None

    def setup(self):
        repo_id = MODEL_REPO_MAP.get(self.model_id)
        print(f"Loading model '{self.model_id}' from '{repo_id}'...")
        self.model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
        self.model = self.model.to(self.device)
        state_dict = timm.models.load_state_dict_from_hf(repo_id)
        self.model.load_state_dict(state_dict)
        print("Loading tag list...")
        self.labels: LabelData = load_labels_hf(repo_id=repo_id)
        print("Creating data transform...")
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def __call__(self, image: Image):
        # ensure image is RGB
        image = pil_ensure_rgb(image)
        # pad to square with white background
        image = pil_pad_square(image)
        # run the model's input transform to convert to tensor and rescale
        inputs: Tensor = self.transform(image).unsqueeze(0)
        # NCHW image RGB to BG!
        inputs = inputs[:, [2, 1, 0]]

        print("Running inference...")
        with torch.inference_mode():
            # move model to GPU, if available
            inputs = inputs.to(self.device )
            # run the model
            outputs = self.model.forward(inputs)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to to cpu if we were on GPU
            outputs = outputs.to("cpu")
        print("Processing results...")
        caption, tag_list, ratings, character, general = get_tags(
            probs=outputs.squeeze(0),
            labels=self.labels,
            gen_threshold=self.gen_threshold,
            char_threshold=self.char_threshold,
        )
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {tag_list}")

        print("--------")
        print("Ratings:")
        for k, v in ratings.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Character tags (threshold={self.char_threshold}):")
        for k, v in character.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"General tags (threshold={self.gen_threshold}):")
        for k, v in general.items():
            print(f"  {k}: {v:.3f}")
        return tag_list
