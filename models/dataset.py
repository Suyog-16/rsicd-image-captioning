import torch
# Dataset class adapted from authors https://github.com/isaaccorley/torchrs 
import os
import json
from typing import List, Dict, Callable, Optional

from torchvision.io import read_image, ImageReadMode


class RSICD(torch.utils.data.Dataset):
    """ Image Captioning Dataset from 'Exploring Models and Data for
    Remote Sensing Image Caption Generation', Lu et al. (2017)
    https://arxiv.org/abs/1712.07835

    'RSICD is used for remote sensing image captioning task. more than ten thousands
    remote sensing images are collected from Google Earth, Baidu Map, MapABC, Tianditu.
    The images are fixed to 224X224 pixels with various resolutions. The total number of
    remote sensing images are 10921, with five sentences descriptions per image.'
    """
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset_rsicd.json"), split)
        self.image_root = "RSICD_images"

    @staticmethod
    def load_captions(path: str, split: str) -> List[Dict]:
        with open(path) as f:
            captions = json.load(f)["images"]
        return [c for c in captions if c["split"] == split]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict:
        captions = self.captions[idx]
        path = os.path.join(self.root, self.image_root, captions["filename"])
        x = read_image(path, mode=ImageReadMode.RGB).float().div(255.0)
        if self.transform is not None:
            x = self.transform(x)
        sentences = [sentence["raw"] for sentence in captions["sentences"]]
        return dict(x=x, captions=sentences)




