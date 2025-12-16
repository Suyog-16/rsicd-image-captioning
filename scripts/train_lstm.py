#!!! RUN THIS AS A MODULE !!
import os
import yaml
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from typing import List, Dict
from models.decoder_lstm import Decoder
from models.encoder import Encoder
from models.cnn_lstm_model import ImageCaptioningModel
from utils.seeds import set_seed
from utils.logging import save_config,create_writer
from utils.load_configs import load_config


config = load_config("config.yaml")


#--------------- Dataset Class Adapted from authors : https://github.com/isaaccorley/torchrs --------

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
        transform: T.Compose = T.Compose([T.ToTensor()])
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
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        sentences = [sentence["raw"] for sentence in captions["sentences"]]
        return dict(x=x, captions=sentences)


#----------------------- Model Defination -------------------------
Encoded_dim = config['model']['encoded_dim']
Projected_dim = config['model']['embedding_dim']
vocab_len = 1721
encoder = Encoder(encoded_dim = Encoded_dim,projection_dim=Projected_dim)
decoder = Decoder(embed_dim=Projected_dim,hidden_dim=512,vocab_size=vocab_len)

base_model = ImageCaptioningModel(encoder,decoder)

#----------------------- Loss functions,optimizers and logging------------------
EPOCHS = config['training']['epochs']
learning_rate = config['training']['learning_rate']
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(),lr = learning_rate)
