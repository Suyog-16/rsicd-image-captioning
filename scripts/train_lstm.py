#!!! RUN THIS AS A MODULE !!
import os
import yaml
import json
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader,Dataset

from PIL import Image
from typing import List, Dict

from models.decoder_lstm import Decoder
from models.encoder import Encoder
from models.cnn_lstm_model import ImageCaptioningModel

from utils.seeds import set_seed
from utils.logging import save_config,create_writer
from utils.load_configs import load_config
from utils.collate_function import collate_fn
from utils.preprocess import build_vocab
from models.dataset import RSICD


set_seed(42)

config = load_config("config.yaml")
#---------- Making it Device agnostic---------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


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
BATCH_SIZE = config['training']['batch_size']
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(),lr = learning_rate)

#--------------- Dataset Class Adapted from authors : https://github.com/isaaccorley/torchrs --------
train_dataset = RSICD(root="data",split='train')
all_train_captions = [cap for item in train_dataset for cap in item['captions']]
word2idx, idx2word = build_vocab(all_train_captions, min_freq=2)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn= lambda batch: collate_fn(batch,word2idx))




