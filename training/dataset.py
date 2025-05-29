import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class RSICD_dataset(Dataset):
    def __init__(self,captions_file,image_dir,word2idx,transform = None):
        self.captions_dir = captions_file
        self.image_dir = image_dir
        self.word2idx = word2idx
        
        self.image_files = []
        self.captions = []
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.images_files)

