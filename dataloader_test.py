import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from training.dataset import RSICD
import sys

transform = transforms.Compose([transforms.ToTensor()])
dataset = RSICD(root = "data",transform=transform)

sample = dataset[19]
image_tensor = sample['x']
captions =  sample['captions']
print(captions)
print(image_tensor.shape)
