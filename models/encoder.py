import torch
from torch import nn
from torchvision import models

efficient_netb0 = models.efficientnet_b0(pretrained=True)


class Encoder(nn.Module):
    def __init__(self,encoded_dim,projection_dim):
        super().__init__(Encoder)
        self.encoder = nn.Sequential(*list(efficient_netb0.children())[:-1])
        self.fc = nn.Linear(encoded_dim,projection_dim) # project to same dim as decoder embedding


    def forward(self,images):
        features = self.encoder(images) #(B,1280,1,1)
        features = features.flatten(start_dim =1)
        #features = features.view(features.size(0),-1) # (B,1280)
        features = self.fc(features)

        return features
