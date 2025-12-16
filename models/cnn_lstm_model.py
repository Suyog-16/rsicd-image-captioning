import torch
from torch import nn
class ImageCaptioningModel(nn.Module):
    def __init__(self,Encoder,Decoder):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        
    def forward(self,image,captions):
        features = self.encoder(image)
        output = self.decoder(captions,features)
        return output