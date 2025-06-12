import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self,embed_dim,hidden_dim,vocab_size,num_layers = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self, captions, features):
        # captions: (B, T)
        embeddings = self.embedding(captions)  # (B, T, embed_dim)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # prepend image feature
        outputs, _ = self.lstm(embeddings)  # (B, T+1, hidden_dim)
        outputs = self.fc(outputs)  # (B, T+1, vocab_size)
        return outputs