import torch
from torch import nn


class Decoder(nn.Module):
	def __init__(
		self,
		embed_dim,
		vocab_size,
		num_layers=3,
		num_heads=8,
		ff_dim=2048,
		dropout=0.1,
		max_len=128,
	):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.pos_embedding = nn.Embedding(max_len, embed_dim)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=ff_dim,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(embed_dim, vocab_size)

	@staticmethod
	def _causal_mask(size, device):
		# Upper-triangular True entries are masked in self-attention.
		return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

	def forward(self, captions, features):
		# captions: (B, T), features: (B, embed_dim)
		batch_size = captions.size(0)
		device = captions.device

		token_embeddings = self.embedding(captions)  # (B, T, embed_dim)
		image_token = features.unsqueeze(1)  # (B, 1, embed_dim)

		# Keep output contract consistent with LSTM: logits for image token + caption tokens.
		x = torch.cat((image_token, token_embeddings), dim=1)  # (B, T+1, embed_dim)

		seq_len = x.size(1)
		positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
		x = self.dropout(x + self.pos_embedding(positions))

		causal_mask = self._causal_mask(seq_len, device)
		x = self.transformer(x, mask=causal_mask)

		outputs = self.fc(x)  # (B, T+1, vocab_size)
		return outputs
