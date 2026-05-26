#!!! RUN THIS AS A MODULE !!
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.decoder_transformer import Decoder
from models.encoder import Encoder
from models.cnn_lstm_model import ImageCaptioningModel

from utils.seeds import set_seed
from utils.logging import save_config, create_writer
from utils.load_configs import load_config
from utils.preprocess import build_vocab
from utils.collate_function import collate_fn
from models.dataset import RSICD


set_seed(42)

config = load_config("config.yaml")
#---------- Making it Device agnostic---------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(config)
#----------------------- Model Defination -------------------------
Encoded_dim = config["model"]["encoded_dim"]
Projected_dim = config["model"]["embedding_dim"]
vocab_len = 1721  # found through EDA

encoder = Encoder(encoded_dim=Encoded_dim, projection_dim=Projected_dim)
decoder = Decoder(
	embed_dim=Projected_dim,
	vocab_size=vocab_len,
	num_layers=config["model"].get("num_layers", 6),
	dropout=config["model"].get("dropout", 0.1),
)

base_model = ImageCaptioningModel(encoder, decoder).to(device)

#----------------------- Loss functions, optimizers and logging------------------
EPOCHS = config["training"]["epochs"] #10
learning_rate = config["training"]["learning_rate"]# 0.0003
BATCH_SIZE = config["training"]["batch_size"]#32
WEIGHT_DECAY = config["training"]['weight_decay']#-0.0001
LABEL_SMOOTHING = config["training"]['label_smoothing']
optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate,weight_decay=WEIGHT_DECAY)
warmup_epoch = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS -warmup_epoch
)

#--------------- Dataset Class Adapted from authors : https://github.com/isaaccorley/torchrs --------
train_dataset = RSICD(root="data", split="train")
all_train_captions = [cap for item in train_dataset for cap in item["captions"]]
word2idx, idx2word = build_vocab(all_train_captions, min_freq=2)
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/vocab_minfreq2.json", "w") as f:
	json.dump({"word2idx": word2idx}, f, indent=2)
train_loader = DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	collate_fn=lambda batch: collate_fn(batch, word2idx),
)

val_dataset = RSICD(root="data", split="val")
val_loader = DataLoader(
	val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	collate_fn=lambda batch: collate_fn(batch, word2idx),
)  # use the same vocab

criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"],label_smoothing = LABEL_SMOOTHING )

#----------------- Logging setup----------------------------------
log_dir = "runs/experiment_transformer"
writer = create_writer(log_dir)
save_config(config, log_dir)

#----------------------Training Loop--------------------------------
print(f"Starting transformer training for {EPOCHS} epochs")
print(f"Using {device}")

for epoch in range(EPOCHS):
	base_model.train()
	total_train_loss = 0
	train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)

	for batch_idx, (images, captions) in enumerate(train_progress):
		images, captions = images.to(device), captions.to(device)
		inputs = captions[:, :-1]
		targets = captions[:, 1:]

		# Calculate output
		outputs = base_model(images, inputs)
		# Compute Loss (skip the prepended image token position)
		loss = criterion(outputs[:, 1:, :].reshape(-1, vocab_len), targets.reshape(-1))
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
		optimizer.step()
		total_train_loss += loss.item()
		train_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

		# Batch-level logging
		writer.add_scalar("Loss/Train_Batch", loss.item(), epoch * len(train_loader) + batch_idx)

	avg_train_loss = total_train_loss / len(train_loader)
	writer.add_scalar("Loss/Train", avg_train_loss, epoch)
	if epoch > warmup_epoch:
		scheduler.step()

	# Validation
	base_model.eval()
	total_val_loss = 0
	val_progress = tqdm(
		val_loader,
		desc=f"Epoch {epoch+1}/{EPOCHS} Validation",
		leave=False,
	)

	with torch.no_grad():
		for images, captions in val_progress:
			images, captions = images.to(device), captions.to(device)
			inputs = captions[:, :-1]
			targets = captions[:, 1:]

			outputs = base_model(images, inputs)
			loss = criterion(outputs[:, 1:, :].reshape(-1, vocab_len), targets.reshape(-1))
			total_val_loss += loss.item()
			val_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

	avg_val_loss = total_val_loss / len(val_loader)

	print(
		f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, "
		f"Val Loss: {avg_val_loss:.4f}"
	)

	writer.add_scalar("Loss/Val", avg_val_loss, epoch)

	# Save checkpoint
	if (epoch + 1) % 5 == 0:
		checkpoints_path = f"checkpoints/_final_ transformer_model_epoch_{epoch+1}.pt"
		os.makedirs("checkpoints", exist_ok=True)
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": base_model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"word2idx": word2idx,
				"train_loss": avg_train_loss,
				"val_loss": avg_val_loss,
			},
			checkpoints_path,
		)
		print(f"Checkpoint saved: {checkpoints_path}")

print("Training completed")
writer.close()
