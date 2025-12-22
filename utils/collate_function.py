import torch
from torch.nn.utils.rnn import pad_sequence
from preprocess import preprocess_fn, caption_to_indices, build_vocab
from models.dataset import RSICD
import random

# Load train split and build vocabulary
train_dataset = RSICD(root="data", split='train')
all_train_captions = [cap for item in train_dataset for cap in item['captions']]
word2idx, idx2word = build_vocab(all_train_captions, min_freq=2)

# Collate function that takes word2idx as argument
def collate_fn(batch, word2idx):
    images = []
    captions = []
    for item in batch:
        img = item['x']
        cap = random.choice(item['captions'])  # randomly pick one caption
        cap_tokens = preprocess_fn(cap)
        cap_indices = caption_to_indices(cap_tokens, word2idx)
        images.append(img)
        captions.append(torch.tensor(cap_indices))
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=word2idx['<PAD>'])
    return images, captions
