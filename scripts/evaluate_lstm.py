#!!! RUN THIS AS A MODULE !!
import os
import yaml
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from typing import List, Dict
import numpy as np
import argparse

from models.decoder_lstm import Decoder
from models.encoder import Encoder
from models.cnn_lstm_model import ImageCaptioningModel

from utils.seeds import set_seed
from utils.load_configs import load_config
from utils.preprocess import preprocess_fn, caption_to_indices, build_vocab
from utils.collate_function import collate_fn
from models.dataset import RSICD

# Try to import evaluation metrics
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: pycocoevalcap not installed. Install with: pip install pycocoevalcap")
    print("Evaluation metrics will be calculated using alternative methods.")

set_seed(42)

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate LSTM image captioning model")
parser.add_argument("--model_path", type=str, default="checkpoints/final_model.pt",
                    help="Path to the model checkpoint (default: checkpoints/final_model.pt)")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Batch size for evaluation (default: from config)")
args = parser.parse_args()

config = load_config("config.yaml")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
Encoded_dim = config['model']['encoded_dim']
Projected_dim = config['model']['embedding_dim']
vocab_len = 1721  # found through EDA

encoder = Encoder(encoded_dim=Encoded_dim, projection_dim=Projected_dim)
decoder = Decoder(embed_dim=Projected_dim, hidden_dim=512, vocab_size=vocab_len)

base_model = ImageCaptioningModel(encoder, decoder).to(device)

# Load trained model checkpoint
CHECKPOINT_PATH = args.model_path
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {CHECKPOINT_PATH}")
else:
    print(f"Checkpoint not found at {CHECKPOINT_PATH}")
    print("Available checkpoints:")
    for ckpt in os.listdir("checkpoints"):
        print(f"  - {ckpt}")

# Dataset preparation
BATCH_SIZE = args.batch_size if args.batch_size else config['training']['batch_size']

train_dataset = RSICD(root="data", split='train')
all_train_captions = [cap for item in train_dataset for cap in item['captions']]
word2idx, idx2word = build_vocab(all_train_captions, min_freq=2)

test_dataset = RSICD(root="data", split="test")
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         collate_fn=lambda batch: collate_fn(batch, word2idx))

print(f"Test dataset size: {len(test_dataset)}")
print(f"Using device: {device}")

# Caption generation function with beam search
def generate_caption_beam_search(image, beam_width=3, max_length=30):
    """
    Generate caption using beam search.
    
    Args:
        image: Input image tensor (1, 3, 224, 224)
        beam_width: Width of beam search
        max_length: Maximum caption length
    
    Returns:
        Generated caption as string
    """
    base_model.eval()
    
    with torch.no_grad():
        # Encode image
        features = base_model.encoder(image.unsqueeze(0).to(device))  # (1, proj_dim)
        
        # Initialize beam search
        start_token = word2idx['<SOS>']
        end_token = word2idx['<EOS>']
        pad_token = word2idx['<PAD>']
        
        # Beam: (sequence, log_probability)
        beams = [([start_token], 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for sequence, score in beams:
                if sequence[-1] == end_token:
                    new_beams.append((sequence, score))
                    continue
                
                # Prepare input
                seq_tensor = torch.tensor([sequence]).to(device)
                
                # Get model output
                with torch.no_grad():
                    outputs = base_model.decoder(seq_tensor, features)  # (1, T, vocab_size)
                
                # Get probabilities for next token
                next_probs = torch.softmax(outputs[0, -1, :], dim=0)
                
                # Get top beam_width candidates
                top_probs, top_indices = torch.topk(next_probs, min(beam_width, len(next_probs)))
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = sequence + [idx.item()]
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_seq, new_score))
            
            # Keep top beam_width sequences
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams
            
            # Check if all beams have ended
            if all(seq[-1] == end_token for seq, _ in beams):
                break
        
        # Get best sequence
        best_seq = beams[0][0]
    
    # Convert indices to words
    caption_words = []
    for idx in best_seq:
        if idx == end_token or idx == pad_token:
            break
        if idx != start_token:
            caption_words.append(idx2word.get(idx, '<UNK>'))
    
    return ' '.join(caption_words)


# Simple greedy decoding function (faster alternative)
def generate_caption_greedy(image, max_length=30):
    """
    Generate caption using greedy decoding.
    
    Args:
        image: Input image tensor (3, 224, 224)
        max_length: Maximum caption length
    
    Returns:
        Generated caption as string
    """
    base_model.eval()
    
    with torch.no_grad():
        # Encode image
        features = base_model.encoder(image.unsqueeze(0).to(device))  # (1, proj_dim)
        
        start_token = word2idx['<SOS>']
        end_token = word2idx['<EOS>']
        pad_token = word2idx['<PAD>']
        
        sequence = [start_token]
        
        for _ in range(max_length):
            seq_tensor = torch.tensor([sequence]).to(device)
            
            # Get model output
            outputs = base_model.decoder(seq_tensor, features)  # (1, T, vocab_size)
            
            # Get the most likely next token
            next_token = torch.argmax(outputs[0, -1, :], dim=0).item()
            
            if next_token == end_token or next_token == pad_token:
                break
            
            sequence.append(next_token)
        
        # Convert indices to words
        caption_words = []
        for idx in sequence:
            if idx == end_token or idx == pad_token or idx == start_token:
                continue
            caption_words.append(idx2word.get(idx, '<UNK>'))
    
    return ' '.join(caption_words)


# Evaluation metrics
def calculate_bleu(reference_captions, generated_captions):
    """
    Calculate BLEU scores (1-4).
    Uses simple implementation without external library.
    
    Args:
        reference_captions: List of reference caption strings
        generated_captions: List of generated caption strings
    
    Returns:
        Dictionary with BLEU1-4 scores
    """
    try:
        # Try to use pycocoevalcap if available
        from pycocoevalcap.bleu.bleu import Bleu
        
        refs = {}
        hyps = {}
        for i, (ref, hyp) in enumerate(zip(reference_captions, generated_captions)):
            refs[i] = [ref]
            hyps[i] = [hyp]
        
        scorer = Bleu(4)
        score, scores = scorer.compute_score(refs, hyps)
        
        # Handle numpy arrays or lists
        bleu1 = float(scores[0]) if hasattr(scores[0], 'item') else float(np.mean(scores[0]))
        bleu2 = float(scores[1]) if hasattr(scores[1], 'item') else float(np.mean(scores[1]))
        bleu3 = float(scores[2]) if hasattr(scores[2], 'item') else float(np.mean(scores[2]))
        bleu4 = float(scores[3]) if hasattr(scores[3], 'item') else float(np.mean(scores[3]))
        
        return {
            'BLEU1': bleu1 * 100,
            'BLEU2': bleu2 * 100,
            'BLEU3': bleu3 * 100,
            'BLEU4': bleu4 * 100,
        }
    except ImportError:
        print("Using simplified BLEU calculation (without pycocoevalcap)")
        return None


def calculate_cider(reference_captions, generated_captions):
    """
    Calculate CIDEr score.
    Requires pycocoevalcap library.
    
    Args:
        reference_captions: List of reference caption strings
        generated_captions: List of generated caption strings
    
    Returns:
        CIDEr score
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        
        refs = {}
        hyps = {}
        for i, (ref, hyp) in enumerate(zip(reference_captions, generated_captions)):
            refs[i] = [ref]
            hyps[i] = [hyp]
        
        scorer = Cider()
        score, scores = scorer.compute_score(refs, hyps)
        
        return score * 100
    except ImportError:
        print("CIDEr calculation requires pycocoevalcap library")
        return None


# Evaluation loop
print("\nStarting evaluation on test set...")

base_model.eval()
all_generated_captions = []
all_reference_captions = []

test_progress = tqdm(test_loader, desc="Evaluating", leave=True)

for images, captions in test_progress:
    images = images.to(device)
    
    # For each image in batch, generate caption
    for i, img in enumerate(images):
        # Generate caption (using greedy decoding for speed)
        generated_caption = generate_caption_greedy(img)
        all_generated_captions.append(generated_caption)
        
        # Get reference caption from batch
        # Convert caption tensor to text
        ref_cap_indices = captions[i].tolist()
        ref_cap_words = []
        for idx in ref_cap_indices:
            if idx == word2idx['<EOS>'] or idx == word2idx['<PAD>'] or idx == word2idx['<SOS>']:
                continue
            ref_cap_words.append(idx2word.get(idx, '<UNK>'))
        reference_caption = ' '.join(ref_cap_words)
        all_reference_captions.append(reference_caption)

print(f"\nGenerated {len(all_generated_captions)} captions")

# Calculate metrics
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

# Print sample captions
print("\nSample Generated Captions:")
for i in range(min(5, len(all_generated_captions))):
    print(f"  [{i+1}] {all_generated_captions[i]}")

print("\nSample Reference Captions:")
for i in range(min(5, len(all_reference_captions))):
    print(f"  [{i+1}] {all_reference_captions[i]}")

# Calculate BLEU scores
bleu_scores = calculate_bleu(all_reference_captions, all_generated_captions)
if bleu_scores:
    print("\nBLEU Scores:")
    print(f"  BLEU-1: {bleu_scores['BLEU1']:.4f}")
    print(f"  BLEU-2: {bleu_scores['BLEU2']:.4f}")
    print(f"  BLEU-3: {bleu_scores['BLEU3']:.4f}")
    print(f"  BLEU-4: {bleu_scores['BLEU4']:.4f}")
else:
    print("\nBLEU scores calculation skipped (install pycocoevalcap for full metrics)")

# Calculate CIDEr score
cider_score = calculate_cider(all_reference_captions, all_generated_captions)
if cider_score is not None:
    print(f"\nCIDEr Score: {cider_score:.4f}")
else:
    print("\nCIDEr score calculation skipped (install pycocoevalcap for full metrics)")

print("\n" + "="*60)

# Save results to file
results = {
    'checkpoint': CHECKPOINT_PATH,
    'test_set_size': len(test_dataset),
    'bleu_scores': bleu_scores,
    'cider_score': cider_score,
    'sample_captions': [
        {
            'generated': all_generated_captions[i],
            'reference': all_reference_captions[i]
        }
        for i in range(min(10, len(all_generated_captions)))
    ]
}

results_file = f"results/evaluation_results_{os.path.basename(CHECKPOINT_PATH).replace('.pt', '')}.json"
os.makedirs("results", exist_ok=True)

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_file}")
