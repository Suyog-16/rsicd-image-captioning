#!!! RUN THIS AS A MODULE !!
import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn_lstm_model import ImageCaptioningModel
from models.decoder_transformer import Decoder
from models.encoder import Encoder
from models.dataset import RSICD
from utils.load_configs import load_config
from utils.preprocess import build_vocab, preprocess_fn
from utils.seeds import set_seed

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
except ImportError as exc:
    raise ImportError(
        "pycocoevalcap is required for publication-grade evaluation. "
        "Install it in your environment before running this script."
    ) from exc


set_seed(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer image captioning model on RSICD")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/_final_transformer_model_epoch_10.pt",
        help="Path to checkpoint containing model_state_dict",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="checkpoints/vocab_minfreq2.json",
        help="Optional path to saved vocabulary JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: config value)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum generated caption length (without <SOS>)",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Decoding strategy",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="Beam width when --decode beam",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.6,
        help="Beam-search length penalty alpha (0 disables normalization)",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Only used when vocabulary has to be rebuilt from train captions",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation JSON",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="How many sample predictions to save",
    )
    return parser.parse_args()


def normalize_caption_text(caption: str) -> str:
    tokens = preprocess_fn(caption)
    text = " ".join(tokens)
    # Remove spaces before punctuation for cleaner readable dumps.
    return re.sub(r"\s+([?.!,;:])", r"\1", text).strip()


def eval_collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, List[List[str]]]:
    images = torch.stack([item["x"] for item in batch], dim=0)
    all_refs = [item["captions"] for item in batch]
    return images, all_refs


def load_checkpoint(checkpoint_path: str, device: str) -> Dict:
    if not os.path.exists(checkpoint_path):
        available = []
        if os.path.isdir("checkpoints"):
            available = sorted(os.listdir("checkpoints"))
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Available checkpoints: {available}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_state_dict'. "
            "Re-save checkpoints with model_state_dict for safe evaluation."
        )
    return checkpoint


def _load_vocab_from_json(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "word2idx" in payload:
        word2idx = {str(k): int(v) for k, v in payload["word2idx"].items()}
    else:
        word2idx = {str(k): int(v) for k, v in payload.items()}

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def load_or_build_vocab(
    checkpoint: Dict,
    vocab_path: str,
    min_freq: int,
) -> Tuple[Dict[str, int], Dict[int, str], str]:
    if "word2idx" in checkpoint:
        word2idx = {str(k): int(v) for k, v in checkpoint["word2idx"].items()}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word, "checkpoint"

    if vocab_path and os.path.exists(vocab_path):
        word2idx, idx2word = _load_vocab_from_json(vocab_path)
        return word2idx, idx2word, f"file:{vocab_path}"

    train_dataset = RSICD(root="data", split="train")
    all_train_captions = [cap for item in train_dataset for cap in item["captions"]]
    word2idx, idx2word = build_vocab(all_train_captions, min_freq=min_freq)
    return word2idx, idx2word, "rebuilt_from_train_split"


def validate_vocab(word2idx: Dict[str, int], model_state_dict: Dict) -> None:
    required_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    missing = [tok for tok in required_tokens if tok not in word2idx]
    if missing:
        raise ValueError(f"Vocabulary missing required special tokens: {missing}")

    expected_vocab_size = model_state_dict["decoder.embedding.weight"].shape[0]
    if expected_vocab_size != len(word2idx):
        raise ValueError(
            f"Vocabulary mismatch: checkpoint expects vocab_size={expected_vocab_size}, "
            f"but loaded vocab has size={len(word2idx)}."
        )


def build_model(config: Dict, vocab_len: int, device: str) -> ImageCaptioningModel:
    encoded_dim = config["model"]["encoded_dim"]
    projected_dim = config["model"]["embedding_dim"]

    encoder = Encoder(encoded_dim=encoded_dim, projection_dim=projected_dim)
    decoder = Decoder(
        embed_dim=projected_dim,
        vocab_size=vocab_len,
        num_layers=config["model"].get("num_layers", 6),
        dropout=config["model"].get("dropout", 0.1),
    )
    return ImageCaptioningModel(encoder, decoder).to(device)


def decode_token_ids(
    token_ids: List[int],
    idx2word: Dict[int, str],
    sos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    words: List[str] = []
    for tid in token_ids:
        if tid == eos_id:
            break
        if tid in (sos_id, pad_id):
            continue
        words.append(idx2word.get(tid, "<UNK>"))

    text = " ".join(words)
    return re.sub(r"\s+([?.!,;:])", r"\1", text).strip()


def _length_normalize(score: float, length: int, alpha: float) -> float:
    if alpha <= 0.0:
        return score
    return score / (((5.0 + float(length)) ** alpha) / (6.0 ** alpha))


def generate_caption_greedy(
    model: ImageCaptioningModel,
    image: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    max_decoder_len: int,
    device: str,
) -> List[int]:
    sequence = [sos_id]

    for _ in range(min(max_length, max_decoder_len)):
        seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        logits = model(image.unsqueeze(0).to(device), seq_tensor)
        logits_last = logits[0, -1, :]

        # Never let generation emit <SOS> again.
        logits_last[sos_id] = -1e9
        next_id = int(torch.argmax(logits_last, dim=-1).item())

        if next_id in (eos_id, pad_id):
            break
        sequence.append(next_id)

    return sequence


def generate_caption_beam_search(
    model: ImageCaptioningModel,
    image: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    beam_size: int,
    alpha: float,
    max_decoder_len: int,
    device: str,
) -> List[int]:
    beams: List[Tuple[List[int], float, bool]] = [([sos_id], 0.0, False)]

    num_steps = min(max_length, max_decoder_len)
    for _ in range(num_steps):
        expanded: List[Tuple[List[int], float, bool]] = []

        for seq, score, ended in beams:
            if ended:
                expanded.append((seq, score, ended))
                continue

            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(image.unsqueeze(0).to(device), seq_tensor)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)

            # Do not emit <SOS> after the first token.
            log_probs[sos_id] = -1e9

            top_log_probs, top_indices = torch.topk(log_probs, k=beam_size)
            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                next_seq = seq + [int(idx)]
                next_ended = int(idx) in (eos_id, pad_id)
                expanded.append((next_seq, score + float(lp), next_ended))

        expanded.sort(
            key=lambda x: _length_normalize(x[1], len(x[0]), alpha),
            reverse=True,
        )
        beams = expanded[:beam_size]

        if all(ended for _, _, ended in beams):
            break

    best_seq = max(
        beams,
        key=lambda x: _length_normalize(x[1], len(x[0]), alpha),
    )[0]
    return best_seq


def compute_metrics(
    references: Dict[int, List[str]],
    predictions: Dict[int, List[str]],
) -> Dict:
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(references, predictions)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, predictions)

    return {
        "BLEU1": float(bleu_score[0]),
        "BLEU2": float(bleu_score[1]),
        "BLEU3": float(bleu_score[2]),
        "BLEU4": float(bleu_score[3]),
        "CIDEr": float(cider_score),
        "BLEU1_percent": float(bleu_score[0] * 100.0),
        "BLEU2_percent": float(bleu_score[1] * 100.0),
        "BLEU3_percent": float(bleu_score[2] * 100.0),
        "BLEU4_percent": float(bleu_score[3] * 100.0),
        "CIDEr_percent": float(cider_score * 100.0),
    }


def main() -> None:
    args = parse_args()
    config = load_config("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.model_path, device)

    word2idx, idx2word, vocab_source = load_or_build_vocab(
        checkpoint=checkpoint,
        vocab_path=args.vocab_path,
        min_freq=args.min_freq,
    )

    validate_vocab(word2idx, checkpoint["model_state_dict"])

    model = build_model(config=config, vocab_len=len(word2idx), device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sos_id = word2idx["<SOS>"]
    eos_id = word2idx["<EOS>"]
    pad_id = word2idx["<PAD>"]

    max_decoder_len = model.decoder.pos_embedding.num_embeddings - 1
    if args.max_length > max_decoder_len:
        print(
            f"Warning: max_length={args.max_length} exceeds decoder positional limit "
            f"({max_decoder_len}). Clamping to {max_decoder_len}."
        )

    test_dataset = RSICD(root="data", split="test")
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )

    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Vocabulary source: {vocab_source}")
    print(f"Test dataset size: {len(test_dataset)}")
    print("\nStarting evaluation...")

    references: Dict[int, List[str]] = {}
    predictions: Dict[int, List[str]] = {}
    sample_rows: List[Dict] = []

    global_idx = 0
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, batch_refs in progress:
            images = images.to(device)
            batch_size_actual = images.shape[0]

            for i in range(batch_size_actual):
                image = images[i]

                if args.decode == "beam":
                    pred_ids = generate_caption_beam_search(
                        model=model,
                        image=image,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        max_length=args.max_length,
                        beam_size=args.beam_size,
                        alpha=args.length_penalty,
                        max_decoder_len=max_decoder_len,
                        device=device,
                    )
                else:
                    pred_ids = generate_caption_greedy(
                        model=model,
                        image=image,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        max_length=args.max_length,
                        max_decoder_len=max_decoder_len,
                        device=device,
                    )

                pred_text = decode_token_ids(
                    token_ids=pred_ids,
                    idx2word=idx2word,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )

                raw_refs = batch_refs[i]
                refs_text = [normalize_caption_text(ref) for ref in raw_refs]
                refs_text = [r for r in refs_text if len(r) > 0]

                if not refs_text:
                    refs_text = ["<empty_reference>"]

                references[global_idx] = refs_text
                predictions[global_idx] = [pred_text if pred_text else "<empty_prediction>"]

                if len(sample_rows) < max(0, args.num_samples):
                    sample_rows.append(
                        {
                            "sample_id": global_idx,
                            "generated": predictions[global_idx][0],
                            "references": refs_text,
                        }
                    )

                global_idx += 1

    if len(references) != len(predictions):
        raise RuntimeError(
            f"Prediction/reference count mismatch: {len(predictions)} preds vs {len(references)} refs"
        )

    metrics = compute_metrics(references=references, predictions=predictions)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS (pycocoevalcap)")
    print("=" * 70)
    print(f"BLEU-1: {metrics['BLEU1']:.4f} ({metrics['BLEU1_percent']:.2f}%)")
    print(f"BLEU-2: {metrics['BLEU2']:.4f} ({metrics['BLEU2_percent']:.2f}%)")
    print(f"BLEU-3: {metrics['BLEU3']:.4f} ({metrics['BLEU3_percent']:.2f}%)")
    print(f"BLEU-4: {metrics['BLEU4']:.4f} ({metrics['BLEU4_percent']:.2f}%)")
    print(f"CIDEr : {metrics['CIDEr']:.4f} ({metrics['CIDEr_percent']:.2f})")

    print("\nSample predictions:")
    for row in sample_rows[: min(5, len(sample_rows))]:
        print(f"[{row['sample_id']}] Pred: {row['generated']}")
        print(f"     Ref1: {row['references'][0]}")

    output = {
        "checkpoint": args.model_path,
        "vocab_source": vocab_source,
        "decode": {
            "method": args.decode,
            "max_length": args.max_length,
            "beam_size": args.beam_size if args.decode == "beam" else None,
            "length_penalty": args.length_penalty if args.decode == "beam" else None,
        },
        "counts": {
            "num_images": len(predictions),
            "num_reference_sets": len(references),
        },
        "metrics": metrics,
        "samples": sample_rows,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_stem = os.path.basename(args.model_path).replace(".pt", "")
    output_path = os.path.join(
        args.results_dir,
        f"evaluation_results_transformer_{checkpoint_stem}_{args.decode}.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
