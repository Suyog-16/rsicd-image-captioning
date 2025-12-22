import re
from collections import Counter

def preprocess_fn(captions: str) -> list:
    captions = captions.lower().strip()
    captions = re.sub(r"([?.!,])", r" \1 ", captions)# add space puntuation

    captions = re.sub(r"[^a-zA-Z?.!,]+", " ", captions)# remove unwanted characters

    captions = re.sub(r'\s+', ' ', captions).strip()

    tokens = captions.split()

    return tokens

def caption_to_indices(tokens, word2idx):
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in tokens]
    indices = [word2idx['<SOS>']] + indices + [word2idx['<EOS>']]
    return indices

def build_vocab(all_captions, min_freq=2):
    tokenized_captions = [preprocess_fn(c) for c in all_captions]
    word_counter = Counter()
    for tokens in tokenized_captions:
        word_counter.update(tokens)

    vocab_words = [word for word, freq in word_counter.items() if freq >= min_freq]
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    vocab = special_tokens + vocab_words

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word



