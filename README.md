# RSICD(Remote Sensing Image Captioning Dataset) image captioning project

A deep learning project for generating natural language captions for remote sensing images using the RSICD dataset.

This project explores the effectiveness of different encoder-decoder architectures in understanding and describing satellite imagery.

### Objective
To build models that can generate accurate and meaningful captions for aerial images, and compare different decoding strategies:

- CNN + LSTM
Classic encoder-decoder approach using convolutional features and recurrent sequence modeling.

- CNN + Transformer
Combines convolutional visual encoders with attention-based language decoders for improved context handling.

### Dataset

RSICD (Remote Sensing Image Caption Dataset)

~10,000 high-resolution images

Each image is annotated with 5 human-written captions

Covers diverse land types: urban, forest, water, farmland, and more

### Scope
Focused on model comparison and caption quality

Evaluation will include standard captioning metrics (BLEU, CIDEr, METEOR)

Later extensions may explore vision-language pretraining or multimodal setups