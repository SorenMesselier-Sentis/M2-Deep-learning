# M2-Deep-learning

## Tiny shakespeare

In the `tiny_shakespeare.ipynb` you will find all the set-up for training the tiny_shakespeare model.

## From DE to EN

## Overview
This project implements a Transformer model for sequence-to-sequence tasks, specifically machine translation. It includes tokenization using Byte Pair Encoding (BPE), data preprocessing, and a PyTorch implementation of the Transformer architecture.
In the `custom_translation.ipynb` file.
## Features
- Implementation of Transformer components such as:
  - Input Embeddings
  - Positional Encoding
  - Layer Normalization
  - Feed-Forward Blocks
  - Multi-Head Attention
  - Residual Connections
  - Encoder and Decoder Blocks
- Training pipeline with dataset loading, tokenization, and padding
- Training loop with loss computation, optimizer, and learning rate scheduler
- Logging with TensorBoard

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install torch datasets tokenizers tqdm tensorboard
```

## Model Architecture
The Transformer model consists of:
- An **Encoder** with multiple layers containing self-attention and feed-forward blocks
- A **Decoder** with similar layers and an additional cross-attention mechanism
- A final **Projection Layer** to map decoder outputs to vocabulary tokens

## Data Preparation
The dataset used is **WMT16 German-English**. (https://huggingface.co/datasets/wmt/wmt16)

The tokenization process includes:
1. **Training a BPE tokenizer**
2. **Tokenizing sentences**
3. **Converting tokens to tensors**
4. **Padding sequences to uniform length**

## Usage
### Training
Run the training script with:

```bash
python train.py
```

### Model Evaluation
To evaluate the model, you can test it on a sentence:

```python
# Example Translation
test_sentence = "Das ist ein Test."
tokenized_input = bpe_tokenizer.encode(test_sentence)
input_tensor = torch.tensor(tokenized_input).unsqueeze(0).to(device)
encoder_output = model.encode(input_tensor, None)
decoder_output = model.decode(encoder_output, None, input_tensor, None)
output_logits = model.project(decoder_output)
```

## Configuration
- `d_model` (embedding size)
- `num_heads` (number of attention heads)
- `num_layers` (number of encoder/decoder layers)
- `dropout` (dropout rate)
- `vocab_size` (size of BPE vocabulary)

