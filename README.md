# super-miniGPT
Implemented Small Language Model from Scratch (Yes, I have used Pytorch don't come jumping guns on me).
This implementation includes the core components of the GPT architecture including multi-head self-attention, feedforward networks, and the transformer decoder blocks.

## Features

- Support for custom model configurations
- Efficient token generation with temperature and top-k sampling
- Weight initialization following GPT-2 style
- Dropout for regularization
- Layer normalization and residual connections

## Model Architecture

The implementation includes the following key components:

- Multi-head self-attention mechanism
- Position-wise feedforward networks
- Layer normalization
- Residual connections
- Dropout for regularization
- Token and position embeddings
- Weight tying between input embeddings and output layer

## Requirements

- Python 3.6+
- PyTorch 1.0+
- NumPy

## Usage

```python
from model import GPT, GPTConfig
import torch

# Initialize model configuration
config = GPTConfig(
    vocab_size=50000,  # size of vocabulary
    block_size=1024,   # context length
    n_embd=768,        # embedding dimension
    n_head=12,         # number of attention heads
    n_layer=12,        # number of transformer layers
    dropout=0.1,       # dropout probability
    bias=True          # whether to use bias in linear layers
)

# Create model instance
model = GPT(config)

# Generate text
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # example input sequence
output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=40)
```
