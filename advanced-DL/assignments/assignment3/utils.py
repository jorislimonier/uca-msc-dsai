import math
import os
import random
import urllib.request
from functools import partial
from urllib.error import HTTPError

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm.notebook import tqdm


def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
  """Return:
  - `output_values` which is the scaled dot product attention : $softmax( \frac{QK^T}{\sqrt{d_k}} ) V$
  - `attention` which is the same as `output_values`, but without multiplying by $V$
  """
  d_k = q.size()[-1]

  # Compute attn_logits
  attention = torch.matmul(q, k.transpose(-2, -1))
  attention /= d_k ** (1 / 2)

  # Apply mask if not None
  if mask is not None:
    attention = attention.masked_fill(mask == 0, -(10**14))

  # Pass through softmax
  attention = F.softmax(attention, dim=1)

  # Weight values accordingly
  output_values = torch.matmul(attention, v)

  return output_values, attention


class MultiheadAttention(nn.Module):
  def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
    super().__init__()

    err_mess = "Embedding dimension must be 0 modulo number of heads."
    assert embed_dim % num_heads == 0, err_mess

    self.embed_dim = embed_dim  # dimension of concatenated heads
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    # Create linear layers for both qkv and output
    # TIP: Stack all weight matrices 1...h together for efficiency
    self.qkv_proj = nn.Linear(in_features=input_dim, out_features=embed_dim * 3)
    self.o_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    self._reset_parameters()

  def _reset_parameters(self):
    # Original Transformer initialization, see PyTorch documentation

    nn.init.xavier_uniform_(self.qkv_proj.weight)
    self.qkv_proj.bias.data.fill_(0)

    nn.init.xavier_uniform_(self.o_proj.weight)
    self.o_proj.bias.data.fill_(0)

  def forward(self, x: torch.Tensor, mask=None, return_attention=False):
    # FILL IT YOURSELF!
    batch_dim, seq_length, input_dim = x.shape

    # Compute linear projection for qkv and separate heads
    # QKV: [Batch, Head, SeqLen, Dims]
    qkv = self.qkv_proj(x)
    qkv = qkv.reshape(batch_dim, seq_length, self.num_heads, 3 * self.head_dim)
    qkv.permute(0, 2, 1, 3)
    q, k, v = qkv.chunk(3, dim=-1)

    # Apply Dot Product Attention to qkv ()
    values, attention = scaled_dot_product(q=q, k=k, v=v)

    # Concatenate heads to [Batch, SeqLen, Embed Dim]
    attention = attention.reshape(batch_dim, seq_length, self.embed_dim)

    # Output projection
    o = self.o_proj(attention)

    if return_attention:
      return o, attention
    else:
      return o


class EncoderBlock(nn.Module):
  def __init__(self, input_dim: int, num_heads: int, dim_feedforward: int, dropout_prob=0.0):
    """
    Args:
        input_dim: Dimensionality of the input
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
    """
    super().__init__()
    # FILL IT YOURSELF!

    # Create Attention layer
    self.mh_attention = MultiheadAttention(
      input_dim=input_dim,
      embed_dim=input_dim,  # We assume it according to exercise
      num_heads=num_heads,
    )

    # Create Two-layer MLP with dropout
    self.ffn = nn.Sequential(
      nn.Linear(in_features=input_dim, out_features=dim_feedforward),
      nn.ReLU(),
      nn.Linear(in_features=dim_feedforward, out_features=input_dim),
    )

    # Layers to apply in between the main layers (Layer Norm and Dropout)
    self.layer_norm_mh_attention = nn.LayerNorm(normalized_shape=input_dim)
    # self.dropout_mh_attention = nn.Dropout(p=dropout_prob)

    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=input_dim)
    # self.dropout_ffn = nn.Dropout(p=dropout_prob)


  def forward(self, x, mask=None):
    # Compute Attention part
    attended = self.mh_attention(x)
    x = self.layer_norm_mh_attention(x + attended)

    # Compute MLP part
    fedforward = self.ffn(x)
    x = self.layer_norm_ffn(x + fedforward)

    return x
