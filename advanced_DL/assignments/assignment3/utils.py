import math
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm.notebook import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
  """
  Perform scaled dot product which links what we are looking for (the queries `q`) to what
  the attention has to offer (the keys `k`), using their relative importance (the values `v`)
  inferred by the attention mechanism.
  Return:
  - `output_values` which is the scaled dot product attention : $softmax( \frac{QK^T}{\sqrt{d_k}} ) V$
  - `attention` which is the same as `output_values`, but without multiplying by $V$
  """
  d_k = q.size()[-1]

  # Compute attn_logits
  attention = torch.matmul(q, k.transpose(-2, -1))
  attention /= d_k ** (1 / 2)

  # Apply mask if not None
  # The mask gives very big negative values to elements
  # that should be ignore my the attention mechanism.
  # Such values get sent to 0 by the softmax function.
  if mask is not None:
    attention = attention.masked_fill(mask == 0, -(10**14))

  # Pass through softmax
  attention = F.softmax(attention, dim=-1)

  # Weight values accordingly
  output_values = torch.matmul(attention, v)
  return output_values, attention


class MultiheadAttention(nn.Module):
  """
  Perform multi-head attention on the `q`, `k` and `v` matrices.

  Each of the heads is responsible for understanding one aspect of the input sequence.
  For example, in a transliteration (such as translation) task, one head may be
  responsible for the order of the words, one head for the grammatical structures... etc.
  """

  def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
    super().__init__()

    err_mess = "Embedding dimension must be 0 modulo number of heads."
    assert embed_dim % num_heads == 0, err_mess

    self.embed_dim = embed_dim  # dimension of concatenated heads
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    # Create linear layers for both qkv and output
    # TIP: Stack all weight matrices 1...h together for efficiency
    # `qkv_proj` outputs `3*embed_dim` because each of `q`, `k` and `v`
    # have their own coefficients in the output layer.
    # This is also why we chunk them in the `forward` method.
    self.o_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    self.qkv_proj = nn.Linear(in_features=input_dim, out_features=3 * embed_dim)

    self._reset_parameters()

  def _reset_parameters(self):
    # Original Transformer initialization, see PyTorch documentation

    nn.init.xavier_uniform_(self.qkv_proj.weight)
    self.qkv_proj.bias.data.fill_(0)

    nn.init.xavier_uniform_(self.o_proj.weight)
    self.o_proj.bias.data.fill_(0)

  def forward(self, x: torch.Tensor, mask=None, return_attention=False):
    """
    Perform a forward pass on the multi-head attention block.

    This outputs the result of the two linear layers, with the scaled dot product in between.
    """
    # FILL IT YOURSELF!
    batch_dim, seq_length, input_dim = x.shape

    # Compute linear projection for qkv and separate heads
    # QKV: [Batch, Head, SeqLen, Dims]
    qkv = self.qkv_proj(x)
    qkv = qkv.reshape(batch_dim, seq_length, self.num_heads, 3 * self.head_dim)
    qkv = qkv.permute(0, 2, 1, 3)
    q, k, v = qkv.chunk(3, dim=-1)  # chunk by 3 because out_features=3 * embed_dim

    # Apply Dot Product Attention to qkv ()
    attention_values, attention = scaled_dot_product(q=q, k=k, v=v)

    # Concatenate heads to [Batch, SeqLen, Embed Dim]
    attention_values = attention_values.reshape(
      batch_dim,
      seq_length,
      self.embed_dim,
    )

    # Output projection
    o = self.o_proj(attention_values)

    if return_attention:
      return o, attention
    else:
      return o


class EncoderBlock(nn.Module):
  def __init__(
    self, input_dim: int, num_heads: int, dim_feedforward: int, dropout_prob=0.0
  ):
    """
    Perform Multi-Head attention + 1-hidden-layer feed-forward network.

    Args:
        input_dim: Dimensionality of the input
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
    """
    super().__init__()
    # FILL IT YOURSELF!

    # Create Attention layer
    self.self_attn = MultiheadAttention(
      input_dim=input_dim,
      embed_dim=input_dim,  # We assume it according to exercise
      num_heads=num_heads,
    )

    # Create Two-layer MLP with dropout
    # Dropout is done on the hidden layer to help with the network make
    # progress and to prevent the vanishing gradient problem.
    self.ffn = nn.Sequential(
      nn.Linear(in_features=input_dim, out_features=dim_feedforward),
      nn.Dropout(p=dropout_prob),
      nn.ReLU(inplace=True),  # Adds non-linearity to the network.
      nn.Linear(in_features=dim_feedforward, out_features=input_dim),
    )

    # Layers to apply in between the main layers (Layer Norm and Dropout)
    # We could probably use the same dropout for attention and ffn
    # because PyTorch most certainly doesn't use any backward operation on them,
    # but it also doesn't hurt to have two different objects.
    self.layer_norm_self_attn = nn.LayerNorm(normalized_shape=input_dim)
    self.dropout_self_attn = nn.Dropout(p=dropout_prob)

    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=input_dim)
    self.dropout_ffn = nn.Dropout(p=dropout_prob)

  def forward(self, x, mask=None):
    # Compute Attention part
    attended = self.self_attn(x)
    x = self.layer_norm_self_attn(x + self.dropout_self_attn(attended))  # add and norm

    # Compute MLP part
    fedforward = self.ffn(x)
    x = self.layer_norm_ffn(x + self.dropout_ffn(fedforward))  # add and norm

    return x


class TransformerEncoder(nn.Module):
  """
  Compute the encoder block.

  This block contains `num_layers` layers of encoder block.
  The forward pass iterates over them.
  """

  def __init__(self, num_layers, **block_args):
    super().__init__()
    self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask=mask)
    return x

  def get_attention_maps(self, x, mask=None):
    """Return the attention maps of each encoder block.
    This can be used to (try to) understand the behaviour of the transformer.
    In particular, this shows how much attention is put on each part of the input,
    then the human can make hypotheses about what pattern is identified by each head.
    """
    attention_maps = []
    for layer in self.layers:
      _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
      attention_maps.append(attn_map)
      x = layer(x)
    return attention_maps


class PositionalEncoding(nn.Module):
  """
  This is a smart way of choosing functions that intend to tell the network the order
  in which the sequence goes. By doing so, it is able to infer the "temporal"
  dimension of the input sequence (e.g. the order of the words in a transliteration task).
  """

  def __init__(self, d_model, max_len=5000):
    """
    Args
        d_model: Hidden dimensionality of the input.
        max_len: Maximum length of a sequence to expect.
    """
    super().__init__()

    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
    # Used for tensors that need to be on the same device as the module.
    # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
    self.register_buffer("pe", pe, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.pe[:, : x.size(1)]
    return x


def display_positional_encoding() -> None:
  """
  To understand how positional encoding works, this function plots the value that is
  passed by the `PositionalEncoding` class as a function of the hidden dimension and
  the position in the input sequence.
  """
  encod_block = PositionalEncoding(d_model=48, max_len=96)
  pe = encod_block.pe.squeeze().T.cpu().numpy()

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
  pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
  fig.colorbar(pos, ax=ax)
  ax.set_xlabel("Position in sequence")
  ax.set_ylabel("Hidden dimension")
  ax.set_title("Positional encoding over hidden dimensions")
  ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
  ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
  plt.show()


class TransformerPredictor(nn.Module):
  """
  Contain all blocks for a transformer encoder + a prediction. In particular:
    - A linear layer with dropout
    - The positional indication given to the network thanks to the `PositionalEncoding` class
    - A `TransformerEncoder` that contains each of the `EncoderBlock`s
    - A linear feed-forward network with:
      - A hidden layer
      - Layer normalization
      - ReLU
      - Dropout
      - A classification layer
  """

  def __init__(
    self,
    input_dim,
    model_dim,
    num_classes,
    num_heads,
    num_layers,
    dropout=0.0,
    input_dropout=0.0,
  ):
    """
    Args:
        input_dim: Hidden dimensionality of the input
        model_dim: Hidden dimensionality to use inside the Transformer
        num_classes: Number of classes to predict per sequence element
        num_heads: Number of heads to use in the Multi-Head Attention blocks
        num_layers: Number of encoder blocks to use.
        lr: Learning rate in the optimizer
        warmup: Number of warmup steps. Usually between 50 and 500
        max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
        dropout: Dropout to apply inside the model
        input_dropout: Dropout to apply on the input features
    """
    super().__init__()
    self.input_dim = input_dim
    self.model_dim = model_dim
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.input_dropout = input_dropout

    # FILL IT YOURSELF!

    # Create a Generic Input Encoder Input dim -> Model dim with input dropout
    self.input_net = nn.Sequential(
      nn.Linear(in_features=self.input_dim, out_features=self.model_dim),
      nn.Dropout(p=input_dropout),
    )

    # Create positional encoding for sequences
    self.positional_encoding = PositionalEncoding(d_model=self.model_dim)

    # Create transformer Encoder
    self.transformer = TransformerEncoder(
      num_layers=self.num_layers,
      input_dim=self.model_dim,
      num_heads=self.num_heads,
      dim_feedforward=self.model_dim,
      dropout_prob=self.input_dropout,
    )

    # Create output classifier per sequence element Model_dim -> num_classes
    self.output_net = nn.Sequential(
      nn.Linear(in_features=self.model_dim, out_features=self.model_dim),
      nn.LayerNorm(normalized_shape=self.model_dim),
      nn.ReLU(),
      nn.Dropout(p=self.dropout),
      nn.Linear(in_features=self.model_dim, out_features=self.num_classes),
    )

  def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    add_positional_encoding: bool = True,
  ):
    """
    Compute in turn each of the elements described above, that is,
    all blocks from input to prediction.

    Args:
        x: Input features of shape [Batch, SeqLen, input_dim]
        mask: Mask to apply on the attention outputs (optional)
        add_positional_encoding: If True, we add the positional encoding to the input.
                                  Might not be desired for some tasks.
    """
    x = self.input_net(x)

    # Add positional encoding if the input is a sequence.
    # This is not needed for orderless inputs.
    if add_positional_encoding:
      x = self.positional_encoding(x)

    x = self.transformer(x, mask=mask)
    x = self.output_net(x)
    return x

  @torch.no_grad()
  def get_attention_maps(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    add_positional_encoding: bool = True,
  ):
    """Function for extracting the attention matrices of the whole Transformer for a single batch.

    Input arguments same as the forward pass.
    """
    x = self.input_net(x)
    if add_positional_encoding:
      x = self.positional_encoding(x)
    attention_maps = self.transformer.get_attention_maps(x, mask=mask)
    return attention_maps


class ReverseDataset(data.Dataset):
  """
  Reverse the order (flip) of the sequences that are passed.
  This allows to test transformers as a proof of concept for a basic task
  that some other networks (e.g. GRU, RNN) cannot solve.
  """

  def __init__(self, num_categories, seq_len, size):
    super().__init__()
    self.num_categories = num_categories
    self.seq_len = seq_len
    self.size = size

    self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    inp_data = self.data[idx]
    labels = torch.flip(inp_data, dims=(0,))
    return inp_data, labels


def train_step(model, x, y, optim):
  """
  Use `model` to make prediction, compute the loss and update the weights with
  one step of gradient descent, according to the strategy defined by `optim` (e.g. Adam).
  """
  model.train()

  # Fetch data and transform categories to one-hot vectors
  inp_data = F.one_hot(x, num_classes=model.num_classes).float()

  # Perform prediction and calculate loss and accuracy
  preds = model(inp_data, add_positional_encoding=True)
  loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
  acc = (preds.argmax(dim=-1) == y).float().mean()

  # Backpropagate and update weights
  loss.backward()
  optim.step()
  model.zero_grad()

  return loss, acc


def eval_step(model, x, y):
  """
  Evaluate the model after it has been trained.
  Use the OneHotEncodings to determine the class with highest probability.
  """
  with torch.no_grad():
    model.eval()

    # Fetch data and transform categories to one-hot vectors
    inp_data = F.one_hot(x, num_classes=model.num_classes).float()

    # Perform prediction and calculate loss and accuracy
    preds = model(inp_data, add_positional_encoding=True)
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
    acc = (preds.argmax(dim=-1) == y).float().mean()

  return loss, acc


def train_model(model, train_loader, val_loader, test_loader, optim, epochs=5):
  """
  Update the weights with gradient descent, that is, making several steps with `train_step`.
  Save the model in terms of average validation accuracy over batches.
  Use tqdm to give a progress bar to the user.
  """
  best_acc = 0.0
  pbar = tqdm(range(epochs))
  for e in range(epochs):
    train_loss, train_acc = 0.0, 0.0
    for x, y in train_loader:
      loss, acc = train_step(model, x, y, optim)
      train_loss += loss
      train_acc += acc

    val_loss, val_acc = 0.0, 0.0
    for x, y in val_loader:
      loss, acc = eval_step(model, x, y)
      val_loss += loss
      val_acc += acc

    if val_acc / len(val_loader) > best_acc:
      torch.save(model.state_dict(), "best_model.pt")
      best_acc = val_acc / len(val_loader)

    pbar.update()
    pbar.set_description(
      f"Train Acc: {train_acc/len(train_loader)* 100:.2f} "
      f"Train Loss: {train_loss/len(train_loader):.2f} "
      f"Val Acc: {val_acc/len(val_loader)* 100 :.2f}  "
      f"Val loss: {val_loss/len(val_loader):.2f} "
    )

  test_loss, test_acc = 0.0, 0.0
  for x, y in test_loader:
    loss, acc = eval_step(model, x, y)
    test_loss += loss
    test_acc += acc

  print(f"Test accuracy: {test_acc/len(test_loader)*100 :.2f}")

  pbar.close()
  model.load_state_dict(torch.load("best_model.pt"))

  return model
