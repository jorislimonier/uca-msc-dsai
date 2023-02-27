from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pio.templates.default = "plotly_white"


def create_sequences(data, seq_length) -> list[tuple[pd.DataFrame, float]]:
  """Create sequences of data for training."""
  sequences = []
  data_size = len(data)

  # create sequences
  for idx in range(data_size - seq_length):
    label_position = idx + seq_length
    sequence = data.iloc[idx:label_position]
    label = data.iloc[label_position]
    sequences.append((sequence, label))

  return sequences


class AirlineDataset(Dataset):
  def __init__(self, sequences: list):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]

    return dict(
      sequence=torch.tensor(sequence.values, dtype=torch.float),
      label=torch.tensor(label.values, dtype=torch.float),
    )


class AirlineDataModule(pl.LightningDataModule):
  def __init__(
    self,
    train_sequences: list,
    val_sequences: list,
    test_sequences: list,
    batch_size: int,
  ):
    super().__init__()
    self.train_sequences = train_sequences
    self.val_sequences = val_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train = AirlineDataset(sequences=self.train_sequences)
    self.val = AirlineDataset(sequences=self.val_sequences)
    self.test = AirlineDataset(sequences=self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
      dataset=self.train,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=8,
    )

  def val_dataloader(self):
    return DataLoader(
      self.val,
      batch_size=1,
      shuffle=False,
      num_workers=8,
    )

  def test_dataloader(self):
    return DataLoader(
      self.test,
      batch_size=1,
      shuffle=False,
      num_workers=1,
    )


class PassengerPredictionModel(nn.Module):
  def __init__(self, n_features, n_hidden=20, n_layers=2):
    super().__init__()

    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,  # number of stacked LSTM layers
      batch_first=True,
      dropout=0.1,
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def forward(self, x):
    self.lstm.flatten_parameters()
    _, (hidden_state, _) = self.lstm(x)

    # get the last hidden state
    out = hidden_state[-1]

    return self.linear(out)


class PassengerPredictor(pl.LightningModule):
  def __init__(
    self,
    n_features: int,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
  ):
    super().__init__()
    self.model = PassengerPredictionModel(
      n_features=n_features, n_layers=1, n_hidden=20
    )
    self.criterion = nn.MSELoss()
    self.lr = lr
    self.weight_decay = weight_decay

  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0

    if labels is not None:
      loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, _ = self(sequences, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, outputs = self(sequences, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, outputs = self(sequences, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.AdamW(
      params=self.parameters(),
      lr=self.lr,
      weight_decay=self.weight_decay,
    )
