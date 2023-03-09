from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

pio.templates.default = "plotly_white"


def ttv_split(
  df: pd.DataFrame,
  train_size: Optional[float] = None,
  val_size: Optional[float] = None,
  test_size: Optional[float] = None,
  shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Split a DataFrame into train, validation and test sets.

  If one of train_size, val_size and test_size is not specified, it will be
  computed as the complement of the other two.

  Args:
    df (pd.DataFrame): DataFrame to split.
    train_size (float, optional): Size of the training set. Defaults to 0.8.
    val_size (float, optional): Size of the validation set. Defaults to 0.1.
    test_size (float, optional): Size of the test set. Defaults to 0.1.
    shuffle (bool, optional): Whether to shuffle the DataFrame before splitting. Defaults to False.

  Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test sets.
  """
  n_na = pd.isna(np.array([train_size, val_size, test_size])).sum()

  if n_na == 0:
    mess = "Train, validation and test sizes must sum to 1."
    assert train_size + val_size + test_size == 1, mess
  elif n_na == 1:
    if pd.isna(train_size):
      train_size = 1 - val_size - test_size
    elif pd.isna(val_size):
      val_size = 1 - train_size - test_size
    elif pd.isna(test_size):
      test_size = 1 - train_size - val_size
  else:
    raise ValueError(
      "At least two of train_size, val_size and test_size must be specified."
    )

  # Compute sizes
  train_val_size = train_size + val_size
  train_val, test = train_test_split(df, train_size=train_val_size, shuffle=shuffle)
  train, val = train_test_split(
    train_val, train_size=train_size / train_val_size, shuffle=shuffle
  )

  return train, val, test


def scale_wrt(
  *dfs: List[pd.DataFrame],
  wrt: pd.DataFrame,
  feature_range=(0, 1),
) -> List[np.ndarray]:
  """
  Scale data frames with respect to a reference array.

  Args:
    dfs (List[pd.DataFrame]): DataFrames to scale.
    wrt (pd.DataFrame): DataFrame to scale with respect to.
    feature_range (tuple, optional): Range to scale to. Defaults to (0, 1).

  Returns:
    List[pd.DataFrame]: Scaled arrays.
  """
  scaler = MinMaxScaler(feature_range=feature_range)
  scaler.fit(wrt)

  scaled_dfs = []
  for df in dfs:
    scaled_df = df = pd.DataFrame(
      data=scaler.transform(df),
      columns=df.columns,
      index=df.index,
    )
    scaled_dfs.append(scaled_df)

  return scaled_dfs


def plot_tts(
  train: pd.DataFrame,
  val: pd.DataFrame,
  test: pd.DataFrame,
  title: str = "Airline passengers",
):
  """
  Plot train, validation and test sets.

  Args:
    train (pd.DataFrame): Train set.
    val (pd.DataFrame): Validation set.
    test (pd.DataFrame): Test set.
    title (str, optional): Title of the plot. Defaults to "Train, validation and test sets".
  """
  fig = go.Figure()

  # Add train trace
  fig.add_trace(
    go.Scatter(
      x=train.index,
      y=train.passengers,
      name="train",
      mode="lines",
    )
  )

  # Add validation trace
  fig.add_trace(
    go.Scatter(
      x=val.index,
      y=val.passengers,
      name="val",
      mode="lines",
    )
  )

  # Add test trace
  fig.add_trace(
    go.Scatter(
      x=test.index,
      y=test.passengers,
      name="test",
      mode="lines",
    )
  )
  fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Passengers")

  return fig


def create_sequences(
  data: pd.DataFrame, seq_length: int, target_col: str = None
) -> list[tuple[pd.DataFrame, float]]:
  """Create sequences of data for training.

  Args:
    data (pd.DataFrame): Data to create sequences from.
    seq_length (int): Length of the sequence.

  Returns:
    list[tuple[pd.DataFrame, float]]: List of sequences and labels.
  """
  sequences = []
  data_size = len(data)

  # create sequences
  for idx in range(data_size - seq_length):
    label_position = idx + seq_length
    sequence = data.iloc[idx:label_position]

    if target_col is None:
      label = data.iloc[label_position]
    else:
      label = data.iloc[label_position].loc[[target_col]]
    sequences.append((sequence, label))

  return sequences


def ohe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
  """
  One-hot-encode the specified columns in the dataframe, dropping the original
  columns.
  """
  for col in columns:
    # Perform one-hot-encoding if the column is in the dataframe
    # Prevents errors on notebook reruns
    if col in df.columns:
      ohe = pd.get_dummies(df[col], prefix=col, drop_first=True)
      df = pd.concat(objs=[df, ohe], axis=1)
      df = df.drop(columns=col)
  return df


def compute_pred_error(
  y_pred: np.ndarray, y_true: np.ndarray, loss_fn: nn.Module, seq_length: int
) -> float:
  """Compute the error between the predictions and the true values."""
  with torch.no_grad():
    error = loss_fn(
      torch.tensor(y_pred).reshape(-1),
      torch.tensor(y_true[seq_length:]),
    ).item()
  return error

class AirlineDataset(Dataset):
  """
  Dataset for the airline data.

  Args:
    sequences (list): List of sequences and labels.
  """

  def __init__(self, sequences: list, device: str = "cuda"):
    self.sequences = sequences
    self.device = device

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]

    return dict(
      sequence=torch.tensor(sequence.values, dtype=torch.float, device=self.device),
      label=torch.tensor(label.values, dtype=torch.float, device=self.device),
    )


class PassengerDataModule:
  """
  DataModule for the airline data.
  """

  def __init__(
    self,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    batch_size: int = 4,
    seq_length: int = 1,
    target_col: str = None,
    device: str = "cuda",
  ):
    self.train = train
    self.val = val
    self.test = test
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.target_col = target_col
    self.device = device
    self.setup()

  def setup(self):
    """
    Create sequences, datasets and dataloaders.
    """
    # Create sequences
    self.train_sequences = create_sequences(
      data=self.train, seq_length=self.seq_length, target_col=self.target_col
    )
    self.val_sequences = create_sequences(
      data=self.val, seq_length=self.seq_length, target_col=self.target_col
    )
    self.test_sequences = create_sequences(
      data=self.test, seq_length=self.seq_length, target_col=self.target_col
    )

    # Create datasets
    self.train_dataset = AirlineDataset(
      sequences=self.train_sequences,
      device=self.device,
    )
    self.val_dataset = AirlineDataset(
      sequences=self.val_sequences,
      device=self.device,
    )
    self.test_dataset = AirlineDataset(
      sequences=self.test_sequences,
      device=self.device,
    )

    # Create dataloaders
    self.train_dataloader = DataLoader(
      dataset=self.train_dataset,
      batch_size=self.batch_size,
      shuffle=False,
    )
    self.val_dataloader = DataLoader(
      dataset=self.val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
    )
    self.test_dataloader = DataLoader(
      dataset=self.test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
    )


class PassengerLSTM(nn.Module):
  """
  A LSTM model for the airline data.
  """

  def __init__(
    self,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    device: str,
    **kwargs,
  ):
    super(PassengerLSTM, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device
    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      device=device,
      **kwargs,
    )
    self.fc = nn.Linear(
      in_features=hidden_size, out_features=output_size, device=device
    )

  def forward(self, x: torch.Tensor):
    """
    Implement the forward pass of the model.

    The hidden state and cell state are initialized to zero, then passed to the LSTM.
    The output of the LSTM (the last hidden state) is passed to the linear layer
    to produce the final output.
    """
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])

    return out


class EarlyStopping:
  def __init__(
    self,
    patience: int = 10,
    verbose: bool = True,
    restore_best: bool = True,
    model_path: str = "best_model.pt",
  ):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_loss = torch.inf
    self.early_stop = False
    self.restore_best = restore_best
    self.model_path = model_path

  def __call__(
    self,
    val_loss: float,
    model: nn.Module,
    epoch: int,
  ):
    """
    Check if the model should stop training.
    """

    if val_loss < self.best_loss:
      self.save_checkpoint(val_loss=val_loss, model=model, model_path=self.model_path)
      self.best_loss = val_loss
      self.best_epoch = epoch
      self.counter = 0

    else:
      self.counter += 1
      if self.verbose:
        print(
          f"EarlyStopping counter: {self.counter} / {self.patience}",
        )
      if self.counter >= self.patience:
        self.early_stop = True

        if self.restore_best:
          self.restore_checkpoint(model=model, model_path=self.model_path)

  def save_checkpoint(self, val_loss: float, model: nn.Module, model_path: str):
    """
    Save the model to disk.
    """
    # if self.verbose:
    # print(f"New best val_loss. Saving model ...")

    torch.save(model.state_dict(), model_path)

  def restore_checkpoint(self, model: nn.Module, model_path: str):
    """
    Restore the model from disk.
    """
    # if self.verbose:
    print(f"Restoring model from epoch {self.best_epoch} ...")

    model.load_state_dict(torch.load(model_path))


class PassengerPredictor:
  def __init__(
    self,
    data_module: PassengerDataModule,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
  ):
    self.data_module = data_module
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def train_step(
    self,
    batch: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
  ):
    """Perform a training step."""
    optimizer.zero_grad()
    seq = batch["sequence"].float()
    pred = model(seq)
    loss = loss_fn(pred, batch["label"])

    loss.backward()
    optimizer.step()

    return loss.item()

  def val_step(self, batch: dict, model: nn.Module, loss_fn: nn.Module):
    """Perform a validation step."""
    with torch.no_grad():
      pred = model(batch["sequence"])
      loss = loss_fn(pred, batch["label"])
    return loss.item()

  def train(
    self,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    n_epochs: int,
    early_stopping: EarlyStopping = None,
  ):
    """Train the model."""
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):

      # Initialize/reset the loss
      train_loss = 0
      val_loss = 0

      # Set the model to training mode
      model.train()

      # Iterate over the training data
      for batch in self.data_module.train_dataloader:
        train_loss += self.train_step(
          batch=batch, model=model, optimizer=optimizer, loss_fn=loss_fn
        )

      # Set the model to evaluation mode
      model.eval()

      # Iterate over the validation data
      for batch in self.data_module.val_dataloader:
        val_loss += self.val_step(batch=batch, model=model, loss_fn=loss_fn)

      # Store the losses
      train_loss_batch = train_loss / len(self.data_module.train_dataloader)
      val_loss_batch = val_loss / len(self.data_module.val_dataloader)

      train_losses.append(train_loss_batch)
      val_losses.append(val_loss_batch)

      if early_stopping is not None:
        early_stopping(val_loss=val_loss_batch, model=model, epoch=epoch)

        if early_stopping.early_stop:
          print(
            f"Early stopping at epoch {epoch}, "
            + f"best val loss {early_stopping.best_loss} "
            + f"(epoch {early_stopping.best_epoch})"
          )

          # if early_stopping.restore_best:
          #   return early_stopping.best_model, train_losses, val_losses
          break

      if epoch % 10 == 0:
        print(
          f"Epoch {epoch}: train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}",
        )

    return train_losses, val_losses

  def predict(self, model: nn.Module, dataloader: DataLoader, future: int = 0):
    """Make predictions."""
    model.eval()
    preds = []
    for batch in dataloader:
      pred = model(batch["sequence"])
      preds.append(pred)

    # Make predictions for the future
    if future > 0:
      with torch.no_grad():
        last_sequence = batch["sequence"][:, -1, :]
        print(last_sequence.shape)
        for _ in range(future):
          pred = model(last_sequence)
          preds.append(pred)
          last_sequence = torch.cat((last_sequence, pred), dim=1)[:, -1, :]

    return torch.cat(preds).detach().cpu().numpy()
