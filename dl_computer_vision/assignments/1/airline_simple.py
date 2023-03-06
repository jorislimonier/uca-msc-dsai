from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CustomLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(CustomLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x: torch.Tensor):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])

    return out


class PassengerPredictor:
  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def train_step(self, batch, model, optimizer, loss_fn):
    optimizer.zero_grad()
    pred = model(batch["sequence"].float())
    loss = loss_fn(pred, batch["label"])
    loss.backward()
    optimizer.step()
    return loss.item()

  def val_step(self, batch, model, loss_fn):
    with torch.no_grad():
      pred = model(batch["sequence"])
      loss = loss_fn(pred, batch["label"])
    return loss.item()

  def train(
    self, model, train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs
  ):
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
      train_loss = 0
      val_loss = 0
      model.train()
      for batch in train_dataloader:
        train_loss += self.train_step(batch, model, optimizer, loss_fn)
      model.eval()
      for batch in val_dataloader:
        val_loss += self.val_step(batch, model, loss_fn)
      train_losses.append(train_loss / len(train_dataloader))
      val_losses.append(val_loss / len(val_dataloader))
      print(
        f"Epoch {epoch}: train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}"
      )
    return train_losses, val_losses
