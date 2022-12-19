"""
Utility functions for assignment 4.
"""

import copy
import os
import time
from typing import Callable
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from plotly.subplots import make_subplots
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

pio.templates.default = "plotly_white"


class Data:
  def __init__(self, dl_num_workers: int = 4, dl_batch_size: int = 16) -> None:
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self._load_data(dl_num_workers=dl_num_workers, dl_batch_size=dl_batch_size)

  def _load_data(
    self,
    dl_num_workers: int,
    dl_batch_size: int,
    use_subset: bool = True,
    subset_n_samples: int = 1024,
  ) -> None:
    self.TRAIN_DIR = "data/cifar10_train"
    self.VAL_DIR = "data/cifar10_val"

    # Define means and stds to normalize data
    self.TRAIN_MEANS = (0.4914, 0.4822, 0.4465)
    self.TRAIN_STDS = (0.247, 0.243, 0.261)

    # Data augmentation and normalization for training
    self.train_transforms = transforms.Compose(
      [
        transforms.RandomResizedCrop(size=224, scale=[0.75, 1.0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.TRAIN_MEANS, std=self.TRAIN_STDS),
      ]
    )
    # Just normalization for validation
    self.val_transforms = transforms.Compose(
      [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.TRAIN_MEANS, std=self.TRAIN_STDS),
      ]
    )
    self.train_dataset = CIFAR10(
      root=self.TRAIN_DIR,
      train=True,
      transform=self.train_transforms,
      download=True,
    )
    self.val_dataset = CIFAR10(
      root=self.VAL_DIR,
      train=False,
      transform=self.val_transforms,
      download=True,
    )
    train_size = len(self.train_dataset)
    val_size = len(self.val_dataset)
    self.class_names = self.train_dataset.classes

    print("Train size:", train_size)
    print("Val size:", val_size)
    print("Class names:", self.class_names)

    # Use subset dataset if instructed to
    if use_subset:
      self.train_dataset = Subset(
        dataset=self.train_dataset, indices=[*range(subset_n_samples)]
      )
      self.val_dataset = Subset(
        dataset=self.val_dataset, indices=[*range(subset_n_samples)]
      )

    self.train_dl = DataLoader(
      self.train_dataset,
      shuffle=True,
      batch_size=dl_batch_size,
      num_workers=dl_num_workers,
    )
    self.val_dl = DataLoader(
      self.val_dataset,
      batch_size=dl_batch_size * 2,
      num_workers=dl_num_workers,
    )

  def _imshow(self, inp: torch.Tensor, title: list = None) -> go.Figure:
    """
    Display `inp` images, along with their labels if `title` is passed.
    """
    inp = inp.numpy().transpose((1, 2, 0))  # reconvert to numpy tensor
    inp = self.TRAIN_STDS * inp + self.TRAIN_MEANS  # take out normalization
    inp = np.clip(inp, 0, 1)
    fig = px.imshow(inp, title=", ".join(title))

    return fig

  def imshow_train_val(self, num_img) -> None:
    """Visualize images from train and val datasets"""
    # Get a batch of training/valid data
    for d in [self.train_dl, self.val_dl]:
      x, classes = next(iter(d))

      # Make a grid from batch
      out = torchvision.utils.make_grid(x[:num_img])
      self._imshow(
        inp=out, title=[self.class_names[c] for c in classes[:num_img]]
      ).show()

  def train_one_epoch(
    self,
    model: nn.Module,
    train_dl: DataLoader,
    loss: Callable,
    optim: optim.Optimizer,
    device: torch.device,
  ):
    """Function to iterate over data while training."""

    model.train()  # Set model to training mode
    cur_loss, cur_acc = 0.0, 0.0
    for x, y in train_dl:
      x, y = x.to(device), y.to(device)

      # zero the parameter gradients
      optim.zero_grad()

      # forward
      outputs = model(x)
      preds = torch.argmax(outputs, 1)
      l = loss(outputs, y)

      # backward + optimize
      l.backward()
      optim.step()

      # statistics
      cur_loss += l.item() * x.size(0)
      cur_acc += torch.sum(preds == y.data)

    epoch_loss = cur_loss / len(train_dl.dataset)
    epoch_acc = cur_acc.double() / len(train_dl.dataset)
    return epoch_loss, epoch_acc

  def eval_one_epoch(
    self, model: nn.Module, val_dl: DataLoader, loss: Callable, device: torch.device
  ):
    """Iterate over data while evaluating"""
    model.eval()  # Set model to training mode
    cur_loss, cur_acc = 0.0, 0.0
    with torch.no_grad():
      for x, y in val_dl:
        x, y = x.to(device), y.to(device)

        # forward
        outputs = model(x)
        preds = torch.argmax(outputs, 1)
        l = loss(outputs, y)

        # statistics
        cur_loss += l.item() * x.size(0)
        cur_acc += torch.sum(preds == y.data)

    epoch_loss = cur_loss / len(val_dl.dataset)
    epoch_acc = cur_acc.double() / len(val_dl.dataset)
    return epoch_loss, epoch_acc

  def train_model(
    self,
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    loss: Callable,
    optim: optim.Optimizer,
    num_epochs: int = 25,
  ):
    """Train the given `model` on specified dataloaders."""
    model.to(self.DEVICE)
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
      print(f"Epoch {epoch}/{num_epochs - 1}")
      print("-" * 10)

      train_loss, train_acc = self.train_one_epoch(
        model=model, train_dl=train_dl, loss=loss, optim=optim, device=self.DEVICE
      )
      print(f"Train Loss: {train_loss:<15f} Acc: {train_acc:<15f}")

      val_loss, val_acc = self.eval_one_epoch(model, val_dl, loss, self.DEVICE)
      print(f"Val Loss:   {val_loss:<15f} Acc: {val_acc:<17f}\n")

      # save the best model
      if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "temp_model.pt")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load("temp_model.pt"))
    return model
