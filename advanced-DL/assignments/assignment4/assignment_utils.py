"""
Utility functions for assignment 4.
"""

import copy
import os
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from plotly.subplots import make_subplots
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets import CIFAR10

pio.templates.default = "plotly_white"


class Data:
  def __init__(self, dl_num_workers: int = 4, dl_batch_size: int = 256) -> None:
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self._load_data(dl_num_workers=dl_num_workers, dl_batch_size=dl_batch_size)

  def _load_data(
    self,
    dl_num_workers: int,
    dl_batch_size: int,
    use_subset: bool = True,
    subset_n_samples: int = 32,
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

  def _imshow(self, inp: torch.Tensor, title: list = None) -> go.Figure:
    """
    Display `inp` images, along with their labels if `title` is passed.
    """
    inp = inp.numpy().transpose((1, 2, 0))  # reconvert to numpy tensor
    inp = self.TRAIN_STDS * inp + self.TRAIN_MEANS  # take out normalization
    inp = np.clip(inp, 0, 1)
    fig = px.imshow(inp, title=", ".join(title))

    return fig
