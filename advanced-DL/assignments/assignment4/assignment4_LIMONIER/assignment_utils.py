"""
Utility functions for assignment 4.
"""

import time
from typing import Callable, Optional

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
from livelossplot import PlotLosses
from plotly.subplots import make_subplots
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, models, transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

pio.templates.default = "plotly_white"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


class Data:
  def __init__(
    self,
    dl_num_workers: int = 4,
    dl_batch_size: int = 16,
    use_subset: bool = True,
    subset_n_samples: int = 2048,
  ) -> None:

    # Load data
    self._load_data(
      dl_num_workers=dl_num_workers,
      dl_batch_size=dl_batch_size,
      use_subset=use_subset,
      subset_n_samples=subset_n_samples,
    )

  def _load_data(
    self,
    dl_num_workers: int,
    dl_batch_size: int,
    use_subset: bool = True,
    subset_n_samples: int = 2048,
  ) -> None:
    """
    Perform all data loading operations.
    """
    self.TRAIN_DIR = "data/cifar10_train"
    self.VAL_DIR = "data/cifar10_val"

    # Define means and stds to normalize data.
    # This prevents having to load the entire dataset, computing
    # its means & std, then having to fo through the whole dataset again
    # to normalize it.
    # Values obtained from https://github.com/kuangliu/pytorch-cifar/issues/19
    self.TRAIN_MEANS = (0.4914, 0.4822, 0.4465)
    self.TRAIN_STDS = (0.247, 0.243, 0.261)

    # Data augmentation and normalization for training.
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

    print(f"{'Train size:':<15}", train_size)
    print(f"{'Val size:':<15}", val_size)
    print(f"{'Class names:':<15}", ", ".join(self.class_names))

    # Use subset dataset if instructed to
    if use_subset:
      indices = [*range(subset_n_samples)]
      self.train_dataset = Subset(dataset=self.train_dataset, indices=indices)
      self.val_dataset = Subset(dataset=self.val_dataset, indices=indices)

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


class Prediction:
  """
  Class for everything related to prediction.
  It contains the functions to train each variation of the models.
  """

  def __init__(self) -> None:
    pass

  def extract_features(self, dl: DataLoader):
    """
    Make dataset with extracted features.
    """
    cnn_model = models.resnet18(weights=torchvision.models.ResNet18_Weights)
    cnn_model.fc = torch.nn.Identity()
    print(next(cnn_model.parameters()).device)
    cnn_model.to(DEVICE)

    x_extr, y_extr = [], []
    with torch.no_grad():
      for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = cnn_model(x)

        x_extr.append(preds)
        y_extr.append(y)

      x_extr = torch.cat(x_extr, dim=0)
      y_extr = torch.cat(y_extr, dim=0)

      dataset_extr = TensorDataset(x_extr, y_extr)

    return dataset_extr

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
    print_epoch_res: bool = False,
    max_train_time: Optional[float | int] = None,
  ) -> nn.Module:
    """
    Train the given `model` on specified dataloaders.
    Also plot the train & val accuracy, as well as the train & val loss.
    """

    model.to(DEVICE)  # Move model to gpu if available

    since = time.time()  # Measure training time

    # best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize variables
    best_acc = 0.0
    liveloss = PlotLosses()
    epoch = 0
    # Use try block to print final results
    # even if it encounters KeyboardInterrupt
    try:
      while epoch < num_epochs:
        logs = {}

        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Compute train metrics
        train_loss, train_acc = self.train_one_epoch(
          model=model,
          train_dl=train_dl,
          loss=loss,
          optim=optim,
          device=DEVICE,
        )
        if print_epoch_res:
          print(f"Train Loss: {train_loss:<15f} Acc: {train_acc:<15f}")

        # Compute val metrics
        val_loss, val_acc = self.eval_one_epoch(
          model=model,
          val_dl=val_dl,
          loss=loss,
          device=DEVICE,
        )
        if print_epoch_res:
          print(f"Val Loss:   {val_loss:<15f} Acc: {val_acc:<17f}\n")

        # Save if best model
        if val_acc > best_acc:
          best_acc = val_acc
          torch.save(model.state_dict(), "temp_model.pt")

        # Compute logs to be passed to liveloss
        logs["loss"] = train_loss
        logs["accuracy"] = train_acc.cpu()
        logs["val_loss"] = val_loss
        logs["val_accuracy"] = val_acc.cpu()

        liveloss.update(logs)
        liveloss.send()

        # Break from training loop if max time is specified and reached.
        if max_train_time is not None and since + max_train_time <= time.time():
          print(f"Max training time reached ({max_train_time}), breaking.")
          break

        epoch += 1

    # Do not raise error for KeyboardInterrupt
    except KeyboardInterrupt:
      pass

    # Print results even if user interrupted training
    finally:
      time_elapsed = time.time() - since  # Compute train time

      # Print final results
      print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
      print(f"Best val Acc: {best_acc:4f}")

      # Load best model weights
      model.load_state_dict(torch.load("temp_model.pt"))

    return model
