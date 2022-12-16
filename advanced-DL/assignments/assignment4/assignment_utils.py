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
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

pio.templates.default = "plotly_white"

train_dir = "data/cifar10_train"
val_dir = "data/cifar10_val"

train_means = (0.4914, 0.4822, 0.4465)
train_stds = (0.247, 0.243, 0.261)

# Data augmentation and normalization for training
train_transforms = transforms.Compose(
  [
    transforms.RandomResizedCrop(224, [0.75, 1.0]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds),
  ]
)
# Just normalization for validation
val_transforms = transforms.Compose(
  [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds),
  ]
)
train_dataset = torchvision.datasets.CIFAR10(
  root=train_dir, train=True, transform=train_transforms, download=True
)
val_dataset = torchvision.datasets.CIFAR10(
  root=val_dir, train=False, transform=val_transforms, download=True
)
