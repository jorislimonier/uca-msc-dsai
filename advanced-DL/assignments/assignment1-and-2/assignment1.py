import gzip
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt  # To plot and display stuff
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.optim as optim  # Where the optimization modules are
import torchvision  # To be able to access standard datasets more easily
from plotly.subplots import make_subplots
from torchvision.transforms import ToTensor

SELF_DOWNLOADED_PATH = pathlib.Path("data/MNIST/self-downloaded").resolve()


def load_data_torch(
  normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Use torchvision to conveniently load some datasets.
  Return X_train, y_train, X_test, y_test
  """
  train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=ToTensor()
  )
  test = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=ToTensor()
  )

  # Extract tensor of data and labels for both the training and the test set
  X_train, y_train = train.data.float(), train.targets
  X_test, y_test = test.data.float(), test.targets

  if normalize:
    X_train /= 255
    X_test /= 255

  return X_train, y_train, X_test, y_test


def load_data_ylc(
  file_name: str,
  is_image: bool,
  image_size: int = 28,
  nb_images: int = 10000,
  normalize: bool = True,
) -> torch.Tensor:
  """Load data from the files downloaded on Yann Le Cun's website (http://yann.lecun.com/exdb/mnist/)
  and convert it to a PyTorch Tensor.
  """
  f = gzip.open(SELF_DOWNLOADED_PATH / file_name)

  # Behave according to whether the file contains images or labels
  if is_image:
    # As per the docs, the first 16 values contain metadata
    offset = 16
    reshape_dims = [nb_images, image_size, image_size]
    read_bits = np.prod(reshape_dims)
  else:
    # As per the docs, the first 8 values contain metadata
    offset = 8
    reshape_dims = [nb_images]
    read_bits = nb_images

  f.read(offset)
  buf = f.read(read_bits)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(*reshape_dims)

  if normalize:
    data /= 255

  return torch.Tensor(data)


def display_digits(
  X_train: torch.Tensor, y_train: torch.Tensor, nb_subplots: int = 12, cols: int = 4
) -> go.Figure:
  # Compute the rows and columns arrangements
  nb_subplots = 12
  cols = 4

  if nb_subplots % cols:
    rem = 1
  else:
    rem = 0

  rows = nb_subplots // cols + rem
  fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f"Label: {int(y_train[idx])}" for idx in range(nb_subplots)],
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.1,
  )

  for idx in range(nb_subplots):
    row = (idx // cols) + 1
    col = idx % cols + 1
    img = X_train[idx]
    img = img.flip([0])
    trace = px.imshow(img=img, color_continuous_scale="gray")
    fig.append_trace(trace=trace.data[0], row=row, col=col)

  fig.update_layout(coloraxis_showscale=False)
  fig.update_xaxes(showticklabels=False)
  fig.update_yaxes(showticklabels=False)

  return fig


def define_net(hidden_sizes: list[int]) -> torch.nn.modules.container.Sequential:
  """Generate a PyTorch dense neural net with the specified hidden layer sizes."""
  layers = []
  for idx, h in enumerate(hidden_sizes):
    if idx == 0:
      layers.append(torch.nn.Linear(28 * 28, h))
      layers.append(torch.nn.Sigmoid())
    else:
      prev_hidden = hidden_sizes[idx - 1]
      layers.append(torch.nn.Linear(prev_hidden, h))
      layers.append(torch.nn.Sigmoid())

    if idx == len(hidden_sizes) - 1:
      layers.append(torch.nn.Linear(h, 10))

  net = torch.nn.Sequential(*layers)

  return net


def plot_errors(
  df_results: pd.DataFrame,
  hidden_sizes: list[int],
  log_y: bool = False,
  write_image: bool = True,
  lr: Optional[float] = None,
) -> go.Figure:
  # Write image without logarithmic scale

  if log_y:
    title = "Cross-entropy loss (logarithmic scale)"
  else:
    title = "Cross-entropy loss (no logarithmic scale)"

  print(title)

  fig = px.line(
    data_frame=df_results,
    log_y=log_y,
    title=title,
  )
  fig.update_xaxes(title_text="Epoch")
  fig.update_yaxes(title_text="Cross entropy loss")

  if write_image:
    filename = "-".join(
      ["data/cross-entropy-comparison", str(len(hidden_sizes))]
      + [str(h) for h in hidden_sizes]
    )

    if lr is not None:
      filename = f"{filename}-lr{lr}"

    if log_y:
      filename = f"{filename}-log.png"
    else:
      filename = f"{filename}.png"

    fig.write_image(filename)

  return fig
