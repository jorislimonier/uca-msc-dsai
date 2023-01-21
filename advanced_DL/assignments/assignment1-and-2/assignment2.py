#%%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import torch.optim as optim  # Where the optimization modules are
import torchvision  # To be able to access standard datasets more easily
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

DEVICE = torch.device("cuda:0")
# %%
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
    X_train /= X_train.max()
    X_test /= X_test.max()

  # Reshape to 2D tensor
  X_train = X_train.reshape(len(X_train), -1)
  X_test = X_test.reshape(len(X_test), -1)

  # Move to approriate device
  X_train = X_train.to(DEVICE)
  X_test = X_test.to(DEVICE)
  y_train = y_train.to(DEVICE)
  y_test = y_test.to(DEVICE)

  return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data_torch()
y_train_one_hot = F.one_hot(y_train, num_classes=10).to(torch.float).to(DEVICE)
y_test_one_hot = F.one_hot(y_test, num_classes=10).to(torch.float).to(DEVICE)

X_train
# %%
class MnistMLP(nn.Module):
  def __init__(
    self,
    hidden_sizes: list[int],
    activation_fn: nn.Module = nn.Sigmoid(),
    input_size: int = 784,
    output_size: int = 10,
    device: torch.device = DEVICE,
  ) -> None:
    super().__init__()

    self.input_size = input_size
    self.output_size = output_size
    self.hidden_sizes = hidden_sizes
    self.activation_fn = activation_fn
    self.device = device

    first_layer = nn.Linear(
      in_features=self.input_size, out_features=self.hidden_sizes[0]
    )

    all_out_sizes = self.hidden_sizes + [self.output_size]

    # Use ModuleList to tell pytorch that the list contains layers
    self.hidden_layers: list[nn.Module] = [first_layer]

    for idx, size in enumerate(all_out_sizes):
      # From second layer onwards
      if idx < len(all_out_sizes) - 1:
        next_size = all_out_sizes[idx + 1]

        # Add layer
        layer = nn.Linear(in_features=size, out_features=next_size)
        self.hidden_layers.append(layer)

    # Move layers to cuda
    self.hidden_layers = nn.Sequential(*[l.to(self.device) for l in self.hidden_layers])

  def forward(self, X):
    output = X

    for idx, layer in enumerate(self.hidden_layers):
      output = layer(output)

      if idx != len(self.hidden_layers) - 1:
        output = self.activation_fn(output)

    return output


#%%
def fit_model(
  train_dl: DataLoader,
  test_dl: DataLoader,
  nb_epochs: int,
  model: MnistMLP,
  optimizer: optim,
  loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(),
):
  error_train = []
  error_test = []

  for epoch in range(nb_epochs):
    model.train()

    for X_batch, y_batch in train_dl:
      optimizer.zero_grad()
      y_pred = model(X_batch)

      loss = loss_fn(y_pred, y_batch)
      loss.backward()

      optimizer.step()

    model.eval()
    with torch.no_grad():
      test_losses = []

      for X_batch, y_batch in test_dl:
        y_pred_one_hot = model(X_batch)
        prediction_loss = loss_fn(y_pred_one_hot, y_batch).cpu()
        test_losses.append(prediction_loss)

      if epoch % 1 == 0:
        print(f"Epoch {epoch},\t Loss: {prediction_loss:.5f}")

      error_train.append(loss.item())
      error_test.append(np.mean(test_losses))

  return {"error_train": error_train, "error_test": error_test}


def set_model_params_to_zero(model):
  """Question 2:
  Set all model parameters to 0 for all layers
  """
  for layer in model.hidden_layers:
    for param in layer._parameters:
      param_shape = layer._parameters[param].shape
      print(layer._parameters[param])
      layer._parameters[param] = torch.zeros(*param_shape, requires_grad=True).to(
        DEVICE
      )
      print(layer._parameters[param])
  return model


#%%
def plot_sorted_train_labels(train_ds: TensorDataset, write_image: bool = False):
  """Plot the sorted training labels."""
  train_ds_sorted = sorted(train_ds, key=lambda x: x[1])
  fig = px.imshow(np.reshape([d[1].item() for d in train_ds_sorted], (200, -1)))
  if write_image:
    fig.write_image("data/sorted-digits.png")

  return fig


def get_data_objects(
  X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = False
):
  ds = TensorDataset(X, y)
  dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)
  return ds, dl


#%%
def get_strat_shuffled_X_y(y, ds):
  # Assign samples to their label
  digit_samples = 10 * [torch.Tensor().to(DEVICE)]

  for X_sample, y_sample in ds:
    digit_samples[y_sample] = torch.cat([digit_samples[y_sample], X_sample])

  # After loop, reshape samples
  digit_samples = [d.reshape(-1, 784) for d in digit_samples]
  strat_shuffled_X = torch.cat(digit_samples)
  strat_shuffled_y = y.sort().values

  return strat_shuffled_X, strat_shuffled_y


# strat_shuffled_X_train, strat_shuffled_y_train = get_strat_shuffled_X_y(
#   y=y_train, ds=train_ds
# )

# strat_shuffled_train_ds, strat_shuffled_train_dl = get_data_objects(
#   X=strat_shuffled_X_train, y=strat_shuffled_y_train, batch_size=64
# )

#%%
# train_ds_sorted = sorted(train_ds, key=lambda x: x[1])
#%%
model = MnistMLP(hidden_sizes=[300])
lr = 0.0005
batch_size = 16
nb_epochs = 101

train_ds, train_dl = get_data_objects(
  X=X_train,
  y=y_train,
  batch_size=batch_size,
  shuffle=True,
)
test_ds, test_dl = get_data_objects(X=X_test, y=y_test, batch_size=batch_size * 16)

sgd = optim.SGD(model.parameters(), lr=lr)

loss_fn = torch.nn.CrossEntropyLoss()
adam = optim.Adam(model.parameters(), lr=lr)

# model = set_model_params_to_zero(model=model)
fit_model(
  train_dl=train_dl,
  test_dl=test_dl,
  nb_epochs=nb_epochs,
  model=model,
  optimizer=adam,
  loss_fn=loss_fn,
)

# %%
