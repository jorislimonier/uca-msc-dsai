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


model = MnistMLP(hidden_sizes=[300])
lr = 0.5
batch_size = 128


nb_epochs = 101

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size * 16)
optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = torch.nn.CrossEntropyLoss()


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


model = set_model_params_to_zero(model=model)

fit_model(
  train_dl=train_dl,
  test_dl=test_dl,
  nb_epochs=nb_epochs,
  model=model,
  optimizer=optimizer,
  loss_fn=loss_fn,
)
# %%


model.hidden_layers[0]._parameters
