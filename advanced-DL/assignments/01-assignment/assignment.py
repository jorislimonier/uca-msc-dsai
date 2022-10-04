#%%
import gzip
import pathlib
from typing import Tuple

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


X_train, y_train, X_test, y_test = load_data_torch()
X_train.max(), X_test.max()
#%%
### Q1
# Try to load the same data directly from the "MINST database" website http://yann.lecun.com/exdb/mnist/
# Be careful that the images can have a different normalization and encoding

SELF_DOWNLOADED_PATH = pathlib.Path("data/MNIST/self-downloaded").resolve()


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


# Set data sets
X_train = load_data_ylc(
    file_name="train-images-idx3-ubyte.gz",
    is_image=True,
    nb_images=60000,
)
y_train = load_data_ylc(
    file_name="train-labels-idx1-ubyte.gz",
    is_image=False,
    nb_images=60000,
    normalize=False,
)
X_test = load_data_ylc(
    file_name="t10k-images-idx3-ubyte.gz",
    is_image=True,
    nb_images=10000,
)
y_test = load_data_ylc(
    file_name="t10k-labels-idx1-ubyte.gz",
    is_image=False,
    nb_images=10000,
    normalize=False,
)
#%%
# Transform labels to one_hot encoding
y_train_one_hot = torch.nn.functional.one_hot(
    y_train.to(torch.int64), num_classes=10
).float()
y_test_one_hot = torch.nn.functional.one_hot(
    y_test.to(torch.int64), num_classes=10
).float()

### Q2
# Using the utilities in plt and numpy display some images
# and check that the corresponding labels are consistent


def display_digits(nb_subplots: int = 12, cols: int = 4) -> go.Figure:
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


fig = display_digits()
fig.show()
fig.write_image("data/labels.png")
# %%

### Q3
# Complete the code below so to have a MLP with one hidden layer with 300 neurons
# Remember that we want one-hot outputs

# Now let us define the neural network we are using


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


hidden_sizes = [16, 16]
net = define_net(hidden_sizes=hidden_sizes)

# Now we define the optimizer and the loss function
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Initialize arrays to track errors
# The test error array is there for informative purposes.
# We do not use it when updating weights.
# In a real world scenario, we shoudln't even look at it to choose when to (early-) stop training.
error_train = []
error_test = []

inputs = torch.flatten(X_train, start_dim=1, end_dim=2)
labels = y_train_one_hot

# %%
sum([p.numel() for p in net.parameters()])

#%%
### Q4
# Complete the code below to perform a GD based optimization


for k in range(20000):
    optimizer.zero_grad()

    outputs = net(inputs)

    # Define the empirical risk
    risk = loss(outputs, labels)

    # Make the backward step (1 line instruction)
    risk.backward()

    # Update the parameters (1 line instruction)
    optimizer.step()

    with torch.no_grad():
        y_pred_one_hot = net(torch.flatten(X_test, start_dim=1, end_dim=2))
        prediction_loss = loss(y_pred_one_hot, y_test_one_hot)

        error_train.append(risk.item())
        error_test.append(prediction_loss.item())

        print(
            f"k = {k}, \tRisk = {risk.item()}, \tPrediction loss = {prediction_loss.item()}"
        )


#%%
df_results = pd.DataFrame({"train_error": error_train, "test_error": error_test})


def plot_errors(
    df_results: pd.DataFrame,
    hidden_sizes: list[int],
    log_y: bool = False,
    write_image: bool = True,
) -> go.Figure:
    # Write image without logarithmic scale

    if log_y:
        title="Cross-entropy loss (logarithmic scale)"
    else:
        title="Cross-entropy loss (no logarithmic scale)"

    print(title)

    fig = px.line(
        data_frame=df_results,
        log_y=log_y,
        title=title,
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Cross entropy loss")
    
    
    if write_image:
        filename_base = "-".join(
            ["data/cross-entropy-comparison", str(len(hidden_sizes))]
            + [str(h) for h in hidden_sizes]
        )

        if log_y:
            filename = f"{filename_base}-log.png"
        else:
            filename = f"{filename_base}.png"
        
        fig.write_image(filename)
    
    return fig


fig = plot_errors(df_results=df_results, hidden_sizes=hidden_sizes)
fig.show()

# Write image with logarithmic scale
fig = plot_errors(df_results=df_results, hidden_sizes=hidden_sizes, log_y=True)
fig.show()
#%%
### Q5
# Compute the final accuracy on test set

y_pred_one_hot = net(torch.flatten(X_test, start_dim=1, end_dim=2))
y_pred = torch.argmax(input=y_pred_one_hot, dim=1)
acc = (y_test == y_pred).sum() / len(y_test)
print("Final accuracy on test", float(acc))
# %%
