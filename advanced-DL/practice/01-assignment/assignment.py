#%%
import gzip
import pathlib

import matplotlib.pyplot as plt  # To plot and display stuff
import numpy as np
import torch
import torch.optim as optim  # Where the optimization modules are
import torchvision  # To be able to access standard datasets more easily
from torchvision.transforms import ToTensor

# Using torchvision we can conveniently load some datasets
train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=ToTensor()
)
test = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=ToTensor()
)

#%%

# Extract tensor of data and labels for both the training and the test set
X_train, y_train = train.data.float(), train.targets
X_test, y_test = test.data.float(), test.targets

#%%
### Q1
# Try to load the same data directly from the "MINST database" website http://yann.lecun.com/exdb/mnist/
# Be careful that the images can have a different normalization and encoding

SELF_DOWNLOADED_PATH = pathlib.Path("data/MNIST/self-downloaded").resolve()


def load_downloaded_data(
    file_name: str, is_image: bool, image_size: int = 28, nb_images: int = 10000
) -> torch.Tensor:
    """Load data from the files downloaded at http://yann.lecun.com/exdb/mnist/
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

    return torch.Tensor(data)


# Set data sets
X_train = load_downloaded_data(
    file_name="train-images-idx3-ubyte.gz",
    is_image=True,
    nb_images=60000,
)
y_train = load_downloaded_data(
    file_name="train-labels-idx1-ubyte.gz",
    is_image=False,
    nb_images=60000,
)
X_test = load_downloaded_data(
    file_name="t10k-images-idx3-ubyte.gz",
    is_image=True,
    nb_images=10000,
)
y_test = load_downloaded_data(
    file_name="t10k-labels-idx1-ubyte.gz",
    is_image=False,
    nb_images=10000,
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
image = np.asarray(X_train[1])
plt.imshow(image)
plt.show()


# ### Q3
# # Complete the code below so to have a MLP with one hidden layer with 300 neurons
# # Remember that we want one-hot outputs

# # Now let us define the neural network we are using
# net = torch.nn.Sequential(
#     torch.nn.Linear(??, ??),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(??, ??),
# )


# # Now we define the optimizer and the loss function
# loss = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1)


# ### Q4
# # Complete the code below to perform a GD based optimization

# for k in range(100):
#     optimizer.zero_grad()

#     inputs = torch.flatten(X, start_dim=1, end_dim=2)
#     outputs = net(inputs)
#     labels = y_one_hot

#     #Define the empirical risk
#     Risk = ??

#     #Make the backward step (1 line instruction)
#     ??

#     #Upadte the parameters (1 line instruction)
#     ??


#     with torch.no_grad():
#         print("k=", k, "   Risk = ", Risk.item())


# ### Q5
# # Compute the final accuracy on test set

# acc = ??

# print("Final accuracy on test", acc)
