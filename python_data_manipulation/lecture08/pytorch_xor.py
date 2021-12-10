# %%
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

# %%

EPOCHS_TO_TRAIN = 50000

# %%


class XOR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcl1 = nn.Linear(2, 2)
        self.fcl2 = nn.Linear(2, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        return x


# %%
xor = XOR()
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.logical_xor(inputs[:,0], inputs[:,1]).astype(int)

inputs, outputs

# %%
loss_fn = nn.MSELoss()
optimizer = optim.SGD(xor.parameters(), lr=0.1)

# %%
inputs[np.random.choice(inputs.shape[0], 4, replace=False), :]


# %%

# %%
XOR.forward()
