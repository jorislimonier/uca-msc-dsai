import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dimension=784, num_classes=10):
        super(LinearModel, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes)

    def forward(self, x):
        bz = x.size(0)
        x = x.view(bz, -1)
        return self.fc(x)


class CNNModel(nn.Module):
    pass

