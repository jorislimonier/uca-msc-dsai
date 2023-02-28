import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.fc = nn.Linear(self.input_dimension, self.num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
