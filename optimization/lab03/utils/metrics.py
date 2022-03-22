import torch


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc / len(y)
