#%%
import torch
import torchvision  #To be able to access standard datasets more easily
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np  # To plot and display stuff
import torch.optim as optim # Where the optimization modules are

# Using torchvision we can conveniently load some datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
trainset
#%%

# Extract tensor of data and labels for both the training and the test set
x, y = trainset.data.float(), trainset.targets
x_test, y_test = testset.data.float(), testset.targets


### Q1
# Try to load the same data directly from the "MINST database" website http://yann.lecun.com/exdb/mnist/
# Be careful that the images can have a different normalization and encoding

# Transform labels to one_hot encoding
y_one_hot = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=10).float()
y_test_one_hot = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=10).float()


### Q2
# Using the utilities in plt and numpy display some images and check that the corresponding labels are consistent


### Q3
# Complete the code below so to have a MLP with one hidden layer with 300 neurons
# Remember that we want one-hot outputs

# Now let us define the neural network we are using
net = torch.nn.Sequential(
    torch.nn.Linear(??, ??),
    torch.nn.Sigmoid(),
    torch.nn.Linear(??, ??),
)


# Now we define the optimizer and the loss function
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)


### Q4
# Complete the code below to perform a GD based optimization

for k in range(100):
    optimizer.zero_grad()

    inputs = torch.flatten(x, start_dim=1, end_dim=2)
    outputs = net(inputs)
    labels = y_one_hot

    #Define the empirical risk
    Risk = ??

    #Make the backward step (1 line instruction)
    ??

    #Upadte the parameters (1 line instruction)
    ??


    with torch.no_grad():
        print("k=", k, "   Risk = ", Risk.item())


### Q5
# Compute the final accuracy on test set

acc = ??

print("Final accuracy on test", acc)

