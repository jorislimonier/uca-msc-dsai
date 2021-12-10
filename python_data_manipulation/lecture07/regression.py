# %% [markdown]
# We're solving a simple linear regression problem.
# 
# $$ y = a + b * x + e $$
# 
# We fix our target to be:
# $$ a = 1; b = 2 $$

# %%
import numpy as np

# %%
np.random.seed(42)

# %% [markdown]
# Let's generate some points and compute the solution.

# %%
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.rand(100,1)

# %%
idx = np.arange(100)
np.random.shuffle(idx)   # shuffle the dataset

# %%
train_idx = idx[:80]
val_idx = idx[80:]

# %%
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# %%
import matplotlib.pyplot as plt
%matplotlib inline

# %%
fig, axs = plt.subplots(1,2,figsize=(12,5))
axs[0].scatter(x_train, y_train)
axs[0].set_title('Training')
axs[1].scatter(x_val, y_val, c='r')
axs[1].set_title('Validation')

# %%
# Let's initialize a and b random
a = np.random.randn(1)
b = np.random.randn(1)
a, b 

# %% [markdown]
# We will use the MSE as our Error function
# 
# $$ MSE = \frac{1}{N} \sum (y - \hat{y})^2 $$
# 
# Substituting with the function we want to "learn" ($\hat{y} = a + b * x$)
# 
# $$ MSE = \frac{1}{N} \sum (y - a - b * x)^2 $$
# 
# To perform gradient descent we need to derive wrt `a` and `b`.
# 
# That is (using chain rule):
# 
# $$ \frac{\partial MSE}{\partial a} = \frac{\partial MSE}{\partial \hat{y_i}} * \frac{\partial \hat{y_i}}{\partial a} = \frac{1}{N}\sum 2(y_i - a - bx_i) * (-1) = -2 \frac{1}{N}\sum(y_i - \hat{y_i})$$
# 
# $$ \frac{\partial MSE}{\partial b} = \frac{\partial MSE}{\partial \hat{y_i}} * \frac{\partial \hat{y_i}}{\partial b} = \frac{1}{N}\sum 2(y_i - a - bx_i) * (-x_i) = -2 \frac{1}{N}\sum x_i(y_i - \hat{y_i})$$
# 
# Finally, to update the parameters we use gradient descent:
# 
# $$ a = a - lr * \frac{\partial MSE}{\partial a} $$
# $$ b = b - lr * \frac{\partial MSE}{\partial b} $$

# %%
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = a + b * x_train    # FF
    error = y_train - yhat
    
    loss = (error ** 2).mean() # MSE
    
    grad_a = -2 * error.mean()  
    grad_b = -2 * (x_train * error).mean()
    
    a = a - lr * grad_a      # Backpropagation
    b = b - lr * grad_b
    
print(a, b)

# %% [markdown]
# Perfect. To be sure we're doing it right let's do the `sklearn` version.

# %%
from sklearn.linear_model import LinearRegression

# %%
linr = LinearRegression()
linr.fit(x_train, y_train)

# %%
linr.intercept_, linr.coef_[0]

# %% [markdown]
# Good. Let's do it with `pytorch`.

# %%
import torch
import torch.optim as optim   # not used right now but later
import torch.nn as nn   # not used right now but later

# %%
# we need to convert `ndarray` to `tensor`
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# %%
type(x_train_tensor)

# %%
x_train_tensor.type()

# %%
# init at random
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
a, b

# %%
torch.manual_seed(42)

# %%
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    loss.backward()
    print(a.grad, b.grad)
    
    a = a - lr * a.grad
    b = b - lr * b.grad
    
    a.grad.zero_()   # forget everything, as the value is already stored.
    b.grad.zero_()

print(a, b)

# %% [markdown]
# Looks like we're losing the parameters. All operations on tensors change the `attached` function to compute gradients.

# %%
# second try
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    loss.backward()
    print(a.grad, b.grad)
    
    a -= lr * a.grad
    b -= lr * b.grad
    
    a.grad.zero_()
    b.grad.zero_()

print(a, b)

# %% [markdown]
# It would have been to good to be true....

# %%
# third try
lr = 1e-1
n_epochs = 1000

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    loss.backward()
    
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    a.grad.zero_()
    b.grad.zero_()

print(a, b)

# %% [markdown]
# Now we're talking. `with torch.no_grad()` context manager assures that `pytorch` steps back and does not mess with our final values.

# %% [markdown]
# We can simplify the code by using optimizers (SGD, Adam, ...). The goal of the optimizer is to update the parameters and zero the gradients between each epoch.

# %%
lr = 1e-1
n_epochs = 1000

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

optimizer = optim.SGD([a,b], lr=lr)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    loss.backward()
    
#     with torch.no_grad():
#         a -= lr * a.grad
#         b -= lr * b.grad
    optimizer.step()
    
#     a.grad.zero_()
#     b.grad.zero_()
    optimizer.zero_grad()

print(a, b)

# %% [markdown]
# We can also use different loss functions.

# %%
lr = 1e-1
n_epochs = 1000

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

loss_fn = nn.MSELoss()
optimizer = optim.SGD([a,b], lr=lr)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    # loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

print(a, b)

# %% [markdown]
# A `pytorch.Module` is a class that lets us describe everything we did so far in a more concise way.

# %%
class ManualLinearRegression(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
         

    def forward(self, x):
        return self.a + self.b * x
    

# %%
model = ManualLinearRegression()

# %%
model.state_dict()

# %%
model.parameters()

# %%
lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()  # note this! it's pytorch
    
    yhat = model(x_train_tensor)  # DO NOT CALL model.forward() explicitly.
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
print(model.state_dict())


