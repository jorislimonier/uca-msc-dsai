# %%
import numpy as np

# %% [markdown]
# ### Generate data
# %% [markdown]
# ### Forward pass
#
# The dimensions are as follows:
# $$\underbrace{y}_{(n \times 1)} = \underbrace{X}_{(n \times p)} \underbrace{W}_{(p \times 1)} + \underbrace{b}_{(n \times 1)}$$
# %% [markdown]
# ### Back propagation
# $$\frac{\partial Loss(y, \hat{y})}{\partial W} = \frac{\partial Loss(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial W}$$
# and
# $$\frac{\partial Loss(y, \hat{y})}{\partial b} = \frac{\partial Loss(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial b}$$
# With:
# - $$Loss(y, \hat{y}) := MLE(y, \hat{y}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
# - $$\frac{\partial Loss(y, \hat{y})}{\partial \hat{y}} = \sum_{i=1}^n -2(y_i - \hat{y}_i)$$
# - $$z := WX + b$$
# - $\sigma$ being the activation function (*e.g.* sigmoid)
# - $$\hat{y} := \sigma(z)$$

# %%


class NN_xor():
    def __init__(self, n_obs=5, n_features=2):
        self.n_obs = n_obs
        self.n_features = n_features
        self.X = self.generate_data()
        self.y_true = np.logical_xor(
            self.X[:, 0], self.X[:, 1]).astype(int).reshape(-1, 1)

        self.learn_rate = 10**-1
        self.W = np.random.randn(self.X.shape[1], 1)
        self.b = np.random.randn(1)

    def generate_data(self):
        np.random.seed(43)
        return np.random.randint(0, 2, size=(self.n_obs, self.n_features))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod
    def loss(y_true, y_hat):
        return (y_true - y_hat)**2

    @staticmethod
    def loss_deriv(y_true, y_hat):
        return -2 * (y_true - y_hat)

    def forward_pass(self):
        self.z = self.X @ self.W + self.b
        self.y_hat = self.sigmoid(self.z)

    def backprop(self):
        "Perform backpropagation"

        # Compute derivatives to update weights
        dloss_dyhat = self.loss_deriv(self.y_true, self.y_hat)
        dyhat_dz = self.sigmoid_deriv(self.y_hat)
        dz_dw = self.X
        dloss_dw = dloss_dyhat * dyhat_dz
        dloss_dw = dz_dw.T @ dloss_dw
        dloss_db = dloss_dyhat * dyhat_dz * 1

        # Update weights
        self.W -= self.learn_rate * dloss_dw
        self.b = self.b - self.learn_rate * np.sum(dloss_db)


# %%
nn = NN_xor(n_obs=7, n_features=2)
for epoch in range(10001):
    nn.forward_pass()
    nn.backprop()
    if epoch % 2000 == 0:
        print(f"{nn.y_hat}")  # print predictions
print(f"---\ntarget: \n{nn.y_true}")

# %%
nn.X, nn.y_true

# %% [markdown]
# **The Following will fail**, as expected, since perceptron cannot
# linearly separate a XOR gate (need second layer) 

# %%
to_pred = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
z = to_pred @ nn.W + nn.b
y_hat = nn.sigmoid(z)
to_pred, y_hat