import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import orth

device = torch.device("cuda")


def shrinkage(x, theta):
  """
  Shrinkage operator.
  """
  return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ista(X, W_d, a, L, max_iter, eps):
  """
  ISTA algorithm.
  """
  eig, eig_vector = np.linalg.eig(W_d.T * W_d)

  assert L > np.max(eig)
  del eig, eig_vector

  recon_errors = []
  Z_old = np.zeros((W_d.shape[1], 1))

  for _ in range(max_iter):
    grad = W_d.T * (W_d * Z_old - X)
    Z_new = shrinkage(Z_old - 1 / L * grad, a / L)
    if np.sum(np.abs(Z_new - Z_old)) <= eps:
      break
    Z_old = Z_new
    recon_error = np.linalg.norm(X - W_d * Z_new, 2) ** 2
    recon_errors.append(recon_error)

  return Z_new, recon_errors


class LISTA(nn.Module):
  def __init__(self, n_meas: int, n_features: int, W_d, max_iter, L, theta):

    """
    # Arguments
        n_meas: int, dimensions of the measurement
        n_features: int, dimensions of the sparse signal
        W_d: array, dictionary
        max_iter:int, max number of internal iteration
        L: Lipschitz const
        theta: Thresholding
    """

    super(LISTA, self).__init__()
    self._W = nn.Linear(in_features=n_meas, out_features=n_features, bias=False)
    self._S = nn.Linear(in_features=n_features, out_features=n_features, bias=False)
    self.shrinkage = nn.Softshrink(theta)
    self.theta = theta
    self.max_iter = max_iter
    self.A = W_d
    self.L = L

  # Initialization of the weights (based on the Dictionary)
  def weights_init(self):
    A = self.A.cpu().numpy()
    L = self.L
    S = torch.from_numpy(np.eye(A.shape[1]) - (1 / L) * np.matmul(A.T, A))
    S = S.float().to(device)
    W = torch.from_numpy((1 / L) * A.T)
    W = W.float().to(device)
    self._S.weight = nn.Parameter(S)
    self._W.weight = nn.Parameter(W)

  def forward(self, Y) -> torch.Tensor:
    X = self.shrinkage(self._W(Y))
    if self.max_iter == 1:
      return X
    for iter in range(self.max_iter):
      X = self.shrinkage(self._W(Y) + self._S(X))
    return X


def train_lista(Y, dictionary, a, L, max_iter=30):
  """
  Train LISTA model.

  # Arguments
    - Y: array, measurements
    - dictionary: array, dictionary matrix (n, m)
    - a: float, thresholding parameter
    - L: float, Lipschitz constant
    - max_iter: int, max number of epochs
  """
  n, m = dictionary.shape
  n_samples = Y.shape[0]
  batch_size = 128
  steps_per_epoch = n_samples // batch_size

  # Convert to tensors
  Y = torch.from_numpy(Y)
  Y = Y.float().to(device)
  w_d = torch.from_numpy(dictionary)
  w_d = w_d.float().to(device)

  net = LISTA(n, m, w_d, max_iter=30, L=L, theta=a / L)
  net = net.float().to(device)
  net.weights_init()

  # build optimizer and criterion
  learning_rate = 1e-1
  criterion1 = nn.MSELoss()
  criterion2 = nn.L1Loss()
  all_zeros = torch.zeros(batch_size, m).to(device)
  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

  loss_list = []
  for epoch in range(max_iter):
    if epoch % 3 == 2:
      print(f"Epoch {epoch + 1:>3}: Loss: {(10**4) * loss.detach().data:6f} * 1e-4")

    index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
    Y_shuffle = Y[index_samples]

    for step in range(steps_per_epoch):
      Y_batch = Y_shuffle[step * batch_size : (step + 1) * batch_size]
      optimizer.zero_grad()

      # get the outputs
      x_h = net(Y_batch)
      y_h = torch.mm(x_h, w_d.T)

      # compute the loss
      loss1 = criterion1(Y_batch.float(), y_h.float())
      loss2 = a * criterion2(x_h.float(), all_zeros.float())
      loss = loss1 + loss2

      loss.backward()
      optimizer.step()

      with torch.no_grad():
        loss_list.append(loss.detach().data)

  return net, loss_list
