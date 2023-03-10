import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

# define the blur operator
def blur(x, h):
  return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))


# define the downsampling and forward operator
def down_sampling(x, M_L):
  return M_L @ x @ np.transpose(M_L)


def forward(x, h, M_L):
  return down_sampling(blur(x, h), M_L)


# fidelity
def fidelity(x, h, M_L, y):
  return 0.5 * np.linalg.norm(forward(x, h, M_L) - y) ** 2


# define the gradient of the fidelity term in terms of convolutions
def gradient(x, h, M_L, y):
  aus = np.transpose(M_L) @ (forward(x, h, M_L) - y) @ M_L
  return np.real(np.fft.ifft2(np.conj(np.fft.fft2(h)) * np.fft.fft2(aus)))


# cost function
def cost_function_norm0(x, h, M_L, y, lmbda):
  return fidelity(x, h, M_L, y) + lmbda * np.count_nonzero(x)


# proximity operator of the \ell_0 norm
def hard_thresholding(x, tau):
  return x * (x**2 > 2 * tau)


def IHT(x0, tau, lmbda, y, h, M_L, epsilon, maxiter):
  """
  Iterative Hard Thresholding algorithm.
  input parameters:
  - x0 is the initialisation point
  - tau is the stepsize
  - lmbda is the regularisation parameter
  - y is the acquisition, M_L is the downsampling matrix, h is the psf
    ---> needed to compute the gradient of f at each iteration
  - epsilon is the tolerance parameter, maxiter is the maximum numer of iterations
    ---> needed for the stopping criterion
  """
  xk = x0
  cost = np.zeros(maxiter)
  norms = np.zeros(maxiter)
  for k in np.arange(maxiter):

    # forward step
    xkk = xk - tau * gradient(xk, h, M_L, y)

    # backward step
    xkk = hard_thresholding(xkk, tau * lmbda)

    # negativity constraint
    xkk = np.maximum(xkk, 0)

    # compute the cost function
    cost[k] = cost_function_norm0(xkk, h, M_L, y, lmbda)
    norms[k] = np.linalg.norm(xkk - xk)

    # update the iteration
    xk = xkk
    if np.abs(cost[k] - cost[k - 1]) / cost[k] < epsilon:
      break
  return xk, cost, norms


def norm0(x):
  return x != 0


def norm1(x):
  return np.abs(x)


def capped_norm1(x, theta):
  return np.minimum(np.abs(x), theta)


def normP(x, p):  # implement the p-norm elevated to the p-power
  return np.abs(x) ** p


def logsum(x, delta):
  return np.log(delta + np.abs(x))


def MCP(x, lambd, gamma):
  return lambd * (
    lambd * gamma * 0.5 * (np.abs(x) > gamma * lambd)
    + (np.abs(x) - x**2 / (2 * gamma * lambd)) * (np.abs(x) <= gamma * lambd)
  )


def SCAD(x, lambd, a):
  return (
    np.abs(x) * (np.abs(x) <= 1)
    - 0.5
    * (x**2 - 2 * lambd * a * np.abs(x) + lambd**2)
    * (np.abs(x) > 1)
    * (np.abs(x) <= 2)
    + 0.5 * (a + 1) * lambd**2 * (np.abs(x) > 2)
  )


def CEL0(x, lambd, a):
  return lambd - 0.5 * a**2 * (np.abs(x) - np.sqrt(2 * lambd) / a) ** 2 * (
    np.abs(x) <= np.sqrt(2 * lambd) / a
  )


# cost function
def cost_function_norm1(x, h, M_L, y, lmbda):
  return np.linalg.norm(y - forward(x, h, M_L)) ** 2 + lmbda * np.sum(norm1(x))


# prox of \ell_1 norm: soft thresholding function
def soft_thresholding(x, gamma):
  return np.sign(x) * np.maximum(0, np.abs(x) - gamma)


def ISTA(x0, tau, lmbda, y, h, M_L, epsilon, maxiter):
  """
  input parameters
  - x0 is the initialisation point
  - tau is the stepsize
  - lambda is the regolarisation parameter
  - y is the acquisition, M_L is the downsampling matrix, h is the psf ---> needed to compute the gradient of f at each iteration
  - epsilon is the tolerance parameter, maxiter is the maximum numer of iterations ---> needed for the stopping criterion
  """
  xk = x0
  cost = np.zeros(maxiter)
  norms = np.zeros(maxiter)
  for k in np.arange(maxiter):
    # forward step: gradient descent of f
    xkk = xk - tau * gradient(xk, h, M_L, y)
    # backward step
    xkk = soft_thresholding(xkk, tau * lmbda)
    # positivity constraints
    xkk = np.maximum(0, xkk)
    # compute the cost function
    cost[k] = cost_function_norm1(xkk, h, M_L, y, lmbda)
    norms[k] = np.linalg.norm(xkk - xk)
    # update the iteration
    xk = xkk
    if np.abs(cost[k] - cost[k - 1]) / cost[k] < epsilon:
      break
  return xk, cost, norms


# prox of \ell_1 norm: soft thresholding function
def soft_thresholding(x, gamma):
  return np.sign(x) * np.maximum(0, np.abs(x) - gamma)


# cost function
def cost_function_CEL0(x, h, M_L, y, lmbda, normai):
  return np.linalg.norm(y - forward(x, h, M_L)) ** 2 + lmbda * np.sum(
    CEL0(x, lmbda, normai)
  )
