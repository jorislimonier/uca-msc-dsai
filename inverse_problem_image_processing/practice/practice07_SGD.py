import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.random import multivariate_normal, randn
from scipy.linalg import norm, svd, toeplitz


class LinReg(object):
  """A class for the least-squares regression with
  Ridge penalization"""

  def __init__(self, X, y, lbda):
    self.X = X
    self.y = y
    self.n, self.d = X.shape
    self.lbda = lbda

  # global gradient \nabla f
  def grad(self, w):
    return self.X.T.dot(self.X.dot(w) - self.y) / self.n + self.lbda * w

  # definition of components f_i
  def f_i(self, i, w):
    return (
      norm(self.X[i].dot(w) - self.y[i]) ** 2 / (2.0) + self.lbda * norm(w) ** 2 / 2.0
    )

  # definition of f
  def f(self, w):
    return (
      norm(self.X.dot(w) - self.y) ** 2 / (2.0 * self.n)
      + self.lbda * norm(w) ** 2 / 2.0
    )

  def grad_i(self, i, w):
    """gradient of component, \nabla f_i"""
    x_i = self.X[i]
    return (x_i.dot(w) - self.y[i]) * x_i + self.lbda * w

  def lipschitz_constant(self):
    """Return the Lipschitz constant of the gradient of the global function"""
    L = norm(self.X, ord=2) ** 2 / self.n + self.lbda
    return L

  def L_max_constant(self):
    """Return the L_max constant among all components"""
    L_max = np.max(np.sum(self.X**2, axis=1)) + self.lbda
    return L_max

  def mu_constant(self):
    """Return the strong convexity constant"""
    mu = min(abs(la.eigvals(np.dot(self.X.T, self.X)))) / self.n + self.lbda
    return mu


def simu_linreg(w, n, std=1.0, corr=0.5):
  """
  Simulation of the least-squares problem

  Parameters
  ----------
  x : np.ndarray, shape=(d,)
      The coefficients of the model

  n : int
      Sample size

  std : float, default=1.
      Standard-deviation of the noise

  corr : float, default=0.5
      Correlation of the features matrix
  """
  d = w.shape[0]
  cov = toeplitz(corr ** np.arange(0, d))
  X = multivariate_normal(np.zeros(d), cov, size=n)
  noise = std * randn(n)
  y = X.dot(w) + noise
  return X, y


def sgd(
  w0: np.ndarray,
  model: LinReg,
  indices: np.ndarray,
  steps: np.ndarray,
  w_min,
  n_samples: int,
  n_features: int,
  n_iter: int = 100,
  averaging_on: bool = False,
  momentum=0,
  verbose: bool = True,
  start_late_averaging: int = 0,
):
  """Stochastic gradient descent algorithm"""
  w = w0.copy()
  w_new = w0.copy()
  # n_samples, n_features = X.shape

  # average x
  w_average = w0.copy()
  w_test = w0.copy()
  w_old = w0.copy()

  # estimation error history
  errors = []
  err = 1.0

  # objective history
  objectives = []

  # Current estimation error
  if np.any(w_min):
    err = norm(w - w_min) / norm(w_min)
    errors.append(err)

  # Current objective
  obj = model.f(w)
  objectives.append(obj)

  if verbose:
    print("Lauching SGD solver...")
    print(" | ".join([name.center(8) for name in ["it", "obj", "err"]]))

  for k in range(n_iter):
    print("Iteration: ", k + 1, " / ", n_iter, end="\r")

    # Write here SGD update
    # w_new[:] = w - steps[k] * model.grad_i(indices[k], w)

    # Write here SGD with momentum update
    w_new[:] = w - steps[k] * model.grad_i(indices[k], w) + momentum * (w - w_old)
    w_old[:] = w

    # Update step
    w[:] = w_new

    # Late averaging condition
    if averaging_on and k >= start_late_averaging:
      # Naive way
      # w_average[:] = w[start_late_averaging:].mean()

      # Optimized way
      k_new = k - start_late_averaging
      w_average[:] = k_new / (k_new + 1) * w_average + 1 / (k_new + 1) * w

    else:
      w_average[:] = w

    if averaging_on:
      w_test[:] = w_average
    else:
      w_test[:] = w

    obj = model.f(w_test)

    if np.any(w_min):
      err = norm(w_test - w_min) / norm(w_min)
      errors.append(err)
    objectives.append(obj)

    if k % n_samples == 0 and verbose:
      if sum(w_min):
        print(
          " | ".join(
            [
              ("%d" % k).rjust(8),
              ("%.2e" % obj).rjust(8),
              ("%.2e" % err).rjust(8),
            ]
          )
        )
      else:
        print(" | ".join([("%d" % k).rjust(8), ("%.2e" % obj).rjust(8)]))

  if averaging_on:
    w_output = w_average.copy()
  else:
    w_output = w.copy()

  return w_output, np.array(objectives), np.array(errors)
