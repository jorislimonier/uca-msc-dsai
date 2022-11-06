import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom, multivariate_normal, norm, uniform
from sklearn import datasets


# Exercise 1
def beta_binomial_distribution(n: int, y: int, alpha: int, beta: int) -> np.array:
  """
  Returns the beta-binomial distribution over the space of
  parameter theta, given fixed parameters alpha, beta, y, and n.
  """
  # Declares a linear space for the parameter theta
  theta_space = np.linspace(0, 1)
  # Computes and return the distribution
  distribution = theta_space ** (y + alpha - 1) * (1 - theta_space) ** (
    n + beta - y - 1
  )
  return theta_space, distribution


def beta_binomial_lapproximation(n: int, y: int, alpha: int, beta: int) -> np.array:
  """
  Returns the Laplace approximation of the beta binomial
  distribution (i.e. Gaussian distribution) with parameters
  derived from the original distribution's parameters alpha,
  beta, y, and n.
  """
  theta_max = (y + alpha - 1) / (n + alpha + beta - 2)
  hessian = -(y + alpha - 1) / (theta_max**2) - (n + beta - y - 1) / (
    (1 - theta_max) ** 2
  )
  variance = -1 / hessian
  return norm(theta_max, np.sqrt(variance))


# Exercise 2
# Exercise 3
# Exercise 4
# Exercise 5
# Exercise 6
