import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom, multivariate_normal, norm, uniform
from sklearn import datasets
import plotly.io as pio

pio.templates.default = "plotly_white"


# Exercise 1
def beta_binom_distr(n: int, y: int, alpha: int, beta: int) -> np.array:
  """
  Compute the beta binomial distribution as seen during lesson 2.
  """
  # Declare theta values
  theta_range = np.linspace(start=0, stop=1, num=100)
  # Compute the distribution
  distr = theta_range ** (y + alpha - 1) * (1 - theta_range) ** (n + beta - y - 1)
  return theta_range, distr


def beta_binom_laplace_approx(n: int, y: int, alpha: int, beta: int) -> np.array:
  """
  Returns the Laplace approximation of the beta binomial distribution
  as computed inexercise 1.
  """
  theta_max = (y + alpha - 1) / (n + alpha + beta - 2)
  hessian = -(y + alpha - 1) / (theta_max**2) - (n + beta - y - 1) / (
    (1 - theta_max) ** 2
  )
  variance = -1 / hessian

  # Scipy takes the std as a second argument, not the variance
  return norm(theta_max, np.sqrt(variance))


def plot_beta_binom_vs_laplace(y: int, n: int, alpha: int, beta: int) -> go.Figure:
  """Plot the comparison between"""
  # Compute the beta-binomial PDF
  theta_values, beta_binomial = beta_binom_distr(n=n, y=y, alpha=alpha, beta=beta)

  # Compute the Laplace approximation
  laplace_approx = beta_binom_laplace_approx(n=n, y=y, alpha=alpha, beta=beta)

  # Plot the results
  fig = go.Figure()

  # Add beta-binomial trace
  beta_binomial_norm = beta_binomial / np.sum(beta_binomial)
  approx_trace = go.Scatter(
    x=theta_values,
    y=beta_binomial_norm,
    name="PDF of Beta-Binomial distribution",
  )

  # Add Laplace approximation trace
  laplace_approx_norm = laplace_approx.pdf(theta_values) / np.sum(
    laplace_approx.pdf(theta_values)
  )
  distr_trace = go.Scatter(
    x=theta_values,
    y=laplace_approx_norm,
    name="Laplace approximation",
  )
  fig.add_traces(data=[approx_trace, distr_trace])
  fig.update_layout(
    showlegend=True,
    title="Comparison ground truth vs Laplace approximation",
  )

  return fig


# Exercise 2
# Exercise 3
# Exercise 4
# Exercise 5
# Exercise 6
