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
def model(x1: float, x2: float) -> np.array:
  """
  Compute the model given in the exercise's instructions.
  """
  # Declare theta values
  theta_range = np.linspace(start=0, stop=1, num=100)
  # Compute the distribution
  distr = np.exp(-((x1 - theta_range) ** 2 + (x2 - theta_range) ** 2) / 2) / np.sqrt(
    2 * np.pi
  )
  return theta_range, distr


def laplace_approx(x1: float, x2: float) -> np.array:
  """
  Returns the Laplace approximation of the given model
  as computed in exercise 1.
  """
  theta_max = (x1 + x2) / 2
  hessian = -2
  variance = -1 / hessian

  # Scipy takes the std as a second argument, not the variance
  return norm(theta_max, np.sqrt(variance))


def plot_model_vs_laplace(x1: float, x2: float) -> go.Figure:
  """Plot the comparison between the given model
  and its Laplace approximation.
  """
  # Compute the PDF of the given model
  theta_values, model_val = model(x1=x1, x2=x2)

  # Compute the Laplace approximation
  approx = laplace_approx(x1=x1, x2=x2)

  # Plot the results
  fig = go.Figure()

  # Add model trace
  model_val_norm = model_val / np.sum(model_val)
  approx_trace = go.Scatter(
    x=theta_values,
    y=model_val_norm,
    name="PDF of the given model",
  )

  # Add Laplace approximation trace
  laplace_approx_norm = approx.pdf(theta_values) / np.sum(approx.pdf(theta_values))
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
def possible_animal_diversity_model() -> go.Figure:
  fig = go.Figure()

  x = np.linspace(0, 90, num=100)
  y = np.flip(x) + np.random.normal(loc=0, scale=4, size=100)
  trace = go.Scatter(x=x, y=y, mode="markers")

  fig.add_trace(trace=trace)
  return fig

# Exercise 3
# Exercise 4
# Exercise 5
# Exercise 6
