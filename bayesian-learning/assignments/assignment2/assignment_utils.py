import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom, multivariate_normal, norm, uniform
from sklearn import datasets

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
def plot_fox_col_distr(col: str) -> go.Figure:
  """Plot the distribution of the `col` column
  of the fox dataset.\
  The plot consists of a histogram and a kernel density estimation.
  """
  fox = pd.read_csv("data.csv")
  fig = ff.create_distplot(hist_data=[fox[col]], group_labels=["weight"])

  fig.update_layout(title=f"Distribution of the {col} of foxes")
  return fig


def plot_dependent_var_vs_covariates() -> go.Figure:
  """Make a plot of two subplots of `area` and `groupsize`
  as a function of weight.
  """
  fox = pd.read_csv("data.csv")
  fig = make_subplots(
    rows=2, shared_xaxes=True, subplot_titles=["area vs weight", "groupsize vs weight"]
  )
  area_vs_weight_trace = go.Scatter(
    x=fox["weight"],
    y=fox["area"],
    mode="markers",
  )
  fig.add_trace(area_vs_weight_trace, row=1, col=1)
  groupsize_vs_weight_trace = go.Scatter(
    x=fox["weight"],
    y=fox["groupsize"],
    mode="markers",
  )
  fig.add_trace(groupsize_vs_weight_trace, row=2, col=1)

  fig.update_xaxes(title_text="weight")
  fig.update_yaxes(title_text="area", row=1, col=1)
  fig.update_yaxes(title_text="groupsize", row=2, col=1)
  fig.update_layout(
    showlegend=False,
    title="Plot of dependent variable vs potential covariates",
  )
  return fig


def sample_posterior_laplace(laplace_sol: list, n_samples: int) -> np.ndarray:
  posterior_samples = np.random.multivariate_normal(
    laplace_sol[0], laplace_sol[1], size=n_samples
  )
  return posterior_samples


def print_posterior_samples_stats(laplace_sol: list, n_samples: int) -> None:
  posterior_samples = sample_posterior_laplace(
  laplace_sol=laplace_sol, n_samples=n_samples
  )
  result_string = f"""\
  Mean of the parameters: {posterior_samples.mean(axis=0)}
  Std of the parameters: {posterior_samples.std(axis=0)}
  90% quantile bounds for the mean of each parameter are:
  {np.quantile(a=posterior_samples, q=[0.05, 0.95], axis=0)}
  """
  print(result_string)


# Exercise 4
