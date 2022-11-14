from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom, multivariate_normal, norm, uniform
from sklearn import datasets
from tools import *

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
  Return the Laplace approximation of the given model
  as computed in exercise 1.
  """
  theta_max = (x1 + x2) / 2
  hessian = -2
  variance = -1 / hessian

  # Scipy takes the std as a second argument, not the variance
  return norm(theta_max, np.sqrt(variance))


def plot_model_vs_laplace(x1: float, x2: float) -> go.Figure:
  """
  Plot the comparison between the given model
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
  """
  Plot an example of what the diversity of animals could look like.
  """
  fig = go.Figure()

  x = np.linspace(0, 90, num=100)
  y = np.flip(x) + np.random.normal(loc=0, scale=4, size=100)
  trace = go.Scatter(x=x, y=y, mode="markers")

  fig.add_trace(trace=trace)
  return fig


# Exercise 3
def plot_fox_col_distr(col: str) -> go.Figure:
  """
  Plot the distribution of the `col` column
  of the fox dataset.\
  The plot consists of a histogram and a kernel density estimation.
  """
  fox = pd.read_csv("data.csv")
  fig = ff.create_distplot(hist_data=[fox[col]], group_labels=["weight"])

  fig.update_layout(title=f"Distribution of the {col} of foxes")
  return fig


def plot_dependent_var_vs_covariates() -> go.Figure:
  """
  Make a plot of two subplots of `area` and `groupsize`
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


def laplace_approx_weight() -> np.ndarray:
  """
  Perform Laplace approximation to estimate mean and variance for the `weight` variable.
  """
  fox = pd.read_csv("data.csv")

  # Declare the likelihood with the sample parameters
  lik = "gaussian"
  mu = fox["weight"].mean()  # 4.53
  sigma = fox["weight"].std()  # 1.18
  params = [mu, sigma]

  # Defines the prior mean
  prior_mu_mean = 4.5
  prior_mu_sigma = 2.0
  prior_mu = ["gaussian", [prior_mu_mean, prior_mu_sigma]]

  # Defines the prior sigma
  sigma_inf = 0.0
  sigma_sup = 2.0
  prior_sigma = ["uniform", [sigma_inf, sigma_sup]]

  # Performs the optimization
  laplace_sol = laplace_solution(
    params=params,
    other_params=[],
    data=fox["weight"],
    lik=lik,
    priors=[prior_mu, prior_sigma],
  )

  return laplace_sol


def laplace_approx_weight_vs_area() -> np.ndarray:
  """
  Perform Laplace approximation using the `area` variable.
  """
  fox = pd.read_csv("data.csv")
  # Create a column for centered area
  fox["centered_area"] = fox["area"] - fox["area"].mean()

  # Declare the regression expression
  expression = "weight ~ centered_area"

  # Declare the likelihood
  lik = "gaussian"

  # Define the prior with hyperparameters
  prior_a_mean = 4.5
  prior_a_sigma = 0.11
  prior_a = ["gaussian", [prior_a_mean, prior_a_sigma]]

  prior_b_mean = 0.0
  prior_b_sigma = 1.0
  prior_b = ["gaussian", [prior_b_mean, prior_b_sigma]]

  sigma_inf = 0.0
  sigma_sup = 2.0
  prior_sigma_unif = [sigma_inf, sigma_sup]
  prior_sigma = ["uniform", prior_sigma_unif]

  priors = [prior_a, prior_b, prior_sigma]

  # Perform the optimization
  laplace_sol = laplace_solution_regression(
    expression=expression,
    data=fox,
    lik=lik,
    priors=priors,
  )

  return laplace_sol


def laplace_approx_weight_vs_groupsize() -> np.ndarray:
  """
  Perform Laplace approximation using the `area` variable.
  """
  fox = pd.read_csv("data.csv")
  fox["centered_groupsize"] = fox["groupsize"] - fox["groupsize"].mean()
  # Declare the regression expression
  expression = "weight ~ centered_groupsize"

  # Declare the likelihood
  lik = "gaussian"

  # Define the prior with hyperparameters
  prior_a_mean = 4.5
  prior_a_sigma = 0.11
  prior_a = ["gaussian", [prior_a_mean, prior_a_sigma]]

  prior_b_mean = 0.0
  prior_b_sigma = 1.0
  prior_b = ["gaussian", [prior_b_mean, prior_b_sigma]]

  sigma_inf = 0.0
  sigma_sup = 2.0
  prior_sigma_unif = [sigma_inf, sigma_sup]
  prior_sigma = ["uniform", prior_sigma_unif]

  priors = [prior_a, prior_b, prior_sigma]

  # Perform the optimization
  laplace_sol = laplace_solution_regression(
    expression=expression,
    data=fox,
    lik=lik,
    priors=priors,
  )

  return laplace_sol


def laplace_approx_weight_vs_area_and_groupsize() -> np.ndarray:
  """
  Perform Laplace approximation using the `area` variable.
  """
  fox = pd.read_csv("data.csv")
  fox["centered_area"] = fox["area"] - fox["area"].mean()
  fox["centered_groupsize"] = fox["groupsize"] - fox["groupsize"].mean()
  # Declare the regression expression
  expression = "weight ~ centered_area + centered_groupsize"

  # Declare the likelihood
  lik = "gaussian"

  # Define the prior with hyperparameters
  prior_a_mean = 4.5
  prior_a_sigma = 0.11
  prior_a = ["gaussian", [prior_a_mean, prior_a_sigma]]

  prior_b_mean = 0.0
  prior_b_sigma = 1.0
  prior_b = ["gaussian", [prior_b_mean, prior_b_sigma]]

  prior_b2_mean = 0.0
  prior_b2_sigma = 1.0
  prior_b2 = ["gaussian", [prior_b2_mean, prior_b2_sigma]]

  sigma_inf = 0.0
  sigma_sup = 2.0
  prior_sigma_unif = [sigma_inf, sigma_sup]
  prior_sigma = ["uniform", prior_sigma_unif]

  priors = [prior_a, prior_b, prior_b2, prior_sigma]

  # Perform the optimization
  laplace_sol = laplace_solution_regression(
    expression=expression,
    data=fox,
    lik=lik,
    priors=priors,
  )

  return laplace_sol


def sample_posterior_laplace(n_samples: int, var_used: list = []) -> np.ndarray:
  """
  Make `n_samples` samples from a multivariate normal ditribution from the Laplace solution.
  """
  if set(var_used) == {"area"}:
    laplace_sol = laplace_approx_weight_vs_area()
  elif set(var_used) == {"groupsize"}:
    laplace_sol = laplace_approx_weight_vs_groupsize()
  elif set(var_used) == {"area", "groupsize"}:
    laplace_sol = laplace_approx_weight_vs_area_and_groupsize()
  else:
    laplace_sol = laplace_approx_weight()

  posterior_samples = np.random.multivariate_normal(
    laplace_sol[0], laplace_sol[1], size=n_samples
  )
  return posterior_samples


def compute_posterior_samples_stats(
  n_samples: int,
  var_used: list = [],
  param_names: list = ["alpha", "beta"],
) -> None:
  """
  Print some statistics about the posterior samples from the Laplace solution.
  """
  print(12 * "-", "Output of Laplace computation", 12 * "-")
  posterior_samples = sample_posterior_laplace(n_samples=n_samples, var_used=var_used)

  print(2 * "\n", 12 * "-", "Stats on Laplace computation", 12 * "-")

  means = posterior_samples.mean(axis=0)
  st_devs = posterior_samples.std(axis=0)
  quantiles = np.quantile(a=posterior_samples, q=[0.05, 0.95], axis=0)

  stats = [means, st_devs, quantiles[0, :], quantiles[1, :]]
  stats = pd.DataFrame(stats, columns=param_names, index=["mean", "std", "10%", "90%"])
  stats = stats.transpose()
  display(stats)

  return means, st_devs, posterior_samples


def plot_samples_from_posterior(
  col: Union[str, list],
  n_samples: int = 1000,
  var_used: Optional[list] = None,
):
  # if isinstance(col, str):
  #   col = []

  if var_used is None:
    var_used = [col]
    param_names = ["alpha", "beta", "sigma"]
  else:
    if len(var_used) > 1:
      param_names = ["alpha", "beta", "beta2", "sigma"]

  fox = pd.read_csv("data.csv")
  post_means, post_stds, post_samples = compute_posterior_samples_stats(
    n_samples=n_samples, var_used=var_used, param_names=param_names
  )

  fox[f"centered_{col}"] = fox[col] - fox[col].mean()
  x_min = fox[f"centered_{col}"].min()
  x_max = fox[f"centered_{col}"].max()
  x_range = np.linspace(start=x_min, stop=x_max, num=100)

  # Create fig
  fig = go.Figure(layout=dict(showlegend=False))
  fig = px.scatter(fox, x=col, y="weight")
  fig.update_layout(showlegend=False)

  # Iterate over samples andadd trace
  for sample in post_samples:
    intercept, slope, *_ = sample
    sample_trace = go.Scatter(
      x=x_range + fox[col].mean(),
      y=intercept + slope * x_range,
      opacity=0.07,
      marker=dict(color="grey"),
    )
    fig.add_trace(sample_trace)

  # Add trace for the mean
  intercept, slope, *_ = post_means
  mean_trace = go.Scatter(
    x=x_range + fox[col].mean(),
    y=intercept + slope * x_range,
    marker=dict(color="black"),
  )
  fig.add_trace(mean_trace)

  return fig


# Exercise 4
