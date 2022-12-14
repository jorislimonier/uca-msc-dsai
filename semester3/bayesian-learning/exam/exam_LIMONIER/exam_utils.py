"""
Utility functions for the exam.
"""
from textwrap import dedent
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import stan
from IPython.display import display
from scipy import stats
from scipy.special import logsumexp
from tools import *
from tqdm.notebook import tqdm, tqdm_notebook, trange

pio.templates.default = "plotly_white"


def plot_distr_heatmap(
  freq: np.ndarray,
  X_values: np.ndarray,
  Y_values: np.ndarray,
  X_label: str = "X",
  Y_label: str = "Y",
) -> go.Figure:
  """Plot a heatmap of the given distribution of X and Y."""
  fig = px.imshow(
    img=freq,
    text_auto=True,
    color_continuous_scale="ylorbr",
    labels=dict(x=X_label, y=Y_label),
    x=X_values,
    y=Y_values,
    title="P(X, Y)",
  )
  return fig


def get_df_long(
  freq: np.ndarray, X_values: np.ndarray, Y_values: np.ndarray
) -> pd.DataFrame:
  """
  Create a df in long format with joint probabilities, that is:
    - One column for X
    - One column for Y
    - One column for P(X, Y)
  """
  df_freq = pd.DataFrame(
    freq,
    columns=[int(val) for val in X_values],
    index=[int(val) for val in Y_values],
  )
  df_freq = df_freq.reset_index(names="Y")
  freq_long = pd.melt(df_freq, id_vars="Y", var_name="X")
  return freq_long


class Exercise1:
  def __init__(self) -> None:

    self.freq = np.array([[0.0, 0.0, 0.2], [0.1, 0.1, 0.2], [0.1, 0.15, 0.15]])
    self.X_values = np.array(["1", "2", "3"])
    self.Y_values = np.array(["5", "10", "15"])
    self.prob_x = self.freq.sum(axis=0)
    self.prob_y = self.freq.sum(axis=1)
    self.freq_long = get_df_long(
      freq=self.freq, X_values=self.X_values, Y_values=self.Y_values
    )

  def check_valid_distribution(self) -> str:
    """
    Return printable information about whether P(X, Y) is a valid distribution.

    In particular, check that:
      - The sum of all probabilities is 1
      - All entries are between 0 and 1
    """

    # Check that probabilities sum up to 1
    msg_sum_1 = f"{'All probabilities sum up to 1':.<50}"
    bool_sum_1 = np.allclose(a=self.freq.sum(), b=1)

    # Check that all probabilities are between 0 and 1
    msg_non_neg = f"{'All probabilities are between 0 and 1':.<50}"
    bool_non_neg = np.all(a=[0 <= self.freq, self.freq <= 1])

    return f"{msg_sum_1}{bool_sum_1}\n{msg_non_neg}{bool_non_neg}"

  def check_independent(self) -> go.Figure:
    """
    Compute the distribution of product of X and Y in order
    to compare it with the joint distribution of X and Y.
    """
    prod = np.dot(self.prob_x.reshape(3, 1), self.prob_y.reshape(1, 3))
    fig = px.imshow(
      img=prod,
      text_auto=True,
      color_continuous_scale="ylorbr",
      labels=dict(x="X", y="Y"),
      x=self.X_values,
      y=self.Y_values,
      title="Product of the distributions of X and Y (P(X) * P(Y))",
    )
    return fig

  def compute_prob4(self) -> str:
    mask = (self.freq_long["Y"] == 10) & (self.freq_long["X"] < 2)
    prob_of = self.freq_long[mask]["value"].sum()

    prob_given = self.freq_long[self.freq_long["X"] < 2]["value"].sum()

    return f"P( Y = 10 | X < 2 ) = {prob_of/prob_given:.3f}"


class Exercise2:
  def __init__(self) -> None:
    self.freq = np.array([[0.05, 0.05, 0.05, 0.15], [0.15, 0.25, 0.05, 0.25]])
    self.X_values = ["1", "2", "3", "4"]
    self.Y_values = ["0", "1"]
    self.X_label = "Days sick before treatment (X)"
    self.Y_label = "Treatment effect (Y)"
    self.prob_x = self.freq.sum(axis=0)
    self.prob_y = self.freq.sum(axis=1)
    self.freq_long = get_df_long(
      freq=self.freq, X_values=self.X_values, Y_values=self.Y_values
    )

  def compute_prob1(self) -> str:
    """Compute the probability that Y = 1."""
    mask = self.freq_long["Y"] == 1
    prob_y_eq_1 = self.freq_long[mask]["value"].sum()

    return f"P( Y = 1 ) = {prob_y_eq_1:.1f}"

  def compute_prob2(self) -> str:
    """Compute the probability that Y = 1, given X = 4."""
    mask = (self.freq_long["Y"] == 1) & (self.freq_long["X"] == 4)
    prob_bayes_num = self.freq_long[mask]["value"].sum()

    prob_bayes_den = self.freq_long[self.freq_long["X"] == 4]["value"].sum()

    return f"P( Y = 1 | X = 4 ) = {prob_bayes_num/prob_bayes_den:.3f}"

  def compute_prob3(self) -> str:
    """Compute the probability that Y = 1, given X <= 4."""
    mask = (self.freq_long["Y"] == 1) & (self.freq_long["X"] <= 4)
    prob_bayes_num = self.freq_long[mask]["value"].sum()

    prob_bayes_den = self.freq_long[self.freq_long["X"] <= 4]["value"].sum()

    return f"P( Y = 1 | X <= 4 ) = {prob_bayes_num/prob_bayes_den:.2f}"

  def compute_prob4(self) -> str:
    """Compute the probability that X = 4, given Y = 1."""
    mask = (self.freq_long["X"] == 4) & (self.freq_long["Y"] == 1)
    prob_bayes_num = self.freq_long[mask]["value"].sum()

    prob_bayes_den = self.freq_long[self.freq_long["Y"] == 1]["value"].sum()

    return f"P( X = 4 | Y = 1 ) \u2248 {prob_bayes_num/prob_bayes_den:.3f}"

  def compute_prob5(self) -> str:
    """Compute the probability that X <= 4, given Y = 0."""
    mask = (self.freq_long["X"] <= 4) & (self.freq_long["Y"] == 0)
    prob_bayes_num = self.freq_long[mask]["value"].sum()

    prob_bayes_den = self.freq_long[self.freq_long["Y"] == 0]["value"].sum()

    return f"P( X <= 4 | Y = 0 ) = {prob_bayes_num/prob_bayes_den:.1f}"


class Exercise3:
  def __init__(self) -> None:
    self.freq = np.array(
      [[0.1, 0.0, 0.35, 0.05], [0.05, 0.15, 0.0, 0.05], [0.10, 0.0, 0.15, 0.0]]
    )
    self.X_values = ["0", "1", "2", "3"]
    self.Y_values = ["0", "1", "2"]
    self.X_label = "Nb 2-beds ambulances used on a given day (X)"
    self.Y_label = "Nb 1-bed ambulances used on a given day (Y)"
    self.prob_x = self.freq.sum(axis=0)
    self.prob_y = self.freq.sum(axis=1)
    self.freq_long = get_df_long(
      freq=self.freq, X_values=self.X_values, Y_values=self.Y_values
    ).infer_objects()

  def compute_prob1(self) -> str:
    """Compute the probability that X = 0 and Y = 0."""
    mask = (self.freq_long["X"] == 0) & (self.freq_long["Y"] == 0)

    prob_joint = self.freq_long[mask]["value"].sum()

    return f"P( X = 0, Y = 0 ) = {prob_joint:.1f}"

  def compute_prob2(self) -> str:
    """Compute the probability that Y = 2."""
    mask = self.freq_long["Y"] == 2
    prob = self.freq_long[mask]["value"].sum()

    return f"P( Y = 2 ) = {prob:.2f}"

  def compute_prob3(self) -> str:
    """Compute the probability that X + Y = 2."""
    prob = self.freq_long.query("X + Y > 1")["value"].sum()

    return f"P( X + Y >= 2 ) = {prob:.2f}"

  def compute_prob4(self) -> str:
    """Compute the probability that Y = 2, given X = 3."""
    mask = (self.freq_long["Y"] == 2) & (self.freq_long["X"] == 3)
    prob_bayes_num = self.freq_long[mask]["value"].sum()

    prob_bayes_den = self.freq_long[self.freq_long["X"] == 3]["value"].sum()

    return f"P( Y = 2 | X = 3 ) = {prob_bayes_num/prob_bayes_den:.1f}"


class Exercise4:
  def __init__(self) -> None:
    self._load_data()

  def _load_data(self):
    self.raw_data = np.load("./data_exercise4.npy")
    self.x = self.raw_data[0, :] - np.mean(self.raw_data[0, :])
    self.y = self.raw_data[1, :]
    self.data = pd.DataFrame({"x": self.x, "y": self.y})

  def plot_data(self) -> go.Figure:
    """Make a scatter plot of y vs x."""
    fig = px.scatter(data_frame=self.data, x="x", y="y", title="Data")
    return fig

  def run_laplace_sol(
    self,
    prior_a_mean=10,  # default from exercise
    prior_a_sigma=1,  # default from exercise
    prior_b_mean=0,  # default from exercise
    prior_b_sigma=1,  # default from exercise
    sigma_inf=20,  # default from exercise
    sigma_sup=50,  # default from exercise
  ) -> list:
    """Compute the Laplace optimization process with provided prior parameters."""
    expr = "y ~ x"

    # Initializing the likelihood
    likelihood = "gaussian"

    # Defining the prior with hyperparameters
    prior_a = ["gaussian", [prior_a_mean, prior_a_sigma]]

    prior_b = ["gaussian", [prior_b_mean, prior_b_sigma]]

    prior_sigma_unif = [sigma_inf, sigma_sup]
    prior_sigma = ["uniform", prior_sigma_unif]

    priors = [prior_a, prior_b, prior_sigma]

    solution_regression = laplace_solution_regression(
      expression=expr, data=self.data, lik=likelihood, priors=priors
    )
    return solution_regression

  def plot_distr(self, data: np.array, data_name: str) -> go.Figure:
    """
    Plot a histogram + KDE + rug of the distribution of the `data_name`
    variable, with data contained in `data`.
    """
    fig = ff.create_distplot(hist_data=[data], group_labels=[data_name])

    title = (
      f"Density plot of {data_name} (mean: {data.mean():.2f}, Std: {data.std():.2f})"
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=data_name)
    fig.update_yaxes(title="Density")

    return fig

  @staticmethod
  def sample_posterior_laplace(laplace_sol: list, n_samples: int) -> np.ndarray:
    """
    Make `n_samples` samples from a multivariate normal ditribution from the Laplace solution.
    """

    posterior_samples = np.random.multivariate_normal(
      laplace_sol[0], laplace_sol[1], size=n_samples
    )
    return posterior_samples

  def compute_posterior_samples_stats(
    self,
    n_samples: int,
    laplace_sol: list,
    param_names: list = ["a", "b", "sigma"],
    level: float = 0.85,
  ) -> tuple:
    """
    Print some statistics about the posterior samples from the Laplace solution.
    """
    print(12 * "-", "Output of Laplace computation", 12 * "-")
    posterior_samples = self.sample_posterior_laplace(
      laplace_sol=laplace_sol, n_samples=n_samples
    )

    means = posterior_samples.mean(axis=0)
    st_devs = posterior_samples.std(axis=0)

    q_low = (1 - level) / 2
    q_high = 1 - (1 - level) / 2

    quantiles = np.quantile(a=posterior_samples, q=[q_low, q_high], axis=0)

    stats = [means, st_devs, quantiles[0, :], quantiles[1, :]]
    stats = pd.DataFrame(
      stats, columns=param_names, index=["mean", "std", f"{q_low:.1%}", f"{q_high:.1%}"]
    )
    stats = stats.transpose()
    display(stats)

    return means, st_devs, posterior_samples


class Exercise5:
  def __init__(self) -> None:
    self.x = np.linspace(0.0, 3.0, 100)
    self.y = np.exp(self.x**2 * np.sin(self.x))

    self.theta_0 = 2.3
    self.mu_lap = self.theta_0

    second_deriv = 4 * np.cos(self.theta_0) + (2 - self.theta_0**2) * np.sin(
      self.theta_0
    )
    self.sigma_lap = np.sqrt(-1 / second_deriv)
    self.y_lap = stats.norm.pdf(self.x, loc=self.mu_lap, scale=self.sigma_lap)

  def plot_distr(self) -> go.Figure:
    """Make a line plot of the data."""
    fig = px.line(x=self.x, y=self.y, title="Distribution of y = x^2 * sin(x)")
    return fig

  def plot_distr_vs_lap(self) -> go.Figure:
    """Plot the data distribution as well as the Laplace approximation."""
    fig = self.plot_distr()

    fig = px.line(x=self.x, y=self.y / np.sum(self.y))
    trace_lap = go.Scatter(
      x=self.x, y=self.y_lap / np.sum(self.y_lap), name="Laplace approximation"
    )
    fig.add_trace(trace_lap)
    return fig


class Exercise6:
  def __init__(self) -> None:
    self.data = pd.read_csv("grades.csv", sep=",")
    self.mu0 = 78
    self.tau = 10
    self.a = 0
    self.b = 30

  def plot_distplot(self) -> go.Figure:
    fig = ff.create_distplot(
      hist_data=self.data[["midterm", "final"]].values.T,
      group_labels=["midterm", "final"],
      show_hist=False,
    )
    fig.update_layout(title="Distplot of `midterm` and `final` variables")
    return fig

  def plot_data(self) -> go.Figure:
    """Make a scatter plot of final vs midterm."""
    fig = px.scatter(
      data_frame=self.data, x="midterm", y="final", title="Final vs midterm scores."
    )
    return fig

  def sample_from_mu(
    self,
    mu0=None,
    tau=None,
    n_samples=10000,
  ):
    if mu0 is None:
      mu0 = self.mu0
    if tau is None:
      tau = self.tau

    # Define prior distribution for mu
    prior_mu = stats.norm(mu0, tau)
    prior_samples_mu = prior_mu.rvs(n_samples)

    # Plot draws
    fig = px.histogram(
      x=prior_samples_mu,
      title=f"Draws from the prior distribution of mu: normal({mu0=}, {tau=})",
    )
    return fig

  def sample_from_sigma(
    self,
    a=None,
    b=None,
    n_samples=10000,
  ):
    if a is None:
      a = self.a
    if b is None:
      b = self.b

    # Define prior distribution for sigma
    prior_sigma = stats.uniform(a, b)
    prior_samples_sigma = prior_sigma.rvs(n_samples)

    # Plot draws
    fig = px.histogram(
      x=prior_samples_sigma,
      title=f"Draws from the prior distribution of sigma: uniform({a=}, {b=})",
    )
    return fig

  def sample_both_priors(
    self,
    mu0=None,
    tau=None,
    a=None,
    b=None,
    n_samples=1000,
  ):
    if mu0 is None:
      mu0 = self.mu0
    if tau is None:
      tau = self.tau
    if a is None:
      a = self.a
    if b is None:
      b = self.b

    # Declare both prior distributions
    # then sample from pairs

    sigma = uniform.rvs(a, b, n_samples)
    mu = norm.rvs(mu0, np.sqrt(tau), n_samples)

    # Sample from both distributions
    midterm_samples = []
    for i in range(n_samples):
      midterm_samples.append(norm.rvs(mu[i], sigma[i]))

    fig = px.histogram(x=midterm_samples)
    return fig

  def lik_mu_sigma(self):
    # Initialize values
    mu = 40
    sigma = 20

    # Initializing the likelihood
    likelihood = "gaussian"
    parameters = [mu, sigma]

    # Defining the prior with hyperparameters
    prior_mu_mean = self.mu0
    prior_mu_sigma = self.tau
    prior_mu = ["gaussian", [prior_mu_mean, prior_mu_sigma]]

    sigma_inf = self.a
    sigma_sup = self.b
    prior_sigma_unif = [sigma_inf, sigma_sup]
    prior_sigma = ["uniform", prior_sigma_unif]

    solution = laplace_solution(
      params=[mu, sigma],
      other_params=[],
      data=self.data["midterm"],
      lik=likelihood,
      priors=[prior_mu, prior_sigma],
    )

    return solution

  def plot_post_params(self, solution: list, n_samples=1000):
    # Samples the posterior distribution of midterm
    post_mean = solution[0]
    post_covariance = solution[1]
    post_samples = stats.multivariate_normal.rvs(
      post_mean, post_covariance, size=n_samples
    )

    # Plots the post distribution for mu
    fig_post_mu = px.histogram(
      x=np.array([post_samples[i][0] for i in range(n_samples)]),
      title="Posterior distribution for the mean of themidterm score.",
    )
    # Plots the post distribution for sigma
    fig_post_sigma = px.histogram(
      x=np.array([post_samples[i][1] for i in range(n_samples)]),
      title="Posterior distribution for the std of the midterm score.",
    )
    return fig_post_mu, fig_post_sigma

  def run_laplace_model(self):
    expr = "final ~ midterm"

    # Intialize likelihood
    likelihood = "gaussian"

    # Define priors
    prior_a_mean = 40
    prior_a_sigma = 20
    prior_a = ["gaussian", [prior_a_mean, prior_a_sigma]]

    prior_b_mean = 0
    prior_b_sigma = 10
    prior_b = ["gaussian", [prior_b_mean, prior_b_sigma]]

    sigma_inf = 0
    sigma_sup = 30
    prior_sigma_unif = [sigma_inf, sigma_sup]
    prior_sigma = ["uniform", prior_sigma_unif]

    priors = [prior_a, prior_b, prior_sigma]

    solution_regression = laplace_solution_regression(
      expr, self.data, likelihood, priors
    )
    return solution_regression

  def sample_laplace_regr(self, solution_regression: list, n_samples=1000):
    # Computes the posterior samples
    posterior_samples = multivariate_normal.rvs(
      solution_regression[0], solution_regression[1], size=n_samples
    )

    # Computes and prints the summary stats
    # 85% confidence interval
    post_quantiles = np.quantile(posterior_samples, q=[0.075, 0.925], axis=0)

    # sd
    post_sd = np.std(posterior_samples, axis=0)

    # mean
    post_mean = np.mean(posterior_samples, axis=0)
    summary_stats = [post_mean, post_sd, post_quantiles[0, :], post_quantiles[1, :]]
    summary = pd.DataFrame(summary_stats).transpose()
    summary.columns = ["mean", "SD", "7.5%", "92.5%"]
    summary.rename(index={0: "a", 1: "b", 2: "sigma"}, inplace=True)

    return summary, posterior_samples

  def plot_post_samples(self, posterior_samples, n_samples = 1000):
    x_range = np.linspace(min(self.data["midterm"]), max(self.data["midterm"]))
    
    post_mean = np.mean(posterior_samples, axis=0)
    mean_prediction = post_mean[0] + post_mean[1] * x_range

    for i in range(n_samples):
      prediction = posterior_samples[i, 0] + posterior_samples[i, 1] * x_range
      plt.plot(x_range, prediction, lw=0.05, color="grey")


    plt.scatter(self.data["midterm"],self.data["final"])
    plt.plot(x_range, mean_prediction, lw = 2, color = 'black')
    plt.title('Final scores vs Midterm scores')
    plt.ylabel('Final scores')
    plt.xlabel('Midterm scores')

    plt.show()

class ExercisePyStan:
  @staticmethod
  def _generate_pystan_code(param_distr: dict[str, str], p_i_formula: str):
    """
    Generate the C code to be passed to the `program_code` argument of `stan.build`. \\
    This function adds the boilerplate around the actual important information: the
    parameter names, their respective distribution and the formula for `p_i`.

    This function enforces that the variables are called `x1`, `x2`, ...etc. The
    coherence of parameter names is not actually checked, but will raise an error
    at runtime (when building the Pystan model) in case some of them are not defined.

    Arguments:
      - `param_distr` : A dict of format `{parameter_name: parameter_distribution}`
      - `p_i_formula` : The formula to be optimized
    """

    # Data
    ## Enforce variable names to be start with "x" and increment an index
    ## from 1 to the number of parameters - 1.
    formatted_var = "\n".join([f"real x{i}[N];" for i in range(1, len(param_distr))])
    data = f"""
    data {{
      int<lower=1> N;
      int y[N];
      {formatted_var}
    }}"""
    data = dedent(data)

    param_names = param_distr.keys()
    formatted_params = "\n".join([f"real {param};" for param in param_names])

    # Parameters
    parameters = f"""
    parameters {{
      {formatted_params}
    }}"""
    parameters = dedent(parameters)

    # Transformed_parameters
    transformed_parameters = f"""
    transformed parameters {{
      vector[N] p_i;
      for (i in 1:N) {{
        {p_i_formula} 
        }}
    }}"""
    transformed_parameters = dedent(transformed_parameters)

    # Model
    formatted_distr = "\n".join([f"{par} ~ {dis};" for par, dis in param_distr.items()])
    model = f"""
    model {{
      {formatted_distr}
      y ~ binomial(1, p_i);
    }}"""
    model = dedent(model)

    return "\n".join([data, parameters, transformed_parameters, model])

  def run_stan_model(
    self,
    features: list[str],
    program_code: str,
    target: str = "DX",
    seed: Optional[int] = None,
    num_chains: int = 4,
    num_samples: int = 1000,
    data_name: str = "all",
  ) -> pd.DataFrame:
    """
    Use PyStan's Markov Chain Monte Carlo (MCMC) to obtain a parametric distribution
    for our prior.

    Arguments:
      - `features` : The features to use for prediction.
      - `program_code` : The C code to be passed to Stan
      - `target` : The target variable we want to predict.
      - `seed` : The random seed used to get reproducible results (default is `None`)
      - `num_chains` : The number of chains used by PyStan
      - `num_samples` : The number of samples per chain used by PyStan
    """

    data = self._get_appropriate_data(data_name=data_name)

    # Format useful data for Stan
    # Initialize dict
    data_to_stan = {}

    # Add features to Stan data
    for feat_idx, feat in enumerate(features):
      var_name = f"x{feat_idx+1}"  # Enforce variable names x1, x2, ...etc.
      data_to_stan[var_name] = np.array(data[feat])

    # Add target & number of data points to Stan
    data_to_stan["y"] = np.array(data[target])
    data_to_stan["N"] = len(data[target])

    # Make the Stan model
    posterior = stan.build(
      program_code=program_code,
      data=data_to_stan,
      random_seed=seed,
    )

    # Fit the Stan model
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)

    return fit

  @staticmethod
  def print_ci_param(fit: stan.fit.Fit, param_name: str, level: float = 0.95):
    """
    Print confidence interval lower and upper thresholds, as well as the median
    for a given parameter `param_name` (which must be in the columns of `fit`).
    """
    q_val = [(1 - level) / 2, 0.5, 1 - (1 - level) / 2]
    quantiles = np.quantile(q=q_val, a=fit[param_name])
    q_display = ["Median" if q == 0.5 else f"{q:.1%} threshold" for q in q_val]
    print(
      f"{level:.0%} confidence interval for {param_name!r}:",
      *[f"\t--> {disp:.<30} {q:>8.5f}" for disp, q in zip(q_display, quantiles)],
      sep="\n",
    )

  def get_waic(
    self, fit: stan.fit.Fit, sample_size_waic: int = 1000, data_name: str = "all"
  ) -> float:
    """
    Compute the WAIC from `fit`.

    Arguments:
      - `fit` : A fit from PyStan modeling.
      - `sample_size_waic` : How many samples should be used to compute the WAIC.
        This parameter defaults to the minimum between the number of samples computed by Stan
        and the value passed. If the value passed it the largest, an informational message is shown.
    """

    data = self._get_appropriate_data(data_name=data_name)

    # Get probabilities
    p_i = fit["p_i"]

    n_data_points, n_samples_computed = p_i.shape

    # Adjust number of samples to be less than maximum available
    if sample_size_waic > n_samples_computed:
      print(
        f"{sample_size_waic = } is greater than {n_samples_computed = }.",
        "Limiting to available number of samples.",
      )
      sample_size_waic = n_samples_computed

    # Initialize lppd and list for variances
    lppd = []
    pwaic = []

    # Iterate over data points
    for post_idx in tqdm(range(n_data_points)):
      id_log_lik = []

      # Iterate over draws
      for sample_idx in range(sample_size_waic):
        p = p_i[post_idx, sample_idx]

        # Compute likelihood (with log for stability)
        id_log_lik.append(binom.logpmf(k=data["DX"].values[post_idx], n=1, p=p))

      # Use logsumexp to stably sum vector of logs
      lppd.append(logsumexp(id_log_lik) - np.log(len(id_log_lik)))
      pwaic.append(np.var(id_log_lik))

    waic = -2 * (np.sum(lppd) - np.sum(pwaic))

    return waic

  @staticmethod
  def pretty_print_waic(waic: dict):
    """
    Format the `waic` dictionary into its model name, a repetition of delimiters,
    then the value of the WAIC.
    """
    print(*[f"{k + ' ':.<50} {v:>8.7f}" for k, v in waic.items()], sep="\n")

  def get_params_box_plot(
    self, fit: stan.fit.Fit, model_params: list[str]
  ) -> go.Figure:
    """
    Compute a box plot of the parameters from PyStan modeling.

    Arguments:
      - `fit` : A fit from PyStan modeling.
      - `model_params` : The name of the parameters computed in Pystan's modeling.
        These must be in the columns of `fit`.
    """

    # Convert fit object to df
    df_post = fit.to_frame()

    # Reverse model params to show from top to bottom
    reversed_params = model_params[::-1]

    # Compute and return box plot
    return px.box(
      data_frame=df_post,
      x=reversed_params,
      points="all",
      title=f"Box plot of the model parameters ({', '.join(model_params)}).",
    )
