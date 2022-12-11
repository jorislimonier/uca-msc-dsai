"""
Utility functions for assignment 3 of the Bayesian Learning course.
"""

from textwrap import dedent
from typing import Optional

import nest_asyncio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import stan
from scipy.special import logsumexp
from scipy.stats import binom, norm
from tqdm.notebook import tqdm, tqdm_notebook, trange

# Make stan work
nest_asyncio.apply()
pd.set_option("mode.chained_assignment", None)

# Set Plotly's background
pio.templates.default = "plotly_white"


class ADNI:
  """
  Class for operations linked to the ADNI dataset.
  """

  def __init__(self) -> None:
    self._load_data()
    self._restrict_data_80_yo()

  @staticmethod
  def _normalize(x: pd.Series) -> pd.Series:
    """
    Standard normalize a column. That is, remove the mean and divide
    by the standard deviation of the data.
    """
    return (x - np.mean(x)) / np.std(x)

  def _load_data(self) -> None:
    """
    Load data and perform a few preparation transformations.
    """
    # Load data
    self.raw_data = pd.read_csv("adni_data.csv")

    # Create dataset with sick vs healthy people (diagnosis)
    self.diag = self.raw_data.query("DX == 1 | DX == 3")
    self.diag["DX"] = self.diag["DX"].map({1: 0, 3: 1})

    # Compute normalized brain size, which is a more useful feature
    # than `WholeBrain.bl` or `ICV` indidivually.
    self.diag["norm_brain"] = self.diag["WholeBrain.bl"] / self.diag["ICV"]
    self.diag["norm_brain"] = self._normalize(self.diag["norm_brain"])
    self.diag.dropna(inplace=True)

  @staticmethod
  def _generate_pystan_code(param_distr: dict[str, str], p_i_formula: str):
    """
    Generate the C code to be passed to the `program_code` argument of `stan.build`.

    Arguments:
      - `param_distr` : A dict of format `{parameter_name: parameter_distribution}`
      - `p_i_formula` : The formula to be optimized
    """

    # data
    data = """
    data {
      int<lower=1> N;
      int y[N];
    }"""
    data = dedent(data)

    param_names = param_distr.keys()
    formatted_params = "\n".join([f"real {param};" for param in param_names])

    # parameters
    parameters = f"""
    parameters {{
      {formatted_params}
    }}"""
    parameters = dedent(parameters)

    # transformed_parameters
    transformed_parameters = f"""
    transformed parameters {{
      vector[N] p_i;
      for (i in 1:N) {{
        {p_i_formula} 
        }}
    }}"""
    transformed_parameters = dedent(transformed_parameters)

    formatted_distr = "\n".join([f"{par} ~ {dis};" for par, dis in param_distr.items()])
    model = f"""
    model {{
      {formatted_distr}
      y ~ binomial(1, p_i);
    }}"""
    model = dedent(model)
    return "\n".join([data, parameters, transformed_parameters, model])

  def _restrict_data_80_yo(self) -> None:
    """
    Restrict the ADNI dataset to people of exactly 80 years old only.
    """
    self.eighty = self.diag.copy()
    self.eighty = self.eighty.query(expr="AGE == 80")

  def plot_kde_vs_norm(self) -> go.Figure:
    """
    Plot the Kernel Density Estimation (KDE) vs a normal distribution with
    similar mean and standard deviation.
    """

    # Create distplot with KDE of age
    fig = ff.create_distplot([self.diag["AGE"]], group_labels=["AGE"], show_hist=False)

    # Compute normal distribution with same mean and std
    age_range = np.linspace(start=self.diag["AGE"].min(), stop=self.diag["AGE"].max())
    age_mean = self.diag["AGE"].mean()
    age_std = self.diag["AGE"].std()
    y_norm = norm.pdf(age_range, loc=age_mean, scale=age_std)

    # Plot normal distribution with same mean and std
    normal_trace = go.Scatter(x=age_range, y=y_norm, name="Normal fit")
    fig.add_trace(normal_trace)

    # Add title
    fig.update_layout(
      title=f"""Comparison between the age distribution and a normal
distribution with same mean (={age_mean:.2f}) and std (={age_std:.2f})."""
    )

    return fig

  def plot_apoe4(self) -> go.Figure:
    return px.histogram(
      self.diag,
      x="APOE4",
      nbins=20,
      histnorm="probability",
      title="Distribution of the APOE4 variable.",
    )

  def _get_appropriate_data(self, data_name: str) -> pd.DataFrame:
    """
    Return either the whole ADNI dataset, or the restriction to
    80 years old only according to `data_name`.
    """
    match data_name:
      case "all":
        return self.diag
      case "80 yo":
        return self.eighty
      case other:
        raise ValueError(f"Name {other} is not an acceptable data name.")

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
    print(type(fit))
    return fit

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
