"""
Utility functions for assignment 3 of the Bayesian Learning course.
"""

from typing import Optional

import nest_asyncio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import stan
from scipy.special import logsumexp
from scipy.stats import binom
from tqdm.notebook import tqdm, tqdm_notebook, trange

nest_asyncio.apply()
pd.set_option("mode.chained_assignment", None)


class ADNI:
  """
  Class for operations linked to the ADNI dataset.
  """

  def __init__(self) -> None:
    self._load_data()

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

  def run_stan_model(
    self,
    features: list[str],
    program_code: str,
    target: str = "DX",
    seed: Optional[int] = None,
    num_chains: int = 4,
    num_samples: int = 1000,
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

    # Format useful data for Stan
    # Initialize dict
    data_to_stan = {}

    # Add features to Stan data
    for feat_idx, feat in enumerate(features):
      var_name = f"x{feat_idx+1}"  # Enforce variable names x1, x2, ...etc.
      data_to_stan[var_name] = np.array(self.diag[feat])

    # Add target & number of data points to Stan
    data_to_stan["y"] = np.array(self.diag[target])
    data_to_stan["N"] = len(self.diag[target])

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

  def get_waic(self, fit: stan.fit.Fit, sample_size_waic: int = 1000) -> float:
    """
    Compute the WAIC from `fit`.
    """
    p_i = fit["p_i"]

    n_data_points, n_samples_computed = p_i.shape

    if sample_size_waic > n_samples_computed:
      print(
        f"{sample_size_waic = } is greater than {n_samples_computed = }.",
        "Limiting to available number of samples.",
      )
      sample_size_waic = n_samples_computed

    lppd = []
    pwaic = []

    for post_idx in tqdm(range(n_data_points)):
      id_log_lik = []
      for sample_idx in range(sample_size_waic):
        p = p_i[post_idx, sample_idx]
        id_log_lik.append(binom.logpmf(k=self.diag["DX"].values[post_idx], n=1, p=p))
      lppd.append(logsumexp(id_log_lik) - np.log(len(id_log_lik)))
      pwaic.append(np.var(id_log_lik))

    waic = -2 * (np.sum(lppd) - np.sum(pwaic))
    return waic
