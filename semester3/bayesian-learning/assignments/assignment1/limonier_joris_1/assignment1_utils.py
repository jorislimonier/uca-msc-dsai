from typing import Type

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from scipy.stats import beta, norm


# Exercise 1
class Football:
  def __init__(self) -> None:
    self.football = pd.read_csv("football-dataset.txt")
    self.football["point_diff"] = self.football["favorite"] - self.football["underdog"]

  @staticmethod
  def prob_X_given_Y(X: pd.Series, Y: pd.Series) -> float:
    """Compute the probability of event X, given event Y.
    Both probabilities are empirical probabilities, based on the dataset at hand.
    """
    prob_Y = Y.mean()
    prob_both = (X & Y).mean()
    return prob_both / prob_Y

  @staticmethod
  def print_result(question: int, prob_of: str, given: str, prob: float) -> None:
    """Print the probabilities in a formated way."""
    prob = round(prob, 4)
    res = "\n\t".join(
      [
        f"{question}. \tThe probability that: \t {prob_of},",
        f"given that: \t\t {given}",
        f"is: \t\t\t {prob}",
      ]
    )
    print(res, end="\n\n")

  def p1_freq(self, print_results: bool = False) -> float:
    """Compute P1 using a frequentist approach."""
    fav_won = self.football["favorite"] > self.football["underdog"]
    self.football["favorite_won"] = fav_won

    X = self.football["favorite_won"]
    Y = self.football["spread"] == 8
    prob = self.prob_X_given_Y(X=X, Y=Y)

    if print_results:
      self.print_result(
        question=1,
        prob_of="the favorite wins",
        given="the point spread is 8",
        prob=prob,
      )
    return prob

  def p2_freq(self, print_results: bool = False) -> float:
    """Compute P2 using a frequentist approach."""
    X = self.football["point_diff"] >= 8
    Y = self.football["spread"] == 8
    prob = self.prob_X_given_Y(X=X, Y=Y)
    if print_results:
      self.print_result(
        question=2,
        prob_of="the favorite wins by at least 8 points",
        given="the point spread is 8",
        prob=prob,
      )
    return prob

  def p3_freq(self, print_results: bool = False) -> float:
    """Compute P3 using a frequentist approach."""
    X = self.football["point_diff"] >= 8
    Y = (self.football["spread"] == 8) & self.football["favorite_won"]
    prob = self.prob_X_given_Y(X=X, Y=Y)
    if print_results:
      self.print_result(
        question=3,
        prob_of="the favorite wins by at least 8 points",
        given="the point spread is 8 and the favorite wins",
        prob=prob,
      )
    return prob

  def plot_approx_distr(self) -> go.Figure:
    """Plot a histogram of the
    superimposed with the normal distribution
    with mean and std inferred from z."""

    x = self.football["spread"]
    y = self.football["point_diff"]
    z = y - x

    z_val = np.linspace(start=z.min(), stop=z.max())

    # Get statistics of z
    z_mean = z.mean()
    z_std = z.std()
    z_val = np.linspace(start=z.min(), stop=z.max())

    # Compute normal distribution based on z
    approx_pdf = norm.pdf(x=z_val, loc=z_mean, scale=z_std)

    # Create figure
    fig = go.Figure()

    # Add line to plot
    line = go.Scatter(
      x=z_val,
      y=approx_pdf,
      name="Approximated PDF",
      mode="lines",
    )
    fig.add_trace(line)

    # Add histogram to plot
    hist = go.Histogram(
      x=z,
      histnorm="probability density",
      opacity=0.5,
      name="Sample Distribution",
    )
    fig.add_trace(hist)

    # Add final touch to layout
    fig.update_xaxes(title="z")
    fig.update_yaxes(title="Probability")
    fig.update_layout(title=f"mean of z: {z_mean:.4f},\t std of z: {z_std:.4f}")

    return fig

  def p1_approx_distr(self, plot: bool = True) -> float:
    """Compute P1 by using the data to fit a normal distribution."""
    x = self.football["spread"]
    y = self.football["point_diff"]
    z = y - x

    # Get statistics of z
    z_mean = z.mean()
    z_std = z.std()

    # Compute normal distribution based on z
    z_norm = norm(loc=z_mean, scale=z_std)

    prob = 1 - z_norm.cdf(-8)

    return prob

  def p2_approx_distr(self, plot: bool = True) -> float:
    """Compute P2 by using the data to fit a normal distribution."""
    x = self.football["spread"]
    y = self.football["point_diff"]
    z = y - x

    # Get statistics of z
    z_mean = z.mean()
    z_std = z.std()

    # Compute normal distribution based on z
    z_norm = norm(loc=z_mean, scale=z_std)

    prob = 1 - z_norm.cdf(0)

    return prob

  def p3_approx_distr(self, plot: bool = True) -> float:
    """Compute P3 by using the data to fit a normal distribution."""
    x = self.football["spread"]
    y = self.football["point_diff"]
    z = y - x

    # Get statistics of z
    z_mean = z.mean()
    z_std = z.std()

    # Compute normal distribution based on z
    z_norm = norm(loc=z_mean, scale=z_std)

    prob_num = z_norm.cdf(0) - z_norm.cdf(-8)
    prob_denom = 1 - z_norm.cdf(-8)
    prob = 1 - prob_num / prob_denom

    return prob

  def display_results(self) -> None:
    """Return a dataframe containing all (2*3) probabilities asked,
    as well as figures of the approximated PDF superimposed with the histogram
    of distributions for values of z.
    """
    p1_freq = self.p1_freq()
    p2_freq = self.p2_freq()
    p3_freq = self.p3_freq()
    p_freq = [p1_freq, p2_freq, p3_freq]

    p1_approx = self.p1_approx_distr()
    p2_approx = self.p2_approx_distr()
    p3_approx = self.p3_approx_distr()
    p_approx = [p1_approx, p2_approx, p3_approx]

    index = ["P1", "P2", "P3"]
    data = data = {"Prob (freq)": p_freq, "Prob (approx)": p_approx}
    results = pd.DataFrame(data=data, index=index)

    display(results)

    fig = self.plot_approx_distr()
    fig.show()


# Exercise 2
class Posterior:
  def __init__(
    self,
    n_obs: int,  # n
    mean_weight_emp: float = 70,  # y_mean
    mean_weight_theo: float = 10,  # sigma
    prior_mean: float = 80,  # mu_0
    prior_std: float = 15,  # tau_0
  ) -> None:
    self.n_obs = n_obs
    self.mean_weight_emp = mean_weight_emp
    self.mean_weight_theo = mean_weight_theo
    self.prior_mean = prior_mean
    self.prior_std = prior_std

  def __str__(self) -> str:
    formatted_ci = f"{[round(bound, 4) for bound in self.compute_posterior_ci()]}"
    return "\n".join(
      [
        f"For {self.n_obs} observations, we have:",
        f"\t- The posterior mean for theta is: {self.posterior_mean:.4f}",
        f"\t- The posterior std for theta is: {self.posterior_std:.4f}",
        f"\t- The centered 95% confidence interval for theta is: {formatted_ci}",
      ]
    )

  @property
  def posterior_mean(self) -> float:
    """Compute the posterior mean"""
    posterior_mean_num = (
      self.n_obs * self.mean_weight_emp * self.prior_std**2
      + self.prior_mean * self.mean_weight_theo**2
    )
    posterior_mean_denom = self.n_obs * self.prior_std**2 + self.mean_weight_theo**2
    return posterior_mean_num / posterior_mean_denom

  @property
  def posterior_std(self) -> float:
    """Compute the posterior standard deviation"""
    posterior_std_num = self.mean_weight_theo**2 * self.prior_std**2
    posterior_std_denom = self.n_obs * self.prior_std**2 + self.mean_weight_theo**2
    return np.sqrt(posterior_std_num / posterior_std_denom)

  def compute_posterior_ci(self, ci_level: float = 0.95) -> tuple[float, float]:
    """Compute the posterior confidence interval at level `ci_level`"""
    return norm.interval(
      confidence=ci_level,
      loc=self.posterior_mean,
      scale=self.posterior_std,
    )


# Exercise 3
def plot_beta_pdf(a: float, b: float) -> go.Figure:
  """Plot the PDF of the beta distribution"""

  # Compute PDF
  theta = np.linspace(start=0, stop=1, num=100, endpoint=False)
  beta_pdf = beta.pdf(theta, a=a, b=b)

  # Make plot
  line = go.Scatter(x=theta, y=beta_pdf)
  go.Figure(line)
  df = pd.DataFrame({"theta": theta, "beta_pdf": beta_pdf})
  fig = px.line(df, x="theta", y="beta_pdf")
  fig.update_xaxes(range=[0, 1])
  fig.update_layout(title=f"PDF of Beta({a}, {b})")

  return fig


def slider_plot(b: int = 2) -> go.Figure:
  """Make a line plot containing a slider to change the value of `a`
  for a given `b`"""
  list_a = np.linspace(0.001, 1000, num=200)
  initial_active_trace = 0

  theta_range = np.linspace(0, 1)

  fig = go.Figure()

  for a in list_a:
    pdf = beta.pdf(theta_range, a=a, b=b)
    trace = go.Line(x=theta_range, y=pdf, visible=False)
    fig.add_trace(trace)

  fig.data[initial_active_trace].visible = True

  # Create and add slider
  steps = []
  for i in range(len(fig.data)):
    step = dict(
      method="update",
      args=[
        {"visible": [True if idx == i else False for idx in range(len(fig.data))]},
        {"title": f"a = {list_a[i]:.2f}, b = {b}"},
        {"label": "[str(val) for val in list_a]"},
      ],  # layout attribute
    )
    steps.append(step)

  sliders = [
    dict(
      active=initial_active_trace,
      currentvalue={"prefix": "a = "},
      pad={"t": 50},
      steps=steps,
    )
  ]
  fig.update_layout(
    sliders=sliders, title=f"a = {list_a[initial_active_trace]:.2f}, b = {b}"
  )
  return fig


def compute_beta_stats() -> pd.DataFrame:
  """Compute the following statistics for the posterior beta distribution:
  - Mean
  - Median
  - 95% centered confidence interval (CI)
  """
  k = 650
  n = 1000

  list_a_prior = [1, 2] + [*range(5, 51, 5)]
  list_b_prior = [1, 2] + [*range(5, 51, 5)]

  pairs_prior = [[a, b] for a in list_a_prior for b in list_b_prior]

  beta_stats = pd.DataFrame(pairs_prior, columns=["prior_alpha", "prior_beta"])
  beta_stats["post_alpha"] = beta_stats["prior_alpha"] + k
  beta_stats["post_beta"] = beta_stats["prior_beta"] + n - k
  beta_stats["post mean"] = beta.mean(
    a=beta_stats["post_alpha"],
    b=beta_stats["post_beta"],
  )
  beta_stats["post median"] = beta.median(
    a=beta_stats["post_alpha"],
    b=beta_stats["post_beta"],
  )
  beta_stats["post 95% CI lower bound"] = beta.ppf(
    a=beta_stats["post_alpha"],
    b=beta_stats["post_beta"],
    q=0.025,
  )
  beta_stats["post 95% CI upper bound"] = beta.ppf(
    a=beta_stats["post_alpha"],
    b=beta_stats["post_beta"],
    q=0.975,
  )
  beta_stats["post 95% CI spread"] = (
    beta_stats["post 95% CI upper bound"] - beta_stats["post 95% CI lower bound"]
  )

  return beta_stats
