import nest_asyncio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import stan

nest_asyncio.apply()


class ADNI:
  def __init__(self) -> None:
    # Load data
    self.raw_data = pd.read_csv("adni_data.csv")
    pd.set_option("mode.chained_assignment", None)

    # Create dataset with sick vs healthy people
    self.diag = self.raw_data.query("DX == 1 | DX == 3")
    self.diag["DX"] = self.diag["DX"].map({1: 0, 3: 1})

    # Compute normalized brain size, which is a more useful feature
    # than `WholeBrain.bl` or `ICV` indidivually.
    self.diag["norm_brain"] = self.diag["WholeBrain.bl"] / self.diag["ICV"]
    self.diag["norm_brain"] = self._normalize(self.diag["norm_brain"])
    self.diag.dropna(inplace=True)

  @staticmethod
  def _normalize(x: pd.Series) -> pd.Series:
    """
    Standard normalize a column. That is, remove the mean and divide
    by the standard deviation of the data.
    """
    return (x - np.mean(x)) / np.std(x)

  def two_feat_pred(self, feat: list[str], target: str = "DX"):
    wrong_num_feat = (
      f"This function intends to predict with 2 features, not {len(feat)}"
    )
    assert len(feat) == 2, wrong_num_feat

    feat1, feat2 = feat

    code_to_stan = """
    data {
      int<lower=1> N;
      int y[N];
      real x1[N];
      real x2[N];
    }
    parameters {
      real a;
      real b;
      real c;
    }
    transformed parameters {
      vector[N] p_i;
      for (i in 1:N) {
        p_i[i] = exp(a + b * x1[i] + c * x2[i])/(1 + exp(a + b * x1[i] + c * x2[i])); 
        }
    }
    model {
      a ~ normal(0, 3);
      b ~ normal(0, 3);
      c ~ normal(0, 3);
      y ~ binomial(1, p_i);
    }
    """
    data_to_stan = dict(
      x1=np.array(self.diag[feat1]),
      x2=np.array(self.diag[feat2]),
      y=np.array(self.diag[target]),
      N=len(self.diag[target]),
    )
    seed = 42
    posterior = stan.build(
      program_code=code_to_stan,
      data=data_to_stan,
      random_seed=seed,
    )
