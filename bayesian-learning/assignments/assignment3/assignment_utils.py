import pandas as pd
import numpy as np


class ADNI:
  def __init__(self) -> None:
    # Load data
    self.raw_data = pd.read_csv("adni_data.csv")
    pd.set_option("mode.chained_assignment", None)

    # Create dataset with sick vs healthy people
    data_ct_ad = self.raw_data.query("DX == 1 | DX == 3")
    data_ct_ad["DX"] = data_ct_ad["DX"].map({1: 0, 3: 1})

    data_ct_ad["norm_brain"] = data_ct_ad["WholeBrain.bl"] / data_ct_ad["ICV"]
    data_ct_ad["norm_brain"] = self._normalize(data_ct_ad["norm_brain"])
    data_ct_ad.dropna(inplace=True)

  @staticmethod
  def _normalize(x: pd.Series) -> pd.Series:
    """
    Standard normalize a column. That is, remove the mean and divide
    by the standard deviation of the data.
    """
    return (x - np.mean(x)) / np.std(x)
