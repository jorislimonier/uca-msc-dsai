# %%
import numpy as np
import pandas as pd
from scipy import stats
# %%
# np.random.seed(42)
data = pd.DataFrame(np.random.randint(0, 10, 100))
data = pd.DataFrame(data.value_counts(), columns=["counts"])
data = data.sort_index()
data["theoretical_counts"] = 1/10
data["chisq_dist"] = (
    data["counts"] - data["theoretical_counts"])**2 / len(data)
data["chisq_dist"].sum()

# %%

# test for Poisson
nb_cars = np.array([6, 15, 40, 42, 37, 30, 10, 9, 5, 3, 2, 1])
df = pd.DataFrame(nb_cars, columns=["counts"])
param_est = np.dot(df.index, df["counts"]) / df["counts"].sum()
df["poisson_pmf"] = stats.poisson.pmf(df.index, param_est)
df.loc[11, "poisson_pmf"] = 1 - stats.poisson.cdf(10, param_est)
df.sum()