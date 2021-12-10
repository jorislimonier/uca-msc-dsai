# %%
import numpy as np
import pandas as pd
from scipy import stats
#%%

obs = [3.23, -2.50, 1.88, -0.68, 4.43, 0.17, 1.03, -0.07, -0.01, 0.76, 1.76, 3.18, 0.33, -0.31, 0.30, -0.61, 1.52, 5.43, 1.54, 2.28, 0.42, 2.33, -1.03, 4.00, 0.39]
n_obs = len(obs)


# %%
mu_ml = np.mean(obs)
sig_ml = np.std(obs) 
tau_ml = mu_ml + sig_ml * stats.norm.ppf(.95)
se_tau_ml = np.sqrt((sig_ml**2 + stats.norm.ppf(.95)**2 * sig_ml**2 / 2) / n_obs)
tau_ml - se_tau_ml


# %%