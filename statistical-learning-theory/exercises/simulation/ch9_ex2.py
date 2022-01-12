# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
a = 1
b = 3
n = 10
print(f"MLE: {((a-b)**2) / (12*n)}")
nb_rep = 1000
tau_ml = np.zeros(nb_rep)

for rep in range(nb_rep):
    draws = np.random.uniform(a, b, n)
    tau_ml[rep] = (draws.min() + draws.max()) / 2
bias_ml = tau_ml.mean() - ((a+b)/2)
tau_ml.var() + bias_ml**2


# %%
import plotly.express as px

px.histogram(tau_ml)


