# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import plotly.graph_objects as go
# np.random.seed(42)
B = 100
n = 20
sample = np.random.standard_cauchy(n)

boot_medians = [np.median(np.random.choice(sample, size=n)) for _ in range(B)]
boot_median = np.mean(boot_medians)

fig = go.Figure()
fig.add_trace(go.Scatter(y=boot_medians))
fig.add_trace(go.Scatter(x=list(range(B)), y=np.repeat(boot_median, B)))
fig.add_trace(go.Scatter(x=list(range(B)), y=np.repeat(boot_median, B)))


# %%
# Chapter 8 - Exercise 2


# %%



