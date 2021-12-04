# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy import stats
import plotly.graph_objects as go


# %%
# np.random.seed(42)
n = 10**4
z = stats.norm.ppf(.975)
rep = 10000
mu = 4
sigma = 5
sample = np.random.normal(mu, sigma, (rep, n))
emp_mean = np.mean(sample, axis=1)
emp_std = np.std(sample, axis=1)

excentr = z * emp_std / np.sqrt(n)
low_b = emp_mean - excentr
upp_b = emp_mean + excentr
fig = go.Figure()
sct = go.Scatter(y=emp_mean)
fig.add_trace(sct)

ci = go.Scatter(
        x=list(range(rep)) + list(range(rep-1,-1,-1)), # x, then x reversed
        y=np.concatenate([upp_b, low_b[::-1]]), # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
fig.add_trace(ci)
fig.show()


# %%
np.all([low_b < 4, upp_b > 4], axis=0).sum() / rep


# %%



