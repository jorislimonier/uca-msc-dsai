# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy import stats
import plotly.graph_objects as go


# %%
x_bound = 4
x = np.linspace(-x_bound, x_bound, 101)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 1)))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 2)))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 3)))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 5)))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 100)))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, 10**9)))
fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, 0,1), name="standard normal", line=dict(width=5)))


# %%
deg_fr = 4
stats.t.cdf(1.7, deg_fr) - stats.t.cdf(-0.31, deg_fr)
stats.t.ppf(0.975, 4)


# %%
rep = 2
sample_size = 5
draws = np.random.standard_t(deg_fr,size=(rep,sample_size))
alpha=0.05
z = stats.t.ppf(alpha/2, sample_size-1)

mean_emp = np.mean(draws, axis=1)
std_emp = np.std(draws, axis=1)

mean_emp, std_emp


# %%

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=mean_emp
))


# %%



# %%



