# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

np.random.seed(42)


# %%
n = 100
rep = 1000


def emp_cdf(x):
    return (draws < x).sum()/n


def ci_exc(x, level):
    z_score = float(norm.ppf([level]))
    return z_score * np.sqrt(emp_cdf(x) * (1 - emp_cdf(x)) / n)


def lb(x, level):
    return emp_cdf(x) - ci_exc(x, level)


def ub(x, level):
    return emp_cdf(x) + ci_exc(x, level)


# N(0,1)
draws = np.random.normal(size=n)
# Cauchy
draws = np.random.standard_cauchy(size=n)

x = .5
level = .95

x_min = -3
x_max = 3
x_range = np.linspace(x_min, x_max, 1000)
lbs = [lb(x, level) for x in x_range]
ubs = [ub(x, level) for x in x_range]
emp_cdfs = [emp_cdf(x) for x in x_range]


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_range, y=emp_cdfs, name="estimate"))
fig.add_trace(go.Scatter(x=x_range, y=lbs, marker=dict(color="red"), name="lower bound"))
fig.add_trace(go.Scatter(x=x_range, y=ubs, marker=dict(color="red"), name="upper bound"))


