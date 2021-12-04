# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


# %%
data1 = pd.read_csv("data1.txt", header=None)
data2 = pd.read_csv("data2.txt", header=None)


# %%
px.histogram(data1).show()
px.histogram(data2).show()

# %% [markdown]
# # Normal confidence interval
# Student confidence interval: $\bar{X} \pm \frac{\hat{\sigma}}{n-1} t_{n-1}$

# %%
data = data1[0].values.copy()
data.mean()


# %%
nb_obv = len(data)

alpha = .001
t_alpha = stats.t.ppf(1 - alpha/2, nb_obv)

emp_mean = np.mean(data)
emp_std = np.std(data, ddof=1)

excentr = emp_std * t_alpha / np.sqrt(nb_obv)
ci = [emp_mean - excentr, emp_mean + excentr]
ci


# %%
fig = px.histogram(data1)
fig.add_trace(go.Scatter(x=np.repeat([emp_mean - excentr], 210), name="ci_lb", marker=dict(color="#f00")))
fig.add_trace(go.Scatter(x=np.repeat([emp_mean + excentr], 210), name="ci_ub", marke=dict(color="#f00"), fill="tonextx"))

# %% [markdown]
# # Data 2

# %%
data = data2[0].values.copy()
data.mean()


# %%
nb_obv = len(data)

alpha = .05
z_alpha = stats.norm.ppf(1 - alpha/2)
print(z_alpha)
emp_mean = np.mean(data)
emp_std = np.std(data, ddof=1)

excentr = emp_std * z_alpha / np.sqrt(nb_obv)
ci = [emp_mean - excentr, emp_mean + excentr]
ci


# %%
fig = px.histogram(data2)
fig.add_trace(go.Scatter(x=np.repeat([emp_mean - excentr], 210),showlegend=False, marker=dict(color="#f00")))
fig.add_trace(go.Scatter(x=np.repeat([emp_mean + excentr], 210), name="ci", marker=dict(color="#f00"), fill="tonextx"))

# %% [markdown]
# # Goodness of fit

# %%
n = 200
data = np.random.normal(size=n)
data = np.sort(data)
data = stats.norm.cdf(data)

fig = go.Figure()
x = np.linspace(0,1,n)
fig.add_trace(go.Scatter(x=data, y=x))
x = np.linspace(0,1,n)
fig.add_trace(go.Scatter(x=stats.norm.ppf(x), y=x)) # WRONG


