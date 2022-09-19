# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
n = 500
theta = 4
means = np.array([])

for i in range(100):
    sample = np.random.uniform(0, theta, n)
    means = np.append(means, ((n+1)/n)*np.max(sample))
plt.boxplot(means)
# plt.show()
px.box(means)


# %%



# %%



