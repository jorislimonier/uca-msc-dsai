# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy import stats
import plotly.graph_objects as go


# %%
n = 10**4
X = np.random.normal(size=(n, 1000))
Y = np.exp(X)
sk_pin = stats.skew(Y)
sk_pin
true_sk = (np.exp(1)+2) * np.sqrt(np.exp(1)-1)

fig = go.Figure()
fig.add_trace(go.Histogram(x=sk_pin))
fig.add_trace(go.Scatter(x=[true_sk for _ in range(200)], y=list(range(200))))
fig


# %%
np.mean(sk_pin), true_sk

# %% [markdown]
# # Bootstrap
# %% [markdown]
# ```R
# ### Exercise 2, unit 8
# 
# # We want to build n confidence intervals
# 
# set.seed(123)
# install.packages("moments")
# library(moments)
# 
# n = 150
# Y = rnorm(n,mean = 0,sd = 1)
# X = exp(Y)
# B = 1000
# theta <- (exp(1)+2)*sqrt(exp(1)-1)
# runs <- 100
# a <- 0.05
# Z_alfa <- qnorm(1-a/2)
# 
# # Bootstraping:
# 
# #CI_N
# 
# store_sk_b <- vector(length = B)
# CI_N <- matrix(0,runs,2)
# 
# for (r in 1:runs){
#   for (i in 1:B){
#     X_B <- sample(X,replace = TRUE)
#     store_sk_b[i] <- skewness(X_B)
#   }
#   sd_b_sk <- sd(store_sk_b)
#   sk_x <- skewness(X)
#   CI_N[r,1] <- sk_x - Z_alfa*sd_b_sk
#   CI_N[r,2] <- sk_x + Z_alfa*sd_b_sk
# }
# 
# CI_N
# alfa_c <- sum(CI_N[,1]>theta | CI_N[,2]<theta )/runs
# ```

# %%
n = 1000
Y = np.random.normal(size=n)
X = np.exp(Y)
nb_boots = 1000

# Display fig
fig = go.Figure(go.Scatter(y=X, mode="markers"))
fig.update_layout(title="Sample from theoretical distribution")
fig.show()

theta = (np.exp(1)+2) * np.sqrt(np.exp(1)-1)
nb_runs = 100
alpha = 0.05
z_alpha = stats.norm.ppf(1 - alpha/2)


# %%
sk_boot = np.array([])
ci = np.array([])
for run in range(nb_runs):
    for boot in range(nb_boots):
        X_boot = np.random.choice(X, size=n, replace=True)
        sk_boot = np.append(sk_boot, stats.skew(X_boot))
    sd_boot_sk = np.std(sk_boot)
    sk_X = stats.skew(X) # center of the CI
    excentr = z_alpha * sd_boot_sk # excentricity of the CI (=width / 2)
    ci = np.append(ci, [sk_X - z_alpha*excentr, sk_X + z_alpha*excentr]).reshape((len(ci)+1, -1))

sk_boot.shape, ci.shape


# %%
plot_ci = np.append(ci[:, 0], np.flip(ci[:, 1]))
fig = go.Figure()
fig.add_trace(go.Scatter(y=ci[:,0], showlegend=False , marker=dict(color="#f00")))
fig.add_trace(go.Scatter(y=ci[:,1], name="ci", marker=dict(color="#f00"), fill="tonexty"))
fig.add_trace(go.Scatter(y=np.repeat([theta], nb_runs), name="true value"))


