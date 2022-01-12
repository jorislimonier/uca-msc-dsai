# %%
import torch

# %%
x = torch.ones(2,2, requires_grad=True)
x

# %%
y = x + 2
y

# %%
y.grad_fn

# %%
z = y * y * 3

# %%
z

# %%
out = z.mean()

# %%
out

# %%
out.backward()

# %%
x.grad

# %% [markdown]
# These are the derivatives:
# 
# 
# $y = x + 2$
#     
# $z = y * y * 3$
# 
# $z_i = 3 (x_i + 2)^2$
# 
# $o = \frac{1}{4} \sum_i z_i$
# 
# $z_i |_{x_i = 1} = 27$
# 
# So:
# 
# $\frac{\partial{o}}{\partial{x_i}} = \frac{3}{2} (x_i + 2)$
# 
# $\frac{\partial{o}}{\partial{x_i}} |_{x_i = 1} = \frac{9}{2} = 4.5$
# 
# 

# %%
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# %%
y = w * x + b   # TH

# %%
y

# %%
y.backward()

# %%
x.grad

# %%
w.grad

# %%
b.grad


