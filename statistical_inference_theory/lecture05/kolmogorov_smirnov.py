# %%
import numpy as np
import plotly.graph_objects as go

np.random.seed(42)
sample = np.random.uniform(0, 1, 5)
sample = np.sort(sample)
x_val = np.linspace(-.1, 1.1, 10)

def assoc_step(x, sample):
    sample = np.sort(sample)
    x_best = 0
    for s in sample:
        if x > s:
            x_best = s
    return x_best
sample
assoc_step(.2, sample)
fig = go.Figure()
fig.add_trace(go.Scatter(y=sample))
fig.show()
fig.write_image("./img.png")



# %%
