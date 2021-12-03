# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import plotly.express as px


# %%
lateness_path = "~/Documents/uca-msc-dsai/misc/lateness/"
data = pd.read_excel(lateness_path + "lateness.ods")
data.tail()


# %%
av_late = data.groupby("teacher").mean().sort_values(
    "start_time", ascending=False)
px.bar(av_late, y="start_time", color="start_time",
       color_continuous_scale="Bluered", barmode="group", title="Mean lateness")


# %%
sum_late = data.groupby("teacher").sum().sort_values(
    "start_time", ascending=False)
px.bar(sum_late, y="start_time", color="start_time",
       color_continuous_scale="Bluered", barmode="group", title="Cumulated lateness")


# %%
px.bar(data.sort_values("start_time", ascending=False), x="teacher", y="start_time", color="start_time",
       color_continuous_scale="Bluered", barmode="group")


# %%



