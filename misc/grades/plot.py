# %%
import pandas as pd
import plotly.express as px
# %%
FILE_NAME = "grades.ods"
df = pd.read_excel(FILE_NAME)

df["submitted_on"] = pd.to_datetime(df["submitted_on"])
# %%
px.scatter(df, x="submitted_on", y="grade", color="subject")