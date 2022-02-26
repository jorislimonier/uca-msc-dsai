# %%
import pandas as pd
import plotly.express as px
# %%
FILE_NAME = "grades.ods"
df = pd.read_excel(FILE_NAME)
df["date"] = pd.to_datetime(df["date"])
df

# %%
px.scatter(df, x="date", y="grade", color="subject")
# %%
