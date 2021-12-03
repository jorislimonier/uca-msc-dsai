# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
sales = pd.read_csv("sales_data.csv", parse_dates=["Date"])
# col = 
sales.shape


# %%
sales.dtypes


# %%
sales["Unit_Cost"].plot(kind="box", vert=False)
plt.show()
ax = sales["Unit_Cost"].hist()
ax.axvline(sales["Unit_Cost"].mean(), color="red")
plt.show()


# %%
sales["Age_Group"].value_counts().plot(kind="bar")


# %%
import seaborn as sns
sns.heatmap(sales.corr())


# %%
fig = plt.figure(figsize=(8,8))
plt.matshow(sales.corr(), cmap="RdBu", fignum=fig.number)
plt.xticks(range(len(sales.corr().columns)), sales.corr().columns, rotation="vertical")
plt.yticks(range(len(sales.corr().columns)), sales.corr().columns, rotation="horizontal")
plt.colorbar()


# %%
pd.plotting.scatter_matrix(sales, alpha=.5, figsize=(16,16))


# %%
sales["Revenue_per_Age"] = sales["Revenue"] / sales["Customer_Age"]
sales["Calculated_Cost"] = sales["Order_Quantity"] * sales["Unit_Price"]


# %%
sales["Revenue_per_Age"].plot(kind="hist")
plt.show()
sales["Cost"].plot(kind="hist")
plt.show()
sales.plot("Profit", "Calculated_Cost", kind="scatter")


