# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Notebook n.3 
# ### NYC complaints 311 Calls
# 
# In this notebook you are asked to perform data analysis on a dataset of calls to 311 (municipal calls, not emergency) in the New York City area.
# 
# Get the data:
#    * A compressed smaller version (~100 MB compressed) can be found [here](https://bit.ly/3b7yATT)
#    * If you want (and have enough memory on your laptop), you can download the original data from [here](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) (11Gb+... Go to Export -> CSV).
# 
# ## Instructions:
#    * The small dataset is enough to finish this notebook.
#    * For each question, add as many code cells as you need, as well as Markdown cells to explain your thought process and answer in text to the questions (where needed).

# %%
# Run this cell to import commonly used modules
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 5)  # large and nice

# %% [markdown]
# ### <a id="point1">1.</a> Load the `csv` file into a `pandas.dataframe` called `complaints`

# %%
complaints = pd.read_csv("311_small.csv")

# %% [markdown]
# ### 2. Basic overview
# 
#    * How many rows?
#    * How many columns?
#    * Type of each column
#    * Number of unique values per column
#    * Show the first 5 rows
#    * [Earlier](#point1) you probably received a warning. If so, why? Investigate (type mismatch, nans, ...)

# %%
print(f"--> Number of rows: {complaints.shape[0]}")


# %%
print(f"--> Number of columns: {complaints.shape[1]}")


# %%
print(f"--> Type of each column:\n{complaints.dtypes}")


# %%
print(f"--> Number of unique values per column:\n{complaints.nunique()}")


# %%
complaints.head()

# %% [markdown]
# **Warning ?**\
# Yes, I had the following warning:
# ```
# /home/joris/.local/lib/python3.9/site-packages/IPython/core/interactiveshell.py:3172: DtypeWarning: Columns (8,31,32,34,35,36,37) have mixed types.Specify dtype option on import or set low_memory=False.
#   has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
#   ```
# 
# It seems to be because some columns have multiple dtypes. Let's investigate further.
# 

# %%
complaints = pd.read_csv("311_small.csv", low_memory=False)


# %%
# Compute the proportion of rows that are NA for these columns
complaints.iloc[:, [8,31,32,34,35,36,37]].isna().sum() / complaints.shape[0] # almost everything is NA, except for columns `Incident Zip`


# %%
inc_zip = complaints.iloc[:,8].copy()
print(f"--> Types in the Incident Zip column: {set(type(e) for e in inc_zip.unique())}")
print(f"--> Number of unique values in Incident Zip: {inc_zip.nunique(dropna=False)}")
print(f"--> Number of unique values of type string: {len(set(e for e in inc_zip.unique() if type(e) == str))}")
print(f"--> Number of unique values of type float: {len(set(e for e in inc_zip.unique() if type(e) == float))}")

# What is the only float value ?
print(f"--> Only float value: {set(e for e in inc_zip.unique() if type(e) == float)}")

# %% [markdown]
# We saw that the type issue arose because the NA's in the Incident Zip column are of type `float`, whereas the other values are of type `str`.
# %% [markdown]
# ### 3. Show the top 10 rows for attributes `Complaint Type` and `Borough`

# %%
for col in ["Complaint Type", "Borough"]:
    print(f"--> Top 10 most common attributes in column {col}: {complaints[col].value_counts(ascending=False).iloc[:10].index}")
# Note that there are less that 10 values for the column `Borough`

# %% [markdown]
# ### 4. How many distinct `Complaint Type` are there?
#    * Count them and show them

# %%
compl_counts = complaints["Complaint Type"].value_counts()
print(f"--> All complaint types, with their respective counts:\n{compl_counts}")
print(f"--> So in total, there are {len(compl_counts)} distinct complaints (and {compl_counts.isna().sum()} NA).")

# %% [markdown]
# ### <a id="point5">5.</a> Clean `all` lines where `Complaint Type` contains the keyword `"Misc."` 
#    * How many lines are dropped?

# %%
mask = complaints["Complaint Type"].str.contains("Misc.")
print(f"{mask.sum()} lines will be dropped")


# %%
complaints = complaints[~mask]

# %% [markdown]
# #### From now on, use the dataframe resulting from point [5](#point5)
# 
# ### <a id="point6">6.</a> Count the number of `Complaint Type`
#    * Show the top 10 most popular complaint types.
#    * Plot the histogram of the 10 most popular complaint types.

# %%
top_comp_type = complaints.value_counts("Complaint Type").index[:10]
mask = complaints["Complaint Type"].isin(top_comp_type)
index = complaints[mask].value_counts("Complaint Type").index
val = complaints[mask].value_counts("Complaint Type").values
plt.barh(index, val) # will plot bar plot since data is categorical

# %% [markdown]
# ### 7. Extract all the rows with the Top 1 Complaint Type into a new dataframe
#    * Top 1 Complaint Type is the single most popular Complaint Type found in [the previous question](#point6) (from now on `Top1`)
#    * Show the top 3 rows of the new dataframe
#    * What is the relation between the Top1 Complaint Type and different boroughs?
#       * Leave out eventual unspecified data
#       * Find the data distribution (i.e., count them)
#       * Plot the histogram of the ratio of Top1 over all complaints, per borough.

# %%
top1_complaint = complaints.value_counts("Complaint Type").index[0]
print(f"Top1 complaint: {top1_complaint}")

df_top1 = complaints[complaints["Complaint Type"] == top1_complaint]
df_top1 = df_top1[complaints["Borough"] != "Unspecified"] # leave unspecified data
df_top1.head(3)


# %%
df_top1.hist("Borough", "Complaint Type")
df_top1.value_counts("Borough")


# %%
ratio = df_top1.value_counts("Borough") / complaints.shape[0]
plt.bar(ratio.index, ratio.values)


