# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import csv
cwd = os.getcwd()
file = cwd + "/data/food-inspections.csv"
with open(file) as f:
    food = csv.reader(f, delimiter=",")
    for row in food:
        print(", ".join(f))


# %%
food


