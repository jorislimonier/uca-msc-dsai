
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10,8)
import pandas as pd
import numpy as np


# %%
from sklearn.datasets import make_blobs, make_moons 
from sklearn.linear_model import LogisticRegression


# %%
header = ["buying", "maint", "doors", "person", "lug_boot", "safety", "class"]
data = pd.read_csv("cars.csv", header=None, names=header)
for col in data.columns:
    # print(f"{data.value_counts(col)}\n")
    print(f"{data[col].unique()}\n")


# %%
X = data.drop(["class"], axis=1)
y = data["class"]


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# %%
try:
    header.remove("class")
except:
    pass


# %%
X


# %%
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=header)
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)


# %%
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)


# %%
y_pred = clf.predict(X_test)
clf.score(X_test, y_test)


