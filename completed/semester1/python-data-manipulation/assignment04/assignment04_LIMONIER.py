# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Assignement n.4
# 
# Collect the ABALONE Dataset from [here](https://archive.ics.uci.edu/ml/datasets/Abalone).
# 
# You"ll get 2 files:
#    * abalone.data  : csv for the dataset
#    * abalone.names : explanation and names of the attributes.
#    
# ## GOALS:
# 
#   1. Provide basic data exploration results on the dataset
#   2. Train a binary classifier on the Abalone dataset for the following target classes:
#       * 0: young snail, number of rings <= 12
#       * 1: old snail, number of rings > 12
# 
# ### Notes:
# 
#   * Use 70-30 as train-test split
#   * Try different classifiers and compare the results
#   * Provide an evaluation of the classifier
#   * *Optional* Tune the classifier"s hyperparameters
#   * Comment with markdown cells everything you code

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Load data

# %%
initial_data = pd.read_csv("abalone.data", header=None)
initial_data.columns = ["sex", "length", "diam", "height", "whole", "shucked", "viscera", "shell", "rings"]

initial_data

# %% [markdown]
# # Exploratory Data Analysis

# %%
initial_data.info()


# %%
initial_data.describe()


# %%
initial_data.isna().sum() # No NA, life is beautiful


# %%
initial_data.hist(figsize=(12,6))


# %%
initial_data["sex"].value_counts().plot(kind="bar", figsize=(6,4))

# %% [markdown]
# # Data preprocessing

# %%
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

data = initial_data.copy()
ord_enc = OrdinalEncoder()
sex_enc = ord_enc.fit_transform(data[["sex"]])

# %% [markdown]
# #### See effect of ordinal encoding

# %%
for item in zip(np.unique(sex_enc), np.unique(data["sex"])):
    print(f"{item[1]} is encoded as {item[0]}")


# %%
pd.DataFrame({"sex_enc": sex_enc[:,0], "sex": data["sex"]}).head()


# %%
data["sex"] = sex_enc


# %%
if "rings" in data.columns:
    data["young"] = np.where(data["rings"] <= 12, 1, 0)
    data = data.drop(columns="rings")

# %% [markdown]
# # Classification

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% [markdown]
# ### Perform train-test split

# %%
X = data.drop(columns="young").copy().values
y = data[["young"]].copy().values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# %% [markdown]
# ### SVC without scaling

# %%
clf_svc = SVC()
clf_svc.fit(X_train, y_train.reshape(-1))
pred = clf_svc.predict(X_test)
print("Classification report\n", classification_report(y_test, pred))
print("Confusion matrix\n", confusion_matrix(y_test, pred))
print("Accuracy\n", accuracy_score(pred, y_test))

# %% [markdown]
# ### SVC with standard scaling

# %%
from sklearn.preprocessing import StandardScaler


# %%
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
clf_svc = SVC()
clf_svc.fit(X_train_scaled, y_train.reshape(-1))
pred = clf_svc.predict(X_test_scaled)
print("Classification report\n", classification_report(y_test, pred))
print("Confusion matrix\n", confusion_matrix(y_test, pred))
print("Accuracy\n", accuracy_score(pred, y_test))

# %% [markdown]
# ### SVC with MinMax scaling

# %%
from sklearn.preprocessing import MinMaxScaler


# %%
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
clf_svc = SVC()
clf_svc.fit(X_train_scaled, y_train.reshape(-1))
pred = clf_svc.predict(X_test_scaled)
print("Classification report\n", classification_report(y_test, pred))
print("Confusion matrix\n", confusion_matrix(y_test, pred))
print("Accuracy\n", accuracy_score(pred, y_test))

# %% [markdown]
# The standard scaler seems to perform best for SVC. Let's now implement cross validation.
# ## $k$-fold cross validation

# %%
from sklearn.model_selection import ShuffleSplit, cross_val_score,GridSearchCV


# %%
cv = ShuffleSplit(test_size=.3)
cv_scores = cross_val_score(clf_svc, X, y.ravel(), cv=cv)
print(cv_scores, cv_scores.mean())
cv_scores = cross_val_score(clf_svc, X, y.ravel(), cv=5)
print(cv_scores, cv_scores.mean())
cv_scores = cross_val_score(clf_svc, X, y.ravel(), cv=10)
print(cv_scores, cv_scores.mean())
clf_svc.get_params()

# %% [markdown]
# ## Grid search SVC

# %%
param_grid = {
    'C': [100, 200, 300],
    'gamma': ['scale', 'auto'],
    'gamma': [100000, 20000, 10000],
    'kernel': ['linear']
}
grid = GridSearchCV(clf_svc, param_grid=param_grid, verbose=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train.ravel())


# %%
grid.best_params_


# %%
grid_pred_svc = grid.predict(X_test_scaled)
print("Classification report\n", classification_report(y_test, grid_pred_svc))
print("Confusion matrix\n", confusion_matrix(y_test, grid_pred_svc))
print("Accuracy\n", accuracy_score(grid_pred_svc, y_test))

# %% [markdown]
# We achieve 88.12% accuracy, which is the best we got so far.
# %% [markdown]
# ## Random forest

# %%
from sklearn.ensemble import RandomForestClassifier


# %%
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train.ravel())


# %%
pred_rf = clf_rf.predict(X_test)
print("Classification report\n", classification_report(y_test, pred_rf))
print("Confusion matrix\n", confusion_matrix(y_test, pred_rf))
print("Accuracy\n", accuracy_score(pred_rf, y_test))

# %% [markdown]
# ## Grid search Random Forest

# %%
param_grid_rf = {'bootstrap': [True, False],
                 'max_depth': [10, 100],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1],
                 'min_samples_split': [4, 5, 6],
                 'n_estimators': [50, 100, 200]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf,
                    refit=True, verbose=3, n_jobs=-1)
grid_rf.fit(X_train, y_train.ravel())


# %%
grid_rf.best_params_


# %%
grid_pred_rf = clf_rf.predict(X_test)
print("Classification report\n", classification_report(y_test, grid_pred_rf))
print("Confusion matrix\n", confusion_matrix(y_test, grid_pred_rf))
print("Accuracy\n", accuracy_score(grid_pred_rf, y_test))

# %% [markdown]
# ## Neural Networks?

# %%
import torch


# %%
X_train_ten, y_train_ten, X_test_ten, y_test_ten = map(
    torch.tensor, (X_train, y_train, X_test, y_test)
)


# %%



