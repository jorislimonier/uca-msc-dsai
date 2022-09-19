# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12,12)


# %%
iris = load_iris()
X, y = iris.data, iris.target


# %%
classifier = KNeighborsClassifier()


# %%
y


# %%
import numpy as np
rng = np.random.RandomState(0)


# %%
permutation = rng.permutation(len(X))


# %%
permutation


# %%
X, y = X[permutation], y[permutation]


# %%
y


# %%
k = 5
n_samples = len(X)
fold_size = n_samples // k


# %%
fold_size


# %%
masks = []
scores = []
for fold in range(k):
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[fold * fold_size : (fold + 1) * fold_size] = True
    masks.append(test_mask)
    X_test, y_test = X[test_mask], y[test_mask]
    X_train, y_train = X[~test_mask], y[~test_mask]
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))


# %%
scores


# %%
plt.matshow(masks, cmap='gray_r')


# %%
np.mean(scores)


# %%
from sklearn.model_selection import cross_val_score


# %%
scores = cross_val_score(classifier, X, y)
np.mean(scores)


# %%
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit


# %%
cv = StratifiedKFold(n_splits=5)
for train, test in cv.split(iris.data, iris.target):
    print(test)


# %%
def plot_cv(cv, features, labels):
    masks = []
    for train, test in cv.split(features, labels):
        mask = np.zeros(len(labels), dtype=bool)
        mask[test] = 1
        masks.append(mask)
        
    plt.matshow(masks, cmap='gray_r')


# %%
plot_cv(StratifiedKFold(n_splits=5), iris.data, iris.target)


# %%
plot_cv(KFold(n_splits=5), iris.data, iris.target)


# %%
plot_cv(KFold(n_splits=15), iris.data, iris.target)


# %%
plot_cv(ShuffleSplit(n_splits=5, test_size=.2), iris.data, iris.target)


# %%
plot_cv(ShuffleSplit(n_splits=25, test_size=.2), iris.data, iris.target)


# %%
cv = ShuffleSplit(n_splits=5, test_size=.2)


# %%
cross_val_score(classifier, X, y, cv=cv)


# %%
classifier.get_params()


# %%
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# %%
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=42)


# %%
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# %%
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, n_jobs=-1)


# %%
grid.fit(X_train, y_train)


# %%
grid.best_params_


# %%
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))


