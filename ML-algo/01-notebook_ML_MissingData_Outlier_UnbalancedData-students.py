# %% [markdown]
# # Work with Missing value, Outlier, Unbalanced Dataset

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports-and-Dataset" data-toc-modified-id="Imports-and-Dataset-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports and Dataset</a></span></li><li><span><a href="#Sampler,--transformer-and-estimator" data-toc-modified-id="Sampler,--transformer-and-estimator-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Sampler,  transformer and estimator</a></span></li><li><span><a href="#Lab-1:-Missing-value" data-toc-modified-id="Lab-1:-Missing-value-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Lab 1: Missing value</a></span></li><li><span><a href="#Outlier-removal" data-toc-modified-id="Outlier-removal-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Outlier removal</a></span></li><li><span><a href="#Unbalance-dataset" data-toc-modified-id="Unbalance-dataset-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Unbalance dataset</a></span></li></ul></div>

# %% [markdown]
# ## Imports and Dataset

# %%
# import warnings
# warnings.filterwarnings('ignore')

# %%
from tqdm import tqdm
import seaborn as sns  # For plotting data
import pandas as pd  # For dataframes
import numpy as np
import matplotlib.pyplot as plt  # For plotting data

# %matplotlib inline

# For splitting the dataset
from sklearn.model_selection import train_test_split

# For setting up pipeline
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# For Missing data
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# For Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# For Unbalanced dataset
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# For classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# For optimization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# %% [markdown]
# The **original ForestCover/Covertype dataset** from UCI machine learning repository is a multiclass classification dataset. This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.
#
# In this notebook you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.
#
# This dataset has 54 attributes :
# * 10 quantitative variables,
# * 4 binary wilderness areas
# * and 40 binary soil type variables).
# Here, outlier detection dataset is created using only 10 quantitative attributes. Instances from class 2 are considered as normal points and instances from class 4 are anomalies. The anomalies ratio is 0.9%. Instances from the other classes are omitted.
#
# Dataset description available on [Kaggle](https://www.kaggle.com/uciml/forest-cover-type-dataset).
# * Elevation: Elevation in meters.
# * Aspect: Aspect in degrees azimuth.
# * Slope: Slope in degrees.
# * Horizontal_Distance_To_Hydrology: Horizontal distance in meters to nearest surface water features.
# * Vertical_Distance_To_Hydrology: Vertical distance in meters to nearest surface water features.
# * Horizontal_Distance_To_Roadways: Horizontal distance in meters to the nearest roadway.
# * Hillshade_9am: hillshade index at 9am, summer solstice. Value out of 255.
# * Hillshade_Noon: hillshade index at noon, summer solstice. Value out of 255.
# * Hillshade_3pm: shade index at 3pm, summer solstice. Value out of 255.
# * Horizontal_Distance_To_Fire_Point*: horizontal distance in meters to nearest wildfire ignition points.
# * Wilderness_Area#: wilderness area designation.
# * Soil_Type#: soil type designation.
#
# Wilderness_Area feature is one-hot encoded to 4 binary columns (0 = absence or 1 = presence), each of these corresponds to a wilderness area designation. Areas are mapped to value in the following way:
# 1. Rawah Wilderness Area
# 1. Neota Wilderness Area
# 1. Comanche Peak Wilderness Area
# 1. Cache la Poudre Wilderness Area
#
# The same goes for Soil_Type feature which is encoded as 40 one-hot encoded binary columns (0 = absence or 1 = presence) and each of these represents soil type designation. All the possible options are:
# 1. Cathedral family - Rock outcrop complex, extremely stony
# 1. Vanet - Ratake families complex, very stony
# 1. Haploborolis - Rock outcrop complex, rubbly
# 1. Ratake family - Rock outcrop complex, rubbly
# 1. Vanet family - Rock outcrop complex complex, rubbly
# 1. Vanet - Wetmore families - Rock outcrop complex, stony
# 1. Gothic family
# 1. Supervisor - Limber families complex
# 1. Troutville family, very stony
# 1. Bullwark - Catamount families - Rock outcrop complex, rubbly
# 1. Bullwark - Catamount families - Rock land complex, rubbly.
# 1. Legault family - Rock land complex, stony
# 1. Catamount family - Rock land - Bullwark family complex, rubbly
# 1. Pachic Argiborolis - Aquolis complex
# 1. ¨unspecified in the USFS Soil and ELU Survey
# 1. Cryaquolis - Cryoborolis complex
# 1. Gateview family - Cryaquolis complex
# 1. Rogert family, very stony
# 1. Typic Cryaquolis - Borohemists complex
# 1. Typic Cryaquepts - Typic Cryaquolls complex
# 1. Typic Cryaquolls - Leighcan family, till substratum complex
# 1. Leighcan family, till substratum, extremely bouldery
# 1. Leighcan family, till substratum - Typic Cryaquolls complex
# 1. Leighcan family, extremely stony
# 1. Leighcan family, warm, extremely stony
# 1. Granile - Catamount families complex, very stony
# 1. Leighcan family, warm - Rock outcrop complex, extremely stony
# 1. Leighcan family - Rock outcrop complex, extremely stony
# 1. Como - Legault families complex, extremely stony
# 1. Como family - Rock land - Legault family complex, extremely stony
# 1. Leighcan - Catamount families complex, extremely stony
# 1. Catamount family - Rock outcrop - Leighcan family complex, extremely stony
# 1. Leighcan - Catamount families - Rock outcrop complex, extremely stony
# 1. Cryorthents - Rock land complex, extremely stony
# 1. Cryumbrepts - Rock outcrop - Cryaquepts complex
# 1. Bross family - Rock land - Cryumbrepts complex, extremely stony
# 1. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony
# 1. Leighcan - Moran families - Cryaquolls complex, extremely stony
# 1. Moran family - Cryorthents - Leighcan family complex, extremely stony
# 1. Moran family - Cryorthents - Rock land complex, extremely stony
#
# Cover_Type: forest cover type designation, its possible values are between 1 and 7, mapped in the following way:
# 1. Spruce/Fir
# 1. Lodgepole Pine
# 1. Ponderosa Pine
# 1. Cottonwood/Willow
# 1. Aspen
# 1. Douglas-fir
# 1. Krummholz
#
# <font color=blue>
# We will use a very small part of this dataset with only classes 1 and 7.
# </font>

# %%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://www.i3s.unice.fr/~riveill/dataset/covtype/"
filename = "covtype.csv"

# %%
# load train and test
train = pd.read_csv(url + "train.csv", delimiter=",")
test = pd.read_csv(url + "test.csv", delimiter=",")

# %%
columns = list(train.columns)
target = "Cover_Type"
columns.remove(target)
cat_columns = [
    c for c in columns if "Soil_Type" in c or "Wilderness_Area" in c
]  # already one hot encode
num_columns = [c for c in columns if c not in cat_columns]

# %%
y_train = np.array(train[target]).reshape(-1, 1)
X_train = train[columns]

y_test = np.array(test[target]).reshape(-1, 1)
X_test = test[columns]

X_train.shape, X_test.shape

# %%
# Class distribution
distribution = pd.Series(y_train.flatten()).value_counts().to_dict()
distribution

# %% [markdown]
# ## Sampler,  transformer and estimator
#
# There are three types of objects in imblearn/scikit-learn design:
#
# **Transformer** transform observation (modify only X_train) and implements:
# * fit: used for calculating the initial parameters on the training data and later saves them as internal objects state.
# * transform: Use the initial above calculated values and return modified training data as output. Do not modify the length of the dataset.
#
# **Predictor** is a "model" and implements:
# * fit: calculates the parameters or weights on the training data and saves them as an internal object state.
# * predict: Use the above-calculated weights on the test data to make the predictions.
#
# **Sampler** is a new element, from imblearn library. A sampler modifies the number of observations in the train set (modify X_train and y_train) and implements:
# * fit_resample
#
# The following cells build a pipeline

# %%
# A sampler
class mySampler(BaseEstimator):
    def fit_resample(self, X, y):
        data = np.concatenate((X, y), axis=1)
        # remove rows with NaN
        data = data[~np.isnan(data).any(axis=1), :]
        return data[:, :-1], data[:, -1]


# %% [markdown]
# It's also possible to build sampler from a function

# %%
def mySamplerFunction(X, y, conta=0.1):
    iforest = IsolationForest(n_estimators=300, max_samples="auto", contamination=conta)
    outliers = iforest.fit_predict(X, y)

    X_filtered = X[outliers == 1]
    y_filtered = y[outliers == 1]

    return X_filtered, y_filtered


# %%
# A transformer
class myTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy
        self.sample = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        return self.sample.fit(X, y)

    def transform(self, X):
        return self.sample.transform(X)


# %% [markdown]
# Like sampler, it's also possible to build transformer from a function see `sklearn.preprocessing.FunctionTransform`

# %%
# A predictor
class myPredictor(BaseEstimator):
    def __init__(self, penalty="l2"):
        self.penalty = penalty
        self.sample = LogisticRegression(
            solver="lbfgs", penalty=self.penalty, max_iter=10000
        )

    def fit(self, X, y):
        return self.sample.fit(X, y)

    def predict(self, X):
        return self.sample.predict(X)


# %%
# Different version of the 2 steps pipeline
# step 1 : remove or imput missing data
# step 2 : remove outlier
# step 3 : predictor
pipeline = Pipeline(
    [
        ("missing_data", None),
        ("outlier", FunctionSampler(func=mySamplerFunction)),
        ("clf", None),
    ]
)

parameters = [
    {
        "missing_data": [mySampler()],
        "clf": [myPredictor()],
        "clf__penalty": ["none"],
    },
    {
        "missing_data": [myTransformer()],
        "missing_data__strategy": ["most_frequent"],
        "clf": [myPredictor()],
        "clf__penalty": ["none"],
    },
]

pipeline

# %%
# GridSearch with pipeline
grid = GridSearchCV(
    pipeline, parameters, cv=2, scoring="f1_micro", refit=True, verbose=2
)
grid

# %% [markdown]
# <font color="green">
# Remember: samplers are only called to perform the "fit" and not to perform the predict. If the data set contains missing values (NaN) in the validation part, a warning may be raised.
# </font>

# %%
# Try to find the best model
grid.fit(
    X_train, y_train
)  # Some data for testing the process... but use all available data

# %%
# Evaluate the model with the whole dataset
y_pred = grid.predict(X_train[:500])
print(f"Best: {round(grid.best_score_, ndigits=2)} using {grid.best_params_}")
print(f"Test set score: {grid.score(X_train[:500], y_train[:500])}")

# %% [markdown]
# ## Lab 1: Missing value
#
# $$[TO DO - Students]$$
#
# Test some algorithms to handle missing data.
# * Choose the classifier that you think is preferable for this job.
#
# 1. with removal of missing data
# 1. with of the following imputation methods
#     * With SimpleImputer
#     * With IterativeImputer
#     * With KNNimputer
#
# Build a 2 step pipeline and use a gridsearch to find the right hyperpameters.
#
# Submit your work in the form of an executable and commented notebook at lms.univ-cotedazur.fr

# %%

pipeline_miss_val = Pipeline(
    [
        ("missing_data", None),
        ("clf", None),
    ]
)
param_miss_val = [
    {
        "missing_data": [mySampler()],
        "clf": [myPredictor()],
        "clf__penalty": ["l1", "l2", "none"],
    },
    {
        "missing_data": [SimpleImputer()],
        "missing_data__strategy": ["mean", "most_frequent"],
        "clf": [myPredictor()],
        "clf__penalty": ["l2", "none"],
    },
]

grid_miss_val = GridSearchCV(
    estimator=pipeline_miss_val,
    param_grid=param_miss_val,
    cv=2,
    scoring="f1_micro",
    refit=True,
    verbose=3,
)
grid_miss_val.fit(X_train, y_train)
# %%
print(
    f"Best: {round(grid_miss_val.best_score_, ndigits=2)} using {grid_miss_val.best_params_}"
)
print(f"Test set score: {grid_miss_val.score(X_test, y_test)}")

# %% [markdown]
# ## Outlier removal
#
# Removing the outliers modifies the data set, so it is a sampler.
#
# <font color='green'>
# IsolationForest or other sklearn detector are not a sampler. You have to read the </font>[imblearn documentation](https://imbalanced-learn.org/dev/references/generated/imblearn.FunctionSampler.html)
#
# A small example with parameters:
# <pre>
# from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
#
# def func(X, y, sampling_strategy, random_state):
#   return RandomUnderSampler(
#       sampling_strategy=sampling_strategy,
#       random_state=random_state).fit_resample(X, y)
#
# sampler = FunctionSampler(func=func,
#                           kw_args={'sampling_strategy': 'auto',
#                                    'random_state': 0})
# X_res, y_res = sampler.fit_resample(X, y)
# print(f'Resampled dataset shape {sorted(Counter(y_res).items())}')
# </pre>
#
# $$[TO DO - Students]$$
#
# Test some algorithms to handle outliers.
# * Choose the classifier that you think is preferable for this job.
#
# 1. Without taking any precautions
# 1. By eliminating outliers with one of the following approaches:
#     * With Isolation Forest (IF)
#     * With Local Outlier Factor (LOF)
#     * With Minimum Covariance Determinant (MCD)
#
# Build a 3 step pipeline and use a gridsearch to find the right hyperpameters.
# The first step, is your best previous "missing data method".
#
# Submit your work in the form of an executable and commented notebook at lms.univ-cotedazur.fr

# %%


class mySamplerClass(BaseEstimator):
    def __init__(self, conta=0.1):
        self.conta = conta

    def fit_resample(self, X, y):
        iforest = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=self.conta,
        )
        outliers = iforest.fit_predict(X, y)

        X_filtered = X[outliers == 1]
        y_filtered = y[outliers == 1]

        return X_filtered, y_filtered


# %%
pipeline_outlier = Pipeline(
    [
        ("missing_data", myTransformer(strategy="most_frequent")),
        ("outlier", None),
        ("clf", myPredictor(penalty="none")),
    ]
)

parameters_outlier = [
    {
        "outlier": [mySamplerClass()],
        "outlier__conta": np.linspace(0.015, 0.18, 5),
    },
]

grid_outlier = GridSearchCV(
    estimator=pipeline_outlier,
    param_grid=parameters_outlier,
    cv=2,
    scoring="f1_micro",
    refit=True,
    verbose=3,
)
grid_outlier.fit(X_train, y_train)
# %%
# %%
print(
    f"Best: {round(grid_outlier.best_score_, ndigits=2)} using {grid_outlier.best_params_}"
)
print(f"Test set score: {grid_outlier.score(X_test, y_test)}")


# %% [markdown]
# ## Unbalance dataset
#
# $$[TO DO - Students]$$
#
# Test some algorithms to work with unbalanced dataset.
# Choose the classifier that you think is preferable for this job.
#
# 1. Without taking any precautions
# 1. With modification of the dataset by Over sampling or Under sampling or SMOTE
# 1. Without modification of the dataset by weight
#
# Build a 4 step pipeline and use a gridsearch to find the right hyperpameters and use a gridsearch to find the right hyperpameters. The first and second step, is your best previous methods.
#
# Submit your work in the form of an executable and commented notebook at lms.univ-cotedazur.fr

# %%
