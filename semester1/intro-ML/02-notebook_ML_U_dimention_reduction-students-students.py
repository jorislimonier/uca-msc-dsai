# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.datasets import load_breast_cancer
from IPython import get_ipython

# %% [markdown]
# # Dimension reduction in Python

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Principal Component Analysis (PCA)
#
# ### Introduction
#
# Principal Component Analysis (PCA) is a **linear dimensionality reduction** technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.
#
# Dimensions are nothing but features that represent the data. For example, A 28 X 28 image has 784 picture elements (pixels) that are the dimensions or features which together represent that image.
#
# One important thing to note about PCA is that it is an **Unsupervised** dimensionality reduction technique, you can cluster the similar data points based on the feature correlation between them without any supervision (or labels), and you will learn how to achieve this practically using Python in later sections of this tutorial!
#
# According to *Wikipedia*, PCA is a **statistical** procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.
#
# **Note**: Features, Dimensions, and Variables are all referring to the same thing. You will find them being used interchangeably.
#
# #### But where can you apply PCA?
#
# - **Data Visualization**: When working on any data related problem, the challenge in today's world is the sheer volume of data, and the variables/features that define that data. To solve a problem where data is the key, you need extensive data exploration like finding out how the variables are correlated or understanding the distribution of a few variables. Considering that there are a large number of variables or dimensions along which the data is distributed, visualization can be a challenge and almost impossible.
#
#     Hence, PCA can do that for you since it projects the data into a lower dimension, thereby allowing you to visualize the data in a 2D or 3D space with a naked eye.
#
# - **Speeding Machine Learning (ML) Algorithm**: Since PCA's main idea is dimensionality reduction, you can leverage that to speed up your machine learning algorithm's training and testing time considering your data has a lot of features, and the ML algorithm's learning is too slow.
#
# At an abstract level,  you take a dataset having many features, and you simplify that dataset by selecting a few ``Principal Components`` from original features.
#
# #### What is a Principal Component?
#
# Principal components are the key to PCA; they represent what's underneath the hood of your data. In a layman term, when the data is projected into a lower dimension (assume three dimensions) from a higher space, the three dimensions are nothing but the three Principal Components that captures (or holds) most of the variance (information) of your data.
#
# Principal components have both direction and magnitude. The direction represents across which *principal axes* the data is mostly spread out or has most variance and the magnitude signifies the amount of variance that Principal Component captures of the data when projected onto that axis. The principal components are a straight line, and the first principal component holds the most variance in the data. Each subsequent principal component is orthogonal to the last and has a lesser variance. In this way, given a set of <i>x</i> correlated variables over <i>y</i> samples you achieve a set of <i>u</i> uncorrelated principal components over the same <i>y</i> samples.
#
# The reason you achieve uncorrelated principal components from the original features is that the correlated features contribute to the same principal component, thereby reducing the original data features into uncorrelated principal components; each representing a different set of correlated features with different amounts of variation.
#
# Each principal component represents a percentage of total variation captured from the data.
#
# In today's tutorial, you will mainly apply PCA on the two use-cases:
# - ``Data Visualization``
# - ``Speeding ML algorithm``
#
# To accomplish the above two tasks, you will use two famous Breast Cancer (numerical) and CIFAR - 10 (image) dataset.
# %% [markdown]
# ## the Data: Breast Cancer
#
# The Breast Cancer data set is a real-valued multivariate data that consists of two classes, where each class signifies whether a patient has breast cancer or not. The two categories are: malignant and benign.
#
# The malignant class has 212 samples, whereas the benign class has 357 samples.
#
# It has 30 features shared across all classes: radius, texture, perimeter, area, smoothness, fractal dimension, etc.
#
# You can download the breast cancer dataset from <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">here</a>, or rather an easy way is by loading it with the help of the ``sklearn`` library.

# %%
# Load the data

# %% [markdown]
# ``load_breast_cancer`` will give you both labels and the data. To fetch the data, you will call ``.data`` and for fetching the labels ``.target``.
#
# The data has 569 samples with thirty features, and each sample has a label associated with it. There are two labels in this dataset.

# %%
breast = load_breast_cancer()

features = breast.feature_names
breast_data = breast.data
breast_labels = breast.target
breast_data.shape, breast_labels.shape


# %%
final_breast_data = np.concatenate(
    [breast_data, breast_labels.reshape(-1, 1)], axis=1)
breast_dataset = pd.DataFrame(final_breast_data)
breast_dataset.columns = np.append(breast.feature_names, 'label')
breast_dataset.head()

# %% [markdown]
# Since the original labels are in 0,1 format, you will change the labels to benign and malignant using .replace function. You will use inplace=True which will modify the dataframe breast_dataset.

# %%
breast_dataset['label'].replace(0, 'Benign', inplace=True)
breast_dataset['label'].replace(1, 'Malignant', inplace=True)
breast_dataset.tail()

# %% [markdown]
# ### Data Visualization using PCA
#
# Now comes the most exciting part of this tutorial. As you learned earlier that PCA projects turn high-dimensional data into a low-dimensional principal component, now is the time to visualize that with the help of Python!
#
# - You start by <b>``Standardizing``</b> the data since PCA's output is influenced based on the scale of the features of the data.
# - It is a common practice to normalize your data before feeding it to any machine learning algorithm.
#
# - To apply normalization, you will import ``StandardScaler`` module from the sklearn library and select only the features from the ``breast_dataset`` you created in the Data Exploration step. Once you have the features, you will then apply scaling by doing ``fit_transform`` on the feature data.
#
# - While applying StandardScaler, each feature of your data should be normally distributed such that it will scale the distribution to a mean of zero and a standard deviation of one.

# %%
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)  # normalizing the features

# %% [markdown]
# Let's check whether the normalized data has a mean of zero and a standard deviation of one.

# %%
np.mean(x), np.std(x)

# %% [markdown]
# Let's convert the normalized features into a tabular format with the help of DataFrame.

# %%
normalised_breast = pd.DataFrame(x, columns=breast_dataset.columns[:-1])
normalised_breast.tail()

# %% [markdown]
# We now apply the PCA to these data using ``sklearn.decomposition.PCA``

# %%
pca = PCA()
x_pca = pca.fit_transform(x)

# %% [markdown]
# Some questions:
# 1. How many principal components do we have?
# 1. What is the main contribution to the first principal component
# 1. What is the explained variance of the first principal component

# %%
print("Number of principal components: {0:d}".format(len(pca.components_)))
print("Composition of the 1rst components: {0}".format(
    feat_cols[np.argmax(pca.components_[0])]))
print("Explained variance of the first principal component: {0}".format(
    pca.explained_variance_[0]))

# %% [markdown]
# 4. If you want to reduce, the reduction dimension of your dataset. How many principal dimensions do you choose?

# %%
# Plot the explained variance by components --> Ratio criterion
plt.figure(figsize=(11, 8.5))
plt.plot(pca.explained_variance_ratio_, "-o", label="explained variance")
plt.plot([0.1]*len(pca.components_), label="10%")
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.legend(loc="best")
plt.ylim(0, 1)
plt.show()


# %%
# Plot the cumulative explained variance --> Cumulative explained variance criterion
plt.figure(figsize=(11, 8.5))
plt.plot(np.cumsum(pca.explained_variance_ratio_),
         "-o", label="cumulative explained variance")
plt.plot([0.9]*len(pca.components_), label="90%")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.legend(loc="best")
plt.show()


# %%
# Select the appropriate number of components --> Cumulative explained variance criterion
def choose_nb_components(pca, threshold):
    sum_expl_variance = 0
    for i, expl_variance in enumerate(pca.explained_variance_ratio_):
        sum_expl_variance += expl_variance
        if sum_expl_variance > threshold:
            return i+1
    return i+1


nb_components = choose_nb_components(pca, 0.9)
nb_components

# %% [markdown]
# By adopting the 90% rule we could keep only 7 main components
# %% [markdown]
# Now comes the critical part, the next few lines of code will be projecting the thirty-dimensional Breast Cancer data to two-dimensional <b>``principal components``</b>.

# %%
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

# %% [markdown]
# Next, let's create a DataFrame that will have the principal component values for all 569 samples.

# %%
principal_breast_Df = pd.DataFrame(data=principalComponents_breast, columns=[
                                   'principal component 1', 'principal component 2'])
principal_breast_Df.tail()

# %% [markdown]
# - Once you have the principal components, you can find the <b>``explained_variance_ratio``</b>. It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.

# %%
print('Explained variation per principal component: {}'.format(
    pca_breast.explained_variance_ratio_))

# %% [markdown]
# From the above output, you can observe that the ``principal component 1`` holds 44.2% of the information while the ``principal component 2`` holds only 19% of the information. Also, the other point to note is that while projecting thirty-dimensional data to a two-dimensional data, 36.8% information was lost.
#
# Let's plot the visualization of the 569 samples along the ``principal component - 1`` and ``principal component - 2`` axis. It should give you good insight into how your samples are distributed among the two classes.

# %%
plt.figure()
plt.figure(figsize=(10, 10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1', fontsize=20)
plt.ylabel('Principal Component - 2', fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset", fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1'],
                principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

plt.legend(targets, prop={'size': 15})

# %% [markdown]
# From the above graph, you can observe that the two classes ``benign`` and ``malignant``, when projected to a two-dimensional space, can be linearly separable up to some extent. Other observations can be that the ``benign`` class is spread out as compared to the ``malignant`` class.
# %% [markdown]
# ### Data visualization using tSNE
#
# Now you will do the same exercise using the t-SNE algorithm. Scikit-learn has an implementation of t-SNE available, and you can check its documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). It provides a wide variety of tuning parameters for t-SNE, and the most notable ones are:
# - **n_components** (default: 2): Dimension of the embedded space.
# - **perplexity** (default: 30): The perplexity is related to the number of nearest neighbors that are used in other manifold learning algorithms. Consider selecting a value between 5 and 50.
# - **early_exaggeration** (default: 12.0): Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
# - **learning_rate** (default: 200.0): The learning rate for t-SNE is usually in the range (10.0, 1000.0).
# - **n_iter** (default: 1000): Maximum number of iterations for the optimization. Should be at least 250.
# - **method** (default: ‘barnes_hut’): Barnes-Hut approximation runs in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time.
#
# Be careful: t-SNE takes much longer to run on the same data sample size than PCA.

# %%
tsne = TSNE(n_components=2)
tsne_breast = tsne.fit_transform(x)


# %%
tsne_breast_Df = pd.DataFrame(data=tsne_breast, columns=['axis 1', 'axis 2'])
tsne_breast_Df.tail()


# %%
plt.figure()
plt.figure(figsize=(10, 10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Axis - 1', fontsize=20)
plt.ylabel('Axis - 2', fontsize=20)
plt.title("t-SNE of Breast Cancer Dataset", fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(tsne_breast_Df.loc[indicesToKeep, 'axis 1'],
                tsne_breast_Df.loc[indicesToKeep, 'axis 2'], c=color, s=50)

plt.legend(targets, prop={'size': 15})

# %% [markdown]
# ### Visualizing data with LDA
#
# Now try to do the same exercise using the LDA algorithm. Scikit-learn has an implementation of LDA which you can consult the documentation [here] (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).
#
# Remember that LDA is a supervised projection.
# %% [markdown]
# If you have an error, it is normal. Why is this?
#
# hit : look at the description of the n_components parameter

# %%
