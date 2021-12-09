# %% [markdown]
# 
# # TP Clustering
# ### Diane LINGRAND 
# 
# diane.lingrand@univ-cotedazur.fr 
# 

# %%
#import the necessary libraries
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

# %% [markdown]
# # Experiments on synthetic data

# %% [markdown]
# ## Generating blobs

# %%
from sklearn.datasets import make_blobs

# %%
n_samples = 1500
random_state = 160
#random_state is the seed for the random generation and let you reproduce the exact same dataset
X, y = make_blobs(centers=3, n_samples=n_samples, random_state=random_state)


# %% [markdown]
# ### Drawing the data

# %%
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c='k')
plt.show()

# %% [markdown]
# ## Clustering method: k-means
# Don't hesitate to read the doc: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html !

# %%
from sklearn.cluster import KMeans

# %%
nbClusters = 3

# %% [markdown]
# ### Initialisation of the centroids

# %%
## basic kmean with random initialisation
km = KMeans(n_clusters=nbClusters, max_iter=10, n_init=1, init='random', algorithm='full')

# %%
## random initialisation but using fixed seed for reproducing the experiments
random_state2 = 150
km = KMeans(n_clusters=nbClusters, max_iter=10, n_init=1, init='random', random_state=random_state2, algorithm='full')

# %%
## fixed initialisation of the centroid (no random)
c = np.array([[2,2],[1,5],[2,-4]]) #this is a bad init. try also c=np.array([[5,5],[8,2],[7,2]])
km = KMeans(n_clusters=nbClusters, max_iter=10, n_init=1, init=c, algorithm='full')

# %%
## random initialisation using heuristic 'k-means++' 
km = KMeans(n_clusters=nbClusters, max_iter=10, n_init=1, init='k-means++', algorithm='full')
# you could also increase n_init parameter for trying different initialisation of the centroids

# %% [markdown]
# ### Learning and prediction

# %%
#learning and computing the result:
y_pred = km.fit_predict(X)

# %% [markdown]
# ### Visualisation of the clustering result.

# %%
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
c=km.cluster_centers_
plt.scatter(c[:,0],c[:,1],c='r',marker="X")
plt.show()

# %% [markdown]
# ## Clustering method: k-medoid
# Don't hesitate to read the doc: https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html !

# %%
from sklearn_extra.cluster import KMedoids
km1 = KMedoids(n_clusters=nbClusters, max_iter=10, init='random')

# %% [markdown]
# Change the init parameters to ‘k-medoids++’ or other parameters and observe the differences.

# %% [markdown]
# Don't forget to visualize the results!

# %% [markdown]
# ## Clustering method: agglomerative clustering
# Don't hesitate to read the doc: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

# %%
from sklearn.cluster import AgglomerativeClustering

# %%
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
#play with the linkages
y_pred = ac.fit_predict(X)

# %% [markdown]
# ## Clustering method: DBSCAN
# Don't hesitate to read the doc: https://scikit-learn.org/dev/modules/generated/sklearn.cluster.DBSCAN.html?highlight=dbscan#sklearn.cluster.DBSCAN

# %%
from sklearn.cluster import DBSCAN

# %%
db = DBSCAN(eps=0.3, min_samples=10)
y_pred=db.fit_predict(X)

#noisy points will be labelled as -1: choose a special color (black ?) for those noisy points
nbClusters = max(y_pred)+1
nbNoise = list(y_pred).count(-1)

print('Number of clusters: ', nbClusters)
print('Number of noisy samples: ', nbNoise)


# %% [markdown]
# ## Clustering method: GMM + EM
# Don't hesitate to read the doc: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html?highlight=gaussianmixture

# %%
from sklearn.mixture import GaussianMixture

# %%
nbGaussians = 3
nbClusters = 3
gm = GaussianMixture(n_components=nbGaussians, covariance_type='spherical', max_iter=50, random_state=0)
y_pred = gm.fit_predict(X)

# %% [markdown]
# ## Evaluation methods
# Don't hesitate to read the doc: https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

# %%
from sklearn import metrics

# %%
print("Calinski Harabasz score: %.2f" %metrics.calinski_harabasz_score(X, y_pred))
print("Davies Bouldin score: %.2f" %metrics.davies_bouldin_score(X, y_pred))
silhouetteScore = metrics.silhouette_score(X, y_pred)
print("Silhouette Coefficient: %.2f" %silhouetteScore)



# %% [markdown]
# **Code for plotting silhouettes.** From https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# %%
import matplotlib.cm as cm

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(18, 7)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(X) + (nbClusters + 1) * 10])
# Compute the silhouette scores for each sample
sample_silhouette_values = metrics.silhouette_samples(X, y_pred)
y_lower = 10
#loop over clusters
for i in range(nbClusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / nbClusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouetteScore, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# %% [markdown]
# # Experiments on Iris dataset or MNIST, FMNIST, ...

# %% [markdown]
# ## Loading the data

# %%
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# %%
# in order to plot this data set on your screen, you need to reduce the dimension to 2

# %% [markdown]
# ### PCA option

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2 = pca.fit(X).transform(X)

# %%
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %% [markdown]
# ### t-SNE option

# %%


# %% [markdown]
# ## Testing K-means and variants
# Remember that you need to find the best configuration: be careful of the configuration of algorithms !

# %%


# %%


# %%
# your work here


