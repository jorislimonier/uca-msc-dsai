# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from keras.datasets import cifar10
from IPython import get_ipython

# %% [markdown]
# # Dimension reduction in Python

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## the lab
#
# You can choose one of the following data sets:
# - **MNIST:** The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.
# - **Fashion MNIST:** Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples.
# - **CIFAR10:** The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
#
# The following cells allow you to load each of the data sets.

# %%
# # Load MNIST dataset
# from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, X_test.shape)

# # In the rest of this exercise, use only the test part of Fashion MINIST which already includes 10.000 images.
# data = X_test
# target = y_test

# # For a better understanding, let's create a dictionary that will have class names
# # with their corresponding categorical class labels.
# label_dict = {
#  0: '0',
#  1: '1',
#  2: '2',
#  3: '3',
#  4: '4',
#  5: '5',
#  6: '6',
#  7: '7',
#  8: '8',
#  9: '8',
# }


# %%
# # Load Fashion MNIST dataset
# from keras.datasets import fashion_mnist
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train.shape, X_test.shape)


# # In the rest of this exercise, use only the test part of Fashion MINIST which already includes 10.000 images.
# data = X_test
# target = y_test

# # For a better understanding, let's create a dictionary that will have class names
# # with their corresponding categorical class labels.
# label_dict = {
#  0: 'T-shirt/top',
#  1: 'Trouser',
#  2: 'Pullover',
#  3: 'Dress',
#  4: 'Coat',
#  5: 'Sandal',
#  6: 'Shirt',
#  7: 'Sneaker',
#  8: 'Bag',
#  9: 'Ankle boot',
# }


# %%
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, X_test.shape)

# In the rest of this exercise, use only the test part of CIFAR which already includes 10.000 images.
data = X_test
target = y_test

# For a better understanding, let's create a dictionary that will have class names
# with their corresponding categorical class labels.
label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# %% [markdown]
# By deleting 2 of the 3 cells in the code above, you can select your dataset.
# %% [markdown]
# - How many classes does the dataset contain?
# - What are the classes?

# %%
classes = label_dict.values()
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# %% [markdown]
# - Draw the first and last image of the data set with its label.

# %%
plt.figure(figsize=[5, 5])

# Display the first image
plt.subplot(121)

plt.imshow(data[0])

print(plt.title("(Label: " + str(label_dict[target[0][0]]) + ")"))

# Display the last image of the dataset
plt.subplot(122)
plt.imshow(data[-1])
print(plt.title("(Label: " + str(label_dict[target[-1][0]]) + ")"))

# %% [markdown]
# - Let's quickly check the maximum and minimum values of the images and <b>``normalize``</b> the pixels between 0 and 1 inclusive.

# %%
print("Before normalization:", np.min(data), np.max(data))
if np.amax(data) == 255:
    data = data / 255
print("After normalization: ", np.min(data), np.max(data))

# %% [markdown]
# - Next, you will create a DataFrame that will hold the pixel values of the images along with their respective labels in a row-column format.
#
# - But before that, let's reshape the image dimensions to one (flatten the images).

# %%
# Flatten the images
data_flat = np.reshape(data, (10000, -1))

# Build DataFrame
feat_cols = ['pixel'+str(i) for i in range(data_flat.shape[1])]
df = pd.DataFrame(data=data_flat, columns=feat_cols)
df['label'] = target
df['label'].replace(label_dict, inplace=True)
print(f'Size of the dataframe: {df.shape}')


# %%
data_flat
data_flat.shape

# %% [markdown]
# ### Data visualization using PCA
#
# - Next, you will create the PCA method and pass the number of components as two and apply ``fit_transform`` on the training data (without the label), this can take few seconds since there are a lot of samples

# %%
df


# %%
pca = PCA()
x_pca = pca.fit_transform(data_flat)
print(x_pca)


# %%
x_pca.shape

# %% [markdown]
# - Then you will convert the principal components for each images from a numpy array to a pandas DataFrame.

# %%
df.columns


# %%
df_pca = pd.DataFrame(
    x_pca, columns=[f"pca_vect{i}" for i in range(x_pca.shape[1])])
df_pca["y"] = df["label"]

# %% [markdown]
# - Let's quickly find out the amount of information or ``variance`` the principal components hold.

# %%
n_pca_vectors = 20
plt.plot(pca.explained_variance_ratio_[:n_pca_vectors])
pca.explained_variance_ratio_[:n_pca_vectors].sum()

# %% [markdown]
# Well, it looks like a decent amount of information was retained by the principal components 1 and 2, given that the data was projected from a lot of dimensions to a mere two principal components.
#
# Its time to visualize the dataset in a two-dimensional space. Remember that there is some semantic class overlap in this dataset which means that a frog can have a slightly similar shape of a cat or a deer with a dog; especially when projected in a two-dimensional space. The differences between them might not be captured that well.
#

# %%
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=df_pca.pca_vect0, y=df_pca.pca_vect1,
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_pca,
    legend="full",
    alpha=0.5
)

# %% [markdown]
# From the above figure, you can observe that some variation was captured by the principal components since there is some structure in the points when projected along the two principal component axis. The points belonging to the same class are close to each other, and the points or images that are very different semantically are further away from each other.
#
#
#
# %% [markdown]
# ### Data visualization using tSNE
#
# Now you will do the same exercise using the t-SNE algorithm. Scikit-learn has an implementation of t-SNE available, and you can check its documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). It provides a wide variety of tuning parameters for t-SNE, and the most notable ones are:
# - **n_components** (default: 2): Dimension of the embedded space.
# - **perplexity** (default: 30): The perplexity is related to the number of nearest neighbors that are used in other manifold learning algorithms. Consider selecting a value between 5 and 50.
# - **early_exaggeration** (default: 12.0): Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
# - **learning_rate** (default: 200.0): The learning rate for t-SNE is usually in the range (10.0, 1000.0).
# - **nc** (default: 1000): Maximum number of iterations for the optimization. Should be at least 250.
# - **method** (default: ‘barnes_hut’): Barnes-Hut approximation runs in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time.
#
# Be careful: t-SNE takes much longer to run on the same data sample size than PCA.

# %%
tsne = TSNE()
x_tsne = tsne.fit_transform(data_flat)

# %% [markdown]
# Then you will convert the projection for each images from a numpy array to a pandas DataFrame.

# %%
df_tsne = pd.DataFrame(x_tsne)

# %% [markdown]
# Its time to visualize the dataset in a two-dimensional space.

# %%
plt.scatter(df_tsne[0], df_tsne[1], s=1)

# %% [markdown]
# - If the 2-D representation is not satisfactory i.e. if the classes are not well separated.It depends on the data set. Do a PCA projection keeping 80% of the explained variance, then apply a t-SNE projection.
# - **Attention:** depending on the dataset the calculation of the PCA for all features can be very long. Limit yourself to **50 components max.**

# %%
'''your code here'''

# %% [markdown]
# ### Data visualization using LDA
#
# Do the same whith LDA projection

# %%
'''your code here'''

# %% [markdown]
# - If the 2-D representation is not satisfactory i.e. if the classes are not well separated. It depends on the data set. Do a PCA 2-D projection in order to select the best components, then apply a LDA projection.

# %%
'''your code here'''
