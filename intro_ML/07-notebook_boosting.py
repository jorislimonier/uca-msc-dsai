# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TP boosting
# ## dataset: MNIST
# Diane Lingrand
# diane.lingrand@univ-cotedazur.fr
# 2021-2022

# %%
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("nb of train samples",len(y_train))

# %% [markdown]
# Display the number of data in the test dataset:

# %%
len(y_test)

# %% [markdown]
# Display the first 100 labels of the train dataset:

# %%
y_train[:100]

# %% [markdown]
# For the binary classification we will choose the class of digit '4' and the class of digit '8' in the MNIST dataset. Feel free to change the classes.

# %%
import numpy as np
from sklearn.utils import shuffle


# %%
# class of '4'
x_train4 = x_train[y_train==4,:]
# class of '8'
x_train8 = x_train[y_train==8,:]

# together
x_trainBinaire = np.append(x_train4,x_train8,axis=0)
# '4' as negative class and '8' as positive class
y_trainBinaire = np.append(np.full(len(x_train4),-1), np.full(len(x_train8),1))

# dimensions ?
print(x_trainBinaire.shape, y_trainBinaire.shape)

# shuffle. why ?
(x_trainBinaire,y_trainBinaire) = shuffle(x_trainBinaire,y_trainBinaire,random_state=0)

# %% [markdown]
# ## binary boosting: directly on image pixels
# An image = a 1-d array of pixels

# %%
n = x_trainBinaire.shape[0]
x_trainBinaire = x_trainBinaire.reshape(n,-1)
print(x_trainBinaire.shape)

# %% [markdown]
# What are the dimensions of x_trainBinaire ? Explain the values.

# %%
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


# %%
x_trainBinaire.shape, y_trainBinaire.shape


# %%
## learning the boosting (Adaboost)
# create the boosting object
myboosting = ensemble.AdaBoostClassifier(n_estimators=20, learning_rate=1, algorithm='SAMME.R')
# learn on the train dataset
myboosting.fit(x_trainBinaire,y_trainBinaire)
# prediction of train data: should be similar to labels
y_predBinaire = myboosting.predict(x_trainBinaire)
print('confusion matrix on train data',confusion_matrix(y_trainBinaire,y_predBinaire))

# %% [markdown]
# We have displayed the confusion matrix on the train dataset. It should be computed on the test dataset. Let's do it!

# %%
# TO DO
# preprocessing of test data(2 classes ....)rain
# class of '4'
x_test4 = x_test[y_test==4,:]
# class of '8'
x_test8 = x_test[y_test==8,:]


# together
x_testBinaire = np.append(x_test4,x_test8,axis=0)
# '4' as negative class and '8' as positive class
y_testBinaire = np.append(np.full(len(x_test4),-1), np.full(len(x_test8),1))

# dimensions ?
print(x_testBinaire.shape, y_testBinaire.shape)
x_testBinaire = x_testBinaire.reshape(x_testBinaire.shape[0],-1)
print(x_testBinaire.shape, y_testBinaire.shape)

# # shuffle. why ?
# (x_testBinaire,y_testBinaire) = shuffle(x_testBinaire,y_testBinaire,random_state=0)


# compute and display the confusion matrix
confusion_matrix(y_testBinaire, myboosting.predict(x_testBinaire))

# %% [markdown]
# How is the result ? And what about modifying the variable n_estimators ?
# %% [markdown]
# Pretty good. Increasing `n_estimators` improves the confusion matrix, but takes more time.
# %% [markdown]
# ## binary boosting using Haar filters
# First step: prepare the data before boosting algorithm.
# %% [markdown]
# ### Haar filters

# %%
from skimage import feature
from skimage import transform

# %% [markdown]
# For Haar filters, you can choose between two options:
# - automatic generation
# - hand-made filters

# %%
# automatic generation from 2 types: 
#       'type-2-x' and 'type-2-y'
# and dimensiosn of images: 28x28
feat_coord, feat_type = feature.haar_like_feature_coord(28,28, ['type-2-x','type-2-y'])
feat_coord.shape, feat_type.shape, x_train[0].reshape(-1).shape

# %% [markdown]
# How many filters ? Compare to the number of pixels ...

# %%
# transformation of images: apply all filters
cpt=0

for image in x_trainBinaire:
    # integral image computation
    int_image = transform.integral_image(image)
    side = int(np.sqrt(int_image.shape[0]))
    int_image = int_image.reshape(side, side)
    # print(int_image.shape)
    # Haar filters computation
    features = feature.haar_like_feature(int_image, 0, 0, 28, 28,feature_type=feat_type,feature_coord=feat_coord)
    if cpt == 0:
        ftrain = [features]
    else:
        ftrain = np.append(ftrain,[features],axis=0)
    cpt += 1

# %% [markdown]
# The previous cell may encounter problem of size. Try to remove some filters. Which ones ? How many ?

# %%
# for you

# %% [markdown]
# Another solution: let's build the list of filters!

# %%
feat_coord = np.array([list([[(0, 0), (27, 13)], [(14, 0), (27, 27)]]),
       list([[(0, 0), (13, 13)], [(14, 0), (27, 13)]])])
# this is just an example: write the list of filters you think you need
feat_type = np.array(['type-2-x', 'type-2-x'])

# %% [markdown]
# ### boosting
# Now compute the binary boosting using the Haar filters representation and compare with the previous one.

# %%
# for you


