# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction-:-a-visual-intuition-of-activation-functions" data-toc-modified-id="Introduction-:-a-visual-intuition-of-activation-functions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction : a visual intuition of activation functions</a></span></li><li><span><a href="#Build-an-MLP-to-classify-MNIST-images" data-toc-modified-id="Build-an-MLP-to-classify-MNIST-images-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Build an MLP to classify MNIST images</a></span></li></ul></div>

# %% [markdown]
# # Introduction to the Multi Layer Perceptron

# %% [markdown]
# ## Introduction : a visual intuition of activation functions

# %% [markdown]
# Using a regression task on the sinus function, we'll try to get an intuition of the effect of activation functions.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# %%
from sklearn.model_selection import train_test_split

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

# %%
X = np.arange(-3*np.pi,3*np.pi,0.01)
y = np.sin(X)

plt.figure(figsize=(13,8))
plt.plot(X, y, label='sinus', color='red')
plt.legend()
plt.show()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

plt.figure(figsize=(13,8))
plt.scatter(X_train, y_train, alpha = 0.03, label='training data')
plt.scatter(X_test, y_test,alpha=0.5,label='testing data')

plt.legend()
plt.show()

# %% [markdown]
# $$[TODO - Students]$$
# 
# * Build a model with 1 hidden layer in 1 dimension and train it on x_train, y_train.
# * What activation should we use for ouput layer ?
# * What loss should we use ?
# 
# * Try different activations for the hidden layer and plot the predictions obtained on x_test
# * Plot also a learning curve

# %%
model = ...

model.compile(...)

history = model.fit(...)

# %%
# Plot leaning cuve
plt.figure(figsize=(13,8))
plt.plot(history.history['loss'])
plt.show()

# %%
def plot_prediction(title, model):
    y_hat_test = model.predict(X_test)

    plt.figure()
    plt.scatter(X_test, y_test, label = 'ground_truth', alpha=0.1)
    plt.scatter(X_test, y_hat_test, label = 'predicted', alpha=0.5)
    plt.legend()
    
    plt.title(title)
    plt.show()

# %%
plot_prediction("my title", model)

# %% [markdown]
# $$[TODO - Students]$$
# 
# Try adding layers and increasing the layers dimension to better fit the test data. You can use the following function to quickly build your models.
# 
# * Try different n_layers (for example 1, 10, 100)
# * Try different hidden_dim (for example 32, 128, 256, 512)
# * Try different bach size
# * Try to understand the `patience` parameters of early stopping

# %%
def build_sin_regression(activation, n_layers, hidden_dim):
    input = Input(shape=(1,), name='input')

    for i in range(n_layers):
      if i==0:
        x = Dense(input_shape=(1,), units=hidden_dim, activation=activation, name='layer_'+str(i))(input)
      else:
        x = Dense(units=hidden_dim, activation=activation, name='layer_'+str(i))(x)

    output = Dense(1, activation='linear', name='output')(x)
    model = Model(input, output, name = 'sinus_regression')
    return model

# %%
model = build_sin_regression(activation = XXX, n_layers = XXX, hidden_dim = XXX)
model.compile(loss='mse', optimizer='adam')
model.summary()

# %%
plot_model(model)

# %%
callbacks_list = [EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=2, mode='min', restore_best_weights=True)
                 ]

history = model.fit(X_train, y_train, validation_split = 0.1, callbacks=callbacks_list, batch_size=32, epochs=20)

# %%
plt.figure(figsize=(13,8))
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend()
plt.show()

# %%
y_hat_test = model.predict(X_test)

plt.scatter(X_test, y_test,label = 'ground_truth', alpha=0.1)
plt.scatter(X_test, y_hat_test,label = 'predicted', alpha=0.5)

plt.legend()
plt.show()

# %% [markdown]
# ## Build an MLP to classify MNIST images
# 
# Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We’ll call the images “x” and the labels “y”. Both the training set and test set contain images and their corresponding labels; for example, the training images are mnist.train.images and the training labels are mnist.train.labels.

# %%
# Load dataset
from tensorflow.keras.datasets import mnist

# the data, shuffled and split between a train and test sets 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, y_train.shape

# %%
# Reshape the image from 3d to 2d (nb_items, other dime)


# %%
# Normalize the data (input between 0 and 1)


# %%
# One hot encode the label


# %%
# Buid MLP model
# You can use the following function
def build_MLP(input_shape, activation, layers, nb_class):
    input = Input(shape=(input_shape,), name='input')

    for i, hidden_size in enumerate(layers):
        if i == 0:
            x = Dense(input_shape=(input_shape,), units=hidden_size, activation=activation, name='layer_'+str(i))(input)
        else:
            x = Dense(units=hidden_size, activation=activation, name='layer_'+str(i))(x)

    output = Dense(nb_class, activation='softmax', name='output')(x)
    model = Model(input, output, name = 'mnist_classifier')
    model.summary()
    return model

model = build_MLP()## TO BE COMPLETED ##)

# %%
# Compile and fit the model
callbacks_list = [EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=20,
                                verbose=2, mode='min', restore_best_weights=True)
                 ]

model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer='adam')
history = model.fit(''' X''', ''' y ''', validation_split = 0.1, callbacks=callbacks_list,
                    batch_size=32, epochs=20)

# %%
# Print history keys
history.history.keys()

# %%
# Babysit your model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26,8))

ax1.plot(history.history['loss'], label="loss")
ax1.plot(history.history['val_loss'], label="val_loss")
ax1.legend()
ax2.plot(history.history['accuracy'], label="accuracy")
ax2.plot(history.history['val_accuracy'], label="val_accuracy")
ax2.legend()
plt.show()

# %%
# Evaluate the model
score = model.evaluate(X_test, y_test_enc)
print('Test loss:', score[0])
print('Test accuracy', score[1])

# %%
# Modify the network in order to obtain better accuracy (better than 0.96)


