# %% [markdown]
# # Sentiment analysis with an MLP and vector representation
# 

# %% [markdown]
# # Case Study: Sentiment Analysis
# 
# In this lab we use part of the 'Amazon_Unlocked_Mobile.csv' dataset published by Kaggle. The dataset contain the following information:
# 
# - Product Name
# - Brand Name
# - Price
# - Rating
# - Reviews
# - Review Votes
# 
# We are mainly interested by the 'Reviews' (X) and by the 'Rating' (y)
# 
# The goal is to try to predict the 'Rating' after reading the 'Reviews'. I've prepared for you TRAIN and TEST set.
# The work to be done is as follows:
# 
# 1. Feature extraction and baseline
#    - read the dataset and understand it
#    - put it in a format so that you can use `CountVectorizer` or`Tf-IDF` to extract the desired features
#    - perform on the desired dates and preprocessing
#    - use one of the classifiers you know to predict the polarity of different sentences
# 1. My first neural network
#    - reuse the features already extracted
#    - proposed a neural network built with Keras
# 1. Hyper-parameter fitting
#    - for the base line: adjust min_df, max_df, ngram, max_features + model's hyper-parameter
#    - for the neural network: adjust batch size, number of layers and number of neuron by layers, use earlystop
# 1. <span style="color:red">Word embedding
#    - stage 1 build a network that uses Keras' embedding which is not language sensitive.
#    - stage 2 build a network that simultaneously uses Keras' embedding and the features extracted in the first weeks.
#    - stage 3 try to use an existing embedding (https://github.com/facebookresearch/MUSE)
#      </span>
# 
# **WARNING:** the dataset is voluminous, I can only encourage you to work first on a small part of it and only at the end, when the code is well debugged and that it is necessary to build the "final model", to use the whole dataset.
# 

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-the-dataset" data-toc-modified-id="Read-the-dataset-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read the dataset</a></span></li><li><span><a href="#Text-normalisation" data-toc-modified-id="Text-normalisation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Text normalisation</a></span></li><li><span><a href="#Approach1---BOW-and-MLP-classifier" data-toc-modified-id="Approach1---BOW-and-MLP-classifier-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Approach1 - BOW and MLP classifier</a></span></li><li><span><a href="#Approach2---Keras-word-embedding-and-MLP-classifier" data-toc-modified-id="Approach2---Keras-word-embedding-and-MLP-classifier-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Approach2 - Keras word embedding and MLP classifier</a></span></li></ul></div>
# 

# %% [markdown]
# ## Read the dataset
# 
# Could you find below a proposal. You can complete them.
# 

# %%
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dense, Embedding, Flatten, Input, TextVectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow_addons.metrics import F1Score


# %%
TRAIN = pd.read_csv(
    "http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/train.csv.gz"
)
TEST = pd.read_csv(
    "http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/test.csv.gz"
)

TRAIN.head()


# %%
# Construct X_train and y_train
X_train = TRAIN["Reviews"]
y_train = np.array(TRAIN["Rating"]).reshape(-1, 1)

X_test = TEST["Reviews"]
y_test = np.array(TEST["Rating"]).reshape(-1, 1)

nb_classes = len(np.unique(y_train))

ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
y_train_ohe = ohe.fit_transform(y_train)
y_test_ohe = ohe.fit_transform(y_test)

X_train.shape, y_train_ohe.shape, np.unique(y_train)


# %% [markdown]
# ## Approach1 - BOW and MLP classifier
# 
# Using the course companion notebook, build a multi-layer perceptron using a BOW representation of the dataset and evaluate the model.
# 
# The dataset being unbalanced the metric will be the f1 score.
# 

# %% [markdown]
# $$TO DO STUDENT$$
# 
# > - Build BOW representation of the train and test set
# > - Fix a value for vocab_size = the maximum number of words to keep, based on word frequency. Only the most common vocab_size-1 words will be kept.
# 

# %%
# Your code
vocab_size = 20000
tokenize = Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(X_train)
X_train_ohe = tokenize.texts_to_matrix(X_train, mode="tfidf")
X_test_ohe = tokenize.texts_to_matrix(X_test, mode="tfidf")


# %% [markdown]
# $$TO DO STUDENT$$
# 
# > - Build an MLP and print the model (model.summary())
# 

# %%
# build sequential model
model = Sequential()
model.add(Input(shape=(vocab_size,), name="input", dtype=tf.float32))
model.add(Dense(64, activation="relu", name="hidden"))
model.add(Dense(5, activation="softmax"))
model.build()
model.summary()


# %% [markdown]
# $$ TO DO STUDENT $$
# 
# > - Compile the network
# > - Fit the network using EarlyStopping
# > - Babysit your model
# > - Evaluate the network with f1 score
# 

# %%
X_train_ohe.shape


# %%
y_train_ohe.shape, y_test_ohe.shape


# %%
## compile the model with f1 metrics
# define F1Score instance

f1_score_name = "f1_score"
f1 = F1Score(
    num_classes=len(np.unique(y_test)),
    name=f1_score_name,
    average="weighted",
)
# compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[f1, "accuracy"],
)

# define early stopping
early_stop = EarlyStopping(
    monitor="val_f1_score",
    patience=10,
    verbose=1,
    restore_best_weights=True,
    mode="max",
)

# fit model using early stopping
history = model.fit(
    x=X_train_ohe,
    y=y_train_ohe,
    validation_data=(X_test_ohe, y_test_ohe),
    # validation_split=.3,
    # batch_size=1,
    epochs=2000,
    verbose=1,
    callbacks=[early_stop],
    workers=6,
    use_multiprocessing=True,
)


# %%
import plotly.express as px

# Babysit the model - use you favourite plot
px.line(
    pd.DataFrame(
        {
            "val_loss": history.history["val_loss"],
            "loss": history.history["loss"],
            "val_f1_score": history.history["val_f1_score"],
            "f1_score": history.history["f1_score"],
            "val_accuracy": history.history["val_accuracy"],
            "accuracy": history.history["accuracy"],
        }
    )
)


# %%
# Evaluate the model with f1 metrics (Tensorflow f1 metrics or sklearn)
model.evaluate(X_test_ohe, y_test_ohe)


# %% [markdown]
# ## Approach2 - Keras word embedding and MLP classifier
# 
# Using the course companion notebook, build a multi-layer perceptron using an Embedding Keras layer and the same classifier as in approach 1. Evaluate the model.
# 

# %% [markdown]
# $$ TO DO STUDENTS $$
# 
# > - fix the max_lengh of a review (max number of token in a review)
# > - use the same vocab_size as previously
# > - fix the embedding dimension (embed_dim variable)
# 

# %%
import nltk

X_train_tok = [nltk.word_tokenize(review) for review in X_train]
max_len = int(
    np.amax([len(review_tok) for review_tok in X_train_tok])
)  # Sequence length to pad the outputs to
# In order to fix it, you have to know the distribution on lengh... see first lab
embed_dim = 300  # embedding dimension


# %% [markdown]
# $$ TO DO STUDENTS $$
# 
# > - Create a vectorizer_layer with TextVectorization function
# > - Fit the vectorizer_layer (adapt function
# 

# %%
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=max_len,
)
vectorize_layer.adapt(X_train)
vectorize_layer(X_test)  # display vectorized test set


# %%
vectorize_layer.get_vocabulary()


# %%
vectorize_layer.get_config()


# %% [markdown]
# $$TO DO STUDENT$$
# 
# > - Build an MLP and print the model (model.summary())
# 

# %%
# Flatten after Embedding in order to reduce the dimension of tensors
model = Sequential()
model.add(Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(Embedding(input_dim=vocab_size, output_dim=64))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation="sigmoid"))
model.build()

# get summary of the model
model.summary()


# %% [markdown]
# $$ TO DO STUDENT $$
# 
# > - Compile the network
# > - Fit the network using EarlyStopping
# > - Babysit your model
# > - Evaluate the network with f1 score
# 

# %%
vectorize_layer(X_train)


# %%
# compile the model with metrics f1 score
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[f1, "accuracy"],
)

# define early stopping
early_stop = EarlyStopping(
    monitor=f"val_{f1_score_name}",
    patience=10,
    verbose=3,
    restore_best_weights=True,
    mode="max",
)

# fit model using ealy stopping
history = model.fit(
    x=X_train,
    y=y_train_ohe,
    epochs=2000,
    validation_data=(X_test, y_test_ohe),
    callbacks=early_stop,
    use_multiprocessing=True,
    workers=6,
)


# %%
# Babysit the model
px.line(
    pd.DataFrame(
        {
            "val_loss": history.history["val_loss"],
            "loss": history.history["loss"],
            "val_f1_score": history.history["val_f1_score"],
            "f1_score": history.history["f1_score"],
            "val_accuracy": history.history["val_accuracy"],
            "accuracy": history.history["accuracy"],
        }
    )
)


# %%
# Evaluate the model
model.evaluate(X_test, y_test_ohe)


# %% [markdown]
# **The model seems to overfit: its results improve on the train set, but (at best) remain stable on the validation set.**
# 

# %% [markdown]
# ## Approach3 - Word embedding and MLP classifier
# 
# Using the course companion notebook, build a multi-layer perceptron using an existing embedding matrix (Word2Vec / Glove or FastText), or on an embedding matrix that you will have built using Gensim.
# 
# Use the same constant as a previous steps.
# 
# Evaluate the model.
# 

# %%
import gensim
from gensim import models, utils

gensim_path = f"{gensim.__path__[0]}/test/test_data/"
corpus = "lee_background.cor"
corpus_path = gensim_path + corpus


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


# %%
sentences = MyCorpus()
model = models.Word2Vec(sentences=sentences, vector_size=150)


# %%
# Same steps as Keras Embedding
max_len = 10  # Sequence length to pad the outputs to.
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(X_train)
X_train_vec = vectorizer(X_train)
X_train_vec


# %%
# Build word dict
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))


# %%
# Make a dict mapping words (strings) to their NumPy vector representation:

path_to_glove_file = "glove.6B/glove.6B.50d.txt"
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")


# %%
# Prepare embedding matrix

num_tokens = len(voc) + 2
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    print(word, i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print(f"Converted {hits} words ({misses} misses)")


# %%
# Define embedding layers

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
)


# %%
# define the model
input_ = Input(shape=(max_len,), dtype=tf.int32)
x = embedding_layer(input_)
x = Flatten()(x)
output_ = Dense(5, activation="sigmoid")(x)
model = Model(input_, output_)
# summarize the model
model.summary()


# %%
# compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[f1, "accuracy"],
)

# define early stopping
early_stop = EarlyStopping(
    monitor=f"val_{f1_score_name}",
    patience=100,
    verbose=2,
    mode="max",
    restore_best_weights=True,
)


# fit model using ealy stopping
history = model.fit(
    x=vectorizer(X_train),
    y=y_train_ohe,
    validation_data=(vectorizer(X_test), y_test_ohe),
    epochs=2000,
    callbacks=early_stop,
)


# %%
# Babysit the model
px.line(
    pd.DataFrame(
        {
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "f1_score": history.history["f1_score"],
            "val_f1_score": history.history["val_f1_score"],
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
        }
    )
)


# %%
# Evaluate the model
model.evaluate(vectorizer(X_test), y_test_ohe)


# %% [markdown]
# ## Approach3 (bis) - Word embedding and MLP classifier
# 
# Using the course companion notebook, build a multi-layer perceptron using an existing embedding matrix (Word2Vec / Glove or FastText), or on an embedding matrix that you will have built using Gensim.
# 
# Use the same constant as a previous steps.
# 
# Evaluate the model.
# 

# %%
# Build gensim model
import gensim
from gensim.test.utils import datapath
from gensim import utils
import gensim.models

gensim_path = f"{gensim.__path__[0]}/test/test_data/"
corpus = "lee_background.cor"
corpus_path = gensim_path + corpus


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, vector_size=150)


# %%
# Export gensim model
import tempfile

with tempfile.NamedTemporaryFile(prefix="gensim-model-", delete=False) as tmp:
    temporary_filepath = tmp.name
    print(temporary_filepath)
    model.save(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
    #
    # To load a saved model:
    #
    new_model = gensim.models.Word2Vec.load(temporary_filepath)


# %%
# Load gensim model
new_model = gensim.models.Word2Vec.load(temporary_filepath)


# %%
# Prepare embedding matrix
num_tokens = len(voc) + 2
embedding_dim = 150
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    try:
        model.wv[word]
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = model.wv[word]
        hits += 1
    except :
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# %%
# Define embedding layers
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
)


# %%
# define the model
input_ = Input(shape=(max_len,), dtype=tf.int32)
x = embedding_layer(input_)
x = Flatten()(x)
output_ = Dense(5, activation='sigmoid')(x)
model = Model(input_, output_)
# summarize the model
model.summary()

# %%
# compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", f1],
)

# define early stopping
early_stop = EarlyStopping(
    monitor="val_f1_score",
    patience=100,
    verbose=1,
    restore_best_weights=True,
    mode="max",
)


# fit model using ealy stopping
history = model.fit(
    x=vectorizer(X_train),
    y=y_train_ohe,
    epochs=2000,
    verbose=1,
    validation_data=(vectorizer(X_test), y_test_ohe),
    callbacks=[early_stop],
    workers=6,
    use_multiprocessing=True,
)


# %%
# Babysit the model
px.line(
    pd.DataFrame(
        {
            "val_loss": history.history["val_loss"],
            "loss": history.history["loss"],
            "val_f1_score": history.history["val_f1_score"],
            "f1_score": history.history["f1_score"],
            "val_accuracy": history.history["val_accuracy"],
            "accuracy": history.history["accuracy"],
        }
    )
)


# %%
# evaluate the model
model.evaluate(vectorizer(X_test), y_test_ohe, verbose=1)



