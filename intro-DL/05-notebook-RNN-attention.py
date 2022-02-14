# %% [markdown]
# # Attention mechanism for sentiment analysis
# 

# %% [markdown]
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

# %%
"""
(Practical tip) Table of contents can be compiled directly in jupyter notebooks using the following code:
I set an exception: if the package is in your installation you can import it otherwise you download it 
then import it.
"""
try:
    from jyquickhelper import add_notebook_menu 
except:
    !pip install jyquickhelper
    from jyquickhelper import add_notebook_menu
    
"""
Output Table of contents to navigate easily in the notebook. 
For interested readers, the package also includes Ipython magic commands to go back to this cell
wherever you are in the notebook to look for cells faster
"""
add_notebook_menu()

# %% [markdown]
# ## Imports
# 

# %%
import os

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.strings
import tensorflow_addons as tfa
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (LSTM, Activation, AveragePooling1D,
                                     Bidirectional, Dense, Dot, Dropout,
                                     Embedding, Flatten, Input, Permute,
                                     RepeatVector, TextVectorization,
                                     TimeDistributed)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import plot_model
from tensorflow_addons.metrics import F1Score


# %% [markdown]
# ## Read the dataset
# 
# Could you find below a proposal. You can complete them.
# 

# %%
BASE_DATASET_PATH = "http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/"
TRAIN = pd.read_csv(f"{BASE_DATASET_PATH}train.csv.gz").fillna(value="")
VAL = pd.read_csv(f"{BASE_DATASET_PATH}val.csv.gz").fillna(value="")
TEST = pd.read_csv(f"{BASE_DATASET_PATH}test.csv.gz").fillna(value="")

# TRAIN = TRAIN[:2042]  # save training time

TRAIN.head()


# %%
""" Construct X_train and y_train """
X_train = np.array(TRAIN["Reviews"]).reshape(-1, 1)

ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
y_train = ohe.fit_transform(np.array(TRAIN["Rating"]).reshape(-1, 1))

X_train.shape, y_train.shape


# %%
""" Do the same for val """
X_val = np.array(VAL["Reviews"]).reshape(-1, 1)
y_val = ohe.transform(np.array(VAL["Rating"]).reshape(-1, 1))

""" Do the same for test """
X_test = np.array(TEST["Reviews"]).reshape(-1, 1)
y_test = ohe.transform(np.array(TEST["Rating"]).reshape(-1, 1))


# %%
X_train.shape, X_test.shape, X_val.shape


# %% [markdown]
# ## Build an a neural network with vectorized embedding and RNN cells.
# 
# The task is to predict the sentiment according to the content of the review. We can treat this kind of task by a Many-to-one model.
# 
# ![LSTM for sentiment analysis](https://www.programmerall.com/images/679/8c/8c66e6ee3b9418358a791b363572bedf.jpeg)
# 
# Implement such a network with :
# 
# - a first layer of type LSTM
# - a second layer of type LSTM, each cell of this layer will be fed by the corresponding output of the first layer (see figure above).
# 

# %% [markdown]
# ### Study of the size of the reviews.

# %%
import plotly.express as px

px.histogram(
    data_frame={"review_length": [len(review[0].split(" ")) for review in X_train]} ,
    cumulative=True,
    histnorm="probability density",
    title="Cumulative histogram of the review lengths (in number of words)"
)


# %% [markdown]
# We simply spitted the reviews by white space, which is a simple, basic approach, but yet informative. The plot above tells us that in order to get $95\%$ of the reviews that are non-truncated, we could consider the `max_len` parameter to be 150.

# %%
# Constants
nb_classes = y_train.shape[1]
vocab_size = 10 ** 4  # Maximum vocab size -- adjust with the size of the vocabulary
embedding_size = 20  # Embedding size (usually <= 300)
recurrent_size = 64  # Recurrent size
hidden_size = recurrent_size // 4  # Hidden layer
dropout_rate = 0.2  # Dropout rate for regularization (usually between 0.1 and 0.25)
max_len = 150  # Sequence length to pad the outputs to (deduced from the length distribution study)
learning_rate = 0.0075


# %% [markdown]
# ### Try to deal with emojis
# Here, I try to make a callable that makes emojis behave as words, but it is not successful. I thought leaving traces of some work is better than showing nothing at all.\
# I dove into the tensorflow documentation to find what the `"strip_and_lower"` keyword actually did. I found that it uses `tf.strings.lower` and `tf.strings.regex_replace`, but I couldnt get it to split my emojis with a blank space.

# %%
emoji = X_train[0,0][-1]
DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
DEFAULT_STRIP_REGEX

# %%
X_train[0,0].encode("unicode_escape")
X_train_transf = tensorflow.strings.regex_replace(X_train, r"\\", "RRR")
X_train_transf = tensorflow.strings.lower(X_train_transf)
# X_train_transf = tensorflow.strings.regex_replace(X_train_transf, DEFAULT_STRIP_REGEX, "")
X_train_transf[0,0].numpy()

# %%
print(r"".format(X_train[0,0][-1]))

# %%
# Create the vectorized layer.
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    standardize="lower_and_strip_punctuation",
    # it is possible to build your own function
    # to transform emoji into text
    # to transform foreign reviews in english one
    # etc.
    output_mode="int",
    output_sequence_length=max_len,
)


# %%
# Fit vectorized layer on train
vectorize_layer.adapt(X_train)


# %%
vectorize_layer(X_train[0])

# %%
# Define the network
def build_model():
    input_ = Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(input_)
    x = Embedding(vocab_size, embedding_size, name="Embedding")(x)
    x = Bidirectional(
        LSTM(
            recurrent_size,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
        )
    )(x)
    x = Dense(hidden_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_ = Dense(nb_classes, activation="softmax", dtype=tf.float64)(x)
    model = Model(input_, output_)
    return model


model = build_model()


# %%
# summarize the model
model.summary()


# %%
plot_model(model, show_shapes=True, show_layer_names=True, to_file="05-model.png")


# %%
# Compile the model
f1 = F1Score(num_classes=nb_classes, average="macro", threshold=None)
op = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=op, loss="categorical_crossentropy", metrics=[f1])


# %%
# fit model using ealy stopping
es = EarlyStopping(
    monitor="val_f1_score",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    epochs=4000,
    callbacks=[es],
    verbose=1,
)


# %%
# plot history
def babysit(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # summarize history for loss
    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.set_title("model loss")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    ax1.legend(["loss train", "loss val"], loc="best")

    # summarize history for loss
    ax2.plot(history.history["f1_score"])
    ax2.plot(history.history["val_f1_score"])
    ax2.set_title("model F1")
    ax2.set_ylabel("F1")
    ax2.set_xlabel("epoch")
    ax2.legend(["f1 train", "f1 val"], loc="best")

    plt.show()


babysit(history)


# %%
# Evaluate the model
f1.update_state(y_test, model.predict(X_test))
print(f"F1: {f1.result().numpy()}")


# %% [markdown]
# <font color='red'>
# To do student
# 
# 1. Understand the code
# 1. Play with LSTM model for sentiment analysis
#    - Replace LSTM by BI-LSTM
#    - Use stacked LSTM or BI-LSTM \* Use all hidden state and average it
#    </font>
# 

# %% [markdown]
# <font color='green'>
# If you want to go further
# 
# If you are interested in the subject, current networks for sentiment prediction combine a part with recurrent networks (LTSM) to capture long dependencies and a part with convolution (CNN) to capture short dependencies. [This resarch paper](https://arxiv.org/pdf/1904.04206.pdf) or [this one](https://hal.archives-ouvertes.fr/hal-03052803/document) describe some accurate networks for sentiment analysis.
# 
# Here, another paper that gives you some indications to go further: [Attention, CNN and what not for Text Classification](https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/)
# 
# You will see next week the CNN with Diane. So there is no need to use them today.
# </font>
# 

# %% [markdown]
# ## Attention with LSTM network
# 

# %%
# ------------------------------------------------------
# MODEL BUILDING
# ------------------------------------------------------
def build_model():
    # Input: a review
    input_ = Input(shape=(1,), name="input", dtype=tf.string)

    # Transform the review in a list of tokenID
    vect = vectorize_layer(input_)

    # Keras embedding
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=None,  # Without pre-learning
        trainable=True,  # Trainable
        name="embedding",
    )(vect)

    # You can try also a Bidirectionnel cell
    rnn = LSTM(
        recurrent_size,
        return_sequences=True,
        return_state=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
    )(embedding)

    # In the case of LSTM, there are two internal states
    #      the hidden state, usually denoted by h,
    #      the cell state usually denoted by c
    # The tuple (c, h) is the internal state of a LSTM
    # return_sequences=True gives you the hidden state (h) of the LSTM for every timestep
    # used in combination with return_state=True, you will only get the tuple (c, h) for the final timestep

    # Attention layer
    attention = Dense(1, activation="tanh")(rnn)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)

    # Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
    attention = RepeatVector(recurrent_size)(attention)  # NORMAL RNN
    attention = Permute([2, 1])(attention)

    # Application de l'attention sur la sortie du RNN
    sent_representation = Dot(axes=1, normalize=False)([rnn, attention])

    # Flatten pour entrer dans le Dense
    flatten = Flatten()(sent_representation)

    # Dense pour la classification avec 1 couche cachee
    hidden_dense = Dense(hidden_size, activation="relu")(flatten)
    hidden_dense = Dropout(dropout_rate)(hidden_dense)

    # Classification et ouput
    output_ = Dense(nb_classes, activation="softmax")(hidden_dense)

    # Build  model
    model = Model(inputs=input_, outputs=output_)

    return model


model = build_model()


# %%
# Plot model
plot_model(
    model=model, show_shapes=True, show_layer_names=True, to_file="LSTM_with_attention.png"
)


# %%
# ------------------------------------------------------
# MODEL BUILDING
# ------------------------------------------------------
def build_hypermodel(hp):
    # Input: a review
    input_ = Input(shape=(1,), name="input", dtype=tf.string)

    # Transform the review in a list of tokenID
    vect = vectorize_layer(input_)

    # Keras embedding
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=None,  # Without pre-learning
        trainable=True,  # Trainable
        name="embedding",
    )(vect)

    # You can try also a Bidirectionnel cell
    rnn = LSTM(
        recurrent_size,
        return_sequences=True,
        return_state=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
    )(embedding)

    # In the case of LSTM, there are two internal states
    #      the hidden state, usually denoted by h,
    #      the cell state usually denoted by c
    # The tuple (c, h) is the internal state of a LSTM
    # return_sequences=True gives you the hidden state (h) of the LSTM for every timestep
    # used in combination with return_state=True, you will only get the tuple (c, h) for the final timestep

    # Attention layer
    attention = Dense(1, activation="tanh")(rnn)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)

    # Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
    attention = RepeatVector(recurrent_size)(attention)  # NORMAL RNN
    attention = Permute([2, 1])(attention)

    # Application de l'attention sur la sortie du RNN
    sent_representation = Dot(axes=1, normalize=False)([rnn, attention])

    # Flatten pour entrer dans le Dense
    flatten = Flatten()(sent_representation)

    # Dense pour la classification avec 1 couche cachee
    hp_hidden_size = hp.Choice('units', values=[2**power for power in range(4, 11)])
    hp_dropout_rate = hp.Choice('rate', values=[0.1 + k*0.05 for k in range(5)])
    
    hidden_dense = Dense(hp_hidden_size, activation="relu")(flatten)
    hidden_dense = Dropout(hp_dropout_rate)(hidden_dense)

    # Classification et ouput
    output_ = Dense(nb_classes, activation="softmax")(hidden_dense)

    # Build  model
    model = Model(inputs=input_, outputs=output_)
    
    # Compile the model
    f1 = F1Score(num_classes=nb_classes, average="macro", threshold=0.5)

    hp_learning_rate = hp.Choice('learning_rate', values=[0.005, 0.001, 0.0005, 0.0001])
    hp_beta_1 = hp.Choice('beta_1', values=[0.8, 0.9, 0.99, 0.999])
    hp_beta_2 = hp.Choice('beta_2', values=[0.99, 0.999, 0.9999])

    op = Adam(learning_rate=hp_learning_rate, beta_1=hp_beta_1, beta_2=hp_beta_2, epsilon=1e-08)
    model.compile(optimizer=op, loss="categorical_crossentropy", metrics=[f1])

    return model




# %%
# Compile the model
f1 = F1Score(num_classes=nb_classes, average="macro", threshold=0.5)
op = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=op, loss="categorical_crossentropy", metrics=[f1])


# %%
tuner = kt.Hyperband(
    hypermodel=build_hypermodel,
    objective=kt.Objective("val_f1_score", direction="max"),
    max_epochs=20,
    factor=3,
)

# fit model using ealy stopping
es = EarlyStopping(
    monitor="val_f1_score",
    mode="max",
    patience=20,
    restore_best_weights=True,
    verbose=2,
)

tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[es])


# %%
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.values

# %%
hp_model = tuner.hypermodel.build(best_hps)

# %%
history = hp_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=4000,
    callbacks=[es],
    verbose=1,
)


# %%
# plot history
babysit(history)


# %%
# Evaluate the model
f1.update_state(y_test, hp_model.predict(X_test))
print(f"F1: {f1.result()}")


# %% [markdown]
# ## Use Attentionnal model
# 

# %% [markdown]
# In the cell below, we reproduce our existing model until the Activation layer only.

# %%
# ------------------------------------------------------
# GET ATTENTION MODEL
# ------------------------------------------------------
def get_attention_model(model):
    attention_layer_indice = 0
    for layer in model.layers:
        print(type(layer))
        if type(layer) is Activation:
            break
        else:
            attention_layer_indice += 1

    # Create an attention model
    return Model(
        inputs=model.layers[0].input,
        outputs=model.layers[attention_layer_indice].output,
    )


# %%
# PLOT ATTENTION MODEL from classifier model with ATTENTION
attention_model = get_attention_model(model)
plot_model(
    attention_model,
    show_shapes=True,
    show_layer_names=True,
    to_file="model_get_attention.png",
)


# %%
# ------------------------------------------------------
# GET ATTENTION
# ------------------------------------------------------
attentions = attention_model.predict(X_val[0])
attentions


# %% [markdown]
# Now we measure the attention given by our model to some sample words. The model gives quite some attention to "pretty" and "good". This makes sense since it is a meaningful indicator which hints that the review will be positive. \
# Had the terms been more vague, the mdoel would have (or at least should have) given them less attention.

# %%
# ------------------------------------------------------
# GET ATTENTION for each WORD
# ------------------------------------------------------
from sklearn import preprocessing


def get_attention(X, y, prediction, attention, N=5):
    # normalize attention (without the padding part)
    normalized_attention = preprocessing.QuantileTransformer().fit_transform(attention)

    results = []
    for i, (X_, y_, p_, a_) in enumerate(zip(X, y, prediction, normalized_attention)):
        if i > N:
            break
        # build result
        result_entry = {}
        result_entry["prediction"] = (np.argmax(y_), np.argmax(p_))
        result_entry["original"] = np.asscalar(X_)
        result_entry["sentence"] = []
        for j, word in enumerate(vectorize_layer(X_).numpy().flatten().tolist()):
            word_obj = {}
            if word == 0:
                break
            word_obj[vectorize_layer.get_vocabulary()[word]] = a_[j].item()
            result_entry["sentence"].append(word_obj)

        results += [result_entry]
    return results


sentences_with_attention = get_attention(
    X_val, y_val, model.predict(X_val), attention_model.predict(X_val), 10
)
sentences_with_attention[0]


# %% [markdown]
# For a clear explanation, we display in green the words with attention > 0.75 (meaning the model gives them a lot of attention), in red the words with attention < 0.25 (meaning the model gives them little attention), and in grey the other words. \
# The `[UNK]` words are words that are unknown to the model (*i.e.* they are not part of the vocabulary). If we increased the vocabulary size (and therefore the model complexity), some of them would stop being unknowned by our model.

# %%
# convert prediction with attention to colored text
from termcolor import colored


def print_text(sentences_with_attention):
    threshold = 0.75
    classes = []
    print(colored("In green, the most important word\n\n", "green", attrs=["bold"]))

    for i, sentence in enumerate(sentences_with_attention):
        # Retrieve the class of this sentence
        # print(sentence)
        original_class, predicted_class = sentence["prediction"]
        # print(original_class, predicted_class)

        # Retrieve all the words and weights of this sentence
        words, weights = [], []
        # print("--", sentence['sentence'])
        for item in sentence["sentence"]:
            for word, weight in item.items():
                words.append(word)
                weights.append(float(weight))

        size = 0
        print(sentence["original"])
        for j, word in enumerate(words):
            if size != 0 and j != 0 and word != "," and word != ".":
                print(" ", end="")
            if weights[j] > threshold:
                print(colored(word, "green", attrs=["bold"]), end=" ")
            elif weights[j] < (1 - threshold):
                print(colored(word, "red", attrs=["bold"]), end=" ")
            else:
                print(colored(word, "grey"), end=" ")
            size += len(word) + 1
            if size > 80:
                print()
            size = 0

        print("\n")


print_text(sentences_with_attention)


# %% [markdown]
# ## Your work
# 
# <font color='red'>
# <br>
# TO DO Students
#     
# 1. Before modifying the code, take the time to understand it well. We use here the global attentions mechanism only from an encoder since the network for sentiment analysis has no decoder part, only a classifier 1
#     
# 1. Improve the f1 score for the **Attentional LSTM** model using BI-LSTM approach, better hyper-parameters and a better preprocessing (the same as in the previous step).
#     * Take inspiration from the course slides to build an original architecture that you will describe
#     * Use your Attention part in order to explain the words taken into account by the network to predict the sentiment.
#     
# 1. **Upload on moodle**
#     * **a clean, documented notebook** containing **your best LSTM attentional model**. The evaluation metric is the f1 score (macro avg).
#     * You can build all sorts of architectures but only using the cells seen in class (i.e. in particular: **CNNs are not yet seen so you should not use them**).
# 
#     * It is of course possible / desirable to use keras tuner to get the best hyper-parameters.
# 
#     * This notebook will be evaluated and the grade will take into account the editorial quality of your text.
# 
#     * Any notebook containing more than 1 model will not be evaluated (score = 0 -> **You have to choose the best one**).
# 
# </font>
# 

# %% [markdown]
# <font color='red'>
# 


