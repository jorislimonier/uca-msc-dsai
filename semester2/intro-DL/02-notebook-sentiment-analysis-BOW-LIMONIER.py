# %% [markdown]
# # Sentiment analysis with BOW representation
# 
# Text classification is a machine learning technique that assigns a set of predefined categories to open-ended text. Text classifiers can be used to organize, structure, and categorize pretty much any kind of text – from documents, medical studies and files, and all over the web.
# 
# For example, new articles can be organized by topics; support tickets can be organized by urgency; chat conversations can be organized by language; brand mentions can be organized by sentiment; and so on.
# 
# Text classification is one of the fundamental tasks in natural language processing with broad applications such as **sentiment analysis**, topic labeling, spam detection, and intent detection.
# 
# **Why is Text Classification Important?**
# 
# It’s estimated that around 80% of all information is unstructured, with text being one of the most common types of unstructured data. Because of the messy nature of text, analyzing, understanding, organizing, and sorting through text data is hard and time-consuming, so most companies fail to use it to its full potential.
# 
# This is where text classification with machine learning comes in. Using text classifiers, companies can automatically structure all manner of relevant text, from emails, legal documents, social media, chatbots, surveys, and more in a fast and cost-effective way. This allows companies to save time analyzing text data, automate business processes, and make data-driven business decisions.
# 
# **How Does Text Classification Work?**
# 
# Instead of relying on manually crafted rules, machine learning text classification learns to make classifications based on past observations. By using pre-labeled examples as training data, machine learning algorithms can learn the different associations between pieces of text, and that a particular output (i.e., tags) is expected for a particular input (i.e., text). A “tag” is the pre-determined classification or category that any given text could fall into.
# 
# The first step towards training a machine learning NLP classifier is feature extraction: a method is used to transform each text into a numerical representation in the form of a vector. One of the most frequently used approaches is bag of words, where a vector represents the frequency of a word in a predefined dictionary of words.
# 
# Then, the machine learning algorithm is fed with training data that consists of pairs of feature sets (vectors for each text example) and tags (e.g. sports, politics) to produce a classification model:
# 
# ![training](https://monkeylearn.com/static/507a7b5d0557f416857a038f553865d1/2ed04/text_process_training.webp)
# 
# Once it’s trained with enough training samples, the machine learning model can begin to make accurate predictions. The same feature extractor is used to transform unseen text to feature sets, which can be fed into the classification model to get predictions on tags (e.g., sports, politics):
# 
# ![prediction](https://monkeylearn.com/static/afa7e0536886ee7152dfa4c628fe59f0/2b924/text_process_prediction.webp)
# 
# Text classification with machine learning is usually much more accurate than human-crafted rule systems, especially on complex NLP classification tasks. Also, classifiers with machine learning are easier to maintain and you can always tag new examples to learn new tasks.

# %% [markdown]
# ## Today lab
# 
# In this lab we use part of the 'Amazon_Unlocked_Mobile.csv' dataset published by Kaggle. The dataset contain the following information:
# * Product Name
# * Brand Name
# * Price
# * Rating
# * Reviews
# * Review Votes
# 
# We are mainly interested by the 'Reviews' (X) and by the 'Rating' (y)
# 
# The goal is to try to predict the 'Rating' after reading the 'Reviews'. I've prepared for you TRAIN and TEST set.

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Today-lab" data-toc-modified-id="Today-lab-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Today lab</a></span></li><li><span><a href="#Load-dataset" data-toc-modified-id="Load-dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Load dataset</a></span><ul class="toc-item"><li><span><a href="#About-Train,-validation-and-test-sets" data-toc-modified-id="About-Train,-validation-and-test-sets-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span><a href="https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7" rel="nofollow" target="_blank">About Train, validation and test sets</a></a></span></li><li><span><a href="#Undestand-the-dataset" data-toc-modified-id="Undestand-the-dataset-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Undestand the dataset</a></span></li></ul></li><li><span><a href="#Build-X-(features-vectors)-and-y-(labels)" data-toc-modified-id="Build-X-(features-vectors)-and-y-(labels)-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Build X (features vectors) and y (labels)</a></span></li><li><span><a href="#Our-previous-baseline" data-toc-modified-id="Our-previous-baseline-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Our previous baseline</a></span></li><li><span><a href="#Build-an-MLP-Classifier" data-toc-modified-id="Build-an-MLP-Classifier-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Build an MLP Classifier</a></span></li></ul></div>

# %% [markdown]
# ## Load dataset

# %%
import pandas as pd
import numpy as np
import nltk
nltk.download('popular')

# %%
TRAIN = pd.read_csv("http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/train.csv.gz")
TEST = pd.read_csv("http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/test.csv.gz")

TRAIN.head()

# %% [markdown]
# ## Build X (features vectors) and y (labels)

# %%
# Construct X_train and y_train
X_train = TRAIN['Reviews'].fillna("")
y_train = TRAIN['Rating']
X_train.shape, y_train.shape

# %%
# Construct X_test and y_test
X_test = TEST['Reviews'].fillna("")
y_test = TEST['Rating']
X_test.shape, y_test.shape

# %% [markdown]
# ## Features extraction
# 
# A bag-of-words model is a way of extracting features from text so the text input can be used with machine learning algorithms or neural networks.
# 
# Each document, in this case a review, is converted into a vector representation. The number of items in the vector representing a document corresponds to the number of words in the vocabulary. The larger the vocabulary, the longer the vector representation, hence the preference for smaller vocabularies in the previous section.
# 
# Words in a document are scored and the scores are placed in the corresponding location in the representation.
# 
# In order to extract feature, you can use `CountVectorizer` or `TfidfVectorizer` and you can perform the desired text cleaning.

# %% [markdown]
# $$[TODO - Students]$$ 
# > * Quickly remind what are `CountVectorizer`, `TfidfVectorizer` and how they work.\
# `CountVectorizer` counts how many times a word occurs in the analysed text.\
# `TfidfVectorizer` also counts how many times a word occurs in the analysed text, but then it compares it with its usual frequency within the whole corpus. This allows to get an idea of how a word is over-represented in the analyzed text and it disregards words that simply occur a lot in the whole corpus.
# 
# > * Build the BOW representation for train and test set

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# %%

preproc_pipe = make_pipeline(
    TfidfVectorizer(),
)
# Extract features
preproc_X_train = preproc_pipe.fit_transform(X_train)
preproc_X_test = preproc_pipe.transform(X_test)

# %% [markdown]
# ## Build a baseline with logistic regression.
# 
# Using the previous BOW representation, fit a logistic regression model and evaluate it.

# %% [markdown]
# $$[TODO - Students]$$ 
# > * Quickly remind what are `LogisticRegression` and how they work.\
# `LogisticRegression` computes a prediction probability for each of the classes

# %%
y_train.value_counts().sort_index()

# %% [markdown]
# > * what are the possible metrics. Choose one and justify your choice.\
# We could use accuracy, recall, precision or f1-score. I choose to use the f1-score because our data set is unbalanced (class 1 ->  885, class 2 ->  291, class 3 ->  385, class 4 ->  747, class 5 -> 2692, )
# 

# %%
# Build your model
clf = LogisticRegression()

param_search = {
    # "penalty": ["l1", "l2"],
    "C": np.linspace(0.01,4),
    "max_iter": [1000],
}

search = RandomizedSearchCV(clf, param_search, n_jobs=-1, verbose=1, scoring="f1_weighted")
search.fit(preproc_X_train, y_train)

# %%
log_reg_params = search.best_params_
log_reg_params

# %%
# Evaluate your model
y_pred_log_reg = search.predict(preproc_X_test)
log_reg_score = f1_score(y_test, y_pred_log_reg, average="weighted")
print(f"Logistic Regression best score: {round(log_reg_score, 4)}")

# %% [markdown]
# ## Build an MLP Classifier
# 

# %%
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# %% [markdown]
# $$[TODO - Students]$$ 
# > * Quickly remind what are `Multi Layer Perceptron` and how they work.\
# `Multi Layer Perceptron` is a class of feed-forward neural network. They contain an input layer, an output layer and one or more hidden layer, each with an activation function.
# > * If necessary, One hot encode the output vectors

# %%
# Encode output vector if necessary.
from sklearn.preprocessing import OneHotEncoder
ohe_y = OneHotEncoder()
y_train_ohe = ohe_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_ohe = ohe_y.transform(y_test.values.reshape(-1, 1))

# %% [markdown]
# $$[TODO - Students]$$ 
# > * What is the size of the input vector and the output vector?

# %%
# Define constant
input_dim = preproc_X_train.shape[1] # number of features of X_train
output_dim = y_train_ohe.shape[1] # number of classes (that we spread into 5 dimensions by OneHotEncod-ing)

# %% [markdown]
# $$[TODO - Students]$$ 
# 
# > * Build a simple network to predict the star rating of a review using the functional API. It should have the following characteristic : one hidden layer with 256 nodes and relu activation.
# > * What is the activation function of the output layer?\
# Softmax since we are tackling a classification problem.

# %%
# Build your MLP
inputs = Input(shape=(input_dim,))
x = Dense(256, activation="relu")(inputs)
outputs = Dense(output_dim, activation="softmax")(x)

# %% [markdown]
# $$[TODO - Students]$$ 
# 
# We are now compiling and training the model.
# > * Using the tensorflow documentation, explain the purpose the EarlyStopping callback and detail its arguments.\
# As per the Tensorflow documentation:\
# *Assuming the goal of a training is to minimize the loss. With this, the metric to be monitored would be 'loss', and mode would be 'min'. A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing, considering the min_delta and patience if applicable. Once it's found no longer decreasing, model.stop_training is marked True and the training terminates.*\
# In other words, `EarlyStopping` stops the training process (*i.e.* the `fit` method) if no more progress is made in terms of the loss.
# 
# > * Compile the model
# > * Fit the model

# %%
# Compile the model and start training
# Stop training with early stopping with patience of 20
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=tfa.metrics.F1Score(5))

callback = EarlyStopping(monitor="loss", patience=20)
history = model.fit(x=preproc_X_train.toarray(), y=y_train_ohe.toarray(), epochs=100, callbacks=[callback], use_multiprocessing=True)

# %%
model.summary()
plot_model(model, show_shapes=True)

# %% [markdown]
# $$[TODO - Students]$$ 
# 
# > * Babysit your model: plot learning curves

# %%
# Plot the learning curves and analyze them
# It's possible to plot them very easily using: pd.DataFrame(history.history).plot(figsize=(8,5))
pd.DataFrame(history.history).plot(figsize=(8,5))

# %% [markdown]
# $$[TODO - Students]$$ 
# 
# > * How do you interpret those learning curves ?\
# There is clearly something wrong since we see close to no improvement over the epochs. This could be due to overfitting.
# 
# The model appears to overfit the training data. Various strategies could reduce the overfitting but for this lab we will just change the number and size of layers. We will do that a little later.
# 
# > * Evaluate the model (on test part) and plot confusion matrix.
# > * Are you doing better or worse than with our first attempt with Logistic regression.

# %%
# Evaluate the model
pred_proba = model.predict(preproc_X_test.toarray())
y_pred = np.argmax(pred_proba, axis=1)+1

# %%
mlp_initial_score = f1_score(y_test, y_pred, average="weighted")
print(f"Initial MLP best score: {round(mlp_initial_score, 4)}")
print(f"Logistic Regression best score: {round(log_reg_score, 4)}")
print("MLP performs worse than linear regression")

# %%
# Print/plot the confusion matrix
print("--> Using tensorflow:\n", tf.math.confusion_matrix(y_test, y_pred))
print("\n--> Using sklearn:")
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=list(range(1,6))).plot()
cm

# %% [markdown]
# ## Hyper-parameters search
# 
# Using [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) and modifying various hyper-parameters, improve your model. Change in particular the number of layers, the number of neurons per layer, the dropout, the regularization.

# %%
import keras_tuner as kt


def build_model(hp):
    # Wrap a model in a function
    NUM_LAYERS = hp.Int("num_layers", 1, 3)
    # Define hyper-parameters
    NUM_DIMS = hp.Int("num_dims", min_value=32, max_value=128, step=32)
    ACTIVATION = hp.Choice("activation", ["relu", "tanh"])
    DROPOUT = hp.Boolean("dropout")
    DROP_RATE = hp.Choice("drop_rate", values=[0.2, 0.25, 0.5])
    # replace static value
    text_input = Input(shape=(input_dim,), name='input')
    h = text_input
    # with hyper-parameters
    for i in range(NUM_LAYERS):
        h = Dense(NUM_DIMS//(2*i+1), activation=ACTIVATION)(h)
        if DROPOUT:
            h = Dropout(rate=DROP_RATE)(h)
    ouputs = Dense(output_dim, activation='softmax', name='output')(h)
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=tfa.metrics.F1Score(5),
    )
    return model


tuner = kt.BayesianOptimization(build_model,
                                objective="loss",
                                max_trials=10,
                                overwrite=True,
                                # directory='my_dir',
                                project_name='BOW_MLP')
ea = EarlyStopping(monitor='loss', mode='max',
                   patience=2, restore_best_weights=True)

tuner.search(preproc_X_train.toarray(), y_train_ohe.toarray(), epochs=10,
             validation_split=0.1,
             callbacks=[ea])


# %%
best_hp = tuner.get_best_hyperparameters()
model = tuner.hypermodel.build(best_hp[0])
H = model.fit(preproc_X_train.toarray(), y_train_ohe.toarray(),
              validation_split=0.1, callbacks=[ea])

# %%
# evaluate the network
pred_proba = model.predict(preproc_X_test.toarray())
y_pred_tuned_MLP = pred_proba.argmax(axis=1)+1
y_test

# %%
mlp_tuned_score = f1_score(y_test, y_pred_tuned_MLP, average="weighted")

print(f"Logistic Regression best score: {round(log_reg_score, 4)}")
print(f"Initial MLP best score: {round(mlp_initial_score, 4)}")
print(f"Tuned MLP best score: {round(mlp_tuned_score, 4)}")


