# %%
from itertools import chain

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn_crfsuite
from sklearn import metrics as mt
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn_crfsuite import metrics, scorers
from sklearn_crfsuite.utils import flatten

TRAIN_FILE_NAME = "04-train.txt"
TEST_FILE_NAME = "04-test.txt"


# %% [markdown]
# A simple sentence NER example:
# 
# [**ORG** U.N. ] official [**PER** Ekeus ] heads for [**LOC** Baghdad ]
# 
# We will concentrate on four types of named entities:
# 
# - persons (**PER**),
# - locations (**LOC**)
# - organizations (**ORG**)
# - Others (**O**)
# 

# %%
def _generate_examples(filepath):
    with open(filepath, encoding="utf-8") as f:
        sent = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if sent:
                    yield sent
                    sent = []
            else:
                splits = line.split(" ")
                token = splits[0]
                pos_tag = splits[1]
                ner_tag = splits[3].rstrip()
                if "MISC" in ner_tag:
                    ner_tag = "O"

                sent.append((token, pos_tag, ner_tag))


# %%
# %%time 
# hint use the above defined function
train_sents = list(_generate_examples(TRAIN_FILE_NAME))
test_sents = list(_generate_examples(TEST_FILE_NAME))

# %%
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "postag": postag,
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:postag": postag1,
            }
        )
    else:
        features["BOS"] = True
    return features


# %%
test_sents[2]


# %%
word2features(test_sents[2], 0)


# %%
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


# %%
# %%time
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# %%
# %%time 
#search for sklearn_crfsuite.CRF, 
# use the lbfgs algorithm, 
# c parameters should be 0.1 and max iterations 100, 
# all possible transactions true
try:
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True,)
    # fit the model
    crf.fit(X_train, y_train)
except AttributeError as e:
    print("Error", e)


# %%
# save a list of all labels in your model, hint crfs have a classes attribute
labels = list(crf.classes_)
labels


# %%
# remove the label 'O' from your list
try:
    labels.remove("O")
except ValueError:
    pass
labels


# %%
# perfrom a prediction on your test set
y_pred = crf.predict(X_test)

metrics.flat_f1_score(
    y_test,
    y_pred,
    average="weighted",
    labels=labels,
)


# %%
# group B and I results, use the sorted function on the list labels with a lambda function as the key
sorted_labels =sorted(labels,key=lambda l1: (l1[1:], l1[0]))


# %%
# Display classification report
print(
    mt.classification_report(
        y_true=flatten(y_test),
        y_pred=flatten(y_pred),
        labels=sorted_labels,
        digits=3,
    )
)


# %%
# what is the number of transition features in our model, crfs have an attribute called transition_features_
len(crf.transition_features_)

# %%
from collections import Counter


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s ->  %-7s %0.6f" % (label_from, label_to, weight))


print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

# top 20 unlikely transitions
print("\nTop unlikely transitions:")
(
    pd.DataFrame(crf.transition_features_, index=["value"])
    .transpose()
    .reset_index()
    .rename(
        columns={
            "level_0": "from",
            "level_1": "to",
        },
    )
    .sort_values(by="value")
    .reset_index(drop=True)
    .head(20)
)


# %%
# number of transition features in our model
len(crf.state_features_)


# %%
# create dataframe to easily sort linked values
df_trans = (
    pd.DataFrame(crf.state_features_, index=["value"])
    .transpose()
    .reset_index()
    .rename(
        columns={
            "level_0": "attr_name",
            "level_1": "label",
        },
    )
)
df_trans = df_trans[["value", "label", "attr_name"]]

# %%
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


# top 30 positive
print("Top positive:")
print(
    df_trans.sort_values(
        by="value",
        ascending=False,
        ignore_index=True,
    ).head(30)
)


# top 30 negative
print("\nTop negative:")
print(
    df_trans.sort_values(
        by="value",
        ignore_index=True,
    ).head(30)
)



