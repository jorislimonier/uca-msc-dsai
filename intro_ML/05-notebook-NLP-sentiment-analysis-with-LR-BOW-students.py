# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Case Study: Sentiment Analysis
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

# %%
"""
(Practical tip) Table of contents can be compiled directly in jupyter notebooks using the following code:
I set an exception: if the package is in your installation you can import it otherwise you download it 
then import it.
"""
try:
    from jyquickhelper import add_notebook_menu 
except:
    get_ipython().system('pip install jyquickhelper')
    from jyquickhelper import add_notebook_menu


# %%
"""
Output Table of contents to navigate easily in the notebook. 
For interested readers, the package also includes Ipython magic commands to go back to this cell
wherever you are in the notebook to look for cells faster
"""
add_notebook_menu()

# %% [markdown]
# ## 0. Today lab
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
# ## 1. Load and study the dataset
# %% [markdown]
# ### 1.1. Imports

# %%
import pandas as pd
import numpy as np
#import nltk
import pylab as pl # package inheriting most of matplotlib package functions with shorter syntax 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ### 1.2. [About Train, validation and test sets](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
# ![test/train/val](https://miro.medium.com/max/1466/1*aNPC1ifHN2WydKHyEZYENg.png)
# 
# * **Training Dataset:** The sample of data used to fit the model.
# * **Validation Dataset:** The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
# * **Test Dataset:** The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.

# %%
TRAIN = pd.read_csv("http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/train.csv.gz")
VAL = pd.read_csv("http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/val.csv.gz")
TEST = pd.read_csv("http://www.i3s.unice.fr/~riveill/dataset/Amazon_Unlocked_Mobile/test.csv.gz")

TRAIN.head(10)#pass an integer value to specify the number of header lines to output. 

# %% [markdown]
# ### 1.3. Understand the train dataset

# %%
# Tokenized the reviews
from nltk.tokenize import word_tokenize
#nltk.download('punkt') # If nltk requires to download 'punkt' depending on your installation
reviews_tokenized = [word_tokenize(review) for review in TRAIN['Reviews']]
for sentence in reviews_tokenized[:10]: print('%s \n'%sentence)


# %%
# Count the vocabulary
flatten_reviews = [item for sublist in reviews_tokenized for item in sublist] # contain all words 
unique_words = list(set(flatten_reviews)) #processed as a list for future analysis
vocabulary_size = len(unique_words) # set allows to get unique words contain in flatten_reviews
vocabulary_size


# %%
# Plot the distribution of words
from nltk import FreqDist

freqDist = FreqDist(flatten_reviews)
print(freqDist.most_common(20))
full_counts = [x[1] for x in freqDist.most_common()]
freqDist.plot(25, cumulative=False)
pl.show()
pl.figure(figsize=(8,5))
#plot
sns.kdeplot(full_counts)
#labeling features for readibility you have to think about while presenting your results
pl.xlabel('word occurrence'); pl.ylabel('probability density'); pl.title('word occurences distribution')
pl.show()

# most of the time, the most common words are linking words, basic verbs, punctuation
# For instance, One can wonder either rare words are more significant for discriminative purposes.


# %%
# Plot the distibution of word length size
len_words = [len(w) for w in flatten_reviews]
freqDist2 = FreqDist(len_words)
pl.figure(figsize=(8,6))
pl.title('word length')
freqDist2.plot(cumulative=False)


# %%
# Plot the distribution of sentence lengths
len_sentences = [len(s) for s in reviews_tokenized]
pl.figure(figsize=(8,6))
sns.kdeplot(len_sentences)
pl.xlabel('sentence length'); pl.ylabel('density');pl.title('sentence lengths distribution')


# %%
# Build a function named get_ngrams that return the list of n grams of a text.
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def get_ngrams(text, n ):
    if len(word_tokenize(text)) < n :
        return []
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

# Use this function to output/print the ngrams of the following text
text = "I love the chocolate but I hate tea"

get_ngrams('This is the simplest text i could think of', 3 )


# %%
# For each reviews calculate its 3grams
ngrams_reviews = [get_ngrams(' '.join(r), 3) for r in reviews_tokenized]
ngrams_reviews[:2]

# %% [markdown]
# ###  1.4. What about your validation split ?
# 
# Questions left to the students:
# 
#     - Is the vocabulary observed in the validation dataset included in the train dataset?
#     - How similar are distributions of word occurences or word lengths?
#     - Same for sentence-wise distributions? 
# 
# In practice you do not have prior knowledge on the classification labels of the test dataset, and you want to infer these i.e. be able to generalize this categorization based on known labels of the train dataset. So you use the validation dataset to measure the generalization ability of your model learned on the train dataset.
# 
# However, you have access to observations from all datasets split (sentences here) in stationary cases (to contrast to online settings where new data comes on the fly).

# %%
VAL.head()


# %%
"""
Tokenized the reviews of the validation dataset
Code previously used for processing the train dataset fails on the validation dataset
This is a common situation while dealing with real life datasets built by non machine-learning practitioners.
So we also have to know how to preprocess row data.
Common problems contain in real life dataset:
    contain NaN values; values type are not consistent among a given dataset feature etc.. 
"""
run_failure = False#True
if run_failure: #This code will give you an error
    reviews_tokenized_val = [word_tokenize(review) for review in VAL['Reviews']]
    for sentence in reviews_tokenized_val[:10]: print('%s \n'%sentence)
else: # Here for practical reasons we just show how to deal with this issue using exceptions
    reviews_tokenized_val= []
    failed_review=[]
    for review in VAL['Reviews']:
        try: # Store the result of the function word_tokenize applied on current review if it does not fail 
            reviews_tokenized_val.append(word_tokenize(review))
        except:
            failed_review.append(review)
            continue
    print('How many fails over the whole Validation reviews ? ', len(VAL['Reviews']) - len(reviews_tokenized_val))
    print('Look at failures to understand why it happened ? \nlist of bugged reviews: %s'%failed_review)


# %%
# validation vocabulary
flatten_reviews_val = [item for sublist in reviews_tokenized_val for item in sublist]
unique_words_val = list(set(flatten_reviews_val))
vocabulary_size_val = len(unique_words_val)
print('TRAIN dataset - vocabulary size: %s \nVAL dataset - vocabulary size: %s '%(vocabulary_size,vocabulary_size_val))


# %%
# Are words in the VALIDATION reviews contained in the TRAINING ones? 
included_words = []
excluded_words = []
for word in unique_words_val:
    if word in unique_words:
        included_words.append(word)
    else:
        excluded_words.append(word)
print('Number of INCLUDED words: %s \nNumber of EXCLUDED words:%s'%(len(included_words),len(excluded_words)))


# %%
# [TO DO - Students] Plot most frequent words present in the validation dataset


# %%
# Plot the distribution of word occurences
## Start with plotting the word occurences with distributions from VAL and TRAIN next to each other
pl.figure(figsize=(12,5))
pl.subplot(121)
sns.kdeplot(full_counts_val)
pl.xlabel('word occurrence'); pl.ylabel('probability density'); pl.title('[VAL] word occurences distribution')
pl.subplot(122)
sns.kdeplot(full_counts)
pl.xlabel('word occurrence'); pl.ylabel('probability density'); pl.title('[TRAIN] word occurences distribution')
pl.show()


# %%
# [TO DO - Students] Then do the same thing for word lengths


# %%
#  [TO DO - Students] Plot the distribution of sentence lengths


# %%
#  [TO DO - Students] Plot the distribution of word lengths

# %% [markdown]
# ## 2. Sentence Classifications
# %% [markdown]
# ### 2.1. Build X (features vectors) and y (labels)

# %%
# Construct X_train and y_train
X_train = TRAIN['Reviews'].fillna("")
y_train = TRAIN['Rating']
X_train.shape, y_train.shape


# %%
# Construct X_val and y_val
X_val = VAL['Reviews'].fillna("")
y_val = VAL['Rating']
X_val.shape, y_val.shape


# %%
# Construct X_test and y_test
X_test = TEST['Reviews'].fillna("")
y_test = TEST['Rating']
X_test.shape, y_test.shape

# %% [markdown]
# ### 2.2. Build a Baseline
# Using a binary `CountVectorizer` and a `LogisticRegression` classifier, learned in a previous lecture, build a first model.
# 
# For this model, you will not pre-process the text and will only use words (not N-grams). Leaves all parameter as default.
# 
# The evaluation metric is accuracy.
# %% [markdown]
# $$[TO DO - Students]$$
# 
# Explain in your own words both classification models (CountVectorizer & Logistic Regression), feel free to specify key features/hyperparameters of these models that seem the most important ones to tune and explain why. You can include it at the end of the notebook for your conclusion. 

# %%
# Encode X_train
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(X_train)
X_train_encoded = cv.transform(X_train)
X_train_encoded.shape


# %%
# What is the vocabulary size ?
# Compare with your previous response
len(cv.vocabulary_)


# %%
# What is the stop words used
cv.stop_words_


# %%
# Transform and then inverse transform the following text
# Interpret the result
text = "I love the chocolate but I hate tea"

cv.inverse_transform(cv.transform([text]))


# %%
# Using LogisticRegression from sklearn, fit a first model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() # Keep default settings we will look into hyperparameters fine-tuning later on
lr.fit(X_train_encoded, y_train)


# %%
# Encode X_test
X_test_encoded = cv.transform(X_test)
X_test_encoded.shape


# %%
# Using classification_report, evaluate the model
from sklearn.metrics import classification_report

y_pred = lr.predict(X_test_encoded)
print(classification_report(y_pred, y_test))

# %% [markdown]
# $$[ TO DO - Students]$$
# 
# What do these outputted classsification metrics mean? Interpret these results.  You can include it at the end of the notebook for your conclusion. 
# %% [markdown]
# ### 2.3. A better classifier with a preprocessing
# 
# It's up to you. Try to get a better score (accuracy) using what we have seen in this course:
# - efficient text pre-processing
# - choice of feature extraction
# - use of a more powerful classifier or better hyper-parameter for LogisticRegression.
# 
# The training of the model must be done on the Train and the evaluation on the Test. You can of course use GridSearchCV or RandomizedSearchCV.
# %% [markdown]
# # Some pre-processing

# %%
# Write a "clean_text" function that accepts a text as input and returns a clean text.
# The possible steps (take what you want) are:
# - removal of emoji
# - lowercase the text
# - remove punctuation
# - remove words containing numbers
# - remove stop words
# - stemming or lemmatization
# - remove words smaller than nfrom nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


stemmer = PorterStemmer().stem

def clean_text(text):
    # [TO DO - Students] Create your own preprocessing function
    return text


# %%
# Clean, X_train and X_test with the previous preprocessing
X_train_cleaned = [clean_text(r) for r in X_train]
X_test_cleaned = [clean_text(r) for r in X_test]


# %%
# Build, Fit and Evaluate a model using the previous preprocessing
# Did you improve the result ? 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

cv = CountVectorizer(analyzer='word', binary=True)
cv.fit(X_train_cleaned)
X_train_encoded = cv.transform(X_train_cleaned)

X_test_encoded = cv.transform(X_test_cleaned)

lr = LogisticRegression()
lr.fit(X_train_encoded, y_train)
y_pred = lr.predict(X_test_encoded)

print('accuracy score: ',accuracy_score(y_pred, y_test))

# %% [markdown]
# ### 2.4. Search hyper-parameters
# 
# ![GridSearch](https://i.stack.imgur.com/81Yoo.png)

# %%
# Using Grid Search or Random Search try to find some better hyperparameters
from nltk import word_tokenize          
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# I use my own tokenizer
class MyTokenizer:
    def __call__(self, doc):
        return [clean_text(t) for t in word_tokenize(doc)]

# I define the pipeline
pipeline = Pipeline([
        ('feature_extraction',  TfidfVectorizer(tokenizer=MyTokenizer())),
        ('classification',  LogisticRegression(multi_class='auto', max_iter=400))
        ])

# I define the parameter space
# [TO DO - Students] Etablissez les paramètres à parcourir de votre choix pour la GridSearchCV 
parameters = {}
parameters['feature_extraction__...'] = [...]
parameters['feature_extraction__...'] = [...]
parameters['feature_extraction__...'] = [...]

parameters['classification__...'] = [...]
parameters['classification__...'] = [...]


# I use GridSearchCV to search best hyper parameter
# I use RandomizedSearchCV to search good hyper parameter


grid = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=3, verbose=2,n_jobs=None)
grid.fit(X_train, y_train);

# %% [markdown]
# ## 3. Summarize your conclusion here

# %%
# What is your best params
grid.best_params_


# %%
# What is your best score
grid.best_score_


# %%
# What is your best estimator
grid.best_estimator_

# %% [markdown]
# A random draw with balanced classes (as much data from each class) would give an accuracy of 20% (1/5). We notice that a very light pre-processing or even no pre-processing already gives much better results (around 65-66%) and that it is not easy to do much better.

# %%
# Print/plot the confusion matrix
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(grid, X_test, y_test, normalize=='false');

# %% [markdown]
# The confusion matrix helps us understand the quality of the results. On the diagonal we find the quality of the predictions for a given class:
# * Class 1 is found around 70 %.
# * Class 2, 3 have a low score < 10%.
# * Class 4 is found around 16-20 %
# * Class 5 is found more than 90 %
# 
# On one line is the way the original class was found. For example for class 2 (depend the run) :
# * at 59%, the predictor says that a data of this class is of class 1
# * at 4% of the right class (the 2)
# * at 2% of class 3
# * at 4% of class 4
# * at 31% of class 5
# The sum is 100%.

