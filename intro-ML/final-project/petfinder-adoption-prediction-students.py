# %% [markdown]
# # Adopt a Pet
# 
# You are in charge of an animal shelter and you want to predict if the animals you have in your possession can be adopted within 30 days or not.
# 
# The dataset at your disposal contains different information about the animals in the shelter: data about the breed or color, data about a cost, data about its health. You even have a short description written by the former owner and a picture of the animal.
# 
# We provide you only with the train part and a small test subset so that you can test the whole process.

# %% [markdown]
# <div class="alert-block alert-danger">
# Deadline: Jannuary 15, 2022.
# <br>
# <br>
# You must submit a zip archive to LMS that contains 3 documents:
# 
# - A pdf report that outlines the various stages of your work. You will insist on the different hyperparameters of your treatment and for each of them, you will specify on which ranges of values you have tested them. This report will also contain the precision obtained on the train set and on the test set.
# 
# - the executable notebook containing only the chosen hyper-parameters and not their research. You will leave in this one the execution traces.
# 
# - A ".joblib" file so that we can execute your code. Of course, the test dataset will be modified and only the predict function of the pipeline will be executed.
# <br>
# <br> 
# The final grade will be based on the quality of the prediction (accuracy score) for 25% and the quality of the work for 75%.
# </div>

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Load-train-data" data-toc-modified-id="Load-train-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load train data</a></span><ul class="toc-item"><li><span><a href="#Load-the-images" data-toc-modified-id="Load-the-images-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Load the images</a></span></li><li><span><a href="#Compute-SIFT-detector-and-descriptors-for-one-image" data-toc-modified-id="Compute-SIFT-detector-and-descriptors-for-one-image-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Compute SIFT detector and descriptors for one image</a></span></li><li><span><a href="#Extract-features-and-build-BOFs" data-toc-modified-id="Extract-features-and-build-BOFs-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Extract features and build BOFs</a></span></li></ul></li><li><span><a href="#Build-a-basic-model" data-toc-modified-id="Build-a-basic-model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Build a basic model</a></span></li><li><span><a href="#Evaluation-of-the-model" data-toc-modified-id="Evaluation-of-the-model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Evaluation of the model</a></span></li></ul></div>

# %%
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# %%
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

# %% [markdown]
# ## Load train data

# %%
path = "https://www.i3s.unice.fr/~riveill/dataset/petfinder-adoption-prediction/"

# %%
breeds = pd.read_csv(path+'breed_labels.csv')
colors = pd.read_csv(path+'color_labels.csv')
states = pd.read_csv(path+'state_labels.csv')

train = pd.read_csv(path+'train.csv')

train['dataset_type'] = 'train'

# %%
len(train)

# %%
# In this example notebook, we will only work with a small part of the dataset
N = 10
train = train[:N]

# %%
if 'dataset_type' in train.columns:
    train = train.drop(labels='dataset_type', axis=1)
train.columns

# %%
y_train = train['target']
X_train = train.drop(['target'], axis=1)
X_train.head()

# %%
y_train.head()

# %%
cat_cols = ['Type', 'Gender', 'Breed', 'Color1', 'Color2', 'Color3', 
       'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health']
num_cols = ['Age', 'Fee']
txt_cols = ['Description']
img_cols = ['Images']

# %% [markdown]
# ### Load the images

# %%
# Build the image list of the training set 
img_dir = "train_images/"
X_train['Images'] = [path+img_dir+img for img in train['Images']]

# %%
from skimage import io

# Read the first image of the list
img = io.imread(X_train['Images'][0])
# have a look to the image
plt.imshow(img)

# %% [markdown]
# ### Compute SIFT detector and descriptors for one image

# %%
# convert the image to grey levels 
import cv2

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# %%
# compute SIFT detector and descriptors
sift = cv2.SIFT_create()
kp,des = sift.detectAndCompute(gray,None)

# %%
# plot image and descriptors
cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img)

# %% [markdown]
# ### Extract features and build BOFs

# %%
# First step, extract the SIFTs of each image
# Be carefull: very long process

def extract_SIFT(img_lst):
    nbSIFTs = 0    # Nomber of SIFTs
    SIFTs = []  # List of SIFTs descriptors 
    #dimImgs = []   # Nb of descriptors associated to each images

    for pathImg in tqdm(img_lst, position=0, leave=True): 
        img = io.imread(pathImg)
        if len(img.shape)==2: # this is a grey level image
            gray = img
        else: # we expect the image to be a RGB image or RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        if len(kp) == 0 and img.shape[2]==4: #some images are mask on alpha channel: we thus extract this channel if not kpts have been detected
            gray = img[:,:,3]
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
        
        nbSIFTs += des.shape[0]
        SIFTs.append(des)
        #dimImgs.append(des.shape[0])
    return nbSIFTs, SIFTs#, dimImgs

# %%
nbSIFTs, SIFTs = extract_SIFT(X_train['Images'])
print('nbSifts: ', nbSIFTs)

# %%
# Step 2: clusterize the SIFT
from sklearn.cluster import MiniBatchKMeans

def clusterize(SIFTs, nb_img_features=5, verbose=False):
    clusterizer = MiniBatchKMeans(n_clusters=nb_img_features)   # nb_img_features is a hyperparameter
    # learning of the clustering
    flat_list = SIFTs[0]
    for des in SIFTs[1:]:
        flat_list = np.concatenate((flat_list, des))
        if verbose:
            print("shape:", des.shape, flat_list.shape)
    clusterizer.fit(flat_list)
    # we now know the label of each SIFT descriptor
    return clusterizer

# %%
clusterizer = clusterize(SIFTs, verbose=True)

# %%
# Step 3: build the BOW representation of each images (i.e. construction of the BOFs)

def build_BOFs(SIFTs, clusterizer, verbose=False):
    ok, nok = 0, 0
    #BOF initialization
    nb_img_features = clusterizer.get_params()['n_clusters']
    BOFs = np.empty(shape=(0, nb_img_features), dtype=int)

    # Build label list
    flat_list = SIFTs[0]
    for des in SIFTs[1:]:
        flat_list = np.concatenate((flat_list, des))
        if verbose:
            print("shape:", des.shape, flat_list.shape)
    labels = clusterizer.predict(flat_list)

    # loop on images
    i = 0 # index for the loop on SIFTs
    for des in SIFTs:
        #initialisation of the bof for the current image
        tmpBof = np.array([0]*nb_img_features)
        j = 0
        # for every SIFT of the current image:
        nbs = des.shape[0]
        while j < nbs:
            tmpBof[labels[i]] += 1
            j+=1
            i+=1
        BOFs = np.concatenate((BOFs, tmpBof.reshape(1,-1)), axis=0)
    if verbose:
        print("BOFs : ", BOFs)
    
    return BOFs

# %%
BOFs = build_BOFs(SIFTs, clusterizer, verbose=True)
BOFs.shape

# %%
from sklearn.base import BaseEstimator,TransformerMixin

def list_comparaison(l1, l2):
    if not l1 is None \
        and not l2 is None \
        and len(l1)==len(l2) \
        and len(l1)==sum([1 for i,j in zip(l1, l2) if i==j]):
        return True
    return False
    
class BOF_extractor(BaseEstimator,TransformerMixin): 
    X = None
    SIFTs = None
    nbSIFTs = 0
    
    def __init__(self, nb_img_features=10, verbose=False):
        self.nb_img_features = nb_img_features
        self.verbose = verbose
        self.path = path
        if self.verbose:
            print("BOF.init()")
        
    def fit(self, X, y=None):
        if self.verbose:
            print("BOF.fit()")
        if list_comparaison(X, self.X):
            SIFTs = self.SIFTs 
            nbSIFTs = self.nbSIFTs
        else:
            if self.verbose:
                print("extract_SIFT")
            nbSIFTs, SIFTs = extract_SIFT(X)
        self.X = X
        self.SIFTs = SIFTs 
        self.nbSIFTs = nbSIFTs
        self.clusterizer = clusterize(SIFTs, self.nb_img_features, self.verbose)
        
    def transform(self, X, y=None):
        if self.verbose:
            print("BOF.transform()")
        if list_comparaison(X, self.X):
            SIFTs = self.SIFTs 
            nbSIFTs = self.nbSIFTs
        else:
            if self.verbose:
                print("extract_SIFT")
            nbSIFTs, SIFTs = extract_SIFT(X)

        if self.verbose:
            print("nbSIFTs:", nbSIFTs)
        return build_BOFs(SIFTs, self.clusterizer, self.verbose)
    
    def fit_transform(self, X, y=None):
        if self.verbose:
            print("BOF.fit_transform()")
        if list_comparaison(X, self.X):
            SIFTs = self.SIFTs 
            nbSIFTs = self.nbSIFTs
        else:
            if self.verbose:
                print("extract_SIFT")
            nbSIFTs, SIFTs = extract_SIFT(X)
        self.X = X
        self.SIFTs = SIFTs 
        self.nbSIFTs = nbSIFTs
        self.clusterizer = clusterize(SIFTs, self.nb_img_features, self.verbose)
        return build_BOFs(SIFTs, self.clusterizer, self.verbose)

# %%
test_BOF_extractor = BOF_extractor(nb_img_features=5, verbose=True)

# %%
test_BOF_extractor.fit(X_train['Images'])

# %%
BOFs = test_BOF_extractor.transform(X_train['Images'])
BOFs.shape

# %%
BOFs = test_BOF_extractor.fit_transform(X_train['Images'])
BOFs.shape

# %%
test = pd.read_csv(path+"test.csv")
y_test = test['target']
X_test = test.drop(['target'], axis=1)

img_dir = "test_images/"
X_test['Images'] = [path+img_dir+img for img in test['Images']]
len(X_test)

# %%
BOFs = test_BOF_extractor.transform(X_test['Images'])
BOFs.shape

# %% [markdown]
# ## Build a basic model
# 
# There are much more interesting things in the dataset and I'm going to explore them, but for now let's build a simple model as a baseline.

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
text_preprocessor = CountVectorizer()
image_preprocessor = BOF_extractor(nb_img_features=3, verbose=False)

preprocessor = ColumnTransformer([
    ("categorical encoding", categorical_preprocessor, cat_cols),
    ("numerical encoding", numerical_preprocessor, num_cols),
    ("text encoding", text_preprocessor, 'Description'),
    ("image encoding", image_preprocessor, 'Images'),
])

classifier = LogisticRegression()

model = make_pipeline(preprocessor, classifier)

# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
accuracy_score(y_train, y_pred)

# %%
# Save the model
from joblib import dump, load

dump(model, 'limonier.joblib') # Put your name as a model name

# %% [markdown]
# ## Evaluation of the model
# 
# <div class="alert alert-block alert-danger">
# We will only execute the following cells.
# </div>

# %%
test = pd.read_csv(path+"test.csv")

y_test = test['target']
X_test = test.drop(['target'], axis=1)

img_dir = "test_images/"
X_test['Images'] = [path+img_dir+img for img in test['Images']]
print("Test size:", len(X_test))

model = load('michel.joblib') 
y_pred = model.predict(X_train)
print("ACC on train", accuracy_score(y_train, y_pred))
y_pred = model.predict(X_test)
print("ACC on test", accuracy_score(y_test, y_pred))

# %%


# %% [markdown]
# 

# %%



