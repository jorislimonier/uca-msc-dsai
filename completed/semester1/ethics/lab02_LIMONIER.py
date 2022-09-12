# %% [markdown]
# <h1 style="font-size:3rem;color:#A3623B;">Lab 2</h1>
# 
# ## Security and Ethical aspects of data
# 
# ### Amaya Nogales Gómez
# 
# %% [markdown]
# ## 2.1 Support Vector Machines
# 

# %%
# we import all the required libraries
import numpy as np
import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal  # for generating synthetic data
from sklearn import datasets  # For real datasets
SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


# %%
# We load the real dataset from Lab 1: iris
iris = datasets.load_iris()
print(iris.data.shape)
print(iris.feature_names)  # variables, features
print(iris.target_names)  # classes
# print(iris)


# %%
X = iris["data"][:, (2, 3)]  # petal length, petal width
# print(X)

y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
print(y.size)


# %%
# Now we normalize the features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
from sklearn.svm import SVC


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=100, facecolors='#FFAAAA', alpha=.5)
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# %%
# SVM Classifier model
# the hyperparameter C control the margin violations
# smaller C leads to more margin violations but wider margin

svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X_scaled, y)

plot_svc_decision_boundary(svm_clf, -2, 2)
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1],
            color='#378661', marker='x', s=30, linewidth=1.5, label="Class +1")
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1],
            color='#A73730', marker='x', s=30, linewidth=1.5, label="Class -1")


plt.xlabel("Petal Length normalized", fontsize=12)
plt.ylabel("Petal Width normalized", fontsize=12)
plt.title("Scaled", fontsize=16)
plt.axis([-2, 2, -2, 2])
plt.show()

# %% [markdown]
# ### Questions:
# 
# 1-Obtain and plot the SVM classifier for the dataset from Lab 1, Part 1.1. (X_syn, y_syn).\
# See below
# 

# %%
# Question 1

def generate_synthetic_data(
    n_samples1=10,  # generate these many data points for class1
    n_samples2=10,  # generate these many data points for class2
    mu1=[2, 2],
    sigma1=[[5, 1], [1, 5]],
    mu2=[-2, -2],
    sigma2=[[10, 1], [1, 3]]
):
    """
        Code for generating the synthetic data.
        We will have two features and a binary class.

    """

    def gen_gaussian(size, mean_in, cov_in, class_label):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(size)
        y = np.ones(size, dtype=int) * class_label
        return nv, X, y

    """ Generate the features randomly """
    # For the NON-protected group (men)
    # We will generate one gaussian cluster for each class

    nv1, X1, y1 = gen_gaussian(
        int(n_samples1), mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(
        int(n_samples2), mu2, sigma2, 0)  # negative class

    # join the positive and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples1 + n_samples2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    return X, y


X_syn, y_syn = generate_synthetic_data()

plt.scatter(X_syn[y_syn == 1][:, 0], X_syn[y_syn == 1][:, 1],
            color='#378661', marker='x', s=30, linewidth=1.5, label="Class +1")
plt.scatter(X_syn[y_syn == 0][:, 0], X_syn[y_syn == 0][:, 1],
            color='#A73730', marker='x', s=30, linewidth=1.5, label="Class -1")
plt.legend(loc=2, fontsize=10)
svm_clf = SVC(kernel="linear", C=999)
svm_clf.fit(X_syn, y_syn)

plot_svc_decision_boundary(svm_clf, np.min(X_syn), np.max(X_syn))
plt.show()

# %% [markdown]
# 2-Which differences do you observe from the SVM classifier for the iris dataset?\
# Sometimes, the algorithm doesn't stop running, which is probably because the two classes are not linearly separable, therefore it keeps trying to separate them with a line, which is impossible.
# 
# 3-How can you make it more "similar" to the iris classifier? Plot different SVM classifiers.\
# By using soft-margin, in order to allow for some data points to be "misclassified". The kernel trick can also be used, but it has not been seen in class.
# 
# 4-Load another real dataset from sklearn library, obtain the SVM classifier and plot both. Note: select only 2 features to be in dimension 2 as we did with the iris dataset.\
# See below
# 

# %%
# Question 4
import pandas as pd

def preproc_cancer():
    # Load data
    cancer = datasets.load_breast_cancer()
    # select columns
    red_cancer = pd.DataFrame(cancer.data[:, [0, 1]]).copy()
    red_cancer.rename(columns=dict(
        zip(red_cancer.columns, cancer.feature_names[0:2])), inplace=True)
    red_cancer["target"] = cancer.target
    return red_cancer

def svm_classify(C, red_cancer):
    # format to feed clf
    X_data = red_cancer[["mean radius", "mean texture"]].values
    y_data = red_cancer["target"].values
    svm_clf = SVC(kernel="linear", C=C)
    svm_clf.fit(X_data, y_data)
    return svm_clf

def plot_red_cancer(red_cancer, svm_clf):
    X_data = red_cancer[["mean radius", "mean texture"]].values
    # plot
    x, y = red_cancer[red_cancer["target"] == 0][[
        "mean radius", "mean texture"]].values.reshape(2, -1)
    plt.scatter(x, y, alpha=.7)
    x, y = red_cancer[red_cancer["target"] == 1][[
        "mean radius", "mean texture"]].values.reshape(2, -1)
    plt.scatter(x, y, alpha=.7)
    plot_svc_decision_boundary(svm_clf, np.min(X_data[:,0]), np.max(X_data[:,0]))
    plt.show()

red_cancer = preproc_cancer()
C_values = [10**k for k in range(7)]
C_accuracies = []
for c in C_values:
    print(f"=====\n\nC = {c}")
    svm_clf = svm_classify(C=c, red_cancer=red_cancer)
    plot_red_cancer(red_cancer, svm_clf)

    X_data = red_cancer[["mean radius", "mean texture"]].values
    y_data = red_cancer["target"].values
    C_accuracies.append(svm_clf.score(X_data, y_data))

# %% [markdown]
# 
# 5-Provide a table with accuracy results for all the classifiers above.\
# See below

# %%
df_acc = pd.DataFrame(data={"C":C_values, "accuracy":C_accuracies})
df_acc.plot("C", "accuracy", logx=True)
df_acc

# %% [markdown]
# ## 2.2 Generating Biased data
# 
# Now you are going to generate a toy example of synthetic biased data. You will reproduce the Representation bias and Aggregation bias defined in Lecture 2.
# 
# As a reminder:
# 
# Representation bias occurs when certain parts of the input space are underrepresented.
# 
# Aggregation bias arises when a one-size-fit-all model is used for groups with different conditional distributions.
# 
# In order to create these two type of bias, we can "play" with the probability of an object $i$ of being protected, and the parameters of the distribution (Gaussian in our example) these protected objects follow.
# 
# In the following, you will obtain a dataset with aggregation bias.
# 

# %%
def generate_synthetic_data_bias(
    n_samples = 50,  # generate these many data points per class
    # For biased data
    # this parameter sets the probability of being protected (sensitive feature=1)
    p_sen = 0.5,
    # This is the increment of the mean for the positive class
    delta1 = [3, -2],
    # This is the increment of the mean for the negative class
    delta2 = [3, -2]):
    """
        Code for generating the synthetic data.
        We will have two features and a binary class.

    """


    def gen_gaussian_sensitive(size, mean_in, cov_in, class_label, sensitive):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(size)
        y = np.ones(size, dtype=int) * class_label
        x_sen = np.ones(size, dtype=float) * sensitive
        return nv, X, y, x_sen

    """ Generate the features randomly """
    # For the NON-protected group (sensitive feature=0, for ex. men)
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1, x_sen1 = gen_gaussian_sensitive(
        int((1-p_sen)*n_samples), mu1, sigma1, 1, 0)  # positive class
    nv2, X2, y2, x_sen2 = gen_gaussian_sensitive(
        int((1-p_sen)*n_samples), mu2, sigma2, 0, 0)  # negative class

    # For the Protected group (sensitive feature=1, for ex. women)
    # We will generate one gaussian cluster for each class
    mu3, sigma3 = np.add(mu1, delta1), [[5, 1], [1, 5]]
    mu4, sigma4 = np.add(mu2, delta2), [[10, 1], [1, 3]]
    nv3, X3, y3, x_sen3 = gen_gaussian_sensitive(
        int(p_sen*n_samples), mu3, sigma3, 1, 1.)  # positive class
    nv4, X4, y4, x_sen4 = gen_gaussian_sensitive(
        int(p_sen*n_samples), mu4, sigma4, 0, 1.)  # negative class

    # join the positive and negative class clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_prot = np.hstack((x_sen1, x_sen2, x_sen3, x_sen4))

    # shuffle the data
    perm = list(range(0, n_samples*2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_prot = x_prot[perm]

    return X, y, x_prot
    
# print(X_syn)
# print(y_syn)
# print(x_bias)

def plot_agg_bias(X_syn, y_syn, x_bias):
    plt.scatter(X_syn[y_syn==1][:, 0], X_syn[y_syn==1][:, 1], color='#378661', marker='x', s=40, linewidth=1.5, label= "Class +1")
    plt.scatter(X_syn[y_syn==0][:, 0], X_syn[y_syn==0][:, 1], color='#A73730', marker='x', s=40, linewidth=1.5, label = "Class -1")
    plt.show()
    X_s_0 = X_syn[x_bias == 0.0]
    X_s_1 = X_syn[x_bias == 1.0]
    y_s_0 = y_syn[x_bias == 0.0]
    y_s_1 = y_syn[x_bias == 1.0]

    plt.scatter(X_s_0[y_s_0 == 1][:, 0], X_s_0[y_s_0 == 1][:, 1],
                color='green', marker='x', s=40, linewidth=1.5, label="Non-prot. +1")
    plt.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1],
                color='red', marker='x', s=40, linewidth=1.5, label="Non-prot. -1")
    plt.scatter(X_s_1[y_s_1 == 1][:, 0], X_s_1[y_s_1 == 1][:, 1],
                color='green', marker='o', facecolors='none', s=40, label="Prot. +1")
    plt.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1],
                color='red', marker='o', facecolors='none', s=40, label="Prot. -1")
    plt.legend(loc=2, fontsize=10)

    plt.show()


# %%
X_syn, y_syn, x_bias = generate_synthetic_data_bias()
plot_agg_bias(X_syn, y_syn, x_bias)

# %% [markdown]
# ### Questions:
# 
# 1-Create and plot a dataset with an aggregation bias (much) stronger than in the previous dataset.\
# See below

# %%
X_syn, y_syn, x_bias = generate_synthetic_data_bias(delta1 = [30, -20], delta2 = [30, -20])
plot_agg_bias(X_syn, y_syn, x_bias)

# %% [markdown]
# 
# 2-Do you think the SVM classifier will perform with the same accuracy for the protected and non-protected groups? Why?\
# No, because in the general case the protected group is less represented.
# 
# 3-Create a dataset with representation bias (hint: you can only change p_sen and/or delta1 and/or delta2).\
# Keeping `delta1` and `delta2` from the previous example, we modify `p_sen` to have the sensitive group being (drastically) underrepresented. See below.

# %%
X_syn, y_syn, x_bias = generate_synthetic_data_bias(p_sen=.1, delta1 = [30, -20], delta2 = [30, -20])
print(X_syn, y_syn, x_bias)

# %% [markdown]
# 
# 4-Find the SVM classifier for the 3 datasets: the one provided, the one from question 1 and the other one from question 3. (Answer: the coefficients (w,b) defining the classifier for each case).
# 
# 5\*-Provide a table reporting accuracy for the 3 cases from question 3.
# 
# 6\*-Plot the datasets and classifiers from question 3.
# 
# **Grouped answer for questions 4, 5 and 6:**

# %%
def svm_classify_bias(C, X_syn, y_syn):
    svm_clf = SVC(kernel="linear", C=C)
    svm_clf.fit(X_syn, y_syn)
    return svm_clf

def plot_agg_bias(X_syn, y_syn, x_bias, svm_clf):
    plt.scatter(X_syn[y_syn==1][:, 0], X_syn[y_syn==1][:, 1], color='#378661', marker='x', s=40, linewidth=1.5, label= "Class +1")
    plt.scatter(X_syn[y_syn==0][:, 0], X_syn[y_syn==0][:, 1], color='#A73730', marker='x', s=40, linewidth=1.5, label = "Class -1")
    plot_svc_decision_boundary(svm_clf, np.min(X_syn[:,0]), np.max(X_syn[:,0]))
    plt.show()
    X_s_0 = X_syn[x_bias == 0.0]
    X_s_1 = X_syn[x_bias == 1.0]
    y_s_0 = y_syn[x_bias == 0.0]
    y_s_1 = y_syn[x_bias == 1.0]

    plt.scatter(X_s_0[y_s_0 == 1][:, 0], X_s_0[y_s_0 == 1][:, 1],
                color='green', marker='x', s=40, linewidth=1.5, label="Non-prot. +1")
    plt.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1],
                color='red', marker='x', s=40, linewidth=1.5, label="Non-prot. -1")
    plt.scatter(X_s_1[y_s_1 == 1][:, 0], X_s_1[y_s_1 == 1][:, 1],
                color='green', marker='o', facecolors='none', s=40, label="Prot. +1")
    plt.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1],
                color='red', marker='o', facecolors='none', s=40, label="Prot. -1")
    plt.legend(loc=2, fontsize=10)
    plot_svc_decision_boundary(svm_clf, np.min(X_syn[:,0]), np.max(X_syn[:,0]))
    plt.show()


# %%
def answer_dataset(i, p_sen = 0.5, delta1 = [3, -2], delta2 = [3, -2]):
    print(f"\n------ Dataset {i} ------")
    X_syn, y_syn, x_bias = generate_synthetic_data_bias(p_sen=p_sen, delta1=delta1, delta2=delta2)
    svm_clf = svm_classify_bias(500, X_syn, y_syn)
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    plot_agg_bias(X_syn, y_syn, x_bias, svm_clf)
    print(f"w = \t {w}")
    print(f"b = \t {b}")
    print(f"Accuracy: \t {svm_clf.score(X_syn, y_syn)}")

# Dataset 1
answer_dataset(i=1)
# Dataset 2
answer_dataset(i=2, delta1 = [30, -20], delta2 = [30, -20])
# Dataset 3
answer_dataset(i=3, p_sen=.1, delta1 = [30, -20], delta2 = [30, -20])


# %%



# %%



# %%



# %%



