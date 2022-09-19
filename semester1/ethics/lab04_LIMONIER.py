# %%
from IPython import get_ipython

# %%


# %% [markdown]
# <h1 style="font-size:3rem;color:#A3623B;">Lecture 4: The COMPAS recividism dataset</h1>
# 
# ## Security and Ethical aspects of data
# 
# ### Amaya Nogales Gómez
# 
# %% [markdown]
# ## 1 Introduction
# 
# First, we import the original COMPAS dataset used in the Propublica analysis from https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv.
# 
# We do the same filtering as in the Propublica analysis and we save our new csv file. For that, we will use R.
# 

# %%
# filter dplyr warnings
import warnings
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
warnings.filterwarnings('ignore')

# %% [markdown]
# Note: if you obtain the following error: "UsageError: Cell magic `%%R` not found."
# Try this solution: pip install rpy2
# 
# ## Filtering of data
# 
# <em>In a 2009 study examining the predictive power of its COMPAS score, Northpointe defined recidivism as “a finger-printable arrest involving a charge and a filing for any uniform crime reporting (UCR) code.” We interpreted that to mean a criminal offense that resulted in a jail booking and took place after the crime for which the person was COMPAS scored.
# 
# It was not always clear, however, which criminal case was associated with an individual’s COMPAS score. To match COMPAS scores with accompanying cases, we considered cases with arrest dates or charge dates within 30 days of a COMPAS assessment being conducted. In some instances, we could not find any corresponding charges to COMPAS scores. We removed those cases from our analysis.
# 
# Next, we sought to determine if a person had been charged with a new crime subsequent to crime for which they were COMPAS screened. We did not count traffic tickets and some municipal ordinance violations as recidivism. We did not count as recidivists people who were arrested for failing to appear at their court hearings, or people who were later charged with a crime that occurred prior to their COMPAS screening.</em>
# 
# Finally we save the filtered csv file.
# 

# %%
get_ipython().run_cell_magic('R', '', 'options(timeout=300)\nlibrary(dplyr)\n# You can choose your favorite option:\n# a)Download the dataset and access it locally\n# raw_data <- read.csv("./compas-scores-two-years.csv")\n# b)Access the dataset directly from the repository\nraw_data <- read.csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")\n\nprint(nrow(raw_data))\n\ndf <- dplyr:: select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count,\n                      days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>%\nfilter(days_b_screening_arrest <= 30) %>%\nfilter(days_b_screening_arrest >= -30) %>%\nfilter(is_recid != -1) %>%\nfilter(c_charge_degree != "O") %>%\nfilter(score_text != \'N/A\')\nwrite.csv(df, "propublica.csv")\nprint(nrow(df))')

# %% [markdown]
# Now we import the same libraries as in the previous labs.
# 

# %%
# we import all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal  # for generating synthetic data
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

# %% [markdown]
# We first read the filtered data.
# 

# %%
df = pd.read_csv("propublica.csv")
df.info()
# We calculate the number of objects in the dataset
print("Size of the dataset: %d" % len(df.index))

# %% [markdown]
# The result reveals a total of 6172 entries in the dataset. From the last column, Dtype, we might observe three different types of variables: the columns with data type “int64” and “float64” denote numerical (integer and real respectively) data while data type “object” denotes categorical data.
# 
# We will work with two different datasets. For basic descriptive analysis, we will use "propublica.csv". Later on, in order to obtain different classifiers, we will transform the dataset in order to binarize the categorical features.
# 
# %% [markdown]
# Questions:
# 
# Do the same preprocessing but for the "compas-scores-two-years-violent.csv" dataset.
# 
# 1-Load the data, select the variables and filter. Please note that in this dataset, there are 2 renamed variables: $score\_text$ becomes $v\_score\_text$ and $decile\_score$ becomes $v\_decile\_score$.
# 
# 2-Save it into a new file, "propublica-violent.csv".
# 
# 3-Which is the size of the dataset before and after filtering?
# 

# %%
get_ipython().run_cell_magic('R', '', 'options(timeout=300)\nlibrary(dplyr)\n# You can choose your favorite option:\n# a)Download the dataset and access it locally\n# raw_data <- read.csv("./compas-scores-two-years.csv")\n# b)Access the dataset directly from the repository\nraw_data <- read.csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv")\n\nprint(nrow(raw_data))\n\ndf <- dplyr:: select(raw_data, age, c_charge_degree, race, age_cat, v_score_text, sex, priors_count,\n                      days_b_screening_arrest, v_decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>%\nfilter(days_b_screening_arrest <= 30) %>%\nfilter(days_b_screening_arrest >= -30) %>%\nfilter(is_recid != -1) %>%\nfilter(c_charge_degree != "O") %>%\nfilter(v_score_text != \'N/A\')\nwrite.csv(df, "propublica_violent.csv")\nprint(nrow(df))')

# %% [markdown]
# We first observe the different type of variables and the values the take:
# 

# %%
for col in df:
    print(col, "\n\t", df[col].unique())

# %% [markdown]
# We do some basic statistic descriptive analysis:
# 

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

(
    df.sex.value_counts(normalize=True)
    .plot(kind='bar', title='Distribution of gender among defendants', ax=axes[0], color='#A3623B')
)

(
    df.race.value_counts(normalize=True)
    .plot(kind='bar', title='Distribution of race among defendants', ax=axes[1], color='#A3623B')
)


# plt.savefig("Hist_sex_race.pdf")

# %% [markdown]
# Questions:
# 
# 4- Plot the histogram for the $age\_cat$ variable.
# 

# %%
df["age_cat"].hist()

# %% [markdown]
# 5- Plot a pie chart for the $age\_cat$ variable.
# 

# %%
df["age_cat"].value_counts().plot(kind="pie")

# %% [markdown]
# Judges are often presented with two sets of scores from the Compas system - one that classifies people into High, Medium and Low risk, and a corresponding decile score. Let us analize this two outputs:
# 

# %%
ax = df.decile_score.value_counts(normalize=True).plot(
    kind='bar', title='Distribution of COMPAS scores', color='#714A41')

# plt.savefig("Hist_score_all.pdf")

# %% [markdown]
# Questions:
# 
# 6-Create a bar chart for the "variable" $score\_text$.
# 

# %%
df["score_text"].value_counts().plot(kind="bar")

# %% [markdown]
# 7-Create a bar chart for the compas score ($decile\_score$) for white defendants (caucasian) and black defendants (African-American) separatedly.
# 

# %%
df[df["race"] == "Caucasian"]["decile_score"].plot(kind="hist", alpha=.6)
df[df["race"] ==
    "African-American"]["decile_score"].plot(kind="hist", alpha=.6)

# %% [markdown]
# 8-Repeat question 7 for the $score\_text$.
# 

# %%
df[df["race"] == "Caucasian"]["score_text"].value_counts().plot(
    kind="bar", color="blue", alpha=.5)
df[df["race"] == "African-American"]["score_text"].value_counts().plot(kind="bar",
                                                                       color="red", alpha=.5)
plt.legend(["Caucasian", "African-American"])

# %% [markdown]
# 9-Repeat question 7 for the $sex$.
# 

# %%
df[df["race"] == "Caucasian"]["sex"].value_counts().plot(
    kind="bar", color="blue", alpha=.5)
df[df["race"] == "African-American"]["sex"].value_counts().plot(kind="bar",
                                                                color="red", alpha=.5)
plt.legend(["Caucasian", "African-American"])

# %% [markdown]
# ProPublica also conducted public records research to determine which defendants re-offended in the two years following their COMPAS screening. They were able to follow up on approximately half the defendants.
# 
# This dataset contains a field $two\_year\_recid$ that is 1 if the defendant re-offended within two years of screening and 0 otherwise. This what we denote $y$, the binary label to be predicted.
# 
# We will concern ourselves with comparing the Black and white populations, as in the article.
# 
# Similarly, we will consider a COMPAS score of either 'Medium' or 'High' to be a prediction that the defendant will re-offend within two years, this is what we denote $\hat{y}$, the predicted binary by the COMPAS algorithm.
# 

# %%
df = (
    pd.read_csv("propublica.csv")
    # We first binarize the categorical feature c_charge_degree
    .assign(COMPAS_Decision=lambda x: x['score_text'].replace({'Low': 0, 'Medium': 1, 'High': 1}))
)

# %% [markdown]
# Now we compute the error table:
# 

# %%
pd.crosstab(df['COMPAS_Decision'], df['two_year_recid'], margins=True)

# %% [markdown]
# Question:
# 
# 10- Based on the previous table, which is the False Positive Rate for the COMPAS algorithm? And the False Negative Rate?
# 

# %%
ct = pd.crosstab(df['COMPAS_Decision'], df['two_year_recid'],
            margins=True, normalize="columns")
print(f"FPR: {ct.iloc[1, 0]}")
print(f"FNR: {ct.iloc[0, 1]}")
ct

# %% [markdown]
# 11-Which is the accuracy of the COMPAS algorithm in the propublica dataset?
# 

# %%
crosst = pd.crosstab(df['COMPAS_Decision'], df['two_year_recid'], margins=True)
(crosst.iloc[0, 0] + crosst.iloc[1, 1]) / crosst.iloc[2, 2]

# %% [markdown]
# 12-Which is the accuracy of the COMPAS algorithm in the propublica-violent dataset?
# 

# %%
df_viol = (
    pd.read_csv("propublica_violent.csv")
    # We first binarize the categorical feature c_charge_degree
    .assign(COMPAS_Decision=lambda x: x['v_score_text'].replace({'Low': 0, 'Medium': 1, 'High': 1}))
)


# %%
crosst_viol = pd.crosstab(
    df_viol["COMPAS_Decision"], df_viol["two_year_recid"], margins=True)
(crosst_viol.iloc[0, 0] + crosst_viol.iloc[1, 1]) / crosst_viol.iloc[2, 2]

# %% [markdown]
# We are going to visualize with a heatmap the different metrics we want to analyze: the number and rates of false/true negatives/positives:
# 

# %%
# Warning: we need to normalize by column to obtain the FPR table
cm = pd.crosstab(df['COMPAS_Decision'], df['two_year_recid'], rownames=[
                 'Predicted recidivism'], colnames=['Actual recidivism'])
cm1 = pd.crosstab(df['COMPAS_Decision'], df['two_year_recid'], rownames=[
                  'Predicted recidivism'], colnames=['Actual recidivism'], normalize='columns')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
(
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                ax=axes[0], cmap='YlOrBr', annot_kws={"size": 18})
)
(
    sns.heatmap(cm1, annot=True, fmt="f", cbar=False,
                ax=axes[1], cmap='YlOrBr', annot_kws={"size": 18})
)
sns.set(font_scale=1.4)
# plt.savefig("FreqTable_all.pdf");

# %% [markdown]
# Question:
# 
# 13-Provide the same tables (heatmaps) for the propublica-violent dataset.
# 

# %%
# Warning: we need to normalize by column to obtain the FPR table
cm = pd.crosstab(df_viol['COMPAS_Decision'], df_viol['two_year_recid'], rownames=[
                 'Predicted recidivism'], colnames=['Actual recidivism'])
cm1 = pd.crosstab(df_viol['COMPAS_Decision'], df_viol['two_year_recid'], rownames=[
                  'Predicted recidivism'], colnames=['Actual recidivism'], normalize='columns')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
(
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                ax=axes[0], cmap='YlOrBr', annot_kws={"size": 18})
)
(
    sns.heatmap(cm1, annot=True, fmt="f", cbar=False,
                ax=axes[1], cmap='YlOrBr', annot_kws={"size": 18})
)
sns.set(font_scale=1.4)
# plt.savefig("FreqTable_all.pdf");

# %% [markdown]
# Now we do are going to do the same analysis by race to see if there is any bias:
# 

# %%
b_recid = df[df['race'] == 'African-American']
w_recid = df[df['race'] == 'Caucasian']

pd.concat([
    pd.crosstab(b_recid['COMPAS_Decision'], b_recid['two_year_recid'],
                normalize='columns', margins=False),
    pd.crosstab(w_recid['COMPAS_Decision'],
                w_recid['two_year_recid'], normalize='columns', margins=False)
], axis=1, keys=['Black', 'White'])

# %% [markdown]
# Question:
# 
# 14- Which is the accuracy of the COMPAS algorithm for black defendants in the propublica dataset?
# 

# %%
crosstab_all = pd.concat([
    pd.crosstab(b_recid['COMPAS_Decision'],
                b_recid['two_year_recid'], margins=True),
    pd.crosstab(w_recid['COMPAS_Decision'],
                w_recid['two_year_recid'], margins=True)
], axis=1, keys=['Black', 'White'])
crosstab_all


# %%
(crosstab_all["Black"].iloc[0, 0] + crosstab_all["Black"].iloc[1, 1]
 ) / crosstab_all["Black"].iloc[2, 2]

# %% [markdown]
# 15- Which is the accuracy of the COMPAS algorithm for white defendants in the propublica dataset?
# 

# %%
(crosstab_all["White"].iloc[0, 0] + crosstab_all["White"].iloc[1, 1]
 ) / crosstab_all["White"].iloc[2, 2]

# %% [markdown]
# 16- Calculate the same table for the propublica-violent dataset.
# 

# %%
b_recid_viol = df_viol[df_viol['race'] == 'African-American']
w_recid_viol = df_viol[df_viol['race'] == 'Caucasian']

crosstab_all_viol = pd.concat([
    pd.crosstab(b_recid_viol['COMPAS_Decision'],
                b_recid_viol['two_year_recid'], margins=True),
    pd.crosstab(w_recid_viol['COMPAS_Decision'],
                w_recid_viol['two_year_recid'], margins=True)
], axis=1, keys=['Black', 'White'])
crosstab_all_viol

# %% [markdown]
# We are going to visualize with a heatmap the different metrics we want to analyze but this time for black defendants and white defendants separatedly:
# 

# %%
FT_black = pd.crosstab(b_recid['COMPAS_Decision'], b_recid['two_year_recid'], rownames=[
                       'Predicted recividism'], colnames=['Actual recividism'], normalize='columns')
FT_white = pd.crosstab(w_recid['COMPAS_Decision'], w_recid['two_year_recid'], rownames=[
                       'Predicted recividism'], colnames=['Actual recividism'], normalize='columns')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].set_title('Black defendants', fontsize=18)
axes[1].set_title('White defendants', fontsize=18)

(
    sns.heatmap(FT_black, annot=True, fmt="f", cbar=False,
                ax=axes[0], cmap='YlOrBr', annot_kws={"size": 18})
)

(
    sns.heatmap(FT_white, annot=True, fmt="f", cbar=False,
                ax=axes[1], cmap='YlOrBr', annot_kws={"size": 18})
)
sns.set(font_scale=1.4)

# plt.savefig("FreqTable_BW.pdf")

# %% [markdown]
# Question:
# 
# 17- Provide the same tables for black and white defendants for the propublica-violent dataset.
# 

# %%
FT_black = pd.crosstab(b_recid_viol['COMPAS_Decision'], b_recid_viol['two_year_recid'], rownames=[
                       'Predicted recividism'], colnames=['Actual recividism'], normalize='columns')
FT_white = pd.crosstab(w_recid_viol['COMPAS_Decision'], w_recid_viol['two_year_recid'], rownames=[
                       'Predicted recividism'], colnames=['Actual recividism'], normalize='columns')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].set_title('Black defendants', fontsize=18)
axes[1].set_title('White defendants', fontsize=18)

(
    sns.heatmap(FT_black, annot=True, fmt="f", cbar=False,
                ax=axes[0], cmap='YlOrBr', annot_kws={"size": 18})
)

(
    sns.heatmap(FT_white, annot=True, fmt="f", cbar=False,
                ax=axes[1], cmap='YlOrBr', annot_kws={"size": 18})
)
sns.set(font_scale=1.4)

# %% [markdown]
# 18- Which is the accuracy of the COMPAS algorithm for black defendants in the propublica-violent dataset?
# 

# %%
(crosstab_all_viol["Black"].iloc[0, 0] + crosstab_all_viol["Black"].iloc[1, 1]
 ) / crosstab_all_viol["Black"].iloc[2, 2]

# %% [markdown]
# 19- Which is the accuracy of the COMPAS algorithm for white defendants in the propublica-violent dataset?
# 

# %%
(crosstab_all_viol["White"].iloc[0, 0] + crosstab_all_viol["White"].iloc[1, 1]
 ) / crosstab_all_viol["White"].iloc[2, 2]

# %% [markdown]
# Optional question:
# 
# 20- In the propublica dataset, we justified unfairness in the FNR and FPR differences between black and white defendants. Can we say the same for the propublica-violent dataset? Why?
# %% [markdown]
# Yes, that conclusion still holds. Indeed, from question 17 we see that in the propublica-violent dataset, the FNR of white people is roughly twice as high as for black people. Similarily, in the propublica-violent dataset, the FPR is twice as high for black people, compared to white people.

# %%
display(crosstab_all, crosstab_all_viol)


# %%



