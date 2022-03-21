# %% [markdown]
# # Recommender Systems - notebook 1 - Traditional Approaches
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
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# %%
from tqdm import tqdm
import pandas as pd
import numpy as np


# %% [markdown]
# ## Dataset description
# 

# %% [markdown]
# We use here the MovieLens 100K Dataset. It contain 100,000 ratings from 1000 users on 1700 movies.
# 
# - u.train / u.test part of the original u.data information
#   - The full u data set, 100000 ratings by 943 users on 1682 items.
#     Each user has rated at least 20 movies. Users and items are
#     numbered consecutively from 1. The data is randomly
#     ordered. This is a tab separated list of
#     user id | item id | rating | timestamp.
#     The time stamps are unix seconds since 1/1/1970 UTC
# - u.info
#   - The number of users, items, and ratings in the u data set.
# - u.item
#   - Information about the items (movies); this is a tab separated
#     list of
#     movie id | movie title | release date | video release date |
#     IMDb URL | unknown | Action | Adventure | Animation |
#     Children's | Comedy | Crime | Documentary | Drama | Fantasy |
#     Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
#     Thriller | War | Western |
#     The last 19 fields are the genres, a 1 indicates the movie
#     is of that genre, a 0 indicates it is not; movies can be in
#     several genres at once.
#     The movie ids are the ones used in the u.data data set.
# - u.genre
#   - A list of the genres.
# - u.user
#   - Demographic information about the users; this is a tab
#     separated list of
#     user id | age | gender | occupation | zip code
#     The user ids are the ones used in the u.data data set.
# 

# %%
path = "https://www.i3s.unice.fr/~riveill/dataset/dataset_movilens_100K/"


# %% [markdown]
# Before we build our model, it is important to understand the distinction between implicit and explicit feedback, and why modern recommender systems are built on implicit feedback.
# 
# - **Explicit Feedback:** in the context of recommender systems, explicit feedback are direct and quantitative data collected from users.
# - **Implicit Feedback:** on the other hand, implicit feedback are collected indirectly from user interactions, and they act as a proxy for user preference.
# 
# The advantage of implicit feedback is that it is abundant. Recommender systems built using implicit feedback allow recommendations to be adapted in real time, with each click and interaction.
# 
# Today, online recommender systems are built using implicit feedback.
# 

# %% [markdown]
# ### Data preprocessing
# 

# %%
# Load data
np.random.seed(123)

ratings = pd.read_csv(
    path + "u.data",
    sep="\t",
    header=None,
    names=["userId", "movieId", "rating", "timestamp"],
)
ratings = ratings.sort_values(["timestamp"], ascending=True)
print("Nb ratings:", len(ratings))
ratings


# %% [markdown]
# ### Data splitting
# 
# Separating the dataset between train and test in a random fashion would not be fair, as we could potentially use a user's recent evaluations for training and previous evaluations. This introduces a data leakage with an anticipation bias, and the performance of the trained model would not be generalizable to real world performance.
# 
# Therefore, we need to slice the train and test based on the timestamp
# 

# %%
# Split dataset
train_ratings, test_ratings = np.split(ratings, [int(0.9 * len(ratings))])

max(train_ratings["timestamp"]) <= min(test_ratings["timestamp"])


# %%
# drop columns that we no longer need
train_ratings = train_ratings[["userId", "movieId", "rating"]]
test_ratings = test_ratings[["userId", "movieId", "rating"]]

len(train_ratings), len(test_ratings)


# %%
# Get a list of all movie IDs
all_movieIds = ratings["movieId"].unique()


# %% [markdown]
# ### Build pivot table
# 

# %%
""" Pivot table for train set """
train_pivot = pd.pivot_table(
    data=train_ratings,
    values="rating",
    index="userId",
    columns="movieId",
)
print("Nb users: ", train_pivot.shape[0])
print("Nb movies:", train_pivot.shape[1])
train_pivot


# %%
train_users = train_pivot.index
train_movies = train_pivot.columns


# %% [markdown]
# ## Collaborative filtering based on Users similarity
# 
# This approach uses scores that have been assigned by other users to calculate predictions.
# 
# In pivot table
# 
# - Rows are users, $u, v$
# - Columns are items, $i, j$
# 
# $$pred(u, i) = \frac{\sum_v sim(u,v)*r_{v,i}}{\sum_v sim(u,v)}$$
# 
# Wich similarity function:
# 
# - Euclidean distance $[0,1]$: $sim(a,b)=\frac{1}{1+\sqrt{\sum_i (r_{a,i}-r_{b,i})^2}}$
# - Pearson correlation $[-1,1]$: $sim(a,b)=\frac{\sum_i (r_{a,i}-r_a)(r_{b,i}-r_b)}{{\sum_i (r_{a,i}-r_a)^2}{\sum_i (r_{b,i}-r_b)^2}}$
# - Cosine similarity $[-1,1]$: $sim(a, b)=\frac{a.b}{|a|.|b|}$
# 
# Which function should we use? The answer is that there is no fixed recipe; but there are some issues we can take into account when choosing the proper similarity function. On the one hand:
# 
# - Pearson correlation usually works better than Euclidean distance since it is based more on the ranking than on the values. So, two users who usually like more the same set of items, although their rating is on different scales, will come out as similar users with Pearson correlation but not with Euclidean distance.
# - On the other hand, when dealing with binary/unary data, i.e., like versus not like or buy versus not buy, instead of scalar or real data like ratings, cosine distance is usually used.
# 

# %% [markdown]
# ### Build predictor
# 

# %%
# Step 1: build the similarity matrix between users
correlation_matrix = train_pivot.transpose().corr("pearson")
correlation_matrix


# %%
# Step2 build rating function
# We want to calculate the rating that a user could have given for an item.

# Il est plus efficace de travailler avec numpy qu'avec pandas.
# On transforme donc la matrice pivot en numpy
pivot = train_pivot.to_numpy()
# idem pour la matrice de correlation
corr = correlation_matrix.to_numpy()
# Malheureusement, on doit utiliser 2 dictionnaires pour passer
# Du nom de la colonne movieId dans son indice en numpy
movie2column = {j: i for i, j in enumerate(train_pivot.columns)}
# Du nom de la ligne userId dans son indice en numpy
user2row = {j: i for i, j in enumerate(train_pivot.index)}


def predict(pivot, corr, userId, movieId):
    if movieId in movie2column.keys():
        movie = movie2column[movieId]
    else:
        return 2.5
    if userId in user2row.keys():
        user = user2row[userId]
    else:
        return 2.5

    # Normalement le rating est inconnu
    if np.isnan(pivot[user, movie]):
        num = 0
        den = 0
        for u in range(len(corr)):
            if not np.isnan(pivot[u, movie]) and not np.isnan(corr[user, u]):
                # Si l'utilisateur u a déjà vu le film movie
                # et si les deux utilisateurs ont au moins vu un même film
                den += abs(corr[user, u])
                num += corr[user, u] * pivot[u, movie]
        if den != 0:
            return num / den
        else:
            return 2.5  # default value
    else:
        # le film a déjà été vu
        print(
            f"l'utilisateur {userId} a déjà vu le film {movieId}",
            f"et lui a donné la note de {pivot[user, movie]}",
        )
        return pivot[user, movie]


predict(pivot=pivot, corr=corr, userId=1, movieId=1)
predict(pivot=pivot, corr=corr, userId=3, movieId=28)


# %% [markdown]
# ### Predict
# 

# %%
# Step 3 add the predicted rating to the test set

test_ratings["User based"] = [
    predict(pivot, corr, userId, movieId)
    for _, userId, movieId, _ in tqdm(
        test_ratings[["userId", "movieId", "rating"]].itertuples()
    )
]
test_ratings


# %% [markdown]
# ### Evaluate the predictor
# 
# Now that we have trained our model, assigned a value to each pair (userId, movieId), we are ready to evaluate it.
# 

# %% [markdown]
# #### Evaluation with classical metrics: RMSE
# 
# In traditional machine learning projects, we evaluate our models using measures such as accuracy (for classification problems) and RMSE (for regression problems). This is what we will do in the first instance.
# 

# %%
test_ratings["rating"]


# %%
# Step 4 evaluate the resulte : with classical metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(
    "RMSE:",
    np.sqrt(mean_squared_error(test_ratings["rating"], test_ratings["User based"])),
)


# %% [markdown]
# #### Hit Ratio @ K
# 
# However, a measure such as RMSE does not provide a satisfactory evaluation of recommender systems. To design a good metric for evaluating recommender systems, we need to first understand how modern recommender systems are used.
# 
# Amazon, Netflix and others uses a list of recommendations. The key here is that we don’t need the user to interact with every single item in the list of recommendations. Instead, we just need the user to interact with at least one item on the list — as long as the user does that, the recommendations have worked.
# 
# To simulate this, let’s run the following evaluation protocol to generate a list of top 10 recommended items for each user.
# 
# - For each user, randomly select 99 items that the user has not interacted with.
# - Combine these 99 items with the test item (the actual item that the user last interacted with). We now have 100 items.
# - Run the model on these 100 items, and rank them according to their predicted probabilities.
# - Select the top 10 items from the list of 100 items. If the test item is present within the top 10 items, then we say that this is a hit.
# - Repeat the process for all users. The Hit Ratio is then the average hits.
# 
# This evaluation protocol is known as **Hit Ratio @ K**, and it is commonly used to evaluate recommender systems.
# 

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# - Fill the gaps
#   </font>
# 

# %%
# Step 2 with hit ratio
def HRatio(test_ratings, predictor, K=10, predict_func=predict):
    # User-item pairs for testing
    test_user_item_set = set(
        list(set(zip(test_ratings["userId"], test_ratings["movieId"])))[:1000]
    )

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby("userId")["movieId"].apply(list).to_dict()

    hits = []
    for (u, i) in tqdm(test_user_item_set):
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]
        predicted_labels = predictor(
            pairs=[np.array([u] * 100), np.array(test_items)],
            predict_func=predict_func,  # added to be able to pass custom predict functions
        ).reshape(-1)
        topK_items = [test_items[i] for i in np.argsort(predicted_labels)[-K:].tolist()]

        if i in topK_items:
            hits.append(1)
        else:
            hits.append(0)
    hr = np.average(hits)
    print("The Hit Ratio @ {} is {:.2f}".format(K, hr))
    return hr


# %%
def predictor(
    pairs,
    predict_func=predict,  # allows to pass custom predict functions
):
    pred = []
    for userId, movieId in zip(pairs[0], pairs[1]):
        pred += [predict_func(pivot, corr, userId, movieId)]
    return np.array(pred)


HR = dict()
hr = HRatio(
    test_ratings=test_ratings,
    predictor=predictor,
    K=25,
)
HR["User based"] = hr


# %% [markdown]
# ## Improve the rating
# 

# %% [markdown]
# ### Trick 1:
# 
# Since humans do not usually act the same as critics, i.e., some people usually rank movies higher or lower than others, this prediction function can be easily improved by taking into account the user mean as follows:
# 
# $$pred(u, i) = \overline{r_u} + \frac{\sum_v sim(u,v)*(r_{v,i} - \overline{r_v})}{\sum_v sim(u,v)}$$
# 

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# - Modify the previous code in order to implement "Trick 1"
#   </font>
# 

# %%
def user_center(pivot):
    """
    Compute train_pivot user centered (uc), which removes the mean of every user
    """
    user_mean = pivot.transpose().mean()
    return (pivot.transpose() - user_mean).transpose()


train_pivot_uc = user_center(
    pivot=train_pivot
)  # user centered version of `train_pivot`
correlation_matrix = train_pivot_uc.transpose().corr("pearson")


def predict_uc(pivot, corr, userId, movieId):
    if movieId in movie2column.keys():
        movie = movie2column[movieId]
    else:
        return 2.5
    if userId in user2row.keys():
        user = user2row[userId]
    else:
        return 2.5

    # Normalement le rating est inconnu
    if np.isnan(pivot[user, movie]):
        num = 0
        den = 0
        for u in range(len(corr)):
            if not np.isnan(pivot[u, movie]) and not np.isnan(corr[user, u]):
                # Si l'utilisateur u a déjà vu le film movie
                # et si les deux utilisateurs ont au moins vu un même film
                den += abs(corr[user, u])

                # remove the mean of user rating
                num += corr[user, u] * (pivot[u, movie] - np.nanmean(pivot[u]))
        if den != 0:
            return (num / den) + np.nanmean(pivot[user])  # add user mean
        else:
            return 2.5  # default value
    else:
        # le film a déjà été vu
        print(
            f"l'utilisateur {userId} a déjà vu le film {movieId}",
            f"et lui a donné la note de {pivot[user, movie]}",
        )
        return pivot[user, movie]


predict_uc(pivot=pivot, corr=corr, userId=1, movieId=1)
predict_uc(pivot=pivot, corr=corr, userId=3, movieId=28)


# %%
# Step 3 add the predicted rating to the test set

test_ratings["User based_uc"] = [
    predict_uc(pivot, corr, userId, movieId)
    for _, userId, movieId, _ in tqdm(
        test_ratings[["userId", "movieId", "rating"]].itertuples()
    )
]
test_ratings


# %%
# Step 4 evaluate the resulte : with classical metrics

print(
    "RMSE:",
    np.sqrt(mean_squared_error(test_ratings["rating"], test_ratings["User based_uc"])),
)


# %%
hr = HRatio(
    test_ratings=test_ratings,
    predictor=predictor,
    predict_func=predict_uc,
    K=25,
)
HR["User based_uc"] = hr


# %% [markdown]
# ### Trick 2:
# 
# If two users have very few items in common, let us imagine that there is only one, and the rating is the same, the user similarity will be really high; however, the confidence is really small. It's possible to add a ponderation coefficient.
# 
# $$newsim(a, b) = sim(a,b) * \frac{min(N, |P_{a,b}|)}{N}$$
# 
# where $|P_{a,b}|$ is the number of common items shared by user a and user b. The coefficient is $< 1$ if the number of common movies is $< N$ and $1$ otherwise.
# 

# %%
# Count the number of common items shared by user 1 and user 2.
a = user2row[1]
b = user2row[2]

thresh_common_nb = 10  # N
common = pivot[a] + pivot[b]  # sum of np arrays propagates nan
common = np.count_nonzero(~np.isnan(common))
enough_common_coeff = np.min([thresh_common_nb, common]) / thresh_common_nb
enough_common_coeff


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# - Modify the previous code in order to implement "Trick 2"
#   </font>
# 

# %%
def predict_thresh(
    pivot,
    corr,
    userId,
    movieId,
    thresh_common_nb=10,  # N
):
    if movieId in movie2column.keys():
        movie = movie2column[movieId]
    else:
        return 2.5
    if userId in user2row.keys():
        user = user2row[userId]
    else:
        return 2.5

    # Normalement le rating est inconnu
    if np.isnan(pivot[user, movie]):
        num = 0
        den = 0
        for u in range(len(corr)):
            if not np.isnan(pivot[u, movie]) and not np.isnan(corr[user, u]):
                # Si l'utilisateur u a déjà vu le film movie
                # et si les deux utilisateurs ont au moins vu un même film
                common = pivot[user] + pivot[u]  # sum of np arrays propagates nan
                common = np.count_nonzero(~np.isnan(common))
                enough_common_coeff = np.min([thresh_common_nb, common]) / thresh_common_nb
                
                num += enough_common_coeff * corr[user, u] * pivot[u, movie]
                den += abs(enough_common_coeff * corr[user, u])

        if den != 0:
            return num / den

        else:
            return 2.5  # default value
    else:
        # le film a déjà été vu
        print(
            f"l'utilisateur {userId} a déjà vu le film {movieId}",
            f"et lui a donné la note de {pivot[user, movie]}",
        )
        return pivot[user, movie]


predict_thresh(pivot=pivot, corr=corr, userId=1, movieId=1)
predict_thresh(pivot=pivot, corr=corr, userId=3, movieId=28)


# %%
for thresh_common_nb in [10, 15, 20, 30, 50]:
    # Step 3 add the predicted rating to the test set

    test_ratings[f"User based_thresh_{thresh_common_nb}"] = [
        predict_thresh(pivot, corr, userId, movieId, thresh_common_nb)
        for _, userId, movieId, _ in tqdm(
            test_ratings[["userId", "movieId", "rating"]].itertuples()
        )
    ]
    # test_ratings

    # Step 4 evaluate the resulte : with classical metrics

    print(
        f"RMSE for N={thresh_common_nb}:",
        np.sqrt(
            mean_squared_error(
                y_true=test_ratings["rating"],
                y_pred=test_ratings[f"User based_thresh_{thresh_common_nb}"],
            )
        ),
    )


# %%
hr = HRatio(
    test_ratings=test_ratings,
    predictor=predictor,
    predict_func=predict_thresh,
    K=25,
)
HR["User based_thresh"] = hr


# %%
test_ratings

# %%
HR


# %% [markdown]
# ## To go further
# 
# 1. Do the same, but with correlation between items. It's Collaborative filtering based on Items similarity. It's also possible to use the 2 previous trick
# 
# 1. Use Matrix factorization: decompose R in P, Q at rank k (i.e. if R is a m.n matrix, P is a m.k matrix and Q is a n.k matrix) the reconstruct R with P and Q (i.e. $\hat{R} = P Q^T$)
# 
# 1. Use Matrix decomposition: do an truncated SVD decomposition in order to obtain U, S and V, build $\hat{R} = U S V^T$
# 

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# - Choose, implement and evaluate one of the above strategies.
#   </font>
# 

# %% [markdown]
# ### Collaborative filtering based on items similarity
# 

# %%
""" Pivot table for train set """
train_pivot = pd.pivot_table(
    data=train_ratings,
    values="rating",
    index="movieId", # used to be userId
    columns="userId", # used to be movieId
) # the changes cause the axes to be reversed
print("Nb movies:", train_pivot.shape[0])
print("Nb users: ", train_pivot.shape[1])
train_pivot


# %%
# Step 1: build the similarity matrix between users

# no need to remove the transpose since we exchanged movieId and userId
# when making the pivot table
correlation_matrix = train_pivot.transpose().corr("pearson")
correlation_matrix


# %%
# Step2 build rating function
# We want to calculate the rating that a user could have given for an item.

# Il est plus efficace de travailler avec numpy qu'avec pandas.
# On transforme donc la matrice pivot en numpy
pivot = train_pivot.to_numpy()
# idem pour la matrice de correlation
corr = correlation_matrix.to_numpy()
# Malheureusement, on doit utiliser 2 dictionnaires pour passer
# Du nom de la colonne movieId dans son indice en numpy
movie2column = {j: i for i, j in enumerate(train_pivot.columns)}
# Du nom de la ligne userId dans son indice en numpy
user2row = {j: i for i, j in enumerate(train_pivot.index)}


# the names of movieId and userId should be reversed
def predict(pivot, corr, userId, movieId):
    if movieId in movie2column.keys():
        movie = movie2column[movieId]
    else:
        return 2.5
    if userId in user2row.keys():
        user = user2row[userId]
    else:
        return 2.5

    # Normalement le rating est inconnu
    if np.isnan(pivot[user, movie]):
        num = 0
        den = 0
        for u in range(len(corr)):
            if not np.isnan(pivot[u, movie]) and not np.isnan(corr[user, u]):
                # Si l'utilisateur u a déjà vu le film movie
                # et si les deux utilisateurs ont au moins vu un même film
                den += abs(corr[user, u])
                num += corr[user, u] * pivot[u, movie]
        if den != 0:
            return num / den
        else:
            return 2.5  # default value
    else:
        # le film a déjà été vu
        print(
            f"le film {userId} a déjà été vu par l'utilisateur {movieId}",
            f"et a reçu la note de {pivot[user, movie]}",
        )
        return pivot[user, movie]


predict(pivot=pivot, corr=corr, userId=1, movieId=1)
predict(pivot=pivot, corr=corr, userId=3, movieId=28)


# %%
# Step 3 add the predicted rating to the test set

test_ratings["items_based"] = [
    predict(pivot, corr, userId, movieId)
    for _, userId, movieId, _ in tqdm(
        test_ratings[["movieId", "userId", "rating"]].itertuples() # invert movieId and userId
    )
]
test_ratings


# %%
# compute RMSE
print(
    "RMSE:",
    np.sqrt(mean_squared_error(test_ratings["rating"], test_ratings["items_based"])),
)


# %% [markdown]
# ### Matrix factorization
# 
# Same approach as in the slides.
# 
# <font color='blue'>
# Below is a simple algorithm for factoring a matrix.
# </font>
# 

# %%
# Matrix factorization from scratch
def matrix_factorization(R, K, steps=10, alpha=0.005):
    """
    R: rating matrix
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter"""

    # N: num of User
    N = R.shape[0]
    # M: num of Movie
    M = R.shape[1]

    # P: |U| * K (User features matrix)
    P = np.random.rand(N, K)
    # Q: |D| * K (Item features matrix)
    Q = np.random.rand(M, K).T

    for step in tqdm(range(steps)):
        for i in range(N):
            for j in range(M):
                if not np.isnan(R[i][j]):
                    # calculate error
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        tmp = P[i][k] + alpha * (2 * eij * Q[k][j])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k])
                        tmp = P[i][k]

    return P, Q.T


# %%
# We try first on a toy example
# R: rating matrix
import math

R = [
    [5, 3, math.nan, 1],
    [4, math.nan, math.nan, 1],
    [1, 1, math.nan, 5],
    [1, math.nan, math.nan, 4],
    [0, 1, 5, 4],
    [2, 1, 3, math.nan],
]

R = np.array(R)
# Num of Features
K = 3

nP, nQ = matrix_factorization(R, K, steps=10)

nR = np.dot(nP, nQ.T)
nR


# %%
""" TRY to predict with matrix factorization """


# %%
""" Evaluate the result """


# %% [markdown]
# ## Decomposition using latent factor.
# 
# We use SVD decomposition
# 

# %%
pivot = train_pivot.fillna(0).to_numpy()
max_components = min(train_pivot.shape) - 1


# %%
from scipy.sparse.linalg import svds

k = 50
assert k < max_components

u, s, v_T = svds(pivot, k=k)
nR = u.dot(np.diag(s).dot(v_T))  # output of TruncatedSVD


# %%
s

# %%
""" TRY to predict with SVD decomposition """


# %%
""" Evaluate the result """



