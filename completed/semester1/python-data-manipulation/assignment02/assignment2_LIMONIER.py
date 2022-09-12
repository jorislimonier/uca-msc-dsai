# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Pandas
# Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. Indeeed, it is great for data manipulation, data analysis, and data visualization.
# 
# ### Data structures
# Pandas introduces two useful (and powerful) structures: `Series` and `DataFrame`, both of which are built on top of `NumPy`.
# 
# **Series**
# 
# A `Series` is a one-dimensional object similar to an array, list, or even column in a table. It assigns a labeled index to each item in the `Series`. By default, each item will receive an index label from `0` to `N-1`, where `N` is the number items of Series.
# 
# We can create a `Series` by passing a list of values, and let pandas create a default integer index.

# %%
import pandas as pd
import numpy as np

# create a Series with an arbitrary list
s = pd.Series([3, 'Machine learning', 1.414259, -65545, 'Happy coding!'])
print(s)

# %% [markdown]
# Or, an `index` can be used explixitly when creating the `Series`.

# %%
s = pd.Series([3, 'Machine learning', 1.414259, -65545, 'Happy coding!'],
             index=['Col1', 'Col2', 'Col3', 4.1, 5])
print(s)

# %% [markdown]
# A `Series` can be constructed from a dictionary too.

# %%
s = pd.Series({
        'Col1': 3, 'Col2': 'Machine learning', 
        'Col3': 1.414259, 4.1: -65545, 
        5: 'Happy coding!'
    })
print(s)

# %% [markdown]
# We can access items in a `Series` in a same way as `Numpy`.

# %%
s = pd.Series({
        'Col1': 3, 'Col2': -10, 
        'Col3': 1.414259, 
        4.1: -65545, 
        5: 8
    })

# get element which has index='Col1'
print("s['Col1']=", s['Col1'], "\n")

# use boolean indexing for selection
print(s[s > 0], "\n")

# modify elements on the fly using boolean indexing
s[s > 0] = 15

print(s, "\n")

# mathematical operations can be done using operators and functions.
print(s*10,  "\n")
print(np.square(s), "\n")

# %% [markdown]
# **DataFrame**
# 
# A `DataFrame` is a tablular data structure comprised of rows and columns, akin to database table, or R's data.frame object. In a loose way, we can also think of a `DataFrame` as a group of `Series` objects that share an `index` (the column names).
# 
# We can create a `DataFrame` by passing a `dict` of objects that can be converted to series-like.

# %%
data = {'year': [2013, 2014, 2015, 2013, 2014, 2015, 2013, 2014],
        'team': ['Manchester United', 'Chelsea', 'Asernal', 'Liverpool', 'West Ham', 'Newcastle', 'Machester City', 'Tottenham'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
football

# %% [markdown]
# We can store data as a CSV file, or read data from a CSV file

# %%
# save data to a csv file without the index
football.to_csv('football.csv', index=False)

from_csv = pd.read_csv('football.csv')
from_csv.head()

# %% [markdown]
# To read a `CSV` file with a custom delimiter between values and custom columns' names, we can use parameters `sep` and `names` relatively. Moreover, Pandas also supports to read and write to `Excel file` , `sqlite` database file, URL, or even clipboard.
# 
# We can have an overview on the data by using functions `info` and `describe`.

# %%
print(football.info(), "\n")
football.describe()

# %% [markdown]
# Numpy's regular slicing syntax works as well.

# %%
print(football[0:2], "\n")

# show only the teams that have won more than 10 matches from 2014
print(football[(football.year >= 2014) & (football.wins >= 10)])

# %% [markdown]
# An important feature that Pandas supports is `JOIN`. Very often, the data comes from multiple sources, in multiple files. For example, we have 2 CSV files, one contains the information of Artists, the other contains information of Songs. If we want to query the artist name and his/her corresponding songs, we have to do joining two dataframe.
# 
# Similar to `SQL`, in Pandas, you can do `inner join`, `left outer join`, `right outer join` and `full outer join`. Let's see a small example. Assume that we have two dataset of singers and songs. The relationship between two datasets is maintained by a constrain on `singer_code`.

# %%
singers = pd.DataFrame({'singer_code': range(5), 
                           'singer_name': ['singer_a', 'singer_b', 'singer_c', 'singer_d', 'singer_e']})
songs = pd.DataFrame({'singer_code': [2, 2, 3, 4, 5], 
                           'song_name': ['song_f', 'song_g', 'song_h', 'song_i', 'song_j']})
print(singers)
print('\n')
print(songs)


# %%
# inner join
pd.merge(singers, songs, on='singer_code', how='inner')


# %%
# left join
pd.merge(singers, songs, on='singer_code', how='left')


# %%
# right join
pd.merge(singers, songs, on='singer_code', how='right')


# %%
# outer join (full join)
pd.merge(singers, songs, on='singer_code', how='outer')

# %% [markdown]
# We can also concat two dataframes vertically or horizontally via function `concat` and parameter `axis`. This function is useful when we need to append two similar datasets or to put them side by site
# 
# 

# %%
# concat vertically
pd.concat([singers, songs], sort=True)


# %%
# concat horizontally
pd.concat([singers, songs], axis=1)

# %% [markdown]
# When computing descriptive statistic, we usually need to aggregate data by each group. For example, to anwser the question "how many songs each singer has?", we have to group data by each singer, and then calculate the number of songs in each group. Not that the result must contain the statistic of all singers in database (even if some of them have no song)

# %%
data = pd.merge(singers, songs, on='singer_code', how='left')

# count the values of each column in group
print(data.groupby('singer_code').count())

print("\n")

# count only song_name
print(data.groupby('singer_code').song_name.count())

print("\n")

# count song name but ignore duplication, and order the result
print(data.groupby('singer_code').song_name.nunique().sort_values(ascending=True))

# %% [markdown]
# ## ==> Your Turn
# %% [markdown]
# We have two datasets about music: [song](data/song.tsv) and [album](data/album.tsv).
# 
# In the following questions, you have to use Pandas to load data and write code to answer these questions.
# 
# ### Question 1
# Load both dataset into two dataframes and print the information of each dataframe.
# Unpack the additional `data.zip` into a folder `data/` where your notebook resides.
# 
# **HINT** The dataset can be load by using function `read_table`. For example: `df = pd.read_table(url, sep='\t')`

# %%
import pandas as pd

songdb_url = 'data/song.tsv'
albumdb_url = 'data/album.tsv'
song_df = pd.read_csv(songdb_url, sep="\t")
album_df = pd.read_csv(albumdb_url, sep="\t")

album_df

# %% [markdown]
# ### Question 2
# 
# How many albums in this datasets ?
# 
# How many songs in this datasets ?
# 
# 

# %%
print("number of albums:", len(album_df))
print("number of songs:", len(song_df))

# %% [markdown]
# ### Question 3
# How many distinct singers in this dataset ?
# 

# %%
print("number distinct singers:", len(song_df.Singer.unique()))

# %% [markdown]
# ### Question 4
# Is there any song that doesn't belong to any album ?
# Is there any album that has no song ?
# 
# **HINT**
#   * To join two datasets on different key names, we use left_on= and right_on= instead of on=.
#   * Funtion `notnull()` and `isnull()` help determining the value of a column is missing or not. 

# %%
fulldf = pd.merge(song_df, album_df, how='outer', left_on='Album', right_on='Album code')
fulldf


# %%
fulldf[fulldf["Album name"].isnull()] # song with no album


# %%
fulldf[fulldf["Song"].isnull()] # album with no song

# %% [markdown]
# 

