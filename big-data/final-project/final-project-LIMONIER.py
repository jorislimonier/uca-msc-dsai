# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import isnan, when, count, col
import pandas as pd
import plotly.express as px

# %%

spark = SparkSession.builder.getOrCreate()
spark.sparkContext

# %%
col_names = ["id", "time", "latitude", "longitude", "direction", "road", "traffic_status",
             "avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration", ]

df = spark.read.option("delimiter", ";").csv("datexDataB40.csv")
df = df.toDF(*col_names)
df.show(2)
NUM_COL = ["avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration"]
CAT_COLS = ["direction", "traffic_status"]

# %% [markdown]
# ## Exploratory Data Analysis
# Show a summary of the data
# %%
df.summary().show()
# %% [markdown]
# Let's see how many unique values are in each column.
# %%
for col_name in df.columns:
    unique_val = df.select(col_name).distinct().collect()
    print(f"--> {col_name}")
    print(f"\tunique values count: {len(unique_val)}")
    if len(unique_val) <= 10000:
        print(f"\tunique values: {[val[col_name] for val in unique_val]}")
# %% [markdown]
# Count the number of missing values
# %%
# df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(isnan(c) | col(c).isNull() | (col(c) == "null") | (col(c) == "unknown"), c)).alias(c)
          for c in df.columns]).show()
# df.filter(df["avg_vehicle_speed"].isNull()).show()
# %%
df.filter((df["traffic_status"] == "unknown")).count()
# %%
distinct_avg_speed = df.select("avg_vehicle_speed").distinct()
distinct_avg_speed.show(999)
distinct_avg_speed.withColumn(
    "avg_speed_double", distinct_avg_speed["avg_vehicle_speed"].cast("double")).show(999)

# %%
df == "null"
# %% [markdown]
# We notice that, even though there is no proper missing value, the `traffic_status` column contains some (string) "unkown" values. We will have to deal with that later. \
# We also notice that the only road is "B40", so we can drop it.
# %%


def drop_if_exists(col):
    """
    Drop the `col` column from `df` if it is in its columns.
    This avoids errors on cell re-run.
    """
    global df
    if col in df.columns:
        df = df.drop(col)


drop_if_exists("road")
# %% [markdown]
# The number of unique values for `id`, `latitude` and `longitude` is the same (10). \
# We strongly suspect that for a given id, the latitude and longitude are always the same. Let's check it.
# %%
unique_ids = df.select("id").distinct().collect()
unique_triples = df.select("id", "latitude", "longitude").distinct(
).collect()  # unique triples (`id`, `latitude`, `longitude`)

print(len(unique_triples) == len(unique_ids) == 10)

# %% [markdown]
# So an `id` represents a unique camera in a unique location, therefore it has a unique pair (`latitude`, `longitude`).\
# This means that the `latitude` and `longitude` columns are redundant, hence we can drop them.
# %%
drop_if_exists("latitude")
drop_if_exists("longitude")

# %%
for num_col in NUM_COL:
    df = df.withColumn(num_col, df[num_col].cast("double"))

df_num = df.select(NUM_COL).toPandas()
for i in range(len(NUM_COL)):
    for j in range(i):
        col_x = NUM_COL[i]
        col_y = NUM_COL[j]
        px.scatter(df_num, col_x, col_y).show()


# %% [markdown]
# There doesn't appear to be (much) correlation between the numerical columns.
# Interestingly enough however, we noticed that the `traffic_concentration` column seems to be, if not categorical, at least fairly discrete.
# %%
df_corr = df.select(NUM_COL)
df_corr.show(2)
for num_col in NUM_COL:
    df_corr = df_corr.withColumn(num_col, df_corr[num_col].cast("double"))
print(df_corr.dtypes)
num_vector_col = "corr_features"
corr_assembler = VectorAssembler(
    inputCols=df_corr.columns, outputCol=num_vector_col)
df_vect = corr_assembler.transform(df_corr).select(num_vector_col)
Correlation.corr(df_vect, num_vector_col)


# %% [markdown]
# Now that we studied the original data set sufficiently (performing only some redundancy removal), we can transform it.\
# We will start by dealing with missing values in the `traffic_status` (counted as "unknown"), as mentioned previously. Let's see how many observations are concerned by this issue.
# %%
df.groupby("traffic_status").count().show()
print(f"Total number of rows in the data: {df.count()}")
# %% [markdown]
# We see that there are 10309 "unkown" rows for the column `traffic_status`. This represents $\approx 5.8\%$ of the total number of rows. \
# It is probably safe to drop these rows since the rest of the data is fairly clean, therefore we shouldn't have to drop more rows.
# %%
df = df.filter(df["traffic_status"] != "unknown")
df.groupby("traffic_status").count().show()
# %%
CAT_COLS_INDEXER = [f"{cat_col}_indexer" for cat_col in CAT_COLS]
CAT_COLS_ONEHOT = [f"{cat_col}_vec" for cat_col in CAT_COLS]

# for cat_col in CAT_COLS:
#     traffic_status_indexer = StringIndexer(
#         inputCol=cat_col, outputCol=f"{cat_col}_index")
#     traffic_status_indexer.fit(df).transform(df).show(2)

onehot_pipe = Pipeline(stages=[
    StringIndexer(inputCols=CAT_COLS, outputCols=CAT_COLS_INDEXER),
    OneHotEncoder(inputCols=CAT_COLS_INDEXER,
                  outputCols=CAT_COLS_ONEHOT, dropLast=True),
    VectorAssembler(inputCols=CAT_COLS_ONEHOT, outputCol="cat_features"),
])

onehot_pipe.fit(df).transform(df).show(2)
df.select("traffic_status").distinct().collect()
df.select("direction").distinct().collect()
