# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import isnan, when, count, col, to_timestamp
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
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

# %%
NUM_COL = ["avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration"]
CAT_COLS = ["direction", "traffic_status"]

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# We perform EDA on the whole data set, then we will perform a train-test split\
# Show a summary of the data
# %%
df.summary().show()
# %%
df.printSchema()
# %% [markdown]
# Let's see how many unique values are in each column.
# %%
for col_name in df.columns:
    unique_val = df.select(col_name).distinct().collect()
    print(f"--> {col_name}")
    print(f"\tunique values count: {len(unique_val)}")
    if len(unique_val) <= 1000:
        print(f"\tunique values: {[val[col_name] for val in unique_val]}")
# %% [markdown]
# We look for missing values
# %%


def show_unknown_counts():
    df.select([count(when(isnan(c) | col(c).isNull() | (col(c) == "null") | (col(c) == "unknown"), c)).alias(c)
               for c in df.columns]).show()


show_unknown_counts()

# %% [markdown]
# We deal with missing values in the `traffic_status`.
# %%
df.groupby("traffic_status").count().show()
print(f"Total number of rows in the data set: {df.count()}")
# %% [markdown]
# We see that there are 10309 "unkown" rows for the column `traffic_status`. This represents $\approx 5.8\%$ of the total number of rows. \
# It is probably safe to drop these rows since the rest of the data is fairly clean, therefore we shouldn't have to drop more rows later on.
# %%
df = df.filter(df["traffic_status"] != "unknown")
df.groupby("traffic_status").count().show()

# %% [markdown]
# We see that the `traffic_status` and `avg_vehicle_speed` have some unknown values
# %%
df = df.filter(df["traffic_status"] != "unknown")
show_unknown_counts()
# %% [markdown]
# Actually, removing the "unknown" values from the `traffic_status` also removed the ones from `avg_vehicle_speed` (*i.e.* they were on the same rows)

# %% [markdown]
# We notice that the only road is "B40", so we can drop it.
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
unique_ids = (
    df.select("id")
    .distinct()
    .collect()
)
# unique triples (`id`, `latitude`, `longitude`)
unique_triples = (
    df.select("id", "latitude", "longitude")
    .distinct()
    .collect()
)

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

# plot
df_num = df.select(NUM_COL+["traffic_status"]).toPandas()
for i in range(len(NUM_COL)):
    for j in range(i):
        col_x = NUM_COL[i]
        col_y = NUM_COL[j]
        px.scatter(data_frame=df_num, x=col_x, y=col_y,
                   color="traffic_status").show()


# %% [markdown]
# There doesn't appear to be (much) correlation between the numerical columns.
# Interestingly enough however, we noticed that the `traffic_concentration` column seems to be, if not categorical, at least fairly discrete.\
# Additionally, we see that the `traffic_status` column is (probably) derived directly from the `avg_vehicle_speed`, so we will remove it for prediction, otherwise, it would be too easy to predict the `traffic_status` column.
# %%
df_corr = df.select(NUM_COL)
# for num_col in NUM_COL:
#     df_corr = df_corr.withColumn(num_col, df_corr[num_col].cast("double"))
num_vector_col = "corr_features"
corr_assembler = VectorAssembler(
    inputCols=df_corr.columns, outputCol=num_vector_col)
df_vect = corr_assembler.transform(df_corr).select(num_vector_col)
Correlation.corr(df_vect, num_vector_col).show(truncate=False)

# %% [markdown]
# We see a strong correlation, but the columns studied are not linearly dependendent. We do not have hundreds of features so we decide not to drop any.\
# Now we deal with the `time` column.

# %%
time = df.select("time")
time.show(2, truncate=False)
df = df.withColumn("datetime", to_timestamp(df.time,))

df.show(2)

# %%
from pyspark.sql.functions import year
# df.withColumn("weekday", [date["datetime"].weekday() for date in df.select("datetime").collect()]).show(2)
df.select(year("datetime")).show(2) # TODO: make columns for year, month, ...
# %% [markdown]
# ## Perform train-test split
# %%
train, test = df.randomSplit(weights=[.8, .2], seed=42)
print(f"Number of observations in the train set: {train.count()}")
print(f"Number of observations in the test set: {test.count()}")

# %%
CAT_COLS = ["direction", "traffic_status"]
CAT_COLS_INDEXER = [f"{cat_col}_indexer" for cat_col in CAT_COLS]
CAT_COLS_ONEHOT = [f"{cat_col}_vec" for cat_col in CAT_COLS]

onehot_pipe = Pipeline(stages=[
    StringIndexer(inputCols=CAT_COLS, outputCols=CAT_COLS_INDEXER),
    OneHotEncoder(inputCols=CAT_COLS_INDEXER,
                  outputCols=CAT_COLS_ONEHOT, dropLast=True),
    VectorAssembler(inputCols=CAT_COLS_ONEHOT, outputCol="cat_features"),
])

onehot_pipe.fit(train).transform(train).show(2)

# %% [markdown]
# Select numerical columns (disregarding `avg_vehicle_speed` as mentioned previously)
# %%
NUM_COL_PRED = ["vehicle_flow_rate", "traffic_concentration"]
# %%

# %%
