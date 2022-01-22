# %%
import pandas as pd
import plotly.express as px
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, to_timestamp, when
from pyspark.sql.types import DoubleType

# %%

spark = SparkSession.builder.getOrCreate()
spark.sparkContext

# %%
col_names = [
    "id",
    "time",
    "latitude",
    "longitude",
    "direction",
    "road",
    "traffic_status",
    "avg_vehicle_speed",
    "vehicle_flow_rate",
    "traffic_concentration",
]

df = spark.read.option("delimiter", ";").csv("datexDataB40.csv")
df = df.toDF(*col_names)
df.show(2)

# %%
NUM_COL = [
    "avg_vehicle_speed",
    "vehicle_flow_rate",
    "traffic_concentration",
]
CAT_COLS_PRED = ["direction", "traffic_status"]

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
    df.select(
        [
            count(
                when(
                    isnan(c)
                    | col(c).isNull()
                    | (col(c) == "null")
                    | (col(c) == "unknown"),
                    c,
                )
            ).alias(c)
            for c in df.columns
        ]
    ).show()


show_unknown_counts()

# %% [markdown]
# We deal with missing values in the `traffic_status`.
# %%
df.groupby("traffic_status").count().show()
print(f"Total number of rows in the data set: {df.count()}")
# %% [markdown]
# We see that there are 10309 "unkown" rows for the column `traffic_status`. This represents $\approx 5.8\%$ of the total number of rows. \
# It is probably safe to drop these rows since the rest of the data is fairly clean. We also remove the 4 "impossible" values, since they are not numerous enough to make a good classifier.
# %%
df = df.filter(df["traffic_status"] != "unknown")
df = df.filter(df["traffic_status"] != "impossible")
df.groupby("traffic_status").count().show()
show_unknown_counts()
# %% [markdown]
# Actually, removing the "unknown" values from the `traffic_status` also removed the ones from `avg_vehicle_speed` (*i.e.* they were on the same rows)\
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
unique_ids = df.select("id").distinct().collect()
# unique triples (`id`, `latitude`, `longitude`)
unique_triples = df.select("id", "latitude", "longitude").distinct().collect()

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
df_num = df.select(NUM_COL + ["traffic_status"]).toPandas()
for i in range(len(NUM_COL)):
    for j in range(i):
        col_x = NUM_COL[i]
        col_y = NUM_COL[j]
        px.scatter(
            data_frame=df_num,
            x=col_x,
            y=col_y,
            color="traffic_status",
        ).show()


# %% [markdown]
# There doesn't appear to be (much) correlation between the numerical columns.
# Interestingly enough however, we noticed that the `traffic_concentration` column seems to be, if not categorical, at least fairly discrete.\
# Additionally, we see that the `traffic_status` column is (probably) derived directly from the `avg_vehicle_speed`, so we will remove it for prediction, otherwise, it would be too easy to predict the `traffic_status` column.
# %%
df_corr = df.select(NUM_COL)
num_vector_col = "corr_features"
corr_assembler = VectorAssembler(
    inputCols=df_corr.columns,
    outputCol=num_vector_col,
)
df_vect = corr_assembler.transform(df_corr).select(num_vector_col)
Correlation.corr(df_vect, num_vector_col).show(truncate=False)

# %% [markdown]
# We see a strong correlation, but the columns studied are not linearly dependendent. We do not have hundreds of features so we decide not to drop any.\
# Now we deal with the `time` column.

# %%
df.select("time").show(2, truncate=False)
# %% [markdown]
# We convert the time column (currently `str`) to a datetime object
# %%
df = df.withColumn(
    colName="datetime",
    col=to_timestamp(df.time),
)

df.show(2)

# %% [markdown]
# Now we split the datetime object into several of its compenents.
# %%
from pyspark.sql.functions import (
    dayofmonth,
    dayofweek,
    dayofyear,
    hour,
    minute,
    month,
    weekofyear,
    year,
)

time_props = [
    dayofweek,
    dayofyear,
    dayofmonth,
    hour,
    minute,
    month,
    weekofyear,
    year,
]

if "datetime" in df.columns:  # prevent errors on cell re-run
    for time_prop in time_props:
        df = df.withColumn(
            colName=time_prop.__name__,
            col=time_prop(col("datetime")),
        )
        df.groupby(time_prop.__name__).count().show()

# %% [markdown]
# We see that the `year` column has only one value (2019). It doesn't bring any extra information so we drop it.\
# We also drop the `time` and `datetime` columns.
# %%
time_props = [time_prop for time_prop in time_props if time_prop != year]
TIME_COLS = [time_prop.__name__ for time_prop in time_props]

df.show(2)
drop_if_exists("year")
drop_if_exists("time")
drop_if_exists("datetime")
df.show(2)
# %% [markdown]
# We drop the `avg_vehicle_speed` as mentioned before
# %%
drop_if_exists("avg_vehicle_speed")
# %% [markdown]
# ## Classification
# ### Perform train-test split
# %%
train, test = df.randomSplit(weights=[0.8, 0.2], seed=42)
print(f"Number of observations in the train set: {train.count()}")
print(f"Number of observations in the test set: {test.count()}")

# %%
CAT_COLS_PRED = TIME_COLS + ["direction", "id"]
NUM_COLS_PRED = ["vehicle_flow_rate", "traffic_concentration"]
TARGET_COL = "traffic_status"

missing_cols = [
    missing_col
    for missing_col in df.columns
    if missing_col not in CAT_COLS_PRED + NUM_COLS_PRED + [TARGET_COL]
]
if missing_cols == []:
    print("All columns are planned for classification")
else:
    print(f"{missing_cols} are not yet planned")
# %% [markdown]
# ### Preprocess categorical columns
# We OneHotEncode the categorical columns
# %%
CAT_COLS_INDEXER = [f"{cat_col}_indexer" for cat_col in CAT_COLS_PRED]
CAT_COLS_ONEHOT = [f"{cat_col}_vec" for cat_col in CAT_COLS_PRED]

stages_cat = [
    StringIndexer(
        inputCols=CAT_COLS_PRED,
        outputCols=CAT_COLS_INDEXER,
    ),
    OneHotEncoder(
        inputCols=CAT_COLS_INDEXER,
        outputCols=CAT_COLS_ONEHOT,
        dropLast=True,
    ),
]
# VectorAssembler(
#     inputCols=CAT_COLS_ONEHOT,
#     outputCol="cat_features",
# ),


Pipeline(stages=stages_cat).fit(train).transform(train).show(2)

# %% [markdown]
# ### Preprocess numerical columns
# We scale the numerical columns
# %%
stages_num = [
    VectorAssembler(
        inputCols=NUM_COLS_PRED,
        outputCol="assembled_num",
    ),
    StandardScaler(
        inputCol="assembled_num",
        outputCol="scaled_num",
    ),
]
train.show(20, truncate=False)
Pipeline(stages=stages_num).fit(train).transform(train).show(20, truncate=False)
# %%

# %%
