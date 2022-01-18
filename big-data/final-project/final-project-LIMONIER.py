# %%
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
import pandas as pd

# %%
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext

# %%
col_names = ["id", "time", "latitude", "longitude", "direction", "road", "traffic_status",
             "avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration", ]

df = spark.read.option("delimiter", ";").csv("datexDataB40.csv")
df = df.toDF(*col_names)
df.show(2)


# %% [markdown]
# ## Exploratory Data Analysis
# %%
for col in df.columns:
    unique_val = df.select(col).distinct().collect()
    print(f"--> {col}")
    print(f"\tunique values count: {len(unique_val)}")
    if len(unique_val) <= 10:
        print(f"\tunique values: {unique_val}")

# %% [markdown]
# The only road is "B40", so we can drop it
# %%


def drop_if_present(col):
    global df
    if col in df.columns:  # avoid errors on notebook re-run
        df = df.drop(col)


drop_if_present("road")
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
print(df.columns)
drop_if_present("latitude")
drop_if_present("longitude")
print(df.columns)

# %%
NUM_COL = ["avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration"]
df_corr = df.select(NUM_COL)
df_corr.show(2)
for col in NUM_COL:
    df_corr = df_corr.withColumn(col, df_corr[col].cast("double"))
print(df_corr.dtypes)
num_vector_col = "corr_features"
corr_assembler = VectorAssembler(inputCols=NUM_COL, outputCol=num_vector_col)
df_vect = corr_assembler.transform(df_corr).select(num_vector_col)
df_vect.show(2)
Correlation.corr(df_vect, num_vector_col)

# %%

# %%
CAT_COLS = ["direction", "traffic_status"]
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

onehot_pipe.fit(df).transform(df).show(5)
df.select("traffic_status").distinct().collect()
df.select("direction").distinct().collect()
