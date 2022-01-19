# %%
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import isnan, when, count, col
import pandas as pd
import plotly.express as px

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
# %%
NUM_COL = ["avg_vehicle_speed", "vehicle_flow_rate", "traffic_concentration"]
df_corr = df.select(NUM_COL)
df_corr.show(2)
for col in NUM_COL:
    df_corr = df_corr.withColumn(col, df_corr[col].cast("double"))
print(df_corr.dtypes)
num_vector_col = "corr_features"
corr_assembler = VectorAssembler(
    inputCols=df_corr.columns, outputCol=num_vector_col)
df_vect = corr_assembler.transform(df_corr).select(num_vector_col)
Correlation.corr(df_vect, num_vector_col)
