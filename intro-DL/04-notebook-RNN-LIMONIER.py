# %% [markdown]
# # Time series prediction with LSTM (student notebook)
# 
# Neural networks like Long Short-Term Memory (LSTM) recurrent neural networks are able to almost seamlessly model problems with multiple input variables.
# 
# This is a great benefit in time series forecasting, where classical linear methods can be difficult to adapt to multivariate or multiple input forecasting problems.
# 
# In this lab, you will discover how you can develop an LSTM model for multivariate time series forecasting with the Keras deep learning library.

# %%
"""
(Practical tip) Table of contents can be compiled directly in jupyter notebooks using the following code:
I set an exception: if the package is in your installation you can import it otherwise you download it 
then import it.
"""
try:
    from jyquickhelper import add_notebook_menu 
except:
    # !pip install jyquickhelper
    from jyquickhelper import add_notebook_menu
    
"""
Output Table of contents to navigate easily in the notebook. 
For interested readers, the package also includes Ipython magic commands to go back to this cell
wherever you are in the notebook to look for cells faster
"""
add_notebook_menu()

# %% [markdown]
# ## Imports

# %%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %%
# plottinh
import matplotlib.pyplot as plt
import plotly.express as px

# %%
# data
import math
import numpy as np
import pandas as pd

# %%
# ML
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (LSTM, AveragePooling1D, Bidirectional,
                                     Dense, Embedding, Flatten, Input,
                                     RepeatVector)
from tensorflow.keras.models import Model, load_model


# %% [markdown]
# ## LSTM network for uni-variate time series
# 
# LSTM can be used to model univariate time series forecasting problems.
# 
# These problems consist of a single set of observations and a model is needed to learn from the past set of observations in order to forecast the next value in the sequence.
# 
# We will demonstrate a number of variations of the LSTM model for univariate time series forecasting.
# 
# Be careful not to draw hasty conclusions about the relative performance of the models. The number of layers or neurons are highly variable between models.

# %% [markdown]
# ### Data preparation
# 
# #### A first example
# 
# Consider a given univariate sequence: `[10, 20, 30, 40, 50, 60, 70, 80, 90]`
# 
# We can divide the sequence into multiple input/output patterns called samples, where three time steps are used as input and one time step is used as output for the one-step prediction that is being learned.
# 
# `X,				y
# 10, 20, 30		40
# 20, 30, 40		50
# 30, 40, 50		60
# ...`
# 
# The `series_to_supervised()` function below implements this behavior and will split a given univariate sequence into multiple samples where each sample has a specified number of time steps (`n_in, by default 3`) and the output has also a specified number of time steps (`n_out, by default 1`).
# 
# By default, the data to predict is the last columns.

# %%
def series_to_supervised(data, n_in=3, n_out=1, output=None, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    output = [data.columns[-1]] if output is None else output
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols += [df.shift(i)]
        names += [f"{col}(t-{i})" for col in data.columns]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols += [df[output].shift(-i)]
        if i == 0:
            names += [f"{j}(t)" for j in output]
        else:
            names += [f"{j}(t+{i})" for j in output]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * create the time series mentioned in the first exemple.
# </font>

# %%
my_time_series = pd.DataFrame(list(range(10,100,10)))
my_time_series

# %%
data = series_to_supervised(my_time_series, n_in=3, n_out=1)
data.head()

# %% [markdown]
# <font color='blue'>
#     <bold>Head of the previous dataset</bold>
# 
# `	0(t-3)	0(t-2)	0(t-1)	0(t)
# 3	0.0	10.0	20.0	30
# 4	10.0	20.0	30.0	40
# 5	20.0	30.0	40.0	50
# 6	30.0	40.0	50.0	60
# 7	40.0	50.0	60.0	70
# `
# </font>

# %% [markdown]
# #### Do the same for a more sophisticated series

# %%
SIZE = 250
time_stamps = range(SIZE)

fct = lambda x: x*math.sin(x)
time_series = pd.DataFrame(data={"data":[fct(x) for x in range(SIZE)]})

n_features = my_time_series.shape[1] # for univariate time series
time_series

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Plot the time series generated thanks to x --> x*sin(x) with 250 timestamps
# * label x and y axis
# </font>

# %%
# plot data
px.scatter(
    data_frame=time_series,
    y="data",
    labels={"data": "time series value", "index": "time"},
    title="Time series of y -> x.sin(x)",
)


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Using `series_to_supervise` split data into samples with n_in = 6 and n_out = 1
# * Put the result in variable data
# </font>

# %%
""" FILL """
n_in = 6
n_out = 1
data = series_to_supervised(data=time_series, n_in=n_in, n_out=n_out)
data.head()

# %% [markdown]
# **Head of the previous dataset**
# 
# |     | data(t-6) | data(t-5) | data(t-4) | data(t-3) | data(t-2) | data(t-1) | data(t)   |
# | --- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
# | 6   | 0.000000  | 0.841471  | 1.818595  | 0.423360  | -3.027210 | -4.794621 | -1.676493 |
# | 7   | 0.841471  | 1.818595  | 0.423360  | -3.027210 | -4.794621 | -1.676493 | 4.598906  |
# | 8   | 1.818595  | 0.423360  | -3.027210 | -4.794621 | -1.676493 | 4.598906  | 7.914866  |
# | 9   | 0.423360  | -3.027210 | -4.794621 | -1.676493 | 4.598906  | 7.914866  | 3.709066  |
# | 10  | -3.027210 | -4.794621 | -1.676493 | 4.598906  | 7.914866  | 3.709066  | -5.440211 |
# 

# %% [markdown]
# Contrary to the approaches used so far, we cannot separate the data into TRAIN, VALID and TEST in a random way, since we are dealing with time series where the order is important.
# 
# The TRAIN data will therefore necessarily be at the beginning, then we will find the VALIDATION data and finally the TEST data.
# 
# Here are the chosen indices.

# %%
testAndValid = 0.1

SPLIT = int(testAndValid*len(data))
idx_train = len(data)-2*SPLIT
idx_test = len(data)-SPLIT

print(f"TRAIN=time_series[:{idx_train}]")
print(f"VALID=time_series[{idx_train}:{idx_test}]")
print(f"TEST=time_series[{idx_test}:]")

TRAIN=data[:idx_train]
VAL=data[idx_train:idx_test]
TEST=data[idx_test:]

# %%
plt.figure(figsize=(10, 4))
plt.plot(TRAIN["data(t)"], label="train")
plt.plot(VAL["data(t)"], label="val")
plt.plot(TEST["data(t)"], label="test")
plt.legend()
plt.xlabel("time stamps")
plt.ylabel("time series")

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * complete the code for preprocessing your train/validation/test datasets.
# </font>

# %%
# split into input and outputs
train_X, train_y = TRAIN.values[:, :-n_out], TRAIN.values[:, -n_out]
val_X, val_y = VAL.values[:, :-n_out], VAL.values[:, -n_out]
test_X, test_y = TEST.values[:, :-n_out], TEST.values[:, -n_out]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((-1, n_in, n_features))
val_X = val_X.reshape((-1, n_in, n_features))
test_X = test_X.reshape((-1, n_in, n_features))

train_X.shape, train_y.shape

# %% [markdown]
# <font color='darkcyan'>
# <bold>
# train_X.shape = (196, 6, 1)
# 
# train_y.shape = (196,)
# </font>

# %% [markdown]
# ### Build a first network using LSTM cells
# <br>
# <font color='red'>
# $TO DO - Students$
# 
# * Look carefully at the following cell
# * What is the impact of the `return_sequences` parameter of the LSTM cell? (change the value: False or True and observe the shape of the output).
# </font>

# %% [markdown]
# If `return_sequences=True`, the LSTM and the Dense layer get an extra dimension, which has value `n_in`.

# %%
LSTM_SIZE = 16

inputs = Input(shape=(n_in, n_features))
hidden = LSTM(LSTM_SIZE, return_sequences=False, activation="relu")(inputs)
# hidden = LSTM(LSTM_SIZE, return_sequences=True, activation='relu')(inputs) # check what changing `return_sequence` does
outputs = Dense(n_out, activation="linear")(hidden)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Complete the function build_and_fit used to train your RNN model.
# * compile : as usual
# * fit : as usual but...
#     * Be careful, you have to set the shuffle parameter to false in order to take the data in order.
#     * Use the validation set to control the overfitting in the earlystopping callback
# </font>

# %%
def build_and_fit(model, X_train, y_train, X_val, y_val, X_test, y_test, patience=150, epochs=200):
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=1,
        restore_best_weights=True,
        mode="min",
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[es],
        use_multiprocessing=True,
        workers=6,
    )  # epochs = 200
    y_pred = model.predict(X_test)

    # plot history
    plt.figure(figsize=(20, 8))

    plt.subplot(311)
    plt.plot(history.history["loss"][3:], label="loss")
    plt.plot(history.history["val_loss"][3:], label="val_loss")
    plt.legend()

    plt.subplot(312)
    plt.plot(history.history["mae"][3:], label="mae")
    plt.plot(history.history["val_mae"][3:], label="val_mae")
    plt.legend()

    plt.subplot(313)
    plt.plot(range(len(y_train)), y_train, label="train")
    plt.plot(range(len(y_train), len(y_train) + len(y_val)), y_val, label="valid")
    plt.plot(
        range(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_pred)),
        y_test,
        label="test",
    )
    plt.plot(
        range(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_pred)),
        y_pred,
        label="predict",
    )

    plt.legend(loc="center left")
    plt.show()

    return model


history = build_and_fit(model, train_X, train_y, val_X, val_y, test_X, test_y)


# %%
# demonstrate prediction
start = 12
y_true = (start + n_in) * (start + n_in)
x_input = np.array([fct(x) for x in range(start, start + n_in)])
x_input = x_input.reshape((1, n_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat, y_true)


# %% [markdown]
# ### Stacked Bi-LSTM
# 
# In order to improve the performance of the model, it's possible to:
# - stack LSTM with `return_sequence=True` for all levels except the last one where `return_sequence=False`
# - use Bi-LSTM

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Build a model stacking 3 BI-LSTM layers 
# </font>

# %%
inputs = Input(shape=(n_in, n_features))
bi_lstm1 = Bidirectional(
    layer=LSTM(
        units=LSTM_SIZE,
        return_sequences=True,
        activation="relu",
    )
)(inputs)

bi_lstm2 = Bidirectional(
    layer=LSTM(
        units=LSTM_SIZE,
        return_sequences=True,
        activation="relu",
    )
)(bi_lstm1)

bi_lstm3 = Bidirectional(
    layer=LSTM(
        units=LSTM_SIZE,
        return_sequences=False,
        activation="relu",
    )
)(bi_lstm2)
outputs = Dense(n_out, activation="linear")(bi_lstm3)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
# Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, activation=‘relu’))


# %%
model = build_and_fit(model, train_X, train_y, val_X, val_y, test_X, test_y)

# %%
# demonstrate prediction
start = 12
y_true = (start+n_in)*(start+n_in)
x_input = np.array([fct(x) for x in range(start,start+n_in)])
x_input = x_input.reshape((1, n_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat, y_true)

# %% [markdown]
# ## LSTM network for multi-variate time series 
# 
# Multivariate time series data means data where there is more than one observation for each time step.
# 
# There are two main models we may need with multivariate time series data. These are the multiple input series or the multiple parallel series depending on whether we want to predict one or more of the variables.
# 
# In this notebook, we focus on the first case: as input, several time series and as output (the prediction), a single time series.

# %% [markdown]
# ### Prepare the data
# 
# We reuse the same `series_to_supervise()` function in order to build a dataset with :
# * n_in elements for each series
# * n_out elements for each series to be predict
# 
# You have also to select one (Multiple Input Series) or many time series (Multiple Parallel Series) to predict.

# %%
# Get the time series
fct2 = lambda x: 2*x
time_series1 = [fct(x) for x in range(SIZE)]
time_series2 = [fct2(x) for x in range(SIZE)]
out_seq = [time_series1[i]+time_series2[i] for i in range(SIZE)]

# %%
# Get the dataset
dataset = pd.DataFrame(data={"f1":time_series1, "f2":time_series2, "output":out_seq}, index=range(SIZE))
n_features = dataset.shape[1] # for multivariate time series
dataset.head()

# %% [markdown]
# As with the univariate time series, we must structure these data into samples with input and output elements.
# An LSTM model needs sufficient context to learn a mapping from an input sequence to an output value. LSTMs can support parallel input time series as separate variables or features. Therefore, we need to split the data into samples maintaining the order of observations across the two input sequences.
# 
# If we chose six input time steps for the three features, we have to transform the dataset in the following way.

# %%
# As previously, prepare the dataset
''' In the followig example, we select 
- n_in number of time steps (6)
- n_out number of time steps of output
- and one serie to predict : output
'''
n_features = dataset.shape[1] # for multivariate time series
n_in = 6
n_out = 1
output = ["output"]

data = series_to_supervised(dataset, n_in, n_out, output=output)
data.head()

# %%
# Split dataset into TRAIN, VAL and TEST
testAndValid = 0.1

SPLIT = int(testAndValid * len(data))
idx_train = len(data) - 2 * SPLIT
idx_test = len(data) - SPLIT

print(f"TRAIN=time_series[:{idx_train}]")
print(f"VALID=time_series[{idx_train}:{idx_test}]")
print(f"TEST=time_series[{idx_test}:]")

TRAIN = data[:idx_train]
VAL = data[idx_train:idx_test]
TEST = data[idx_test:]


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * build train_X, val_X, test_X and train_y, val_y and test_y as before. Then print the shapes of tensors
#     * train_X is a 3D-tensor (196, 6, 3) for me
#     * train_y is a 1D-tensor (196,)
# </font>

# %%
# split into input and outputs
train_X, train_y = TRAIN.values[:, :-n_out], TRAIN.values[:, -n_out]
val_X, val_y = VAL.values[:, :-n_out], VAL.values[:, -n_out]
test_X, test_y = TEST.values[:, :-n_out], TEST.values[:, -n_out]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((-1, n_in, n_features))
val_X = val_X.reshape((-1, n_in, n_features))
test_X = test_X.reshape((-1, n_in, n_features))

train_X.shape, train_y.shape

# %% [markdown]
# ### Build a neuronal model

# %% [markdown]
# Any of the varieties of LSTMs in the previous section can be used, such as a Vanilla, Stacked, Bidirectional. It's also possible to use CNN or mixed CNN and LSTM.
# 
# We will use a Vanilla LSTM where the number of time steps and parallel series (features) are specified for the input layer via the input_shape argument.

# %%
# Build model
inputs = Input(shape=(n_in, n_features))
hidden = LSTM(LSTM_SIZE, return_sequences=False, activation="relu")(inputs)
outputs = Dense(n_out, activation="linear")(hidden)
model = Model(inputs, outputs)
model.summary()


# %%
model = build_and_fit(model, train_X, train_y, val_X, val_y, test_X, test_y)

# %% [markdown]
# ## Lab work: Air Pollution Forecasting
# 
# This is a dataset that reports on the weather and the level of pollution each hour for five years at the US embassy in Beijing, China.
# 
# The data includes the date-time, the pollution called PM2.5 concentration, and the weather information including dew point, temperature, pressure, wind direction, wind speed and the cumulative number of hours of snow and rain. The complete feature list in the raw data is as follows:
# 
# 1. No: row number
# 1. year: year of data in this row
# 1. month: month of data in this row
# 1. day: day of data in this row
# 1. hour: hour of data in this row
# 1. pm2.5: PM2.5 concentration
# 1. DEWP: Dew Point
# 1. TEMP: Temperature
# 1. PRES: Pressure
# 1. cbwd: Combined wind direction
# 1. Iws: Cumulated wind speed
# 1. Is: Cumulated hours of snow
# 1. Ir: Cumulated hours of rain
# 
# We can use this data and frame a forecasting problem where, given the weather conditions and pollution for prior hours, we forecast the pollution at the next hour.
# 
# This dataset can be used to frame other forecasting problems.

# %% [markdown]
# ## Load the data

# %%
DATAPATH = "https://www.i3s.unice.fr/~riveill/dataset/pollution.csv"

# %%
# Read the dataset
data = pd.read_csv(DATAPATH, sep=",", header=0, index_col=0)
data.head()

# %%
plt.figure(figsize=(16,9))
for i, column in enumerate(data.columns):
    plt.subplot(len(data.columns), 1, i+1)
    plt.plot(data[column].to_numpy())
    plt.title(column, y=0.5, loc='right')
plt.show()

# %% [markdown]
# ### Construct the dataset

# %% [markdown]
# The first step is to prepare the pollution dataset for the LSTM.
# 
# This involves framing the dataset as a supervised learning problem and normalizing the input variables.
# 
# We will frame the supervised learning problem as predicting the pollution at the current hour (t) given the pollution measurement and weather conditions at the prior time step.

# %%
n_features = data.shape[1] # for multivariate time series
n_in = 6
n_out = 1
output = ["pollution"]

n_features

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * using `series_to_supervised` function build the dataset
# </font>

# %%
dataset = series_to_supervised(data=data, n_in=n_in, n_out=n_out, output=output)
dataset.head()


# %%
dataset.dtypes

# %% [markdown]
# <font color='darkcyan'>
# <bold>dataset.dtypes gives the following result for me</bold>
# 
# <pre>
# pollution(t-6)    float64
# dew(t-6)          float64
# temp(t-6)         float64
# press(t-6)        float64
# wnd_dir(t-6)       object
# wnd_spd(t-6)      float64
# snow(t-6)         float64
# rain(t-6)         float64
# pollution(t-5)    float64
# dew(t-5)          float64
# temp(t-5)         float64
# press(t-5)        float64
# wnd_dir(t-5)       object
# wnd_spd(t-5)      float64
# snow(t-5)         float64
# rain(t-5)         float64
# pollution(t-4)    float64
# dew(t-4)          float64
# temp(t-4)         float64
# press(t-4)        float64
# wnd_dir(t-4)       object
# wnd_spd(t-4)      float64
# snow(t-4)         float64
# rain(t-4)         float64
# pollution(t-3)    float64
# dew(t-3)          float64
# temp(t-3)         float64
# press(t-3)        float64
# wnd_dir(t-3)       object
# wnd_spd(t-3)      float64
# snow(t-3)         float64
# rain(t-3)         float64
# pollution(t-2)    float64
# dew(t-2)          float64
# temp(t-2)         float64
# press(t-2)        float64
# wnd_dir(t-2)       object
# wnd_spd(t-2)      float64
# snow(t-2)         float64
# rain(t-2)         float64
# pollution(t-1)    float64
# dew(t-1)          float64
# temp(t-1)         float64
# press(t-1)        float64
# wnd_dir(t-1)       object
# wnd_spd(t-1)      float64
# snow(t-1)         float64
# rain(t-1)         float64
# pollution(t)      float64
# dtype: object
# <pre>
# </font>

# %% [markdown]
# First, we must split the prepared dataset into train and test sets. To speed up the training of the model for this demonstration, we will only fit the model on the first year of data, then evaluate it on the remaining 4 years of data. 
# 
# The example below splits the dataset into train and test sets, then splits the train and test sets into input and output variables. Finally, the inputs (X) are reshaped into the 3D format expected by LSTMs, namely [samples, timesteps, features].

# %%
# get the values
values = dataset.values

# split into train and test sets
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
val = values[n_train_hours:2*n_train_hours, :]
test = values[2*n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-n_out], np.array(train[:, -n_out], dtype="float64")
val_X, val_y = val[:, :-n_out], np.array(val[:, -n_out], dtype="float64")
test_X, test_y = test[:, :-n_out], np.array(test[:, -n_out], dtype="float64")

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
val_X = val_X.reshape((val_X.shape[0], n_in, n_features))
test_X = test_X.reshape((test_X.shape[0], n_in, n_features))
train_X.shape, train_y.shape

# %% [markdown]
# ### Encode and normalize dataset

# %% [markdown]
# Data encoding and normalization
# * The wind direction feature is label encoded (integer encoded).
# * All features are normalized
# 
# And then the dataset is transformed into a supervised learning problem. The weather variables for the hour to be predicted (t) are then removed.

# %%
numeric_features = [
    i for i, t in enumerate(dataset.dtypes[:-n_out]) if t in ["float64", "int32"]
]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = [
    i for i in range(len(dataset.columns) - n_out) if i not in numeric_features
]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

train_X_enc = preprocessor.fit_transform(train_X.reshape(len(train_X), -1)).reshape(
    len(train_X), n_in, -1
)
val_X_enc = preprocessor.fit_transform(val_X.reshape(len(val_X), -1)).reshape(
    len(val_X), n_in, -1
)
test_X_enc = preprocessor.transform(test_X.reshape(len(test_X), -1)).reshape(
    len(test_X), n_in, -1
)

n_features = train_X_enc.shape[2]  # Change with oneHotEncode
n_features


# %% [markdown]
# Running the code below prepare the data. Executing the next cell, prints the first 5 rows of the transformed dataset. We can see the 8 input variables (input series) and the 1 output variable (pollution level at the current hour).

# %% [markdown]
# ### Build, Compile, Fit, Predict and Evaluate a model
# <br>
# <font color='red'>
# $TO DO - Students$
# 
# * Build your model
#     * Put the number of hidden layers you want. If possible more than one.
# </font>

# %%
LSTM_SIZE = 256

inputs = Input(shape=(n_in, n_features))
hidden = LSTM(LSTM_SIZE, return_sequences=False, activation="relu")(inputs)
outputs = Dense(n_out, activation="linear")(hidden)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Compile your model
# </font>

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Fit your model using `EarlyStopping`
# </font>

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Plot learning curve
# </font>

# %%
# compile, fit and plot
model = build_and_fit(model, train_X_enc, train_y, val_X_enc, val_y, test_X_enc, test_y, patience=20)


# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Use your model to predict test set data
# </font>

# %%
# make a prediction
y_test_pred = model.predict(test_X_enc)
y_test_pred

# %% [markdown]
# <font color='red'>
# $TO DO - Students$
# 
# * Evaluate your model with RMSA
# </font>

# %%
# calculate RMSE
rmse = np.sqrt(np.mean((y_test_pred.flatten() - test_y.flatten())**2))
print(f"Test RMSE: {rmse}")

# %% [markdown]
# ### Predict next day
# 
# Generally, what we are trying to predict is a pollution indicator for the day or per 12-hour period. 
# 
# Modify the datasets to create a new column giving a pollution indicator per half day: little pollution, moderate pollution, heavy pollution.

# %% [markdown]
# ___
# We make a `date_id` column of shape YYYYMMDD with padding zeros on the left to make sure that:
# - years have 4 characters
# - months have 2 characters
# - days have 2 characters
# 
# We also add a digit indicating whether the given hour is in the morning or in the afternoon.

# %%
data.index = pd.to_datetime(data.index)
idx = data.index


def date_to_id(date):
    """converts a date to its id"""
    return (
        str(date.year).zfill(4)
        + str(date.month).zfill(2)
        + str(date.day).zfill(2)
        + str(int(date.hour >= 12))  # 0 if morning, else 1
    )


# Convert dates to date id's
day_id = pd.Series(idx).apply(date_to_id)

# Convert to list, otherwise all entries become nan when added to the df
data["day_id"] = day_id.values

# Group data by max of `day_id` instead of something like mean
# because it is common in meteorology to consider whether a certain threshold is passed,
# rather than the mean over hour.
grouped_data = data.groupby("day_id").max()
print(f"\n---> Data grouped by day id\n{grouped_data.head()}")

pollution_halfday = grouped_data.copy()["pollution"].to_dict()
print(f"\n---> Dictionary conveting day id to max pollution that halfday:\n{pollution_halfday}")

data["pollution"] = [pollution_halfday[idx] for idx in data["day_id"]]
data = data.drop(columns="day_id")
data.head(20)


# %%
train_y

# %%
# get the values
dataset = series_to_supervised(data=data, n_in=n_in, n_out=n_out, output=output)
values = dataset.values

# split into train and test sets
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
val = values[n_train_hours:2*n_train_hours, :]
test = values[2*n_train_hours:, :]

# redefine y values (X doesn't change)
train_y = np.array(train[:, -n_out], dtype="float64")
val_y = np.array(val[:, -n_out], dtype="float64")
test_y = np.array(test[:, -n_out], dtype="float64")

# %%
train_y.shape

# %%
history = build_and_fit(model, train_X_enc, train_y, val_X_enc, val_y, test_X_enc, test_y, patience=30)

# %%
# make a prediction
y_test_pred = model.predict(test_X_enc)
y_test_pred

# %%
# calculate RMSE
rmse = np.sqrt(np.mean((y_test_pred.flatten() - test_y.flatten())**2))
print(f"Test RMSE: {rmse}")


