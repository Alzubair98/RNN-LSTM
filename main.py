# Part 1 - Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Inserting the training set as dataframe

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')  # dataFrame
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1), copy=True)
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output

X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# build RNN ( stacked LSTM )
# Importing the Keras libraries and packages

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LTSM
from tensorflow.keras.layers import Droput