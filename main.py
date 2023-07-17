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
