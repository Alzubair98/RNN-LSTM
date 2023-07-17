# Part 1 - Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Inserting the training set as dataframe

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1]
