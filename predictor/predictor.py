# This file first analyzes stock price data and then performs preliminary prediction

import time
import datetime
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Technical indicators
import ta

# Helper function to format date
def parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')

# 1. Get the historical stock data file
#name = input("\nPlease enter the stock historical data file name: \n") 
name = "GOOGL_20190930.csv"
dataset_ex_df = pd.read_csv('../data/invest_data/' + name, header=0, parse_dates=[1], date_parser=parser)
print("\n There are {} days in the dataset.".format(dataset_ex_df.shape[0]))
print(dataset_ex_df[['Date', 'Close']].head(3))

# Visualization
plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label=name + ' stock')
plt.vlines(datetime.date(2016,4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 1: ' + name + ' stock price')
plt.legend()
plt.show()

# Training sets and testing set
num_training_days = int(dataset_ex_df.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, \
                                                                    dataset_ex_df.shape[0]-num_training_days))

# 2. Correlated Assets
## Specified for only one stock