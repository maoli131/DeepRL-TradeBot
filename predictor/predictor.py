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
file_name = "GOOGL_20190930.csv"
stock_name = "GOOGL"
dataset_ex_df = pd.read_csv('../data/invest_data/' + file_name, header=0, parse_dates=[1], date_parser=parser)
print("\n There are {} days in the dataset.".format(dataset_ex_df.shape[0]))
print(dataset_ex_df[['Date', 'Close']].head(3))

# Visualization
plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label=stock_name + ' stock')
plt.vlines(datetime.date(2016,4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 1: ' + stock_name + ' stock price')
plt.legend()
plt.show()

# Training sets and testing set
num_training_days = int(dataset_ex_df.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, \
                                                                    dataset_ex_df.shape[0]-num_training_days))

# 2. Correlated Assets
## Specified for only one stock

# 3. Technical Indicators: to be hand picked
## Directly get from Alpha Vintage API

# 4. Fourier Transform: extract trends and denoise it
data_FT = dataset_ex_df[['Date', 'Close']]

close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT['Close'],  label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 3: ' + stock_name + ' (close) stock prices & Fourier transforms')
plt.legend()
plt.show()


