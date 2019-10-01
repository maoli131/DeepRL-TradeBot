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

# 4. Fourier Transform: extract short and long term trends and denoise it
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

# 4.1: components for fourier transform
from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Figure 4: Components of Fourier transforms')
plt.show()

# 5. ARIMA: Auto Regressive Integrated Moving Average
# Simple time series forecast
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

series = data_FT['Close']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show() 

# 5.1 Predict with ARIMA and show the MSE
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# 5.2 Plot the predicted from ARIMA and real prices.
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on ' + stock_name + ' stock')
plt.legend()
plt.show() 
