# This implements an Auto Regressivve Integrated Moving Average predictor
# Simple Time Series Predictor
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ArimaPredictor:
    
    def __init__(self):
        self.train_df = None # Training set
        self.model = None  # Model
        self.model_fit = None # ARIMA results

    # Trains the ARIMA Model with respective column
    def train(self, train_df, column='Close'):
        self.train_df = train_df
        series = self.train_df[column]
        # the p, d, q may need to be specifically determined
        self.model = ARIMA(series, order=(5, 1, 0))
        self.model_fit = self.model.fit(disp=0)
        print(self.model_fit.summary())

    # Performs prediction based on train model
    # Return an array of forecasted results
    def predict(self, steps=1):
        fc, se, conf_int = self.model_fit.forecast(steps=steps)
        return fc