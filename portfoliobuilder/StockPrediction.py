import pandas as pd
import pandas_datareader as pdr
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
import pandas_ta as ta
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from chart_studio import plotly as py
import plotly.tools as tls
from plotly import figure_factory as FF
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
import pydot
import graphviz
import math
from datetime import date
from datetime import timedelta
from tensorflow.python.keras.models import load_model
import plotly.express as px
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model
from statistics import mean


def getClosePricePrediction(ticker, data):
    # Settings for using cudNN (GPU acceleration)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # get data from yahoo finance for last 10 years
    df_stockData = pdr.DataReader(ticker, data_source='yahoo', start=str(date.today() - timedelta(days=365 * 10)),
                                  end=str(date.today() - timedelta(days=1)))
    # Add indicators using pandas ta lib
    # add Exponential Moving Average (EMA) indicator
    df_stockData.ta.ema(close='Close', length=3, append=True)
    # add Relative Strength Index (RSI) Indicator
    df_stockData.ta.rsi(close='Close', length=7, append=True)
    # add Average Directional Index (ADX) indicator
    df_stockData.ta.adx(high='High', low='Low', close='Close', length=3, append=True)
    # Add Moving Average Convergence Divergence (MACD) indicator
    df_stockData.ta.macd(close='Close', append=True)
    # Add On-Balance Volume indicator
    df_stockData.ta.obv(close='Close', volume='Volume', append=True)
    # Add Daily Percent Return
    df_stockData.ta.percent_return(length=1, append=True)
    # Add Stochastic Momentum Index (SMI)
    df_stockData.ta.smi(close='Close', append=True)
    # Average of open, high, low and close price
    df_stockData.ta.ohlc4(open='Open', high='High', low='Low', close='Close', append=True)

    df_stockData.dropna(inplace=True)

    # Perform Recursive Feature Elimination to get important features for this stock data
    selected_features = performRFE(df_stockData)
    if len(selected_features) > 3:
        selected_features = selected_features[:2]
    selected_features.append('Close')  # add back Close feature

    # Normalizing stock data
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    df_stockData_x = scaler_x.fit_transform(df_stockData[selected_features])
    df_stockData_y = scaler_y.fit_transform(np.array(df_stockData['Close'].values).reshape(-1, 1))

    # perform PCA to ensure 95% variance is maintained in data
    pca = PCA(n_components=0.95)
    df_stockData_x = pca.fit_transform(df_stockData_x)

    # Use 21 days of data to predict next 3 days
    numPastDays, numFutureDays = 21, 3
    # Use only last 3 days as test data which is to be predicted
    train_size = len(df_stockData) - numFutureDays
    x_train = df_stockData_x[0:train_size]
    y_train = df_stockData_y[0:train_size]
    x_test = df_stockData_x[train_size - numPastDays - numFutureDays + 1:]
    y_test = df_stockData_y[train_size - numPastDays - numFutureDays + 1:]
    x_train, y_train = prepData(x_train, y_train, numPastDays, numFutureDays)
    x_test, y_test = prepData(x_test, y_test, numPastDays, numFutureDays)

    # Build model
    model = Sequential()
    model.add(
        LSTM(600, activation='tanh', recurrent_activation='sigmoid', input_shape=(x_train.shape[1], x_train.shape[2]),
             return_sequences=True))
    model.add(LSTM(300, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(y_train.shape[1]))
    model.add(Reshape((y_train.shape[1], y_train.shape[2])))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    print("starting epochs for {}".format(ticker))
    # Use checkpoint callback to save model with lowest val_loss and avoid overfitting
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto', period=1)
    # Use this callback to dynamically lower learning rate depending upon val_loss
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=0.00001, mode='auto')
    # Train/fit the model using the data
    history = model.fit(x_train, y_train, epochs=30, batch_size=5, validation_split=0.3, verbose=1,
                        callbacks=[checkpoint, lr_reducer])
    # load the best model which was saved
    model = load_model("model.h5")
    print("Training finished for {}".format(ticker))
    # predict the 3 days
    predictions = model.predict(x_test)
    unscaled_pred = scaler_y.inverse_transform(predictions[-1].reshape(-1, 1))
    unscaled_target = scaler_y.inverse_transform(y_test[-1].reshape(-1, 1))

    # prepare dataframe to view the actual and predicted results for last 3 days
    compareDF = df_stockData.filter(['Close'])[train_size:]
    compareDF['Predicted'] = unscaled_pred
    print("Actual and predicted prices comparison")
    print(compareDF)
    # plot prediction and save graph
    prediction_graph = plt.figure(figsize=(12, 6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)
    plt.plot(df_stockData[train_size - 20:train_size + 1]['Close'])
    plt.plot(compareDF[['Close', 'Predicted']])
    plt.legend(['Training', 'Test (actual)', 'Predictions'], loc='lower right')
    prediction_graph = tls.mpl_to_plotly(prediction_graph)
    data["{}_pred".format(ticker)] = prediction_graph.to_html(full_html=False)
    sorted_prediction = list(unscaled_pred)
    sorted_prediction.sort()
    # Check for bullish prediction
    if unscaled_pred[2] == sorted_prediction[2] and unscaled_pred[2] > df_stockData.iloc[train_size - 1:train_size,
                                                                       3:4].values:
        print("Bullish prediction for {}".format(ticker))
        return True
    else:
        print("Bearish prediction for {}".format(ticker))
        return False


def performRFE(df_stockData):
    rfecv = RFECV(
        estimator=RandomForestRegressor(),
        min_features_to_select=1,
        step=2,
        n_jobs=-1,
        scoring="r2",
        cv=5,
    )
    cols_rfe = list(df_stockData.columns)
    # Exclude 'Close' feature
    cols_rfe.remove('Close')
    rfecv.fit(StandardScaler().fit_transform(df_stockData.loc[:, cols_rfe]),
              np.array(df_stockData['Close'].values).reshape(-1, ))

    feature_importance_df = pd.DataFrame()
    # Get the important features and their scores
    feature_importance_df['Features'] = list(df_stockData.loc[:, cols_rfe].columns[rfecv.support_])
    feature_importance_df['Importance'] = rfecv.estimator_.feature_importances_
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # return list of selected features in descending order of importance
    return list(feature_importance_df['Features'].values)


def prepData(data_x, data_y, numPastDays, numFutureDays):
    x = []
    y = []
    for i in range(numPastDays, len(data_x) - numFutureDays + 1):
        x.append(data_x[i - numPastDays:i, :])
        y.append(data_y[i:i + numFutureDays])
    x = np.array(x)
    y = np.array(y)
    return x, y
