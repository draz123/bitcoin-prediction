import json

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def forecast_price(start_date, stop_date, data):
  if (start_date == None and stop_date == None):
    df = data[['Close']]

    train_size = int(len(df[['Close']]) * 0.7)
    test_size = len(df[['Close']]) - train_size
    # train, test = df[['Close']][0:train_size, :], df[['Close']][train_size:len(scaled), :]

    forecast_out = int(test_size)  # predicting 30 days into future
    df['Prediction'] = df[['Close']].shift(-forecast_out)  # label column with data shifted 30 units up

    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)

    X_forecast = X[-forecast_out:]  # set X_forecast equal to last 30
    X = X[:-forecast_out]  # remove last 30 from X

    y = np.array(df['Prediction'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=train_size/len(df[['Close']]))

    # Training
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    # Testing
    confidence = clf.score(X_test, y_test)
    print("confidence: ", confidence)

    forecast_prediction = clf.predict(X_forecast)
    print(forecast_prediction)

    formattedRealValues = pd.Series(df['Close'].tail(len(forecast_prediction)))
    formattedForecastValues = pd.Series(forecast_prediction)

    predictDates = df.tail(len(forecast_prediction)).index
    mse = mean_squared_error(formattedRealValues,formattedForecastValues)
    forecast_errors = [formattedRealValues[i] - formattedForecastValues[i] for i in range(len(formattedForecastValues))]
    bias = sum(forecast_errors) * 1.0 / len(formattedForecastValues)


    data = {}
    data['predictedValues'] = []
    data['predictedValues'] = pd.Series(forecast_prediction).to_json(orient='values')
    data['realValues'] = []
    data['realValues'] = pd.Series(df['Close'].tail(len(forecast_prediction))).to_json(orient='values')
    data['dates'] = []
    data['dates'] = pd.Series(predictDates).to_json(orient='values')
    data['mse'] = str(mse)
    data['bias'] = str(bias)

    with open('resources/lr.json', 'w') as outfile:
      json.dump(data, outfile)
    return True
  else:
    print("Parameters inside")
    return True


def get_lr_json():
  with open('resources/lr.json') as f:
    data = json.load(f)
  return json.dumps(data)
