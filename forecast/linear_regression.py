import json

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import functools


def forecast_price(start_date, stop_date, data):
  if (start_date == None and stop_date == None):
    df = data[['Close']]

    train_size = int(len(df[['Close']]) * 0.7)
    test_size = len(df[['Close']]) - train_size

    df['Prediction'] = df[['Close']].shift(-test_size)

    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)

    X_forecast = X[-test_size:]
    X = X[:-test_size]

    y = np.array(df['Prediction'])
    y = y[:-test_size]
    new_X = preprocessing.scale(np.array(data['Close'].head(int(0.7*len(data['Close'])))))


    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=train_size/len(df[['Close']]))

    values = data['Close'].head(train_size).values.reshape(-1, 1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)



    new_y = data.head(train_size).index.values.astype(float)
    # Training
    clf = LinearRegression()
    clf.fit(np.array(list(range(0,train_size))).reshape(-1,1),data['Close'].head(train_size).values.reshape(-1, 1))
    # Testing
    confidence = clf.score(np.array(list(range(train_size,len(data)))).reshape(-1,1),np.array(data['Close'].tail(test_size)))
    print("confidence: ", confidence)

    forecast_prediction = clf.predict(np.array(list(range(0,train_size))).reshape(-1,1))
    print(forecast_prediction)

    formattedRealValues = pd.Series(df['Close'].tail(len(forecast_prediction)))
    formattedForecastValues = pd.Series(functools.reduce(list.__add__,forecast_prediction.tolist()))

    predictDates = data.tail(test_size).index
    mse = mean_squared_error(formattedRealValues,formattedForecastValues)
    forecast_errors = [formattedRealValues[i] - formattedForecastValues[i] for i in range(len(formattedForecastValues))]
    bias = sum(forecast_errors) * 1.0 / len(formattedForecastValues)


    jsonData = {}
    jsonData['predictedValues'] = []
    jsonData['predictedValues'] = pd.Series(formattedForecastValues).to_json(orient='values')
    jsonData['realValues'] = []
    jsonData['realValues'] = pd.Series(data['Close'].tail(test_size)).to_json(orient='values')
    jsonData['dates'] = []
    jsonData['dates'] = pd.Series(predictDates).to_json(orient='values')
    jsonData['mse'] = str(mse)
    jsonData['bias'] = str(bias)

    with open('resources/lr.json', 'w') as outfile:
      json.dump(jsonData, outfile)
    return True
  else:
    print("Parameters inside")
    return True


def get_lr_json():
  with open('resources/lr.json') as f:
    data = json.load(f)
  return json.dumps(data)
