from __future__ import print_function

import json
from math import sqrt

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)



def forecast_price(start_date, stop_date, data):
  print("forecasting price")
  if (start_date == None and stop_date == None):
    try:
      data['Weighted Price'].replace(0, np.nan, inplace=True)
      data['Weighted Price'].fillna(method='ffill', inplace=True)

      values = data['Weighted Price'].values.reshape(-1,1)
      values = values.astype('float32')
      scaler = MinMaxScaler(feature_range=(0, 1))
      scaled = scaler.fit_transform(values)

      train_size = int(len(scaled) * 0.7)
      test_size = len(scaled) - train_size
      train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
      print(len(train), len(test))

      look_back = 1
      trainX, trainY = create_dataset(train, look_back)
      testX, testY = create_dataset(test, look_back)

      trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
      testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

      model = Sequential()
      model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
      model.add(Dense(1))
      model.compile(loss='mae', optimizer='adam')
      model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)


      yhat = model.predict(testX)

      yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
      testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

      rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
      print('Test RMSE: %.3f' % rmse)

      predictDates = data.tail(len(testX)).index

      testY_reshape = testY_inverse.reshape(len(testY_inverse))
      yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

      yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
      testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

      predictDates = data.tail(len(testX)).index
      print(predictDates)

      testY_reshape = testY_inverse.reshape(len(testY_inverse))
      yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
    except TypeError:
      return False
    # actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual Price')
    # predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')



    # fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
    # ax.plot(predictDates, yhat_inverse, 'o-')
    # ax.set_title("Default")
    # fig.autofmt_xdate()
    # plt.show()
    formattedRealValues = pd.Series(testY_reshape)
    formattedForecastValues = pd.Series(yhat_reshape)
    mse = mean_squared_error(formattedRealValues,formattedForecastValues)
    forecast_errors = [formattedRealValues[i] - formattedForecastValues[i] for i in range(len(formattedForecastValues))]
    bias = sum(forecast_errors) * 1.0 / len(formattedForecastValues)

    data = {}
    data['predictedValues'] = []
    data['predictedValues'] = formattedForecastValues.to_json(orient='values')
    data['realValues'] = []
    data['realValues'] = formattedRealValues.to_json(orient='values')
    data['dates'] = []
    data['dates'] = pd.Series(predictDates).to_json(orient='values')
    data['mse'] = str(mse)
    data['bias'] = str(bias)
    with open('resources/lstm.json', 'w') as outfile:
      json.dump(data, outfile)
    return True
  else:
    print("Parameters inside")
    return True

def get_lstm_json():
  with open('resources/lstm.json') as f:
    data = json.load(f)
  return json.dumps(data)
