import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


plt.rcParams["figure.figsize"] = (15, 7)

from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from scipy import stats
import statsmodels.api as sm
from itertools import product

import warnings

warnings.filterwarnings('ignore')


def forecast_price(start_date, stop_date, data):
  print("forecasting price")
  if (start_date == None and stop_date == None):
    btc_month = data.resample('M').mean()
    btc_month.head()

    print("Dickey–Fuller test: p=%f" % adfuller(btc_month['Weighted Price'])[1])

    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(btc_month['Weighted Price'])[1])

    # Box-Cox Transformations
    btc_month['Weighted Price'].replace(0, np.nan, inplace=True)
    btc_month['Weighted Price'].fillna(method='ffill', inplace=True)
    btc_month['wp_box'], lmbda = stats.boxcox(btc_month['Weighted Price'])
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.wp_box)[1])

    # Seasonal differentiation (12 months)
    btc_month['box_diff_seasonal_12'] = btc_month.wp_box - btc_month.wp_box.shift(12)
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff_seasonal_12[12:])[1])

    # Regular differentiation
    btc_month['box_diff2'] = btc_month.box_diff_seasonal_12 - btc_month.box_diff_seasonal_12.shift(1)

    # STL-decomposition
    # seasonal_decompose(btc_month.box_diff2[13:]).plot()
    # print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff2[13:])[1])

    # autocorrelation_plot(btc_month.close)
    # plot_acf(btc_month['Weighted Price'][13:].values.squeeze(), lags=12)

    # Initial approximation of parameters
    qs = range(0, 3)
    ps = range(0, 3)
    d = 1
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model Selection
    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')
    for param in parameters_list:
      try:
        model = SARIMAX(btc_month.wp_box, order=(param[0], d, param[1])).fit(disp=-1)
      except ValueError:
        print('bad parameter combination:', param)
        continue
      aic = model.aic
      if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
      results.append([param, model.aic])

    # Best Models
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    print(result_table.sort_values(by='aic', ascending=True).head())

    #  print(best_model.summary())

    # print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])

    # Prediction
    btc_month_pred = btc_month[['Weighted Price']]

    # Future

    # date_list = [datetime(2018, 3, 31), datetime(2018, 4, 30), datetime(2018, 5, 31), datetime(2018, 6, 30)]
    # future = pd.DataFrame(index=date_list, columns=btc_month.columns)
    # btc_month_pred = pd.concat([btc_month_pred, future])

    btc_month_pred['forecast'] = invboxcox(best_model.predict(start=datetime(2017, 1, 31), end=datetime(2018, 4, 30)),
                                           lmbda)
    formattedForecastValues = btc_month_pred['forecast'][~np.isnan(btc_month_pred['forecast'])]
    formattedRealValues = pd.Series(btc_month['Weighted Price']).tail(len(formattedForecastValues))
    formattedDates = pd.Series(btc_month.index).tail(len(formattedForecastValues))
    # predictDates = btc_month_pred.tail(len(btc_month_pred['forecast'])).index
    mse = mean_squared_error(formattedRealValues,formattedForecastValues)
    forecast_errors = [formattedRealValues[i] - formattedForecastValues[i] for i in range(len(formattedForecastValues))]
    bias = sum(forecast_errors) * 1.0 / len(formattedForecastValues)

    data = {}
    data['predictedValues'] = []
    data['predictedValues'] = formattedForecastValues.to_json(orient='values')
    data['realValues'] = []
    data['realValues'] = formattedRealValues.to_json(orient='values')
    data['dates'] = []
    data['dates'] = formattedDates.to_json(orient='values')
    data['mse'] = str(mse)
    data['bias'] = str(bias)

    with open('resources/arima.json', 'w') as outfile:
      json.dump(data, outfile)
    return True
  else:
    print("Parameters inside")
    return True


def get_arima_json():
  with open('resources/arima.json') as f:
    data = json.load(f)
  return json.dumps(data)

# Inverse Box-Cox Transformation Function
def invboxcox(y, lmbda):
  if lmbda == 0:
    return (np.exp(y))
  else:
    return (np.exp(np.log(lmbda * y + 1) / lmbda))
