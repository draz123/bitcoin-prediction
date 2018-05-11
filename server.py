import json

from flask import Flask, request
from flask_cors import CORS
from flask_jsonpify import jsonify
from flask_restful import Api

from forecast import arima
from forecast import linear_regression
from forecast import lstm
from forecast.data import get_btc_data

app = Flask(__name__)
api = Api(app)

CORS(app)


@app.route("/")
def hello():
  return jsonify({'text': 'Hello World!'})


@app.route("/forecast_price")
def simulate():
  data = get_btc_data()
  start_date = request.args.get('start_date')
  stop_date = request.args.get('stop_date')
  arimaStatus = arima.forecast_price(start_date, stop_date, data)
  lrStatus = linear_regression.forecast_price(start_date, stop_date, data)
  lstmStatus = lstm.forecast_price(start_date, stop_date, data)
  response = {}
  response['arima'] = arimaStatus
  response['lr'] = lrStatus
  response['lstm'] = lstmStatus
  return json.dumps(response)


@app.route('/linear_regression', methods=('GET',))
def linearRegressionForecast():
  return linear_regression.get_lr_json()


@app.route('/lstm', methods=('GET',))
def lstmForecast():
  return lstm.get_lstm_json()


@app.route('/arima', methods=('GET',))
def arimaForecast():
  return arima.get_arima_json()


if __name__ == '__main__':
  app.run(port=5002)
