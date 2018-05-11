import quandl


def get_btc_data():
    quandl.ApiConfig.api_key = "7nx8VgA8ahjtyif-oUTW"
    data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
    return data