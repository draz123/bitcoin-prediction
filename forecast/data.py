import quandl
import pandas

def get_btc_data():
    # quandl.ApiConfig.api_key = "7nx8VgA8ahjtyif-oUTW"
    # data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
    # with open('resources/data.json', 'w') as outfile:
    #   data.to_csv(outfile)
    with open('resources/data.json', 'r') as outfile:
      data = pandas.DataFrame.from_csv(outfile)
    return data
