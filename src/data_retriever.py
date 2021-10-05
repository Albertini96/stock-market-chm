import pandas as pd
from pandas.core.frame import DataFrame
import config
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

class DataRetriever:
    def __init__(self) -> None:
        pass

    def download_yahoo_data(self) -> DataFrame:
        #Getting dictionary of stocks and associating with stock market
        tsmm = config.Config.ticker_stock_market_map
        
        #Downloading data from yFinance
        stock_data_set = pd.DataFrame()

        for i in tsmm.keys():
            data_temp = self._get_yahoo_stock_data(tsmm[i])
            data_temp['stock_market'] = i
            stock_data_set = stock_data_set.append(data_temp)
        
        return stock_data_set

    def _get_yahoo_stock_data(self, in_tickers) -> DataFrame:
        data = pdr.get_data_yahoo( 
            # tickers list or string as well
            tickers = in_tickers,

            # use "period" instead of start/end
            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            # (optional, default is '1mo')
            #period = "max",

            start = config.Config.period['from'], 
            
            end = config.Config.period['to'],

            # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            # (optional, default is '1d')
            interval = config.Config.candle_interval,

            # group by ticker (to access via data['SPY'])
            # (optional, default is 'column')
            group_by = 'ticker',

            # adjust all OHLC automatically
            # (optional, default is False)
            auto_adjust = True,

            # download pre/post regular market hours data
            # (optional, default is False)
            prepost = True,

            # use threads for mass downloading? (True/False/Integer)
            # (optional, default is True)
            threads = True,

            # proxy URL scheme use use when downloading?
            # (optional, default is None)
            proxy = None
        )

        # Transforming data to one column value

        #Case number of tickers = 1
        if len(in_tickers) == 1:
            # Removing Date from index
            data.reset_index(inplace=True)

            # Moving OHLC values to rows
            data = pd.melt(data, id_vars= data.columns[:1], value_vars=data.columns[1:])

            # Renaming column variables
            data.rename(columns={'level_1' : 'Var',
                            'variable' : 'Var'}
                            , inplace=True)

            # Adding Ticker name
            data['Ticker'] = in_tickers[0]
        else:

            # Moving OHLC values to rows
            data = data.stack().reset_index()

            # Moving ticker to rows
            data = pd.melt(data, id_vars = data.columns[:2] , value_vars = data.columns[2:])

            # Renaming column variables
            data.rename(columns={'level_1' : 'Var',
                        'variable' : 'Ticker'}
                        , inplace=True)

        return data