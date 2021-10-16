from numpy.lib.function_base import copy
import pandas as pd
from pandas.core.frame import DataFrame
import config
from pandas_datareader import data as pdr
from helpers import Helper as hp
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

class DataRetriever:
    def __init__(self) -> None:
        self.stock_ds = None
        pass

    def get_stock_ds(self) -> DataFrame:
        return self.stock_ds.copy()

    def get_yahoo_stock_data(self) -> None:
        #Getting dictionary of stocks and associating with stock market
        tickers = config.Config.get_tickers()
        
        yahoo_stock_data = self._download_yahoo_data(tickers)
        yahoo_stock_data.columns = ['_'.join(col).strip() for col in yahoo_stock_data.columns.values]

        self.stock_ds = yahoo_stock_data.reset_index()

    def _download_yahoo_data(self, in_tickers) -> DataFrame:
        data = pdr.get_data_yahoo( 
            # tickers list or string as well
            tickers = in_tickers,

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

        return data