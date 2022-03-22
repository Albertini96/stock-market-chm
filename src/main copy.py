from config import Config
from data_retriever import DataRetriever
from pre_processing import PreProcessing as pp
from scalers.min_max import MinMax

if __name__ == "__main__":
    
    #Obtaining stock market data
    a = DataRetriever()
    b = a.get_yahoo_stock_data()    

    #Filling missings
    pp.fill_stock_data_missings(b, True)

    #Adding features
    clse_cols = [x for x in b.columns.difference(['Date']) if x[-5:] == 'Close']
    pp.add_bollinger_bands(b, clse_cols, True, False)
    pp.add_sma(b, clse_cols, True, 5, True)
    pp.add_rsi(b, clse_cols, True, save_pictures=True)
    pp.add_stochastic_oscilator(b, Config.Config.get_tickers(), inplace=True, save_pictures=True)

    #Scaling values
    pp.values_scaler(b, b.columns.difference(['Date']), MinMax, True)
