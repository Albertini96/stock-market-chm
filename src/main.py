from data_retriever import DataRetriever
from pre_processing import PreProcessing as pp
from scalers.min_max import MinMax

if __name__ == "__main__":
    
    a = DataRetriever()
    b = a.get_yahoo_stock_data()    

    pp.values_scaler(b, b.columns.difference(['Date']), MinMax, True)
    pp.fill_stock_data_missings(b, True)