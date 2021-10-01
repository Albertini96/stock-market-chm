import pandas
from chm import CascadeHierarquicalModel
from decomposers.hilbert_huang import EMDDecomposition
from regressors.lstm_regressor import LSTMRegressor
from data_retriever import DataRetriever
from pre_processing import PreProcessing
from scalers.min_max import MinMax
from scalers.scaler import Scaler
from scalers.standard import Standard

if __name__ == "__main__":
    
    dr = DataRetriever()
    pp = PreProcessing()
    yahoo_ds = dr.download_yahoo_data()
    scaled_yahoo_ds = pp.values_scaler_partition_by_2(yahoo_ds, 'Ticker', 'Var', 'value', MinMax)

    print(yahoo_ds)
    print(scaled_yahoo_ds)