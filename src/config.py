from helpers import Helper as hp
from regressors.lstm_regressor import LSTMRegressor
from regressors.esn_regressor import ESNRegressor
from decomposers.wavelet_transform import WaveletDecomposition
from decomposers.hilbert_huang     import EMDDecomposition
from chm.chm import CascadeHierarquicalModel
class Config():
    def __init__(self) -> None:
        pass    
    
    period = {
            'from' : '2010-01-01',
            'to'   : '2021-01-31'
    }

    candle_interval = '1d'

    ticker_stock_market_map = {
                # Brasil Bolsa Balcao
                'B3' : ['ITUB3.SA', 'ABEV3.SA', 'B3SA3.SA', ],
                # New York Stock Exchange
                'NYSE' : ['AAPL', 'MSFT', 'GOOG',],# 'AMZN', ],
                # Tokyo Stock Exchange (Toyota, Keyence, Sony)
                'TSE' : ['7203.T', '6861.T', '6758.T'], # ['7011.T',  Mitsubishi
                # Shangai Stock Exchange (Kweichow Moutai Co., Ltd., China Construction Bank Corporation, Agricultural Bank of China)
                'SSE' : ['600519.SS', '601939.SS', '601288.SS'],
                # Hong Kong Stock Exchange (Tencent Holdings Limited, Industrial and Commercial Bank of China Limited, China Merchants Bank Co., Ltd.)
                'HKSE' : ['0700.HK', '1398.HK', '3968.HK'],
                # London Stock Exchange (Unilever, AstraZeneca, HSBC)
                'LSE' : ['ULVR.L', 'AZN.L', 'HSBA.L'],
                # National Stock Exchange of India Ltd (Reliance Industries Limited, Tata Consultancy Services, HDFC Bank Limited)
                'NSEI' : ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
                # Frankfurt Stock Exchange (SAP, Siemens Aktiengesellschaft, Volkswagen)
                'FSE' : ['SAP.DE', 'SIE.DE', 'VOW3.DE'],
    }

    k_fold_variations = {
        '2010-2013T-2014V':{
            'start_train_year':2010,
            'end_train_year':2013,
            'validation_year' : 2014
        },
        '2013-2016T-2017V':{
            'start_train_year':2013,
            'end_train_year':2016,
            'validation_year' : 2017
        },
        '2016-2020T-2021V':{
            'start_train_year':2016,
            'end_train_year':2020,
            'validation_year' : 2021
        },
        '33p_train_test':{
            'start_train_year':None,
            'end_train_year':None,
            'validation_year' : None
        }
    }

    model_variations =  {
        'CHM-LSTM-EMD-Freq' :{
            'CHM' : True,
            'Regressor' : LSTMRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : True
        },
        'CHM-LSTM-DWT' :{
            'CHM' : True,
            'Regressor' : LSTMRegressor,
            'Decomposer' : WaveletDecomposition,
            'use_freq' : False
        },

        'CHM-ESN-DWT' :{
            'CHM' : True,
            'Regressor' : ESNRegressor,
            'Decomposer' : WaveletDecomposition,
            'use_freq' : False
        },

        'CHM-LSTM-EMD' :{
            'CHM' : True,
            'Regressor' : LSTMRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : False
        },

        'CHM-ESN-EMD' :{
            'CHM' : True,
            'Regressor' : ESNRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : False
        },


        'CHM-ESN-EMD-Freq' :{
            'CHM' : True,
            'Regressor' : ESNRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : True
        },

        'LSTM-EMD' :{
            'CHM' : False,
            'Regressor' : LSTMRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : False
        },

        'LSTM-DWT' :{
            'CHM' : False,
            'Regressor' : LSTMRegressor,
            'Decomposer' : WaveletDecomposition,
            'use_freq' : False
        },

        'ESN-EMD' :{
            'CHM' : False,
            'Regressor' : ESNRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : False
        },

        'ESN-DWT' :{
            'CHM' : False,
            'Regressor' : ESNRegressor,
            'Decomposer' : WaveletDecomposition,
            'use_freq' : False
        },

        'ESN-EMD-Freq' :{
            'CHM' : False,
            'Regressor' : ESNRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : True
        },

        'LSTM-EMD-Freq' :{
            'CHM' : False,
            'Regressor' : LSTMRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : True
        },
    }

    def get_tickers():
        tsmm = Config.ticker_stock_market_map
        return hp.flatten_list(tsmm.values())
        