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
                'NYSE' : ['AAPL', 'MSFT', 'AMZN', ],
                # Tokyo Stock Exchange
                'TSE' : ['7011.T', ],
                # Shangai Stock Exchange
                # 'SSE' : ['ITUB4', 'ITUB4', 'ITUB4'],
                # # Hong Kong Stock Exchange
                # 'HKSE' : ['ITUB4', 'ITUB4', 'ITUB4'],
                # # London Stock Exchange
                # 'LSE' : ['ITUB4', 'ITUB4', 'ITUB4'],
                # # National Stock Exchange of India Ltd
                # 'NSEI' : ['ITUB4', 'ITUB4', 'ITUB4'],
                # # Frankfurt Stock Exchange
                # 'FSE' : ['ITUB4', 'ITUB4', 'ITUB4'],
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

        'LSTM-EMD' :{
            'CHM' : False,
            'Regressor' : ESNRegressor,
            'Decomposer' : EMDDecomposition,
            'use_freq' : False
        },

        'LSTM-DWT' :{
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
        