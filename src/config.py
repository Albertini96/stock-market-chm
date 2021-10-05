from helpers import Helper as hp
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
                # 'SSE' : ['ITUB4', ],
                # # Hong Kong Stock Exchange
                # 'HKSE' : ['ITUB4', ],
                # # London Stock Exchange
                # 'LSE' : ['ITUB4', ],
                # # National Stock Exchange of India Ltd
                # 'NSEI' : ['ITUB4', ],
                # # Frankfurt Stock Exchange
                # 'FSE' : ['ITUB4', ],
    }

    def get_tickers():
        tsmm = Config.ticker_stock_market_map
        return hp.flatten_list(tsmm.values())
        