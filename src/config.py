class Config():
    def __init__(self) -> None:
        pass    
    
    period = {
            'from' : '2010-01-01',
            'to'   : '2017-04-30'
    }

    candle_interval = '1d'

    ticker_stock_market_map= {
                #Brasil Bolsa Balcao
                'B3' : ['ITUB3.SA', 'ABEV3.SA', 'B3SA3.SA'],
                #New York Stock Exchange
                'NYSE' : ['AAPL', 'MSFT', 'AMZN'],
                #Tokyo Stock Exchange
                'TSE' : ['7011.T', ],
                #Shangai Stock Exchange
                #ticker_stock_market_map['SSE'] = ['ITUB4', ]
                #Hong Kong Stock Exchange
                #ticker_stock_market_map['HKSE'] = ['ITUB4', ]
                #London Stock Exchange
                #ticker_stock_market_map['LSE'] = ['ITUB4', ]
                #National Stock Exchange of India Ltd
                #ticker_stock_market_map['NSEI'] = ['ITUB4', ]
                #Frankfurt Stock Exchange
                #ticker_stock_market_map['FSE'] = ['ITUB4', ]
    }