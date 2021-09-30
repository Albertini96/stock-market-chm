from regressor import ContextRegressor

class ESNRegressor(ContextRegressor):
    """
    Performs the regression of time series using LSTMs

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """