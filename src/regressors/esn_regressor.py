from regressor import BaseRegressor

class ESNRegressor(BaseRegressor):
    """
    Performs the regression of time series using ESNs

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """

