from abc import ABC, abstractclassmethod


class BaseRegressor(ABC):
    """
    Performs the decomposition of time series using hilbert huang decomposition

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def fit(self, X, y) -> object:
        pass

    @abstractclassmethod
    def predict(self, X) -> object:
        pass