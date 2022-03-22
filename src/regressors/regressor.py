from abc import ABC, abstractclassmethod
from typing import Any
from pandas.core.frame import DataFrame

class BaseRegressor(ABC):
    """
    Performs the decomposition of time series using hilbert huang decomposition

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def fit(self, X, y) -> None:
        pass

    @abstractclassmethod
    def fit_generator(self, X) -> None:
        pass

    @abstractclassmethod
    def predict(self, X) -> Any:
        pass

    @abstractclassmethod
    def predict_generator(self, X, predict_last = False) -> Any:
        pass