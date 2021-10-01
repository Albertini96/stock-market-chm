from typing import Any
from scalers.scaler import Scaler
from sklearn.preprocessing import MinMaxScaler

class MinMax(Scaler):
    """
    Performs scaling of data using min max scaler

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """
    def __init__(self) -> None:
        self._scaler = MinMaxScaler()

    def fit(self, X) -> MinMaxScaler:
        return self._scaler.fit(X)

    def transform(self, X) -> Any:
        return self._scaler.transform(X)

    def fit_transform(self, X) -> Any:
        return self._scaler.fit_transform(X) 


