from typing import Any
from scalers.scaler import Scaler
from sklearn.preprocessing import StandardScaler

class Standard(StandardScaler):
    """
    Performs scaling of data using standard scaler

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """
    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit(self, X) -> StandardScaler:
        return self._scaler.fit(X)

    def transform(self, X) -> Any:
        return self._scaler.transform(X)

    def fit_transform(self, X) -> Any:
        return self._scaler.fit_transform(X) 