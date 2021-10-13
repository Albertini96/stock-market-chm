from abc import ABC, abstractclassmethod
from typing import Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BaseScaler(ABC):
    """
    Performs the scaling of data

    """

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def fit(self, X) -> Union[MinMaxScaler, StandardScaler]:
        pass

    @abstractclassmethod
    def transform(self, X) -> Any:
        pass

    @abstractclassmethod
    def fit_transform(self, X) -> Any:
        pass

    