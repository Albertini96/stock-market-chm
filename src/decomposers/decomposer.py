from abc import ABC, abstractclassmethod
import array
from typing import List
from pandas.core.frame import DataFrame
class BaseDecomposer(ABC):
    """
    Abstract class for the decomposition methods

    parameters : 
                 
    """
    def __init__(self) -> None:
        self._dict_waves = dict()

    @abstractclassmethod
    def decompose_series(self, 
                        ds:DataFrame, 
                        apply_cols:List[str]
                        ) -> object:
        pass