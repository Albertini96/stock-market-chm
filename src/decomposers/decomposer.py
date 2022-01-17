import abc
import array
from pandas.core.frame import DataFrame
class BaseDecomposer(abc.ABC):
    """
    Abstract class for the decomposition methods

    parameters : 
                 
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def decompose_series(self, 
                        ds:DataFrame, 
                        apply_cols:array[str]
                        ) -> object:
        pass