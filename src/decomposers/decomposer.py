import abc

class BaseDecomposer(abc.ABC):
    """
    Abstract class for the decomposition methods

    parameters : 
                 
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def decompose_series(self, X) -> object:
        pass