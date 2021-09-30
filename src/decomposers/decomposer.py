import abc

class SeriesDecomposer(abc.ABC):
    """
    Abstract class for the decomposition methods

    parameters : 
                 
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def decompose_series(self) -> object:
        pass