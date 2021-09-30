from abc import abstractclassmethod


class ContextRegressor():
    """
    Performs the decomposition of time series using hilbert huang decomposition

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def extract_context(self) -> object:
        pass

