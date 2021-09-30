class CascadeHierarquicalModel():
    """
    Performs the extraction of time series context

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 bool   (use frequency extracted from time series as a feature)
    
    returns : int (the result is the addition of the two numbers)"""
    
    def __init__(self, context_regressor, decomposer, use_frequency=False):
        self._context_regressor = context_regressor
        self._decomposer        = decomposer
        self._use_frequency     = use_frequency

