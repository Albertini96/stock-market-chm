from decomposers.decomposer import SeriesDecomposer
from regressors.regressor   import ContextRegressor
class CascadeHierarquicalModel():
    """
    Performs the extraction of time series context

    parameters : ContextRegressor (regressor used to extract context)
                 SeriesDecomposer (decomposition used to extract frequency from time series)
                 int    (number of levels in the hierarquical model)
                 int    (number of stages in the hierarquical model)
                 bool   (use frequency extracted from time series as a feature)
    
    """
    
    def __init__(self, context_regressor:ContextRegressor , 
                        decomposer:SeriesDecomposer       , 
                        num_levels:int=10                 ,
                        num_stages:int=1                  ,
                        use_frequency:bool=False
                ):
        self._context_regressor = context_regressor
        self._decomposer        = decomposer
        self._num_levels        = num_levels
        self._num_stages        = num_stages
        self._use_frequency     = use_frequency

    def extract_context(self) -> None:
        pass

    def __sintetize_series(self):
        pass

    def __recompose_series(self):
        pass

    