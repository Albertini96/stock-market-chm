from typing                 import List
from chm.level              import CHMLevel
from chm.stages             import CHMStage
from decomposers.decomposer import BaseDecomposer
from regressors.regressor   import BaseRegressor
from copy import copy

class CascadeHierarquicalModel():
    """
    Cascade Hierarquical Model used to extract context from timeseries and predict non stationary timeseries

    parameters : BaseRegressor    (Instanciated base regressor used to extract context)
                 BaseRegressor    (Instanciated base regressor used to join context created from the stages)
                 SeriesDecomposer (decomposition used to extract frequency from time series)
                 int              (number of levels in the hierarquical model)
                 int              (number of stages in the hierarquical model)
                 bool             (use frequency extracted from time series as a feature)
    
    """
    
    def __init__(self,  context_regressor:BaseRegressor   , 
                        stage_regressor:BaseRegressor     ,
                        decomposer:BaseDecomposer         , 
                        num_levels:int=10                 ,
                        num_stages:int=1                  ,
                        use_frequency:bool=False
                ):
        self._context_regressor = context_regressor
        self._decomposer        = decomposer
        self._num_levels        = num_levels
        self._num_stages        = num_stages
        self._use_frequency     = use_frequency
        self._list_stages       = list[CHMStage]

        for stage in range(self._num_stages):
            temp_level_list = list[CHMLevel]
            for level in range(self._num_levels):
                temp_level_list.append(CHMLevel(copy(context_regressor)))
            self._list_stages.append(CHMStage(temp_level_list, copy(stage_regressor)))

        

    def extract_context(self) -> None:
        pass

    def __sintetize_series(self):
        pass

    def __recompose_series(self):
        pass

    