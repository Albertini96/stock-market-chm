from typing import List
from pandas.core.frame import DataFrame
from chm.level import CHMLevel
from regressors.regressor import BaseRegressor

class CHMStage():
        
    """
        Stage object of a CHM structure

        parameters : List[CHMLevel]  (List of CHM Levels contained on this stage)    
                     BaseRegressor   (Regressor used to join context created for this stage)
                     DataFrame       (X used on regressor)
                     DataFrame       (y used on regressor)

        returns : 
    """

    def __init__(self, level_list:List[CHMLevel],
                       stage_regressor:BaseRegressor,
                       _max_level_context_len:int = None
                    ) -> None:
        self._level_list            = level_list
        self._regressor             = stage_regressor
        self._max_level_context_len = _max_level_context_len