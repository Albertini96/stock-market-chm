from typing import List
from pandas.core.frame import DataFrame
from chm.level import CHMLevel
from regressors.regressor import BaseRegressor

class CHMStage():
        
    """
        
    
    """

    def __init__(self, level_list:List[CHMLevel],
                       stage_regressor:BaseRegressor,
                       X:DataFrame,
                       y:DataFrame
                    ) -> None:
        self._level_list = level_list
        self._regressor  = stage_regressor