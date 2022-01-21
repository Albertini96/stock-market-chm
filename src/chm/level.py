from pandas.core.frame import DataFrame
from regressors.regressor import BaseRegressor

class CHMLevel():
        
    """
        Level object of a CHM structure

        parameters : BaseRegressor (Regressor used to extract context for this level)
                     DataFrame     (X used on regressor)
                     DataFrame     (y used on regressor)

        returns : 
    """


    def __init__(self,
                    context_regressor:BaseRegressor
                ) -> None:
        self._regressor = context_regressor
        