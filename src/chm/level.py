from pandas.core.frame import DataFrame
from regressors.regressor import BaseRegressor

class CHMLevel():
        
    """
        docstring for ClassName.
    
    """


    def __init__(self,
                    context_regressor:BaseRegressor,
                    X:DataFrame,
                    y:DataFrame
                ) -> None:
        self._regressor = context_regressor
        