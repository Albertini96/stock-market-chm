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
                    context_regressor:BaseRegressor,
                    start_index:int = None
                ) -> None:
        self._regressor           = context_regressor
        self.gerenerated_context  = None
        self.generated_prediction = None
        self._start_index         = start_index

    def train_regressor():
        pass

    def predict_regressor():
        pass
    
    def train(self):
        self._regressor.fit_generator()

    def train2(self, X, y):
        self._regressor.fit(X, y)

    def create_context(self, X:DataFrame, predict_last = False):
        self.gerenerated_context = self._regressor.predict_generator(X, predict_last)

    def create_context2(self, X:DataFrame, predict_last = False):
        self.gerenerated_context = self._regressor.predict(X)

    def create_prediction(self, X:DataFrame, predict_last = False):
        self.generated_prediction = self._regressor.predict_generator(X, predict_last)

    def create_prediction2(self, X:DataFrame, predict_last = False):
        self.generated_prediction = self._regressor.predict(X)