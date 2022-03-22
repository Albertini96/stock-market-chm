from typing import Any

from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from regressors.regressor import BaseRegressor
from keras.models import Sequential
from keras.layers import Dense

class ANNRegressor(BaseRegressor):
    """
    Performs the regression of time series using LSTMs

    parameters : 
                 
    """
    def __init__(self, 
                    ds = None, 
                    x_cols = None, 
                    y_cols = None, 
                    epochs = None
                ) -> None:
        self._train_history = None
        self._epochs        = epochs
        self.ds             = ds
        self.x_cols         = x_cols
        self.y_cols         = y_cols


        self._regressor = Sequential()
        self._regressor.add(Dense(100))
        self._regressor.add(Dense(1))
    
        self._regressor.compile(
            loss='mse',
            optimizer='adam'
            )

    def fit(self, X, y) -> None:
        self._train_history = self._regressor.fit(
                                                X,
                                                y,
                                                epochs = self._epochs,
                                                verbose = 0
                                            )
    def fit_generator(self) -> None:
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self, X) -> Any:
        return  self._regressor.predict(X)

    def predict_generator(self, X, predict_last = False) -> Any:
        raise NotImplementedError("Subclasses should implement this!")