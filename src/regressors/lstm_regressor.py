from typing import Any
from regressors.regressor import BaseRegressor
from pandas.core.frame import DataFrame
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SpatialDropout1D

class LSTMRegressor(BaseRegressor):
    """
    Performs the regression of time series using LSTMs

    parameters : 
                 
    """
    def __init__(self, n_inputs, n_features, epochs) -> None:
        self._train_history = None
        self._epochs        = epochs

        self._regressor = Sequential()
        self._regressor.add(Dense(50))
        self._regressor.add(LSTM(100, input_shape=(n_inputs, n_features)))
        self._regressor.add(Dense(40))
        self._regressor.add(Dense(1))
    
        self._regressor.compile(
            loss='mse',
            optimizer='adam'
            )

    def fit(self, X, y) -> None:
        self._train_history = self._regressor.fit(
                                                X,
                                                y,
                                                epochs = 8
                                            )
    def fit_generator(self, gen) -> None:
        self._train_history = self._regressor.fit(
                                                gen,
                                                epochs = self._epochs
                                            )

    def predict(self, X) -> Any:
        return  self._regressor.predict(X)

    def predict_generator(self, X) -> Any:
        return  self._regressor.predict(X)