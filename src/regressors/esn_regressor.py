from typing import Any

from keras.preprocessing.sequence import TimeseriesGenerator
from regressors.regressor import BaseRegressor
from keras.models import Sequential
from keras.layers import Dense
from tensorflow_addons.layers import ESN

class ESNRegressor(BaseRegressor):
    """
    Performs the regression of time series using ESNs

    parameters : object (regressor used to extract context)
                 object (decomposition used to extract frequency from time series)
                 
    """

    def __init__(self, ds, x_cols, y_cols, n_inputs, n_features, epochs) -> None:
        self._train_history = None
        self.n_inputs       = n_inputs
        self.n_features     = n_features
        self._epochs        = epochs
        self.ds             = ds
        self.x_cols         = x_cols
        self.y_cols         = y_cols


        self._regressor = Sequential()
        self._regressor.add(ESN(600, input_shape=(n_inputs, n_features)))
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
    def fit_generator(self) -> None:
        train_ts_gen = TimeseriesGenerator(self.ds[self.x_cols].to_numpy(), self.ds[self.y_cols].to_numpy(), self.n_inputs)

        self._train_history = self._regressor.fit(
                                                train_ts_gen,
                                                epochs = self._epochs,
                                                verbose = 0
                                            )

    def predict(self, X) -> Any:
        return  self._regressor.predict(X)

    def predict_generator(self, X) -> Any:
        test_ts_gen  = TimeseriesGenerator(X[self.x_cols].to_numpy(), X[self.y_cols].to_numpy(), self.n_inputs)
        return  self._regressor.predict(test_ts_gen)
