from typing import Any

from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
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

    def __init__(self, ds= None, 
                       x_cols= None, 
                       y_cols= None, 
                       n_inputs= None, 
                       n_features= None, 
                       epochs= None,
                ) -> None:
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
                                                epochs = self._epochs,
                                                verbose = 0
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

    def predict_generator(self, X, predict_last = False) -> Any:
        test_ts_gen = None
        if predict_last == True:
            test_ts_gen  = timeseries_dataset_from_array(X[self.x_cols].to_numpy(), None, self.n_inputs)
        else:
            test_ts_gen  = timeseries_dataset_from_array(X[self.x_cols].to_numpy(), None, self.n_inputs, end_index=len(X[self.x_cols].to_numpy())-1)
            
        return  self._regressor.predict(test_ts_gen)