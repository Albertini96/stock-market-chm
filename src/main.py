
from sklearn.model_selection import train_test_split
from chm.chm import CascadeHierarquicalModel
import config
from data_retriever import DataRetriever
from pre_processing import PreProcessing as pp
from regressors.esn_regressor import ESNRegressor
from regressors.lstm_regressor import LSTMRegressor
from regressors.ann_regressor import ANNRegressor

from scalers.min_max import MinMax

if __name__ == "__main__":

    predictions_dict = dict()

    for model_type in config.Config.model_variations.keys():
        print('+++++++++ ', model_type, ' +++++++++')
        print()

        model_chm = config.Config.model_variations[model_type]['CHM']
        model_regressor = config.Config.model_variations[model_type]['Regressor']
        model_decomposer = config.Config.model_variations[model_type]['Decomposer']
        model_use_frq = config.Config.model_variations[model_type]['use_freq']


        #Retrieving data from yahoo API
        a = DataRetriever()
        a.get_yahoo_stock_data()

        #Defining which stocks will be predicted
        #x_cols_ = [x for x in b.columns.difference(['Date']) if x[-5:] == 'Close']
        predict_cols = 'ITUB3.SA_Close'

        #Obtaining yahoo dataset
        dataset = a.get_stock_ds()

        #Preprocessing yahoo data
        pp = pp(dataset, MinMax)
        ds = pp.pre_process_once()

        #Setting up column to be predicted
        y_cols = predict_cols
        #Setting up column to be used as features
        x_cols = ds.columns.difference(['Date'])

        # Decomposing series
        dec = model_decomposer()
        dec.decompose_series(
                                ds         = ds,
                                apply_cols = x_cols,
                                add_freq   = model_use_frq
                            )

        train, test = train_test_split(ds, test_size=0.33, shuffle=False)

        if model_chm:
            model = CascadeHierarquicalModel(
                                    train.copy(deep= True),
                                    y_cols            = predict_cols,
                                    context_regressor = model_regressor,
                                    stage_regressor   = ANNRegressor,
                                    decomposer        = dec,
                                    dec_cols          = x_cols,
                                    use_frequency     = model_use_frq
                                )

            model.train_context_extraction()
            pred = model.predict(test)
        else:
            model = model_regressor(train, x_cols, y_cols, 7, 77, 150)
            model.fit_generator()

            pred = model.predict_generator(test).ravel()

        prediction = pp._scalers[y_cols].inverse_transform(pred.reshape(-1,1))

        prediction = prediction.ravel()

        predictions_dict[model_type] = prediction
