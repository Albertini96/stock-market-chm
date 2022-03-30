
from sklearn.model_selection import train_test_split
from chm.chm import CascadeHierarquicalModel
import config
from data_retriever import DataRetriever
from pre_processing import PreProcessing
from regressors.esn_regressor import ESNRegressor
from regressors.lstm_regressor import LSTMRegressor
from regressors.ann_regressor import ANNRegressor
import pandas as pd
import csv
import os
import warnings
warnings.filterwarnings("ignore")


from scalers.min_max import MinMax

if __name__ == "__main__":

    num_models = len(config.Config.model_variations.keys())
    kfolds_dict = config.Config.k_fold_variations

    res = dict()
    for pred_ticker in config.Config.ticker_stock_market_map['B3']:  
        if not os.path.exists('results_comp'):
            os.makedirs('results_comp')

        for kfold in kfolds_dict.keys():
            for model_type, model_index in zip(config.Config.model_variations.keys(), range(1,num_models -1)):
                print('________________________________________________________________')
                print('+++++++++', pred_ticker, ' +++++++++')
                print('+++++++++', kfold, ' +++++++++')
                print('+++++++++ ', model_type,  model_index , ' of ', num_models, ' +++++++++')
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
                predict_cols = pred_ticker  + '_Close'

                #Obtaining yahoo dataset
                dataset = a.get_stock_ds()

                #Preprocessing yahoo data
                pp = PreProcessing(dataset, MinMax)
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

                if kfold == '33p_train_test':
                    train, test = train_test_split(ds, test_size=0.33, shuffle=False)
                else:
                    train = ds[ds.apply(lambda x: (x['Date'].year >= kfolds_dict[kfold]['start_train_year']) & (x['Date'].year <= kfolds_dict[kfold]['end_train_year']), axis=1)]
                    test = ds[ds.apply(lambda x: (x['Date'].year == kfolds_dict[kfold]['validation_year']), axis=1)]
                

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
                    model = model_regressor(train, x_cols, y_cols, 7, len(x_cols), 300)
                    model.fit_generator()

                    pred = model.predict_generator(test).ravel()

                prediction = pp._scalers[y_cols].inverse_transform(pred.reshape(-1,1))

                prediction = prediction.ravel()

                res[model_type] = prediction

            with open('results_comp/' + kfold + '_' + pred_ticker + '.csv', 'w') as testfile:
            # pass the csv file to csv.writer function.
                writer = csv.writer(testfile)
            
                writer.writerow(res.keys())
            
                writer.writerows(zip(*res.values()))
    
