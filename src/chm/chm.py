from typing                 import List

import numpy as np
from chm.level              import CHMLevel
from chm.stages             import CHMStage
from decomposers.decomposer import BaseDecomposer
from regressors.regressor   import BaseRegressor
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from pandas.core.frame import DataFrame
from copy import deepcopy
import seaborn as sns
from matplotlib import pyplot as plt
import os
import pandas as pd
from chm.level              import CHMLevel
from chm.stages             import CHMStage
from decomposers.decomposer import BaseDecomposer
from regressors.regressor   import BaseRegressor


class CascadeHierarquicalModel():
    """
    Cascade Hierarquical Model used to extract context from timeseries and predict non stationary timeseries

    parameters : BaseRegressor    (Instanciated base regressor used to extract context)
                 BaseRegressor    (Instanciated base regressor used to join context created from the stages)
                 SeriesDecomposer (decomposition used to extract frequency from time series)
                 int              (number of levels in the hierarquical model)
                 int              (number of stages in the hierarquical model)
                 bool             (use frequency extracted from time series as a feature)
    
    """
    
    def __init__(self,  ds:DataFrame                      ,
                        # x_cols:List[str]                  ,
                        y_cols:str                        ,
                        context_regressor:BaseRegressor   , 
                        stage_regressor:BaseRegressor     ,
                        decomposer:BaseDecomposer         ,
                        dec_cols:List[str]                ,
                        num_stages:int=1                  ,
                        use_frequency:bool=False          ,
                        verbose=True,
                        epochs= 150
                ):
        self._ds                                   = ds
        # self._x_cols                             = x_cols
        self._y_cols                               = y_cols
        self._context_regressor                    = context_regressor
        self._stage_regressor                      = stage_regressor
        self._dec                                  = decomposer
        self._dec_cols                             = dec_cols
        self._num_stages                           = num_stages
        self._use_frequency                        = use_frequency
        self._list_stages_bottom_up:List[CHMStage] = list()
        self._list_levels_frequency:List[CHMLevel] = list()
        self._verbose                              = verbose
        self._n_inputs                             = 7
        self._epochs                               = epochs
        
        # Obtaining the biggest decomposition
        self._max_wave = len(self._dec.dict_waves[y_cols])
        
    def train_context_extraction(self) -> None:
        self._set_bottom_up_block()
        self._set_frequency_predictor_block()
        self._train_frequency_block()
        self._set_top_down_block()
        self._train_top_down_block()
    
    def predict(self, X:DataFrame) -> List[float]:
        self._predict_bottom_up(X)
        self._predict_frequencies(X)
        return self._predict_top_down()


    def _predict_bottom_up(self, X:DataFrame):
        if(self._verbose):
            print('============================================================')
            print('Predicting bottom up block ...')

        # Flatting subwave column name dictionary
        sub_waves = list() 
        for i in self._dec.dict_waves.keys():
            sub_waves = sub_waves + self._dec.dict_waves[i]

        # For each stage
        for stage, stage_num in zip(self._list_stages_bottom_up, range(len(self._list_stages_bottom_up))):
            max_len = 0

            # Copying temporary dict of waves
            temp_dec_dict = deepcopy(self._dec.dict_waves)

            if self._use_frequency:
                # Copying temporary dict of frequencies
                temp_freq_dict = deepcopy(self._dec.dict_freq)

            # List of levels for this stage
            temp_level_list:List[CHMLevel] = list()

            # Set last level var
            last_level:CHMLevel = None

            # For each level in the current stage
            for level, level_num in zip(stage._level_list, range(len(stage._level_list))):
                if(self._verbose):
                    print('Creating Context > ', 'Level > ', level_num, ' of Stage > ', stage_num)
                
                temp_ds = X.copy(deep=True)

                # Creating dataset for this level
                for wave in temp_dec_dict.keys():

                    # Initiating sintetized column of wave
                    temp_ds[wave + '_sintetized'] = 0   

                    # For each decomposed subwave of current wave
                    for col in temp_dec_dict[wave]:
                        temp_ds[wave + '_sintetized'] = temp_ds[wave + '_sintetized'] + temp_ds[col]

                if self._use_frequency:
                    # For each frequency subwave of current wave
                    for col in temp_freq_dict.keys():
                        temp_ds[wave + '_frequency'] = temp_ds[col][0]

                # Leaving only sintetized columns
                # Dropping used subwave
                temp_ds.drop(sub_waves, axis=1, inplace=True)
                
                # Dropping original waves
                temp_ds.drop(self._dec_cols, axis=1, inplace=True)
                ##

                # List of features of current level (X Cols)
                curr_feat_cols = temp_ds.columns.difference(['Date'])

                # Saving Y from dataset
                temp_ds_y = temp_ds[self._y_cols + '_sintetized']
                
                # Transforming X Cols features to 3 dimensional tensor
                temp_ds_x  = timeseries_dataset_from_array(data=temp_ds[curr_feat_cols].to_numpy(), 
                                                                targets=None, 
                                                                sequence_length=self._n_inputs,
                                                                sequence_stride=1,
                                                                batch_size=len(temp_ds[curr_feat_cols].to_numpy()))

                # Casting Tensor to list
                temp_ds_x = next(temp_ds_x.as_numpy_iterator())
                temp_ds_x = temp_ds_x.tolist()

                curr_n_features = len(curr_feat_cols)

                # Adding prediction from last level if not first level
                if last_level is not None:
                    curr_n_features += 1
                    for line, gen_ctxt in zip(temp_ds_x, last_level.generated_prediction.ravel().tolist()):
                        for line_in_line in line:
                            line_in_line.append(gen_ctxt) 
                
                # Creating context for level
                level.create_prediction2(temp_ds_x[:-1])
    
                last_level = level

                for k in temp_dec_dict.keys():
                    if len(temp_dec_dict[k]) > 1:
                        temp_dec_dict[k].pop()

                if self._use_frequency:
                    for k in temp_freq_dict.keys():
                        if len(temp_freq_dict[k]) > 1:
                            temp_freq_dict[k] = temp_freq_dict[k][1:]

    def _predict_frequencies(self, X:DataFrame) -> None:
        if(self._verbose):
            print('============================================================')
            print('Setting up frequency predictor block ...')

        sub_waves = list() 
        for i in self._dec.dict_waves.keys():
            sub_waves = sub_waves + self._dec.dict_waves[i]
        
        # For each level of frequency of target
        for level, level_num in zip(self._list_levels_frequency, range(len(self._list_levels_frequency))):
            temp_ds = X.copy(deep=True)

            if(self._verbose):
                print('Setting up predictor of ', self._y_cols, ' frequency > ', level_num)

            # Creating dataset for this level
            cols = list()
            for wave in self._dec.dict_waves.keys():
                max_len = len(self._dec.dict_waves[wave])
                if level_num >= max_len:
                    cols.append(self._dec.dict_waves[wave][max_len-1])
                else:
                    cols.append(self._dec.dict_waves[wave][level_num])

            drop_cols = list(filter(lambda x: x not in cols, sub_waves))
            
            # Dropping unused frequency waves
            temp_ds.drop(drop_cols, axis=1, inplace=True)
            
            # Dropping original waves
            temp_ds.drop(self._dec_cols, axis=1, inplace=True)

            # List of features of current level
            curr_feat_cols = temp_ds.columns.difference(['Date'])

            level.create_prediction(temp_ds[curr_feat_cols])

    def _predict_top_down(self) -> List[float]:
        if(self._verbose):
            print('============================================================')
            print('Setting top-down regressors ...')
        for stage, stage_number in zip(self._list_stages_bottom_up, range(len(self._list_stages_bottom_up))):
            stage_ds = pd.DataFrame()
            if self._verbose:
                print('Creating topdown regressor of Stage > ', stage_number)

            for level, level_number in zip(stage._level_list, range(len(stage._level_list))):
                stage_ds['context_level_' + str(level_number)] = level.generated_prediction.ravel()
            
            for frequency, frequency_number in zip(self._list_levels_frequency, range(len(self._list_levels_frequency))):
                stage_ds['frequency_level_' + str(frequency_number)] = frequency.generated_prediction.ravel()

            top_down_ds = pd.DataFrame()
            for level in range(len(stage._level_list)):
                top_down_ds['input_level_' + str(level)] = stage_ds['context_level_' + str(level)]
                for frequency_level, frequency_reversed in zip(range(len(stage._level_list)), reversed(range(len(stage._level_list)))):
                    if frequency_level < level :
                        top_down_ds['input_level_' + str(level)] = top_down_ds['input_level_' + str(level)] + stage_ds['frequency_level_' + str(frequency_reversed)]
            
            curr_feat_cols = top_down_ds.columns.difference([self._y_cols])

            return stage._regressor.predict(top_down_ds[curr_feat_cols])


    def _train_top_down_block(self) -> None:
        if(self._verbose):
            print('============================================================')
            print('Training top-down regressors ...')
        for stage, stage_number in zip(self._list_stages_bottom_up, range(len(self._list_stages_bottom_up))):
            stage._regressor.fit(stage._regressor.ds[stage._regressor.x_cols], 
                                stage._regressor.ds[stage._regressor.y_cols])
            if self._verbose:
                print('Training topdown regressor of Stage > ', stage_number)

    def _set_top_down_block(self) -> None:
        if(self._verbose):
            print('============================================================')
            print('Setting top-down regressors ...')
        for stage, stage_number in zip(self._list_stages_bottom_up, range(len(self._list_stages_bottom_up))):
            stage_ds = pd.DataFrame()
            if self._verbose:
                print('Creating topdown regressor of Stage > ', stage_number)

            for level, level_number in zip(stage._level_list, range(len(stage._level_list))):
                stage_ds['context_level_' + str(level_number)] = level.gerenerated_context.ravel()[-stage._max_level_context_len:]
            
            for frequency, frequency_number in zip(self._dec.dict_waves[self._y_cols], range(len(self._dec.dict_waves[self._y_cols]))):
                stage_ds['frequency_level_' + str(frequency_number)] = self._ds[frequency].to_list()[-stage._max_level_context_len:]

            top_down_ds = pd.DataFrame()
            for level in range(len(stage._level_list)):
                top_down_ds['input_level_' + str(level)] = stage_ds['context_level_' + str(level)]
                for frequency_level, frequency_reversed in zip(range(len(stage._level_list)), reversed(range(len(stage._level_list)))):
                    if frequency_level < level :
                        top_down_ds['input_level_' + str(level)] = top_down_ds['input_level_' + str(level)] + stage_ds['frequency_level_' + str(frequency_reversed)]
            
            top_down_ds[self._y_cols] = self._ds[self._y_cols].to_list()[-stage._max_level_context_len:]
            curr_feat_cols = top_down_ds.columns.difference([self._y_cols])
    
            stage._regressor = self._stage_regressor(
                                                        ds=top_down_ds, 
                                                        x_cols=curr_feat_cols, 
                                                        y_cols=self._y_cols, 
                                                        epochs=self._epochs
                                                    )
        

    def _set_bottom_up_block(self) -> None:
        if(self._verbose):
            print('============================================================')
            print('Setting up bottom up block ...')

        # Flatting subwave column name dictionary
        sub_waves = list() 
        for i in self._dec.dict_waves.keys():
            sub_waves = sub_waves + self._dec.dict_waves[i]

        # For each stage
        for stage in range(self._num_stages):
            max_len = 0

            # Copying temporary dict of waves
            temp_dec_dict = deepcopy(self._dec.dict_waves)

            if self._use_frequency:
                # Copying temporary dict of frequencies
                temp_freq_dict = deepcopy(self._dec.dict_freq)

            # List of levels for this stage
            temp_level_list:List[CHMLevel] = list()

            # Set last level var
            last_level:CHMLevel = None

            # For each level in the current stage
            for level in range(self._max_wave):
                if(self._verbose):
                    print('Creating Context > ', 'Level > ', level, ' of Stage > ', stage)
                
                temp_ds = self._ds.copy(deep=True)

                # Creating dataset for this level
                for wave in temp_dec_dict.keys():

                    # Initiating sintetized column of wave
                    temp_ds[wave + '_sintetized'] = 0   

                    # For each decomposed subwave of current wave
                    for col in temp_dec_dict[wave]:
                        temp_ds[wave + '_sintetized'] = temp_ds[wave + '_sintetized'] + temp_ds[col]

                if self._use_frequency:
                    # For each frequency subwave of current wave
                    for col in temp_freq_dict.keys():
                        temp_ds[wave + '_frequency'] = temp_ds[col][0]

                # Leaving only sintetized columns
                # Dropping used subwave
                temp_ds.drop(sub_waves, axis=1, inplace=True)
                
                # Dropping original waves
                temp_ds.drop(self._dec_cols, axis=1, inplace=True)
                ##

                # List of features of current level (X Cols)
                curr_feat_cols = temp_ds.columns.difference(['Date'])

                # Saving Y from dataset
                temp_ds_y = temp_ds[self._y_cols + '_sintetized']
                
                # Transforming X Cols features to 3 dimensional tensor
                temp_ds_x  = timeseries_dataset_from_array(data=temp_ds[curr_feat_cols].to_numpy(), 
                                                                targets=None, 
                                                                sequence_length=self._n_inputs,
                                                                sequence_stride=1,
                                                                batch_size=len(temp_ds[curr_feat_cols].to_numpy()))

                # Casting Tensor to list
                temp_ds_x = next(temp_ds_x.as_numpy_iterator())
                temp_ds_x = temp_ds_x.tolist()

                curr_n_features = len(curr_feat_cols)

                # Adding prediction from last level if not first level
                if last_level is not None:
                    curr_n_features += 1
                    for line, gen_ctxt in zip(temp_ds_x, last_level.gerenerated_context.ravel().tolist()):
                        for line_in_line in line:
                            line_in_line.append(gen_ctxt) 

                
                # Instantiating CHM Level
                temp_level = CHMLevel(self._context_regressor(
                                                                n_inputs=self._n_inputs, 
                                                                n_features=curr_n_features, 
                                                                epochs=self._epochs
                                                             )
                                        )

                # Training Level # Ignoring last sequence of features
                temp_level.train2(temp_ds_x[:-1], temp_ds_y[self._n_inputs:].to_list())

                # Creating context for level
                temp_level.create_context2(temp_ds_x)

                # Appending level to list of levels of stage
                temp_level_list.append(temp_level)

                last_level = temp_level

                max_len = len(temp_level.gerenerated_context)

                for k in temp_dec_dict.keys():
                    if len(temp_dec_dict[k]) > 1:
                        temp_dec_dict[k].pop()

                if self._use_frequency:
                    for k in temp_freq_dict.keys():
                        if len(temp_freq_dict[k]) > 1:
                            temp_freq_dict[k] = temp_freq_dict[k][1:]
                    

            self._list_stages_bottom_up.append(CHMStage(temp_level_list, None, max_len))

    def _train_frequency_block(self) -> None:
        if(self._verbose):
            print('============================================================')
            print('Training frequencies regressors ...')
        for level, level_number in zip(self._list_levels_frequency, range(len(self._list_levels_frequency))):
            if self._verbose:
                print('Creating Context > ', 'Level > ', level_number, ', of Frequency Block')  
            level.train()
            # Creating context for level
            level.create_context(level._regressor.ds[level._regressor.x_cols])


    def _set_frequency_predictor_block(self) -> None:
        if(self._verbose):
            print('============================================================')
            print('Setting up frequency predictor block ...')

        sub_waves = list() 
        for i in self._dec.dict_waves.keys():
            sub_waves = sub_waves + self._dec.dict_waves[i]
        
        # For each level of frequency of target
        for level in range(self._max_wave):
            temp_ds = self._ds.copy(deep=True)

            if(self._verbose):
                print('Setting up predictor of ', self._y_cols, ' frequency > ', level)

            # Creating dataset for this level
            cols = list()
            for wave in self._dec.dict_waves.keys():
                max_len = len(self._dec.dict_waves[wave])
                if level >= max_len:
                    cols.append(self._dec.dict_waves[wave][max_len-1])
                else:
                    cols.append(self._dec.dict_waves[wave][level])

            drop_cols = list(filter(lambda x: x not in cols, sub_waves))
            
            # Dropping unused frequency waves
            temp_ds.drop(drop_cols, axis=1, inplace=True)
            
            # Dropping original waves
            temp_ds.drop(self._dec_cols, axis=1, inplace=True)

            # List of features of current level
            curr_feat_cols = temp_ds.columns.difference(['Date'])

            # Instantiating CHM Level
            temp_level = CHMLevel(self._context_regressor(
                                                            ds=temp_ds.copy(deep=True), 
                                                            x_cols=curr_feat_cols, 
                                                            y_cols=self._dec.dict_waves[self._y_cols][level], 
                                                            n_inputs=self._n_inputs, 
                                                            n_features=len(curr_feat_cols), 
                                                            epochs=self._epochs
                                                            )
                                    )

            self._list_levels_frequency.append(temp_level)

    def _print_bottom_up_series(self, show_picture=False, save_picture=True):
        sns.set_style("whitegrid")
        index_col = 'Date'
        if(self._verbose):
            print('============================================================')
            print('Printing bottom up block ...')

        # Flatting subwave column name dictionary
        sub_waves = list() 
        for i in self._dec.dict_waves.keys():
            sub_waves = sub_waves + self._dec.dict_waves[i]

        # For each stage
        for stage in range(self._num_stages):

            # Copying temporary dict of waves
            temp_dec_dict = deepcopy(self._dec.dict_waves)

            # For each level in the current stage
            for level in range(self._max_wave):
                if(self._verbose):
                    print('Printing series of input for > ', 'Level > ', level, ' of Stage > ', stage)
                
                temp_ds = self._ds.copy(deep=True)

                # Creating dataset for this level
                for wave in temp_dec_dict.keys():

                    # Initiating sintetized column of wave
                    temp_ds[wave + '_sintetized'] = 0   

                    # For each decomposed subwave of current wave
                    for col in temp_dec_dict[wave]:
                        temp_ds[wave + '_sintetized'] = temp_ds[wave + '_sintetized'] + temp_ds[col]

                # Leaving only sintetized columns
                # Dropping used subwave
                temp_ds.drop(sub_waves, axis=1, inplace=True)
                
                # Dropping original waves
                temp_ds.drop(self._dec_cols, axis=1, inplace=True)
                ##

                # List of features of current level (X Cols)
                curr_feat_cols = temp_ds.columns.difference(['Date'])
                
                for col in temp_ds[curr_feat_cols].columns:
                        

                    #Plotting Y
                    plt.plot(temp_ds[col], label=col)

                    #Adding legends to plot
                    plt.legend()

                    #Setting to show only n of dates on X-Axis
                    n = 30
                    num_dates = np.arange(0, len(temp_ds), n)
                    plt.xticks(num_dates, [(temp_ds[index_col][i]).strftime('%Y-%m-%d') for i in num_dates], rotation='vertical')

                    if save_picture:
                        folder = '../figs/CHM/bottom_up_block/stage/' + str(stage) + '/level' + str(level) + '/' 
                        if not os.path.exists(folder):
                            os.makedirs(folder)

                        plt.savefig(folder + col + '.png')
                    
                    if show_picture:
                        plt.show()

                    plt.clf()

                for k in temp_dec_dict.keys():
                    if len(temp_dec_dict[k]) > 1:
                        temp_dec_dict[k].pop()

    