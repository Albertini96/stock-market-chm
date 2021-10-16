
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame
from requests.sessions import dispatch_hook
from config import Config
from scalers.scaler import BaseScaler
from sklearn.model_selection import train_test_split

class PreProcessing():
    def __init__(self, ds, base_scaler:BaseScaler) -> None:
        self._scalers:Dict[str, BaseScaler] = None
        self._base_scaler                   = base_scaler
        self.ds:DataFrame                   = ds
    # List of columns  dataframe separator (number of scalers = number of unique from all items)

   
    def values_scaler_partition_by_2(self,
                                        ds:DataFrame, 
                                        first_col:str, 
                                        second_col:str,
                                        column_to_scale:str,
                                        used_scaler: BaseScaler
                                    ) -> DataFrame:
        """
        Performs scaling of variables in a dataframe for each partition of by list

        parameters : DataFrame (dataframe to scale with by partition variables)
                     str       (first column to partition by)
                     str       (second column to partition by)
                     str       (name of column to scale)
                     List[str] (list of names of columns to partition by)
                     Scaler    (used scaler to transform)
        
        """
        # Setting return variable to None
        ds_scaled = None

        # Defining number of scalers
        self._scalers = ds[[first_col, second_col]].groupby(by=[first_col, second_col], dropna=False).sum().reset_index()

        #Initializing one scaler per row of scaler
        self._scalers['scaler'] = None
        for index, row in self._scalers.iterrows():  
            row['scaler'] = used_scaler()

        # Copying stock data set to create a scaled version
        ds_scaled = ds.copy()

        # For each row in scalers
        for index, row in self._scalers.iterrows():
            # Fechting sub dataset from stock data based on ticker and variable
            sub_ds = ds_scaled[ (ds_scaled[first_col] == row[first_col]) & (ds_scaled[second_col] == row[second_col] )]

            # Fit sub scaler to subset and transform value in stock data
            scaler_temp = row['scaler']    
            ds_scaled.loc[(ds_scaled[first_col] == row[first_col]) & (ds_scaled[second_col] == row[second_col] ), column_to_scale] = scaler_temp.fit_transform(sub_ds[[column_to_scale]])
            row['scaler'] = scaler_temp

        return ds_scaled

   
    def values_scaler( self,
                        ds:DataFrame, 
                        columns_to_scale:List[str],
                        used_scaler:BaseScaler,
                        transform_inplace:bool
                    ) -> Dict[str, BaseScaler]:
        """
        Performs scaling of list of columns from dataframe

        parameters : DataFrame (dataframe to scale with by partition variables)
                     List[str] (list of names of columns to partition by)
                     Scaler    (used scaler to transform)
                     bool      (whether function applies tranformation or just fit)
        
        returns : dictionary of Scaler, one for each scaled column
        
        """
        # Setting return variable to None
        scalers:dict[str, BaseScaler] = dict[str, BaseScaler]()

        #Fitting all scalers
        for col in columns_to_scale:
            scalers[col] = used_scaler().fit(ds[[col]])
        
        #Transform dataframe
        if transform_inplace:
            for col in columns_to_scale:
                ds[[col]] = scalers[col].transform(ds[[col]])

        return scalers

   
    def fill_stock_data_missings(self,
                                ds:DataFrame, 
                                transform_inplace:bool
                            ) -> DataFrame:
        """
        Performs missing filling

        parameters : DataFrame (dataframe to fill missings)
                     bool      (whether function applies inplace or copy)
        
        returns : DataFrame
        
        """

        ret = None

        if transform_inplace:
            self._replace_row_cell_with_last(ds)
        else:
            ret = ds.copy()
            self._replace_row_cell_with_last(ret)

        return ret

    def _replace_row_cell_with_last(self,
                                    ds:DataFrame) -> None:
        """
        Replaces an empty cell with last non Null value from dataframe

        parameters : DataFrame (dataframe to fill missings)
        
        returns : None
        
        """

        #for each line of the dataset
        for index, row in ds.iterrows():

            index_temp = index
            ds_len = len(ds)
            
            #for each column of the dataset
            for col in ds.columns:
                #if is first row and is null, find the next value that is not null to replace
                if(index_temp == 0 and pd.isna(ds.loc[index_temp, col])):
                    i = 1
                    while((index_temp+i < ds_len) and pd.isna(ds.loc[index_temp+i, col])):
                        i += 1

                    ds.loc[index_temp, col] = ds.loc[index_temp+i, col]
                else:
                    #if row is null find the first value that comes before it, that is not null to replace it
                    if pd.isna(row[col]):
                        i = 1
                        while((index_temp-i > 0) and pd.isna(ds.loc[index_temp-i, col])):
                            i += 1

                        ds.loc[index_temp, col] = ds.loc[index_temp-i, col]

    
    def add_bollinger_bands(self,
                            ds:DataFrame, 
                            cols:List[str],
                            inplace:bool = True,
                            save_pictures:bool = False
                            ) -> DataFrame:
        """
        Adds bollinger bands value to dataframe

        parameters : DataFrame (dataframe to add values)
                     List[str] (list of name of cols to apply bollinger bands)
                     bool      (weather the original dataframe will be replaced or copied)
                     bool      (saves figures of results)

        returns : DataFrame with bollinger bands analysis
        
        """

        ds_copy = None

        if inplace:
            ds_copy = ds
        else:
            ds_copy = ds.copy()
             
        for i in cols:

            bands = ta.bbands(ds[i])
            bands = bands[['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0']]
            bands.columns = [i + '_' + col for col in bands.columns]

            ds_copy[bands.columns] = bands
            
            if save_pictures:
                print('Saving bbands picture of : ', i)

                bands['close'] = ds[i]
                plot = bands.plot(figsize=(100, 40), grid=True, legend=True)

                fig = plot.get_figure()
                fig.savefig("../figs/bollinger_bands/output" + i + ".png")
        
        return ds_copy

   
    def add_sma(self,
                ds:DataFrame, 
                cols:List[str],
                inplace:bool = True,
                sma_length:int = 5,
                save_pictures:bool = False
                ) -> DataFrame:
        """
        Adds SMA values to dataframe

        parameters : DataFrame (dataframe to add values)
                     List[str] (list of name of cols to apply SMA)
                     bool      (weather the original dataframe will be replaced or copied)
                     int       (length of SMA)
                     bool      (saves figures of results)

        returns : DataFrame with SMA analysis
        
        """

        ds_copy = None

        if inplace:
            ds_copy = ds
        else:
            ds_copy = ds.copy()
             
        for i in cols:
            
            sma = ta.sma(ds[i], sma_length)
            
            ds_copy[i + '_SMA_' + str(sma_length)] = sma
            
            if save_pictures:
                print('Saving SMA picture of : ' + i + '_SMA_' + str(sma_length))

                sma_plot = pd.DataFrame()
                sma_plot['close'] = ds[i]
                sma_plot[i + '_SMA_' + str(sma_length)] = sma

                plot = sma_plot.plot(figsize=(100, 40), grid=True, legend=True)

                fig = plot.get_figure()
                fig.savefig("../figs/sma/output" + i + ".png")
        
        return ds_copy

    def add_rsi(self,
                ds:DataFrame, 
                cols:List[str],
                inplace:bool = True,
                rsi_length:int = None,
                save_pictures:bool = False
                ) -> DataFrame:
        """
        Adds RSI (Relative Strength Index) values to dataframe

        parameters : DataFrame (dataframe to add values)
                     List[str] (list of name of cols to apply RSI)
                     bool      (weather the original dataframe will be replaced or copied)
                     bool      (saves figures of results)

        returns : DataFrame with RSI values
        
        """

        ds_copy = None

        if inplace:
            ds_copy = ds
        else:
            ds_copy = ds.copy()
             
        for i in cols:
            
            rsi = ta.rsi(ds[i], lenght=rsi_length)
            
            ds_copy[i + '_RSI'] = rsi
            
            if save_pictures:
                print('Saving RSI picture of : ' + i + '_RSI')

                rsi_plot = pd.DataFrame()
                rsi_plot['close'] = ds[i]
                rsi_plot[i + '_RSI'] = rsi

                plot = rsi_plot.plot(figsize=(100, 40), grid=True, legend=True)

                fig = plot.get_figure()
                fig.savefig("../figs/rsi/output" + i + ".png")
        
        return ds_copy

    def add_stochastic_oscilator(self,
                                ds:DataFrame, 
                                cols:List[str],
                                high_posfix:str ='_High',
                                low_posfix:str ='_Low',
                                close_posfix:str ='_Close',
                                inplace:bool = True,
                                save_pictures:bool = False
                                ) -> DataFrame:
        """
        Adds stochastic oscillators values to dataframe, the function expects a list of str, each str must have a high, low, close pair str. (e.g stock -> stock_High, stock_Low, stock_Close)

        parameters : DataFrame (dataframe add values)
                     List[str] (list of name of cols to apply bollinger bands)
                     str       (high posfix of columns)
                     str       (low posfix of columns)
                     str       (close posfix of columns)
                     bool      (weather the original dataframe will be replaced or copied)
                     bool      (saves figures of results)

        returns : DataFrame with stochastic oscillators values
        
        """

        ds_copy = None

        if inplace:
            ds_copy = ds
        else:
            ds_copy = ds.copy()
             
        for i in cols:

            stoch = ta.stoch(ds[i+high_posfix], ds[i+low_posfix], ds[i+close_posfix])
            stoch = stoch[['STOCHk_14_3_3', 'STOCHd_14_3_3']]
            stoch.columns = [i + '_' + col for col in stoch.columns]

            ds_copy[stoch.columns] = stoch
            
            if save_pictures:
                print('Saving stochastic oscillator picture of : ', i)

                stoch['close'] = ds[i+close_posfix]
                plot = stoch.plot(figsize=(100, 40), grid=True, legend=True)

                fig = plot.get_figure()
                fig.savefig("../figs/stochastic_oscillator/output" + i + ".png")

    def pre_process(self):
        #For training dataset
        train, test = train_test_split(self.ds, test_size=0.33, shuffle=False)
        test = test.reset_index(drop=True)

        #Adding features
        clse_cols = [x for x in train.columns.difference(['Date']) if x[-5:] == 'Close']
        self.add_bollinger_bands(train, clse_cols, True, False)
        self.add_sma(train, clse_cols, True, 5, False)
        self.add_rsi(train, clse_cols, True, save_pictures=False)
        self.add_stochastic_oscilator(train, Config.get_tickers(), inplace=True, save_pictures=False)

        #Scaling values
        self._scalers = self.values_scaler(train, train.columns.difference(['Date']), self._base_scaler, True)

        #Filling missings
        self.fill_stock_data_missings(train, True)


        #For testing dataset
        #Adding features
        clse_cols = [x for x in test.columns.difference(['Date']) if x[-5:] == 'Close']
        self.add_bollinger_bands(test, clse_cols, True, False)
        self.add_sma(test, clse_cols, True, 5, False)
        self.add_rsi(test, clse_cols, True, save_pictures=False)
        self.add_stochastic_oscilator(test, Config.get_tickers(), inplace=True, save_pictures=False)

        #Scaling values
        for col in self._scalers.keys():
            test[[col]] = self._scalers[col].transform(test[[col]])

        #Filling missings
        self.fill_stock_data_missings(test, True)

        return train, test