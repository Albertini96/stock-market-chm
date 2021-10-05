
from typing import List, Union
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from requests.sessions import dispatch_hook
from scalers.scaler import Scaler

class PreProcessing():
    def __init__(self) -> None:
        pass
    # List of columns  dataframe separator (number of scalers = number of unique from all items)

    @staticmethod
    def values_scaler_partition_by_2(self,
                                        ds:DataFrame, 
                                        first_col:str, 
                                        second_col:str,
                                        column_to_scale:str,
                                        used_scaler: Scaler
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

    @staticmethod
    def values_scaler(
                        ds:DataFrame, 
                        columns_to_scale:List[str],
                        used_scaler:Scaler,
                        transform_inplace:bool
                    ) -> dict[str, Scaler]:
        """
        Performs scaling of list of columns from dataframe

        parameters : DataFrame (dataframe to scale with by partition variables)
                     List[str] (list of names of columns to partition by)
                     Scaler    (used scaler to transform)
                     bool      (whether function applies tranformation or just fit)
        
        returns : dictionary of Scaler, one for each scaled column
        
        """
        # Setting return variable to None
        scalers:dict[str, Scaler] = dict[str, Scaler]()

        #Fitting all scalers
        for col in columns_to_scale:
            scalers[col] = used_scaler().fit(ds[[col]])
        
        #Transform dataframe
        if transform_inplace:
            for col in columns_to_scale:
                ds[[col]] = scalers[col].transform(ds[[col]])

        return scalers

    @staticmethod
    def fill_stock_data_missings(ds:DataFrame, 
                                transform_inplace:bool
                            ) -> None or DataFrame:
        """
        Performs missing filling

        parameters : DataFrame (dataframe to fill missings)
                     bool      (whether function applies inplace or copy)
        
        returns : dataframe if transform_inplace is false, none if transform_inplace is true
        
        """

        ret = None

        if transform_inplace:
            PreProcessing._replace_row_cell_with_last(ds)
        else:
            ret = ds.copy()
            PreProcessing._replace_row_cell_with_last(ret)

        return ret

    def _replace_row_cell_with_last(ds):
        
        last_row = None
        row = None
        
        for index, row in ds.iterrows():
            if last_row is not None:
                for col in ds.columns:
                    if pd.isna(row[col]):
                        ds.loc[index, col] = last_row[col]
            last_row = row  
        
        
        