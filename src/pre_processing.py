
from typing import List
from pandas.core.frame import DataFrame
from scalers.scaler import Scaler


class PreProcessing():
    def __init__(self, ) -> None:
        self._scalers:DataFrame = None
        self._data:dict = None
        pass
# List of columns  dataframe separator (number of scalers = number of unique from all items)
        
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
                     List[str] (list of names of columns to partition by)
        
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
        
    #def expand_data_frame_by_date():
