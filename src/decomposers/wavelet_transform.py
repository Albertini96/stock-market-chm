from typing import List
from xmlrpc.client import Boolean
from pandas.core.frame import DataFrame
import pywt
import copy
from decomposers.decomposer import BaseDecomposer

class WaveletDecomposition(BaseDecomposer):
    """
    Performs the decomposition of time series using wavelets decomposition

    parameters : 
                 
    """

    def __init__(self) -> None:
        self.wavelet     = pywt.Wavelet('db4')
        self.dict_waves  = dict()

    def decompose_series(self, 
                        ds:DataFrame,
                        apply_cols:List[str]
                        ) -> object:
        
        for col in apply_cols:
                
            #Finding CA and CDs of wave
            coeffs = pywt.wavedec(ds[col], self.wavelet)

            #Building dictionary of number of waves found
            self.dict_waves[col] = list()

            # For each array of coefficients found
            for i in range(len(coeffs)):

                coeffs2 = copy.deepcopy(coeffs)
                # For each array of coefficients found 
                for j in range(len(coeffs)):
                    if i != j:
                        # For each element of array of coefficients found
                        for k in range(len(coeffs[j])):
                            coeffs2[j][k] = 0

                #Adding decomposed wave to the dataset
                rec = pywt.waverec(coeffs2, self.wavelet)
                
                new_col = col + '_wave_' + str(i)
                self.dict_waves[col].append(new_col)
                ds[new_col] = rec[:len(rec)-1]


    