from array import array
import string
from typing import List
from decomposers.decomposer import BaseDecomposer
import emd
import numpy as np
from scipy.signal import hilbert
from pandas.core.frame import DataFrame
class EMDDecomposition(BaseDecomposer):
    """
    Performs the decomposition of time series using hilbert huang decomposition

    parameters : 
                 
    """
    def __init__(self) -> None:
        self.dict_waves  = dict()
        self.dict_freq = dict()

    def decompose_series(self, 
                        ds:DataFrame, 
                        apply_cols:List[str],
                        add_freq:bool = False
                        ) -> object:
        
        for col in apply_cols:
            imf = emd.sift.sift(ds[col])    
            self.dict_waves[col]  = list()
            self.dict_freq[col] = list()

            for f, fr in zip(range(len(imf[0])), reversed(range(len(imf[0])))):
                new_col = col + '_imf_' + str(f)
                self.dict_waves[col].append(new_col)

                #Adding IMF column to the dataset
                ds[new_col] = imf[:, fr]

                #Add instantaneous frequencies if requested
                if add_freq:
                    #Extracting analytical signal
                    analytic_signal = hilbert(imf[:, fr])

                    #Extracting amplitude evelope
                    #amplitude_envelope = np.abs(analytic_signal)
                    
                    #Extracting instantaneous phase
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

                    #Extracting instantaneous frequency
                    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * 20)
                    #Replicating last frequency to match number of sample of original dataset
                    instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[len(instantaneous_frequency)-1])

                    #Adding frequency column to the dataset
                    new_col = col + '_imf_' + str(f) + '_freq'
                    self.dict_freq[col].append(new_col)
                    ds[col + '_imf_' + str(f) + '_freq'] = instantaneous_frequency