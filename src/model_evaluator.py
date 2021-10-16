from typing import Dict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
import os

class ModelEvaluator():
    """docstring for ModelEvaluator."""
    def __init__(self,
                 model_name,
                 data_set: DataFrame,
                 pred_col: str,
                 y_col: str,
                 x_col: str
                    ) -> None:
        self.model_name   = model_name
        self.data_set     = data_set
        self.y_col        = y_col
        self.x_col        = x_col
        self.pred_col     = pred_col

    def plot_results(self, show_picture=False, save_picture=False):
        sns.set_style("whitegrid")

        plt.figure(figsize=(40,10))
        plt.plot(self.data_set[self.y_col], label=self.y_col)
        plt.plot(self.data_set[self.pred_col], 'r--', label=self.pred_col)

        plt.legend()
        num_dates = np.arange(1, len(self.data_set), 30)
        plt.xticks(num_dates, [str(self.data_set[self.x_col][i]) for i in num_dates], rotation='vertical')

        
        if save_picture:
            folder = '../figs/' + self.model_name + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            print('Saving picture from results of prediction of :' + self.y_col + ' , using model : ' + self.model_name)
            plt.savefig(folder + self.y_col + '.png')
        
        if show_picture:
            plt.show()