from typing import Dict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
import os
from math import floor

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

    #Plot results of model
    def plot_results(self, show_picture=False, save_picture=False):
        sns.set_style("whitegrid")

        #Setting width of figure based on how many observations it has
        plt.figure(figsize=(floor(len(self.data_set)*0.5),10))
        
        #Plotting Y
        plt.plot(self.data_set[self.y_col], label=self.y_col)
        #Plotting predicted Y
        plt.plot(self.data_set[self.pred_col], 'r', label=self.pred_col)
        #Adding legends to plot
        plt.legend()

        #Setting to show only n of dates on X-Axis
        n = 30
        num_dates = np.arange(0, len(self.data_set), n)
        plt.xticks(num_dates, [(self.data_set[self.x_col][i]).strftime('%Y-%m-%d') for i in num_dates], rotation='vertical')

        if save_picture:
            folder = '../figs/' + self.model_name + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            print('Saving picture from results of prediction of :' + self.y_col + ' , using model : ' + self.model_name)
            plt.savefig(folder + self.y_col + '.png')
        
        if show_picture:
            plt.show()

    #Plot results only where predicted is not null
    def plot_results_predicted(self, show_picture=False, save_picture=False):
        sns.set_style("whitegrid")

        pred_ds = self.data_set[~pd.isna(self.data_set[self.pred_col])].copy().reset_index()
        
        #Setting width of figure based on how many observations it has
        plt.figure(figsize=(floor(len(pred_ds)*0.5),20))

        #Plotting Y
        plt.plot(pred_ds[self.y_col], label=self.y_col)
        #Plotting predicted Y
        plt.plot(pred_ds[self.pred_col], 'r', label=self.pred_col)
        #Adding legends to plot
        plt.legend()

        #Adding dates to X-Axis
        num_dates = range(0, len(pred_ds))
        plt.xticks(num_dates, [(pred_ds[self.x_col][i]).strftime('%Y-%m-%d') + '  -  ' + str( round((( (pred_ds[self.pred_col][i] / pred_ds[self.y_col][i]) - 1) * 100 ), 2) ) + ' %' for i in num_dates], rotation='vertical')
        plt.tick_params(labelsize=12)
        
        if save_picture:
            folder = '../figs/' + self.model_name + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            print('Saving picture from results of prediction of predicted only :' + self.y_col + ' , using model : ' + self.model_name)
            plt.ioff()
            plt.savefig(folder + self.y_col + '_predicted.png')
        
        if show_picture:
            plt.show()