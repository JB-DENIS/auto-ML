import datetime
from abc import ABCMeta
from typing import DefaultDict, List
import logging
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class PandasEDA(metaclass=ABCMeta):
    """
    Pandas engine for data operation
    """

    def __init__(self):
        self.status = bool


    def df_head(self, df: pd.DataFrame, nb_rows: int=5):
        return df.head(nb_rows)
        
    def df_describe(self, df: pd.DataFrame, show_plot: bool = False):

        # Output describe
        describe_shape = f"Your selected dataframe has {str(df.shape[1])} columns and {str(df.shape[0])} inputs"
        print(describe_shape)

        descibe_cols = ["col_name",
                        "col_type",
                        "unique_values",
                        "duplicated_values",
                        "max",
                        "min",
                        "mean",
                        "median",
                        # "mode",
                        "var",
                        "std",
                        "skew",
                        "kurtosis",
                        "Nan",
                        "Nan_percent"
                        ]  # TODO add percentile table

        describe_intel = DefaultDict()
        describe_intel["col_name"] = []
        describe_intel["col_type"] = []
        describe_intel["unique_values"] = []
        describe_intel["duplicated_values"] = []
        describe_intel["max"] = []
        describe_intel["min"] = []
        describe_intel["mean"] = []
        describe_intel["median"] = []
        # describe_intel["mode"]=[]
        describe_intel["var"] = []
        describe_intel["std"] = []
        describe_intel["skew"] = []
        describe_intel["kurtosis"] = []
        describe_intel["Nan"] = []
        describe_intel["Nan_percent"] = []

        # for cat in df.select_dtypes(['int32', 'int64', 'float64']).columns:
        for cat in df.columns:
            if df[cat].dtype in ['int32', 'int64', 'float64']:
                describe_intel["col_name"].append(cat)
                describe_intel["col_type"].append(df[cat].dtypes)
                describe_intel["unique_values"].append(df[cat].nunique())
                describe_intel["duplicated_values"].append(
                    df[cat].duplicated().sum())
                describe_intel["max"].append(df[cat].max())
                describe_intel["min"].append(df[cat].min())
                describe_intel["mean"].append(df[cat].mean())
                describe_intel["median"].append(df[cat].median())
                # describe_intel["mode"].append(df[cat].mode())
                describe_intel["var"].append(df[cat].var(ddof=0))
                describe_intel["std"].append(df[cat].std(ddof=0))
                describe_intel["skew"].append(df[cat].skew())
                describe_intel["kurtosis"].append(df[cat].kurtosis())
                describe_intel["Nan"].append(df[cat].isnull().sum())
                describe_intel["Nan_percent"].append(
                    df[cat].isnull().sum()/len(df)*100)
                
                if show_plot:
                    df[cat].hist(bins=25)
                    plt.show()
                    df.boxplot(column=cat, vert=False, showfliers=False)
                    plt.show()
                

            else:
                describe_intel["col_name"].append(cat)
                describe_intel["col_type"].append(df[cat].dtypes)
                describe_intel["unique_values"].append(df[cat].nunique())
                describe_intel["duplicated_values"].append(
                    df[cat].duplicated().sum())
                describe_intel["max"].append(None)
                describe_intel["min"].append(None)
                describe_intel["mean"].append(None)
                describe_intel["median"].append(None)
                # describe_intel["mode"].append(df[cat].mode())
                describe_intel["var"].append(None)
                describe_intel["std"].append(None)
                describe_intel["skew"].append(None)
                describe_intel["kurtosis"].append(None)
                describe_intel["Nan"].append(df[cat].isnull().sum())
                describe_intel["Nan_percent"].append(
                    df[cat].isnull().sum()/len(df)*100)

        describe_df = pd.DataFrame(describe_intel, columns=descibe_cols)

        return describe_df


    def plot_violon_box(self, df: pd.DataFrame, columns: list, x, hue):
        
        if columns:
            ls_col = columns
        else:
            ls_col = df.columns
        
        for cat in ls_col:
            if cat != x:
                vio = sns.violinplot(x=x, y=cat, data=df, hue=hue, orient="v")
                vio.minorticks_on()
                vio.xaxis.set_minor_locator(AutoMinorLocator(2))
                vio.grid(which='minor', axis='x', linewidth=1)
                plt.show()
                        
                
    def plot_hist_chart(self, df: pd.DataFrame, columns: list):
        
        if columns:
            ls_col = columns
        else:
            ls_col = df.columns
        
        for cat in ls_col:
            print(cat)
            sns.distplot(df[cat],color = 'green', rug = False, kde_kws = {'color': 'red', 'lw': 1}) 
            plt.show()
    
    def plot_heatmap_corr(self, df: pd.DataFrame):
        
        df_pears=df.corr(method='pearson', numeric_only= True)
        #coupe en deux le heatmap
        mask = np.triu(np.ones_like(df_pears, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))

        # couleur
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(df_pears, mask=mask, cmap=cmap, vmax=0.9, center=0.05,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
                    