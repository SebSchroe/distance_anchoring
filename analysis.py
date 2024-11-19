import pathlib
import os
import re
import LinearRegDiagnostic
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.power as smp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DIR = pathlib.Path(os.curdir)
speaker_dict = {0: 2.00,
                1: 3.00,
                2: 4.00,
                3: 5.00,
                4: 6.00,
                5: 7.00,
                6: 8.00,
                7: 9.00,
                8: 10.00,
                9: 11.00,
                10: 12.00}

# analysation of statistical power (predict sample size)
def predict_sample_size(effect_size, alpha=0.05, power=0.8, alternative='two-sided'):
    '''
    effect_size = Cohen's d
    # alternative = two-sided', 'larger' or 'smaller'
    '''
    
    analysis = smp.TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)
    
    print(f'Predicted sample size per condition: {sample_size}')

# linear regression diagnostics
def create_diagnostic_plots(df, x, y):
    '''
    1. .residual_plot() -> Residuals vs Fittes values: checks if relationship between x and y is linear (linearity)
    2. .qq_plot() -> Normal Q-Q: checks if errors/residuals are normally distibuted (normality)
    3. .scale_location_plt() -> Scale-location: checks if the residual-variance is the same for every value of x (homoskedasticity)
    4. .leverage_plot() -> Residuals vs Leverage: checks if observations are independent of each other (outliers)
    '''
    
    # fitting linear model
    model = smf.ols(formula=f'{y} ~ {x}', data=df).fit() # formula = y ~ x
    print(model.summary())
    
    # generate diagnostic plots
    cls = LinearRegDiagnostic.LinearRegDiagnostic(model)
    vif, fig, ax = cls()
    print(vif)

def plot_data(df, x, y, col, row, hue, kind='scatterplot', baseline=True):
    
    # data plotting
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, palette='tab10')
    
    if kind == 'scatterplot':
        g.map(sns.scatterplot, x, y).add_legend()
    elif kind == 'lineplot':
        g.map(sns.lineplot, x, y, marker='o').add_legend()
    elif kind == 'regplot':
        g.map(sns.regplot, x, y, order=2).add_legend()
    
    g.add_legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    
    # adjust layout
    for ax in g.axes.flat:
        ax.set_aspect('equal', adjustable='box')
        
        if baseline:
            # add 1:1 line through the origin
            ax.plot([2, 12], [2, 12], ls='--', color='grey', label='1:1 Line')
    
    plt.show()
    
def plot_boxplot(df, x, y, col, hue):
    
    # create FacetGrit
    g = sns.FacetGrid(df, col=col, hue=hue, palette='tab10')
    
    # map data to the grid
    g.map_dataframe(sns.boxplot, x=x, y=y)
    
    # layout
    g.tight_layout()
    plt.show()
    
    # # prepare figure size and grid
    # fig, axes = plt.subplots(2, 2)
    # axes = axes.flatten()
    
    # # loop through block_ids
    # for i, block_id in enumerate(block_ids):
    #     # filter data for current block
    #     block_data = df[df['block_id'] == block_id]
        
    #     # create boxplot
    #     sns.boxplot(data=block_data, x=x, y=y, hue='cond_id', palette='tab10', ax=axes[i])
        
    # plt.tight_layout()
    # plt.show()
    
def plot_with_error_bars(df, x, y, yerr, col, row):
    
    # create FacetGrit
    g = sns.FacetGrid(df, col=col, row=row, palette='tab10')
    
    # map data to the grid
    g.map_dataframe(sns.lineplot, x=x, y=y, marker='o')
    
    for ax, (_, sub_df) in zip(g.axes.flat, g.facet_data()):
        # add error bars
        ax.errorbar(sub_df[x], sub_df[y], yerr=sub_df[yerr], fmt='none', color='black', capsize=4)
        
        # add 1:1 line
        ax.plot([2, 12], [2, 12], ls='--', color='grey', label='1:1 Line')
        
        # set equal aspect
        ax.set_aspect('equal', adjustable='box')
        
    # layout
    g.tight_layout()
    plt.show()
    
# TODO: plot signed error distribution

# TODO: plot mean data with standarderror    

# help functions
def get_concat_df(sub_ids):
    
    # create empty dataframe
    concat_df = pd.DataFrame()
    
    # loop through all sub_ids
    for sub_id in sub_ids:
        sub_dir = DIR / 'results' / f'sub-{sub_id}'
        
        # load all containing result files
        for file_path in sub_dir.glob("*.txt"):
            try:
                # get csv file and concatenate
                new_df = pd.read_csv(file_path)
                concat_df = pd.concat([concat_df, new_df], axis=0, ignore_index=True)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_path}")
                
    return concat_df

def calc_experiment_duration(n_reps, mean_response_time):
    n_reps = n_reps
    n_speaker = [5, 11]
    isi = [0.3, 2]

    n_trials_1 = n_reps * n_speaker[0]
    n_trials_2 = n_reps * n_speaker[1]

    time_per_trial_1 = mean_response_time + isi[0]
    time_per_trial_2 = isi[1]

    experiment_duration_m = (3 * (n_trials_1 * time_per_trial_1) + 2 * (n_trials_1 * time_per_trial_2) + (n_trials_2 * time_per_trial_1))/60
    return experiment_duration_m

def get_speaker_distance(key, speaker_dict):
    return speaker_dict.get(key, None)
    
def get_task_id(cond_id, block_id):
        if cond_id == 1:
            if block_id in [1, 2, 3, 5]:
                task_id = 1
            elif block_id == 4:
                task_id = 2
            elif block_id == 6:
                task_id = 3
            else:
                print('block_id can only be 1 to 6')

        elif cond_id == 2:
            if block_id in [1, 2, 3, 4, 5]:
                task_id = 2
            elif block_id == 6:
                task_id = 1
            else:
                print('block_id can only be 1 to 6')
        else:
            print('cond_id can only be 1 or 2')
        return task_id
