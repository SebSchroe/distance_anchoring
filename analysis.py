import pathlib
import os
import LinearRegDiagnostic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    # alternative can be two-sided', 'larger' or 'smaller'
    '''
    
    analysis = smp.TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)
    
    print(f'Predicted sample size per condition: {sample_size}')

def create_diagnostic_plots(sub_id, cond_id, block_id):
    '''
    1. .residual_plot() -> Residuals vs Fittes values: checks if relationship between x and y is linear (linearity)
    2. .qq_plot() -> Normal Q-Q: checks if errors/residuals are normally distibuted (normal districution)
    3. .scale_location_plt() -> Scale-location: checks if the residual-variance is the same for every value of x (homoskedasticity)
    4. .leverage_plot() -> Residuals vs Leverage: checks if observations are independent of each other (outliers)
    '''
    
    # load data
    df = get_df(sub_id, cond_id, block_id)
    
    # transform data
    df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    
    # fitting linear model
    model = smf.ols(formula='response ~ speaker_distance', data=df).fit() # formula = y ~ x
    print(model.summary())
    
    # generate diagnostic plots
    cls = LinearRegDiagnostic.LinearRegDiagnostic(model)
    vif, fig, ax = cls()
    print(vif)
    
def split_data(sub_id, cond_id):
    
    # load data depending on condition
    if cond_id == 1:
        block_id = 1
    if cond_id == 2:
        block_id = 6
    df = get_df(sub_id, cond_id, block_id)
    
    # transform speaker_id to real speaker distance
    df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    
    # create df subsets of certain speaker range
    df_1 = get_response_subset_df(df, 1, 6)
    df_2 = get_response_subset_df(df, 4, 9)
    
    # do linear regression and get necessary values + coefficients
    x_1, y_1, y_1_pred, regression_coefficients_1 = get_linear_regression_values(df_1, 'speaker_distance', 'response_subset')
    x_2, y_2, y_2_pred, regression_coefficients_2 = get_linear_regression_values(df_2, 'speaker_distance', 'response_subset')
    
    
    # plotting
    plt.scatter(x_1, y_1, color='blue', alpha=0.25, edgecolors='None', label='Data points')
    plt.plot(x_1, y_1_pred, color='blue', linewidth=1.5, label='Linear regression')
    plt.text(0.03, 0.97, regression_coefficients_1, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    plt.scatter(x_2, y_2, color='red', alpha=0.25, edgecolors='None', label='Data points')
    plt.plot(x_2, y_2_pred, color='red', linewidth=1.5, label='Linear regression')
    plt.text(0.97, 0.03, regression_coefficients_2, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    plt.plot([0, 13], [0, 13], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
    
    # layout
    plt.xlabel('speaker_distance')
    plt.ylabel('response')
    plt.title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
    plt.xlim(0, 13)
    plt.ylim(0, 13)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    
    return x_1, y_1, y_1_pred, x_2, y_2, y_2_pred


def plot_presented_vs_percieved_distance(sub_id, cond_id, x_values, y_values, split=False):
    
    # prepare multi plots
    fig, axes = plt.subplots(1, 3, figsize=(19, 4))
    index = 0
    
    # build each subplot per block
    for block_id in [1, 4, 6]:
        
        # load data
        df = get_df(sub_id, cond_id, block_id)
        
        # transform speaker_id to corresponding speaker distance
        df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))

        # split data of block 1 or 6 (depending on cond) in desired response subsets
        if split and ((cond_id == 1 and block_id == 1) or (cond_id == 2 and block_id == 6)):
               
            # get new dataframes for subset
            df_1 = get_response_subset_df(df, 1, 6)
            df_2 = get_response_subset_df(df, 4, 9)
            
            # do linear regression for each subset and get necessary values
            x_1, y_1, y_pred_1, regression_coefficients_1 = get_linear_regression_values(df_1, f'{x_values}', f'{y_values}')
            x_2, y_2, y_pred_2, regression_coefficients_2 = get_linear_regression_values(df_2, f'{x_values}', f'{y_values}')

            # plot baseline
            axes[index].plot([0, 13], [0, 13], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
            
            # plot subset 1
            axes[index].scatter(x_1, y_1, color='blue', alpha=0.25, edgecolors='None', label='Data points')
            axes[index].plot(x_1, y_pred_1, color='blue', linewidth=1.5, label='Linear regression')
            axes[index].text(0.03, 0.97, regression_coefficients_1, transform=axes[index].transAxes,
                             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
            
            # plot subset 2
            axes[index].scatter(x_2, y_2, color='red', alpha=0.25, edgecolors='None', label='Data points')
            axes[index].plot(x_2, y_pred_2, color='red', linewidth=1.5, label='Linear regression')
            axes[index].text(0.97, 0.03, regression_coefficients_2, transform=axes[index].transAxes,
                             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
            
            # layout
            axes[index].set_xlabel(f'{x_values}')
            axes[index].set_ylabel(f'{y_values}')
            axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
            axes[index].set_xlim(0, 13)
            axes[index].set_ylim(0, 13)
            axes[index].set_aspect('equal', adjustable='box')
         
        # if split is false show default datapoints
        else:
            
            # do linear regression and get necessary values + coefficients
            x, y, y_pred, regression_coefficients = get_linear_regression_values(df, f'{x_values}', f'{y_values}')
            
            # plotting
            axes[index].scatter(x, y, color='blue', alpha=0.25, edgecolors='None', label='Data points')
            axes[index].plot(x, y_pred, color='blue', linewidth=1.5, label='Linear regression')
            axes[index].plot([0, 13], [0, 13], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
            axes[index].text(0.03, 0.97, regression_coefficients, transform=axes[index].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
            
            # layout
            axes[index].set_xlabel(f'{x_values}')
            axes[index].set_ylabel(f'{y_values}')
            axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
            axes[index].set_xlim(0, 13)
            axes[index].set_ylim(0, 13)
            axes[index].set_aspect('equal', adjustable='box')
        
        index += 1
    
    plt.tight_layout()
    plt.show()

def plot_differences(sub_id, cond_id, block_id):
    
    # load data
    df = get_df(sub_id, cond_id, block_id)
    
    # transform data
    df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    calc_diff_presented_percieved(df)
    df['delta_presented'] = get_delta_presented(df)
    df['delta_presented'] = df['delta_presented'].shift(1)
    df = df.dropna(subset=['delta_presented'])
    
       
    x = df['delta_presented']
    y = df['diff_presented_percieved']
    
    x, y, y_pred, regression_coefficients = get_linear_regression_values(df, 'delta_presented', 'diff_presented_percieved')
    
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()

def get_response_subset_df(df, nearest_speaker, farthest_speaker):
    df['response_subset'] = np.where((df['speaker_id'] >= nearest_speaker) & (df['speaker_id'] <= farthest_speaker), df['response'], np.nan)
    response_subset = df[['speaker_id', 'speaker_distance', 'response_subset']].copy()
    response_subset.dropna(subset='response_subset', inplace=True)
    response_subset.rename(columns={'response_subset': 'response'}, inplace=True)
    return response_subset

def get_df(sub_id, cond_id, block_id):
    task_id = get_task_id(cond_id=cond_id, block_id=block_id)
    file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
    df = pd.read_csv(file_path)
    return df

def calc_diff_presented_percieved(df):
    df['diff_presented_percieved'] = df['speaker_distance'] - df['response']
    
def get_delta_presented(df):
    delta_presented = []
    for row in range(len(df) - 1):
        pre = df.iloc[row]['speaker_distance']
        post = df.iloc[row + 1]['speaker_distance']
        delta = post - pre
        delta_presented.append(delta)
    delta_presented.append(np.nan)
    return delta_presented
    

def get_linear_regression_values(df, x_values, y_values):
    
    # calculate linear regression
    x = df[f'{x_values}'].values.reshape(-1, 1)
    y = df[f'{y_values}']
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # calculate linear regression coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = r2_score(y, y_pred)
    regression_coefficients = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nR²: {r_squared:.2f}'
    return x, y, y_pred, regression_coefficients

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