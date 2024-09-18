import pathlib
import os
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
cond_1_sub_ids = [1, 3, 4, 7]
cond_2_sub_ids = [1, 2, 6]

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
def create_diagnostic_plots(sub_id, cond_id, block_id):
    '''
    1. .residual_plot() -> Residuals vs Fittes values: checks if relationship between x and y is linear (linearity)
    2. .qq_plot() -> Normal Q-Q: checks if errors/residuals are normally distibuted (normality)
    3. .scale_location_plt() -> Scale-location: checks if the residual-variance is the same for every value of x (homoskedasticity)
    4. .leverage_plot() -> Residuals vs Leverage: checks if observations are independent of each other (outliers)
    '''
    
    # load data
    if sub_id == 'all':
        df, sub_ids = get_concat_df(cond_id, block_id)
    else:
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

def plot_data(cond_id, block_ids):
    
    # load data and concatenate all data
    
    
    # prepare multiplot
    n_plots = len(block_ids)
    fig, axes = plt.subplots(1, n_plots, figsize=(19, 4), squeeze=False)
    index = 0
    
    for block_id in block_ids:
    
        # load data
        df, sub_ids = get_concat_df(cond_id, block_id)
        
        # transform speaker_id to corresponding speaker distance
        df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
        
        # get means per participant
        df_means = df.groupby(['sub_id', 'speaker_distance'], as_index=False).agg(
            mean_response=('response', 'mean'))
        
        sns.lineplot(data=df_means, x='speaker_distance', y='mean_response', hue='sub_id', ax=axes[0, index], palette='tab10')
        
        index += 1
    
    plt.tight_layout()    
    plt.show()

# plot presented vs. percieved distances of test blocks 1, 4 and 6
def plot_presented_vs_percieved_distance(sub_id, cond_id, block_ids, split=False):
    '''
    sub_id: can be individual sub_id or 'all' for all data of certain condition
    split: splits data of first block (condition 1) or last block (condition 2) in speaker subsets 1-6 and 4-9
    '''
    
    # set variables
    x_values = 'speaker_distance'
    y_values = 'response'
    
    # prepare multi plots
    n_plots = len(block_ids)
    fig, axes = plt.subplots(1, n_plots, figsize=(19, 4))
    index = 0
    
    # build each subplot per block
    for block_id in block_ids:
        
        # load data
        if sub_id == 'all':
            df, sub_ids = get_concat_df(cond_id, block_id)
        else:
            df = get_df(sub_id, cond_id, block_id)
        
        # transform speaker_id to corresponding speaker distance
        df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))

        # split data of block 1 or 6 (depending on cond) in desired response subsets
        if split and ((cond_id == 1 and block_id == 2) or (cond_id == 2 and block_id == 6)):
               
            # get new dataframes for subset
            df_1 = get_response_subset_df(df, 1, 6)
            df_2 = get_response_subset_df(df, 4, 9)
            
            # do linear regression for each subset and get necessary values
            x_1, y_1, y_pred_1, regression_coefficients_1 = get_linear_regression_values(df_1, f'{x_values}', f'{y_values}')
            x_2, y_2, y_pred_2, regression_coefficients_2 = get_linear_regression_values(df_2, f'{x_values}', f'{y_values}')

            # plot baseline
            axes[index].plot([2, 12], [2, 12], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
            
            # plot subset 1
            axes[index].scatter(x_1, y_1, color='blue', alpha=0.05, edgecolors='None', label='Data points')
            axes[index].plot(x_1, y_pred_1, color='blue', linewidth=1.5, label='Linear regression')
            axes[index].text(0.03, 0.97, regression_coefficients_1, transform=axes[index].transAxes,
                             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.5))
            
            # plot subset 2
            axes[index].scatter(x_2, y_2, color='red', alpha=0.05, edgecolors='None', label='Data points')
            axes[index].plot(x_2, y_pred_2, color='red', linewidth=1.5, label='Linear regression')
            axes[index].text(0.97, 0.03, regression_coefficients_2, transform=axes[index].transAxes,
                             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.5))
            
            # layout
            axes[index].set_xlabel('Source distance [m]')
            axes[index].set_ylabel('Perceived distance [m]')
            axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
            axes[index].set_xlim(0, 13)
            axes[index].set_ylim(0, 13)
            axes[index].set_xticks(range(0, 13, 2))
            axes[index].set_yticks(range(0, 13, 2))
            axes[index].set_aspect('equal', adjustable='box')
         
        # if split is false show default datapoints
        else:
            
            # do linear regression and get necessary values + coefficients
            x, y, y_pred, regression_coefficients = get_linear_regression_values(df, f'{x_values}', f'{y_values}')
            
            # plotting
            axes[index].scatter(x, y, color='blue', alpha=0.05, edgecolors='None', label='Data points')
            axes[index].plot(x, y_pred, color='blue', linewidth=1.5, label='Linear regression')
            axes[index].plot([2, 12], [2, 12], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
            axes[index].text(0.03, 0.97, regression_coefficients, transform=axes[index].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.5))
            
            # layout
            axes[index].set_xlabel('Source distance [m]')
            axes[index].set_ylabel('Perceived distance [m]')
            axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
            axes[index].set_xlim(0, 13)
            axes[index].set_ylim(0, 13)
            axes[index].set_xticks(range(0, 13, 2))
            axes[index].set_yticks(range(0, 13, 2))
            axes[index].set_aspect('equal', adjustable='box')
        
        index += 1
    
    # finalise
    plt.tight_layout()
    plt.show()
    print('Total n:', len(sub_ids))
    print('Total trials:', len(df))
    return df

def plot_means(sub_id, cond_id, block_ids):
    
    # prepare multiplot
    n_plots = len(block_ids)
    fig, axes = plt.subplots(1, n_plots, figsize=(19, 4))
    index = 0
    
    for block_id in block_ids:
        # get data
        if sub_id == 'all':
            df, sub_ids = get_concat_df(cond_id, block_id)
        else:
            df = get_df(sub_id, cond_id, block_id)
        
        # transform speaker_id to corresponding speaker distance
        df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
        
        # get new df with means at each speaker position
        means_df = get_means_df(df)
        
        # extraxt values from df
        x = means_df['speaker_distance']
        means = means_df['mean_response']
        std = means_df['std_response']
        
        # plotting
        axes[index].errorbar(x, means, yerr=std, fmt='o', capsize=5, color='blue', ecolor='blue', label='mean with std')
        axes[index].plot([2, 12], [2, 12], color='grey', linewidth=1.5, linestyle='--', label='Optimum')
        
        #layout
        axes[index].set_xlabel('Source distance [m]')
        axes[index].set_ylabel('Averaged perceived distance [m]')
        axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
        axes[index].set_xlim(0, 13)
        axes[index].set_ylim(0, 13)
        axes[index].set_xticks(range(0, 13, 2))
        axes[index].set_yticks(range(0, 13, 2))
        axes[index].set_aspect('equal', adjustable='box')
        
        index += 1
    
    plt.tight_layout()
    plt.show()
    
    return means_df

# plot average slopes and standard deviation
def seperate_slope_model():
    
    # load data
    df_1, sub_ids = get_concat_df(cond_id=1, block_id=4)
    df_2, sub_ids = get_concat_df(cond_id=2, block_id=4)
    
    # transform speaker_id to corresponding speaker distance
    df_1['speaker_distance'] = df_1['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    df_2['speaker_distance'] = df_2['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    
    # get regression coefficients
    x_1 = df_1['speaker_distance'].values.flatten()
    y_1 = df_1['response']
    x_2 = df_2['speaker_distance'].values.flatten()
    y_2 = df_2['response']
    
    # simulate datapool for condition 3
    slope = 0.77
    intercept = 1.6
    n = 2860
    sd = 1.68
    
    x_3 = np.linspace(2, 12, n).flatten()
    y_3 = slope * x_3 + intercept + np.random.normal(0, sd, n)
    y_3 = y_3.flatten()
    
    # combine all data in one dataframe
    data = pd.DataFrame({'x': np.concatenate([x_1, x_2, x_3]),
                         'y': np.concatenate([y_1, y_2, y_3]),
                         'Group': ['Condition 1']*len(x_1) + ['Condition 2']*len(x_2) + ['Control']*len(x_3)
                         })
    
    # plotting
    model = smf.ols('y ~ x * Group', data=data).fit()
    print(model.summary())
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    
def plot_signed_error_distribution_at_x(cond_id, block_id, x=2):
    
    # load data
    df, sub_ids = get_concat_df(cond_id, block_id)
    
    # transform speaker_id to corresponding speaker distance
    df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_speaker_distance(x, speaker_dict))
    df['signed_error'] = df['response'] - df['speaker_distance']
    
    # filter data for specific x value
    df_filtered = df[df['speaker_distance'] == x]
    
    # plotting
    plt.figure(figsize=(15, 4))
    sns.histplot(df_filtered['signed_error'], kde=True, binwidth=0.1, color='blue')
    
    # layout
    plt.title(f'signed error distribution at presented distance = {x}, block {block_id}')
    plt.xlabel('Signed error')
    plt.ylabel('Count')
    plt.xlim([-7, 7])
    plt.ylim([0, 6])
    
    # finalise
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

# help functions
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

def get_concat_df(cond_id, block_ids):
    
    # create empty dataframe
    concat_df = pd.DataFrame()
    
    # get sub_ids depending on cond_id
    if cond_id == 1:
        sub_ids = cond_1_sub_ids
    else:
        sub_ids = cond_2_sub_ids
        
    # loop through all block_ids
    for block_id in block_ids:
        # get task_id of current block
        task_id = get_task_id(cond_id, block_id)
        
        # concatenate data for all subjects for the current block_id 
        for sub_id in sub_ids:
            file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
            new_df = pd.read_csv(file_path)
            concat_df = pd.concat([concat_df, new_df], axis=0, ignore_index=True)
    
    return concat_df, sub_ids

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

def get_means_df(df):
    
    # group by speaker distances and get mean of responses
    grouped_df = df.groupby('speaker_distance').agg(
        mean_response=('response', 'mean'),
        std_response=('response', 'std')
        ).reset_index()
    
    # rename columns
    grouped_df.columns = ['speaker_distance', 'mean_response', 'std_response']
    return grouped_df

def get_linear_regression_values(df, x_values, y_values):
    
    # calculate linear regression
    x = df[f'{x_values}'].values.reshape(-1, 1)
    y = df[f'{y_values}']
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # calculate linear regression coefficients
    slope = model.coef_[0]
    # intercept = model.intercept_
    r_squared = r2_score(y, y_pred)
    regression_coefficients = f'Slope: {slope:.2f}\nR²: {r_squared:.2f}'
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
