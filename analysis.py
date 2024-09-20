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

def plot_data(df, kind='scatterplot'):
    
    # data plotting
    g = sns.FacetGrid(df, col='block_id', row='cond_id', hue='sub_id', height=4, aspect=1.5, palette='tab10')
    
    if kind == 'scatterplot':
        g.map(sns.scatterplot, 'speaker_distance', 'response').add_legend()
    if kind == 'lineplot':
        g.map(sns.lineplot, 'speaker_distance', 'mean_response').add_legend()
    if kind == 'regplot':
        g.map(sns.regplot, 'speaker_distance', 'mean_response', order=2).add_legend()
    
    # adjust layout
    for ax in g.axes.flat:
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 13)
        ax.set_xticks(np.arange(0, 13, 1))
        ax.set_yticks(np.arange(0, 13, 1))
        ax.set_aspect('equal', adjustable='box')
        
        # add 1:1 line through the origin
        ax.plot([2, 12], [2, 12], ls='--', color='grey', label='1:1 Line')
    
    plt.tight_layout()
    plt.show()

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

# help functions
def get_response_subset_df(df, nearest_speaker, farthest_speaker):
    df['response_subset'] = np.where((df['speaker_id'] >= nearest_speaker) & (df['speaker_id'] <= farthest_speaker), df['response'], np.nan)
    response_subset = df[['speaker_id', 'speaker_distance', 'response_subset']].copy()
    response_subset.dropna(subset='response_subset', inplace=True)
    response_subset.rename(columns={'response_subset': 'response'}, inplace=True)
    return response_subset

def get_concat_df(cond_ids, sub_ids_dict, block_ids):
    
    # create empty dataframe
    concat_df = pd.DataFrame()
    
    # loop through all conditions and get sub_ids
    for cond_id in cond_ids:
        sub_ids = sub_ids_dict.get(cond_id, [])
                
        # loop through all block_ids and get task_ids
        for block_id in block_ids:
            
            # get task_id of current block
            task_id = get_task_id(cond_id, block_id)
            
            # loop through all sub_ids and concatenate data file
            for sub_id in sub_ids:
                file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
                
                try:
                    new_df = pd.read_csv(file_path)
                    concat_df = pd.concat([concat_df, new_df], axis=0, ignore_index=True)
                except FileNotFoundError:
                    print(f'No data found for sub_id {sub_id}, cond_id {cond_id} and block_id {block_id}.')
                    
    return concat_df

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
