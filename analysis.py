import pathlib
import os
import LinearRegDiagnostic
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.power as smp
from scipy.stats import norm
from scipy.stats import shapiro, kstest

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

def plot_raw_data(df, sub_id):
    
    # filter df per sub_id
    sub_id = int(sub_id)
    filtered_df = df[df['sub_id'] == sub_id]
    
    # calculate mean and std data per speaker_distance
    means_df = (
        filtered_df
        .groupby(['block_id', 'speaker_distance'], as_index=False)
        .agg(
            mean_led_distance=('led_distance', 'mean'),
            std_led_distance=('led_distance', 'std')
            )
        )
    
    # create FacetGrid
    g = sns.FacetGrid(filtered_df, col='block_id')
    
    # scatterplot
    g.map_dataframe(sns.scatterplot, x='speaker_distance', y='led_distance', color='grey', alpha=0.5)
    
    for ax, (block_id, sub_df) in zip(g.axes.flat, means_df.groupby('block_id')):
        
        # add lineplot and errorbars
        ax.errorbar(x=sub_df['speaker_distance'], y=sub_df['mean_led_distance'], yerr=sub_df['std_led_distance'],
                    fmt='-o', capsize=3)
        
        ax.set_aspect('equal', adjustable='box')
        
        ax.plot([2, 12], [2, 12], ls='--', color='grey', label='1:1 Line') # add 1:1 line through the origin
        
    plt.show()

def plot_data(df, x, y, col, row, hue, kind='scatterplot', baseline=True):
    
    # data plotting
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, palette='tab10')
    
    if kind == 'scatterplot':
        g.map(sns.scatterplot, x, y).add_legend()
    elif kind == 'lineplot':
        g.map(sns.lineplot, x, y, marker='o', alpha=0.7).add_legend()
    elif kind == 'regplot':
        g.map(sns.regplot, x, y, order=2).add_legend()
    
    g.add_legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    
    # adjust layout
    for ax in g.axes.flat:
        ax.set_aspect('equal', adjustable='box')
        
        if baseline == 'one_one':
            # add 1:1 line through the origin
            ax.plot([2, 12], [2, 12], ls='--', color='grey', label='1:1 Line') # add 1:1 line through the origin
        elif baseline == 'zero':
            ax.plot([2, 12], [0, 0], ls='--', color='grey', label='zero line') # add baseline at y = 0
        else:
            continue
        
    plt.show()

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
    
def plot_boxplot(df, x, y, col, hue):
    
    # create FacetGrit
    g = sns.FacetGrid(df, col=col, col_wrap=2)
    
    # # map data to the grid
    g.map_dataframe(sns.boxplot, x=x, y=y, hue=hue, dodge=True, palette='tab10')

    # adjust layout
    for ax in g.axes.flat:
        # Add 1:1 line
        ax.plot([0, 10], [2, 12], ls='--', color='grey', label='1:1 Line')

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    g.add_legend()
    g.tight_layout()
    plt.show() 

# distributions
def show_data_distribution(df, x):
    
    # create array of data
    array = df[x].to_numpy()
    
    # prepare multiplot
    fig,axes = plt.subplots(1, 2)
        
    # plot histogram
    sns.histplot(data=df, x=x, bins=7, kde=True, ax=axes[0])
    axes[0].set_title('Histogram with KDE')
    
    # plot QQ-Plot
    sm.qqplot(array, line='s', ax=axes[1])
    axes[1].set_title('QQ-Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Shapiro-Wilk Test
    test_stats, p_value = shapiro(array)
    print("Result of Shapiro-Wilk Test:")
    print(f"Statistic: {test_stats:.3f}, p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Reject H0: Data is not Gaussian.")
    else:
        print("Fail to reject H0: Data is Gaussian.")
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = kstest(array, 'norm', args=(np.mean(array), np.std(array)))
    print("\nResult of Kolmogorov-Smirnov Test:")
    print(f"Statistic: {ks_stat:.3f}, p-value: {ks_p_value:.3f}")
    if ks_p_value < 0.05:
        print("Reject H0: Data is not Gaussian.")
    else:
        print("Fail to reject H0: Data is Gaussian.")


# statistical power analysis
def calculate_cohens_d(mean_1, std_1, n_1, mean_2, std_2, n_2):    
    pooled_std = np.sqrt(((n_1 - 1) * std_1 ** 2 + (n_2 - 1) * std_2 ** 2) / (n_1 + n_2 - 2))
    d = (mean_1 - mean_2) / pooled_std
    return d

def predict_sample_size(group_1, group_2, alpha=0.05, power=0.8, alternative='two-sided'):
    '''
    alternative = two-sided', 'larger' or 'smaller'
    '''
    mean_1, std_1, n_1 = group_1
    mean_2, std_2, n_2 = group_2
    
    effect_size = calculate_cohens_d(mean_1, std_1, n_1, mean_2, std_2, n_2)
    
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
    
    # get number of sub_ids per condition
    print('n cond_1:', sum(int(sub_id) % 2 != 0 for sub_id in sub_ids))
    print('n cond_2:', sum(int(sub_id) % 2 == 0 for sub_id in sub_ids))
         
    return concat_df

def get_questionair_df():
    
    file_path = DIR / 'results' / 'questionair_results.csv'
    df = pd.read_csv(file_path, sep=';')
    df['sub_id'] = df['sub_id'].str.replace('sub-', '').astype(int)
    return df

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
