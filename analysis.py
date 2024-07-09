import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sub_id = 2
cond_id = 2
block_id = 1
task_id = 2

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

DIR = pathlib.Path(os.curdir)
file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
df = pd.read_csv(file_path)

# data manipulation
# convert speaker_id to real distance
def get_value_from_dict(key, speaker_dict):
    return speaker_dict.get(key, None)

df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_value_from_dict(x, speaker_dict))

# data analysis
def linear_regression(x_values, y_values):
    
    # linear regression
    x  = df[f'{x_values}'].values.reshape(-1, 1)
    y = df[f'{y_values}'].values
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # calculate regression coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = r2_score(y, y_pred)
    textstr = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nR²: {r_squared:.2f}'
    
    # plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, y_pred, color='red', linewidth=1.5, label='Linear regression')
    plt.text(0.03, 0.97, textstr, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    # appearence
    plt.xlabel(f'{x_values}')
    plt.ylabel(f'{y_values}')
    plt.title(f'sub {sub_id}, cond {cond_id}, block {block_id}')
    plt.axis([0, 13, 0 ,13])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')
    
    plt.show()
    
    
linear_regression('speaker_distance', 'response')
    

# create figure
# plt.figure(figsize=(8, 8))
# plt.plot(df['event_id'], df['response'], marker='o', linestyle='None')

# plt.title(f'sub = {sub_id}, cond = {cond_id}, block = {block_id}, task = {task_id}')
# plt.xlabel('presented distance [m]')
# plt.ylabel('slider value [m]')
# plt.axis([0, 100, 0, 13])
# plt.gca().set_aspect('equal', adjustable='box')

# plt.show()
