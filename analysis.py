import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sub_id = 1
cond_id = 1
block_id = 6
task_id = 3

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
file_path = DIR / 'results' / f'results_cond-{cond_id}_sub-{sub_id}_block-{block_id}_task-{task_id}.csv'
df = pd.read_csv(file_path)

# data manipulation
# convert speaker_id to real distance
def get_value_from_dict(key, speaker_dict):
    return speaker_dict.get(key, None)

df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_value_from_dict(x, speaker_dict))

# data analysis
def linear_regression(x_values, y_values):
    x  = df[f'{x_values}'].values.reshape(-1, 1)
    y = df[f'{y_values}'].values
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, y_pred, color='red', linewidth=1.5, label='Linear regression')
    
    plt.xlabel(f'{x_values}')
    plt.ylabel(f'{y_values}')
    plt.title(f'Regression of sub {sub_id}, cond {cond_id}, block {block_id}')
    plt.axis([0, 12, 0 ,12])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left')
    
    plt.show()
    print(f'Slope: {model.coef_[0]}')
    print(f'R^2: {model.score(x, y)}')
    
    
linear_regression('speaker_id', 'response')
    

# create figure
# plt.figure(figsize=(8, 8))
# plt.plot(df['event_id'], df['response'], marker='o', linestyle='None')

# plt.title(f'sub = {sub_id}, cond = {cond_id}, block = {block_id}, task = {task_id}')
# plt.xlabel('presented distance [m]')
# plt.ylabel('slider value [m]')
# plt.axis([0, 100, 0, 13])
# plt.gca().set_aspect('equal', adjustable='box')

# plt.show()
