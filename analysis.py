import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
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

# data analysis
def single_plot(sub_id, cond_id, block_id, x_values, y_values):
    
    # load data
    task_id = get_task_id(cond_id=cond_id, block_id=block_id)
    file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
    df = pd.read_csv(file_path)
    
    # manipulate data
    df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_value_from_dict(x, speaker_dict))
        
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
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', linewidth=1.5, label='Linear regression')
    plt.text(0.03, 0.97, textstr, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    # layout
    plt.xlabel(f'{x_values}')
    plt.ylabel(f'{y_values}')
    plt.title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
    plt.axis([0, 13, 0 ,13])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')
    
    plt.show()

def multi_plot(sub_id, cond_id, x_values, y_values):
    
    # prepare multi plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    index = 0
    
    # build each subplot
    for block in [1, 4, 6]:
        
        # load data
        block_id = block
        task_id = get_task_id(cond_id=cond_id, block_id=block_id)
        file_path = DIR / 'results' / f'results_sub-{sub_id}_cond-{cond_id}_block-{block_id}_task-{task_id}.csv'
        df = pd.read_csv(file_path)
        
        # manipulate data
        df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_value_from_dict(x, speaker_dict))
        
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
        axes[index].scatter(x, y, color='blue', label='Data points')
        axes[index].plot(x, y_pred, color='red', linewidth=1.5, label='Linear regression')
        axes[index].text(0.03, 0.97, textstr, transform=axes[index].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
        
        # layout
        axes[index].set_xlabel(f'{x_values}')
        axes[index].set_ylabel(f'{y_values}')
        axes[index].set_title(f'sub: {sub_id}, cond: {cond_id}, block: {block_id}')
        axes[index].set_xlim(0, 13)
        axes[index].set_ylim(0, 13)
        axes[index].set_aspect('equal', adjustable='box')
        axes[index].legend(loc='lower right')
        index += 1
    
    plt.tight_layout()
    plt.show()
    

def get_value_from_dict(key, speaker_dict):
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