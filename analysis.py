import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt

sub_id = 1
cond_id = 2
block_id = 5
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
file_path = DIR / 'results' / f'results_cond-{cond_id}_sub-{sub_id}_block-{block_id}_task-{task_id}.csv'
df = pd.read_csv(file_path)

# convert speaker_id to real distance
def get_value_from_dict(key, speaker_dict):
    return speaker_dict.get(key, None)

df['speaker_distance'] = df['speaker_id'].apply(lambda x: get_value_from_dict(x, speaker_dict))

# create figure
plt.figure(figsize=(8, 8))
plt.plot(df['event_id'], df['response'], marker='o', linestyle='None')

plt.title(f'sub = {sub_id}, cond = {cond_id}, block = {block_id}, task = {task_id}')
plt.xlabel('presented distance [m]')
plt.ylabel('slider value [m]')
plt.axis([0, 100, 0, 13])
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
