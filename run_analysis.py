# %% prepare data
# import modules
import analysis
import math
import seaborn as sns
import pandas as pd
from analysis import speaker_dict


# set global variables
sub_ids = ['01', '02', '03', '06', '09', '10', '12', '14', '16', '24']
cond_ids = [1, 2]
block_ids = [1, 2, 4, 6]

# load and transform data
df = analysis.get_concat_df(sub_ids=sub_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# filter for specific cond_ids and block_ids
df = df[df['cond_id'].isin(cond_ids) & df['block_id'].isin(block_ids)]

# create dataframe with mean results at each distance per participant
means_df = df.groupby(['sub_id', 'cond_id', 'block_id', 'speaker_distance'], as_index=False).agg(mean_led_distance=('led_distance', 'mean'))

# convert speaker_distance in mean_df to categorical variable
distance_order = sorted(means_df['speaker_distance'].unique())
means_df['speaker_distance'] = pd.Categorical(means_df['speaker_distance'], categories=distance_order, ordered=True)

# %% plot individual results per sub_id
n_sub_ids_per_plot = 5
num_plots = math.ceil(len(sub_ids) / n_sub_ids_per_plot)
for i in range(num_plots):
    current_sub_ids = list(map(int, sub_ids[i * n_sub_ids_per_plot:(i + 1) * n_sub_ids_per_plot]))
    temp_df = df[df['sub_id'].isin(current_sub_ids)]
    analysis.plot_data(df=temp_df, x='speaker_distance', y='led_distance',
                       col='block_id', row='sub_id', hue='cond_id', kind='scatterplot')
    
# %% plot all datapoints per cond_id and block_id 
analysis.plot_data(df=df, x='speaker_distance', y='led_distance',
                   col='block_id', row='cond_id', hue='sub_id', kind='scatterplot')

# %% plot mean results of each sub_id per cond_id and block_id
analysis.plot_data(df=means_df, x='speaker_distance', y='mean_led_distance',
                   col='block_id', row='cond_id', hue='sub_id', kind='lineplot')

# %% plot boxplot of mean results
# TODO: add a separate function for boxplotting

# %% fitting mean results of each sub_id
analysis.plot_data(df=means_df, x='speaker_distance', y='mean_led_distance',
                   col='block_id', row='cond_id', hue='sub_id', kind='regplot')

# %% calculate raw experiment duration
mean_response_time_df = df.groupby(['sub_id', 'block_id', 'cond_id', 'event_id'], as_index=False).agg(mean_response_time=('response_time', 'mean'))
analysis.plot_data(df=mean_response_time_df, x='event_id', y='mean_response_time',
                   col='block_id', row='cond_id', hue='sub_id', kind='scatterplot', baseline=False)

# exclude all response times with event_id = 1
filtered_mean_response_time_df = mean_response_time_df.query('event_id != 1')

# calculate mean response time
mean_response_time = filtered_mean_response_time_df['mean_response_time'].mean()

# %% calculate experiment duration
n_reps_list = [8, 9, 10, 11, 12, 13, 14, 15]
exp_dur_list = []

for n_reps in n_reps_list:
    exp_dur = analysis.calc_experiment_duration(n_reps=n_reps, mean_response_time=mean_response_time)
    exp_dur_list.append(exp_dur)
    
sns.scatterplot(x=n_reps_list, y=exp_dur_list)
    
# %%
# predict sample size
# analysis.predict_sample_size(effect_size=1.071)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

