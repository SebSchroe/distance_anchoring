# %% prepare data
# import modules
import analysis
import math
import pandas as pd
from analysis import speaker_dict


# set global variables
sub_ids = ['01', '02', '03', '05', '06', '09', '10', '12', '14', '15', '16', '18', '19', '20', '21', '24', '25', '26']
cond_ids = [1, 2]
block_ids = [1, 2, 4, 6]

# load and transform data
df = analysis.get_concat_df(sub_ids=sub_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# filter for specific cond_ids and block_ids
df = df[df['cond_id'].isin(cond_ids) & df['block_id'].isin(block_ids)]

# TODO: remove first trial of each block + remove trials with less than 300 ms response time and more than 15(?) seconds

# calculate mean led_distance responses and convert speaker_id to categorical values
means_df = (
    df.groupby(['sub_id', 'cond_id', 'block_id', 'speaker_distance'], as_index=False)
    .agg(mean_led_distance=('led_distance', 'mean'),
         std_led_distance=('led_distance', 'std'))
    .assign(speaker_distance=lambda x: pd.Categorical(
        x['speaker_distance'].astype(int), 
        categories=sorted(x['speaker_distance'].unique().astype(int)), 
        ordered=True
    ))
)

# calculate mean and std of mean led_distance responses
mean_of_means_df = (
    means_df.groupby(['cond_id', 'block_id', 'speaker_distance'])
    .agg(mean_mean_led_distance=('mean_led_distance', 'mean'),
         std_mean_led_distance=('mean_led_distance', 'std')
         )
    .reset_index()
)

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

# %% plot with error bars
analysis.plot_with_error_bars(df=mean_of_means_df, x='speaker_distance', y='mean_mean_led_distance', 
                              yerr='std_mean_led_distance', col='block_id', row='cond_id')

# %% plot boxplot of mean results
analysis.plot_boxplot(df=df, x='speaker_distance', y='led_distance', col='block_id', hue='cond_id')

 # %% fitting mean results of each sub_id
# analysis.plot_data(df=means_df, x='speaker_distance', y='mean_led_distance',
#                    col='block_id', row='cond_id', hue='sub_id', kind='regplot')

# %% calculate raw experiment duration
# mean_response_time_df = df.groupby(['sub_id', 'block_id', 'cond_id', 'event_id'], as_index=False).agg(mean_response_time=('response_time', 'mean'))
# analysis.plot_data(df=mean_response_time_df, x='event_id', y='mean_response_time',
#                    col='block_id', row='cond_id', hue='sub_id', kind='scatterplot', baseline=False)

# # exclude all response times with event_id = 1
# filtered_mean_response_time_df = mean_response_time_df.query('event_id != 1')

# # calculate mean response time
# mean_response_time = filtered_mean_response_time_df['mean_response_time'].mean()

# # %% calculate experiment duration
# n_reps_list = [8, 9, 10, 11, 12, 13, 14, 15]
# exp_dur_list = []

# for n_reps in n_reps_list:
#     exp_dur = analysis.calc_experiment_duration(n_reps=n_reps, mean_response_time=mean_response_time)
#     exp_dur_list.append(exp_dur)
    
# sns.scatterplot(x=n_reps_list, y=exp_dur_list)
    
# %% predict sample size
analysis.predict_sample_size(effect_size=1.501, alpha=0.05, power=0.8, alternative='two-sided')

# %% diagnostic plots
# analysis.create_diagnostic_plots(df=df, x='speaker_distance', y='led_distance')

