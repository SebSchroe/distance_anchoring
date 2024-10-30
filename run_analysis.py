# %% import modules
import analysis
from analysis import speaker_dict
import seaborn as sns

# %% set global variables
cond_ids = [1]
sub_ids_dict = {1: ['Basti']} # 1: [1, 3, 4, 7], 2:[1, 2, 6]
block_ids = [1, 2, 3, 4, 5, 6]

# %% load and transform data
df = analysis.get_concat_df(cond_ids=cond_ids, sub_ids_dict=sub_ids_dict, block_ids=block_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# %% plot raw datapoints as they are
analysis.plot_data(df=df, x='speaker_distance', y='response',
                   col='block_id', row='cond_id', hue='sub_id', kind='scatterplot')

# %% create new dataframe with mean values grouped by different variables
means_df = df.groupby(['sub_id', 'speaker_distance', 'block_id', 'cond_id'], as_index=False).agg(mean_response=('response', 'mean'))
analysis.plot_data(df=means_df, x='speaker_distance', y='mean_response',
                   col='block_id', row='cond_id', hue='sub_id', kind='lineplot')

# %% fitting a model
analysis.plot_data(df=means_df, x='speaker_distance', y='mean_response',
                   col='block_id', row='cond_id', hue='sub_id', kind='regplot')

# %% calculate raw experiment duration
mean_response_time_df = df.groupby(['sub_id', 'block_id', 'cond_id', 'event_id'], as_index=False).agg(mean_response_time=('response_time', 'mean'))
analysis.plot_data(df=mean_response_time_df, x='event_id', y='mean_response_time',
                   col='block_id', row='cond_id', hue='sub_id', kind='scatterplot', baseline=False)

# exclude all response times with event_id = 1
filtered_mean_response_time_df = mean_response_time_df.query('event_id != 1')

# calculate mean response time
mean_response_time = filtered_mean_response_time_df['mean_response_time'].mean()

# calculate experiment duration
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

