# %% import modules
import analysis
from analysis import speaker_dict

# %% set global variables
cond_ids = [1, 2]
sub_ids_dict = {1: [1, 3, 4], 2:[1, 2]}
block_ids = [2, 4, 6]

# %% load and transform data
df = analysis.get_concat_df(cond_ids=cond_ids, sub_ids_dict=sub_ids_dict, block_ids=block_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# %% plot raw datapoints as they are
analysis.plot_data(df=df, kind='scatterplot')

# %% create new dataframe with mean values grouped by different variables
means_df = df.groupby(['sub_id', 'speaker_distance', 'block_id', 'cond_id'], as_index=False).agg(mean_response=('response', 'mean'))
analysis.plot_data(df=means_df, kind='lineplot')

# %% fitting a model
analysis.plot_data(df=means_df, kind='regplot')

# %%
# predict sample size
# analysis.predict_sample_size(effect_size=1.071)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

