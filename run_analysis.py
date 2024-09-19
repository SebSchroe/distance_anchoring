# import modules
import analysis
from analysis import speaker_dict

# global variables
cond_ids = [1, 2]
sub_ids_dict = {1: [1, 3, 4, 7], 2:[1, 2, 6]}
block_ids = [2, 4, 6]

# load all dataframes with specific block_ids
df = analysis.get_concat_df(cond_ids=cond_ids, sub_ids_dict=sub_ids_dict, block_ids=block_ids)

# transform speaker_id to corresponding speaker distance
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# create new dataframe with mean values grouped by different variables
means_df = df.groupby(['sub_id', 'speaker_distance', 'block_id', 'cond_id'], as_index=False).agg(mean_response=('response', 'mean'))

# plot data: use 'line' for means_df and 'scatter' for df
analysis.plot_data(df=df, block_ids=block_ids, kind='scatter')

# predict sample size
# analysis.predict_sample_size(effect_size=1.071)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

# plot presented vs percieved distance of block 1, 4 and 6
# df = analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, block_ids=block_ids, split=split)

