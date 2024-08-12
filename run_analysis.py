# import modules
import analysis

# global variables
sub_id = 'all' # 'all' for concatenated data points
cond_id = 1
split = False

# predict sample size
# analysis.predict_sample_size(effect_size=0.161)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

# plot presented vs percieved distance of block 1, 4 and 6
analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, split=split)

# plot differences
# df = analysis.plot_differences(sub_id=sub_id, cond_id=cond_id, block_id=1) # weiß ich nicht...

# merged_df = analysis.get_merged_df(cond_id = cond_id, block_id = 1)
