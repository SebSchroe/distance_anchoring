# import modules
import analysis

# global variables
sub_id = 'all' # 'all' for concatenated data points
cond_id = 1
block_ids = [2, 4, 6]
split = False

# predict sample size
# analysis.predict_sample_size(effect_size=1.071)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

# plot presented vs percieved distance of block 1, 4 and 6
# df = analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, block_ids=block_ids, split=split)

# plot signed error distribution at specific speaker distance
# analysis.plot_signed_error_distribution_at_x(cond_id=cond_id, block_id=2, x=12)

# seperate slope model
# analysis.seperate_slope_model() # more work has to be done

# plot differences
# df = analysis.plot_differences(sub_id=sub_id, cond_id=cond_id, block_id=1) # weiß ich nicht...

# calculate mean
# means_df = analysis.plot_means(sub_id=sub_id, cond_id=cond_id, block_ids=block_ids)

# plot data
analysis.plot_data(cond_id=cond_id, block_ids=block_ids)
