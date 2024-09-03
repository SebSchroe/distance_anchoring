# import modules
import analysis

# global variables
sub_id = 'all' # 'all' for concatenated data points
cond_id = 1
split = False

# predict sample size
# analysis.predict_sample_size(effect_size=1.071)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=4)

# plot presented vs percieved distance of block 1, 4 and 6
# analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, split=split)

# plot signed error distribution at specific speaker distance
# analysis.plot_signed_error_distribution_at_x(cond_id=cond_id, block_id=6, x=12)

# plot one-way ANOVA
analysis.plot_slopes_ANOVA() #-> wirkt auch wie Schwachsinn -> nochmal überarbeiten

# plot differences
# df = analysis.plot_differences(sub_id=sub_id, cond_id=cond_id, block_id=1) # weiß ich nicht...
