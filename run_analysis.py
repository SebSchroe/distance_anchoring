# import modules
import analysis

# global variables
sub_id = 1
cond_id = 1

x_values = 'speaker_distance'
y_values = 'response'

# predict sample size
# analysis.predict_sample_size(effect_size=0.161)

# diagnostic plots
# analysis.create_diagnostic_plots(sub_id=sub_id, cond_id=cond_id, block_id=1)

# plot presented vs percieved distance of block 1, 4 and 6
# analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, x_values=x_values, y_values=y_values, split=False)

# plot differences
df = analysis.plot_differences(sub_id=sub_id, cond_id=cond_id, block_id=1) # weiß ich nicht...