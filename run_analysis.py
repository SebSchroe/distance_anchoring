# import modules
import analysis

# global variables
sub_id = 1
cond_id = 2
block_id = 6

x_values = 'speaker_distance'
y_values = 'response'

# single plot
# analysis.single_plot(sub_id=sub_id, cond_id=cond_id, block_id=block_id, x_values=x_values, y_values=y_values)

# multi plot
analysis.multi_plot(sub_id=sub_id, cond_id=cond_id, x_values=x_values, y_values=y_values)