# import modules
import analysis

# global variables
sub_id = 1
cond_id = 2

x_values = 'speaker_distance'
y_values = 'response'

# multi plot
# analysis.plot_test_blocks(sub_id=sub_id, cond_id=cond_id, x_values=x_values, y_values=y_values)

analysis.split_data(sub_id=sub_id, cond_id=cond_id)