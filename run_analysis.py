# import modules
import analysis

# global variables
sub_id = 4
cond_id = 1

x_values = 'speaker_distance'
y_values = 'response'

# multi plot
#analysis.plot_tests(sub_id=sub_id, cond_id=cond_id, x_values=x_values, y_values=y_values)

analysis.split_block_1(sub_id=sub_id, cond_id=cond_id)