# import modules
import analysis

# global variables
sub_id = 1
cond_id = 2

x_values = 'speaker_distance'
y_values = 'response'

# plot presented vs percieved distance of block 1, 4 and 6
analysis.plot_presented_vs_percieved_distance(sub_id=sub_id, cond_id=cond_id, x_values=x_values, y_values=y_values, split=True)