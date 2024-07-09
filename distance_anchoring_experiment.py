# import necessary modules
import freefield
import distance_anchoring

# set global variables
sub_id = 2
cond_id = 2
block_id = 6

#TODO: change order of sub_id and cond_id everywhere
"""
cond_id:    1 -> task_id = 1 in block 1, 2, 3, 5
                 task_id = 2 in block 4
                 task_id = 3 in block 6
            2 -> task_id = 1 in block 6
                 task_id = 2 in block 1, 2, 3, 4, 5

task_id:    1 -> nearest_speaker = 0, farthest_speaker = 10
            2 -> nearest_speaker = 1, farthest_speaker = 6
            3 -> nearest_speaker = 4, farthest_speaker = 9
"""
# initialize setup
distance_anchoring.initialize_setup()

# start experiment block by block
distance_anchoring.start_block(sub_id=sub_id, cond_id=cond_id, block_id=block_id, kind='experiment') # 'check' or 'experiment'
freefield.flush_buffers(processor='RX81')
