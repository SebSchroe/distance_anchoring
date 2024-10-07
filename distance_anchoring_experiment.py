# import necessary modules
import freefield
import distance_anchoring

# set global variables
sub_id = 6
cond_id = 2
block_id = 6
kind = 'experiment'

"""
kind:       experiment -> ... trials per speaker distance
            check -> ... trials per speaker distance

cond_id:    1 -> task_id = 2 in block 1, 2, 3, 4, 5
                 task_id = 1 in block 6
            2 -> task_id = 3 in block 1, 2, 3, 4, 5
                 task_id = 1 in block 6

task_id:    1 -> nearest_speaker_id = 0, farthest_speaker_id = 10
            2 -> nearest_speaker_id = 1, farthest_speaker_id = 5
            3 -> nearest_speaker_id = 5, farthest_speaker_id = 9
"""
# initialize setup
distance_anchoring.initialize_setup()

# start experiment block by block and flush all buffers after each block
distance_anchoring.start_block(sub_id=sub_id, cond_id=cond_id, block_id=block_id, kind=kind)
freefield.flush_buffers(processor='RX81')
