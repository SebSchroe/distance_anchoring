# import necessary modules
import freefield
import distance_anchoring
import LedControl

# set global variables
sub_id = '11'  # choose 'test_run' for functionality check (training with task_id 1 and n_reps 11)
cond_id = 1
block_id = 1

"""
cond_id:    1 -> block 1, 2, 3, 4, 5 with task_id = 2
                 block 6 with task_id = 1
            2 -> block 1, 2, 3, 4, 5 with task_id = 3
                 block 6 with task_id = 1

task_id:    1 -> nearest_speaker_id = 0, farthest_speaker_id = 10
            2 -> nearest_speaker_id = 1, farthest_speaker_id = 5
            3 -> nearest_speaker_id = 5, farthest_speaker_id = 9
"""
# initialize setup
distance_anchoring.initialize_setup()
LedControl.start_led_control()

# start experiment block by block and flush all buffers after each block
# waiting for participant pressing 'A' to start
while LedControl.CURR_COMMAND != 'green':
    print('Waiting for input', end='\r')
    continue

distance_anchoring.start_block(sub_id=sub_id, cond_id=cond_id, block_id=block_id)
freefield.flush_buffers(processor='RX81')

LedControl.stop_led_control()

