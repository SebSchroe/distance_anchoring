# run this code line by line and only set limited variables
import freefield
import distance_anchoring

sub_id = 0
cond_id = 1

"""
condition:  1 -> task_id = 1 in block 1, 2, 3, 5
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

# block 1: test visual limited
block_id, task_id = set_ids(cond_id=cond_id, block_id=1)
distance_anchoring.test(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id, n_reps=8, isi=0.3) # n_reps of 8 equals 88 stimuli in condition 1, n_reps of 15 equals 90 stimuli in condition 2
freefield.flush_buffers(processor='RX81')

# block 2: test visual
# block_id, task_id = set_ids(cond_id=cond_id, block_id=2)
# distance_anchoring.test(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id, n_reps=2, isi=0)

# block 3: training visual
block_id, task_id = set_ids(cond_id=cond_id, block_id=3)
distance_anchoring.training(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id, n_reps=90, isi=2) # 90 trials with 2 seconds isi equals 180 seconds (3 minutes) testing
freefield.flush_buffers(processor='RX81')

# block 4: visual unlimited test
block_id, task_id = set_ids(cond_id=cond_id, block_id=4)
distance_anchoring.test(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id, n_reps=15, isi=0.3) # n_reps of 15 equals 90 stimuli in both conditions
freefield.flush_buffers(processor='RX81')

# block 5: visual unlimited training
# distance_anchoring.training()

# block 6: visual unlimited test
# distance_anchoring.test()
