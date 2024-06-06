# run this code line by line and only set limited variables

import distance_anchoring
sub_id = 0

"""
task_id:    1 -> nearest_speaker = 0, farthest_speaker = 10
            2 -> nearest_speaker = 1, farthest_speaker = 6
            3 -> nearest_speaker = 0, farthest_speaker = 10 + manipulation of reverb
"""

distance_anchoring.initialize_setup()

# block 1: visual limited test
distance_anchoring.test(sub_id=sub_id, block_id=1, task_id=1, n_reps=1, play_via='cathedral')

# block 2: visual unlimited test
distance_anchoring.test(sub_id=sub_id, block_id=2, task_id=1, n_reps=1, play_via='cathedral')

# block 3: visual unlimited training
distance_anchoring.training()

# block 4: visual unlimited test
distance_anchoring.test()

# block 5: visual unlimited training
distance_anchoring.training()

# block 6: visual unlimited test
distance_anchoring.test()
