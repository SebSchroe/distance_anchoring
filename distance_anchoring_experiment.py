# run this code line by line and only set limited variables

import distance_anchoring
sub_id = 0

distance_anchoring.initialize_setup()
distance_anchoring.test_1(sub_id=1, test_condition=1, n_reps=1, environment='cathedral')






'''
# block 1: test blind -> test sighted
distance_anchoring.start_test(sub_id=sub_id, test_condition=1)
input("Press Enter to continue Experiment...")
distance_anchoring.start_test(sub_id=sub_id, test_condition=1)

# block 2: all sighted, training -> test -> training -> test with special condition
distance_anchoring.start_training(sub_id=sub_id, training_condition=1)

distance_anchoring.start_test(sub_id=sub_id, test_condition=2)

distance_anchoring.start_training(sub_id=sub_id, training_condition=1)

distance_anchoring.start_test(sub_id=sub_id, test_condition=3)




# training conditions
distance_anchoring.start_training(sub_id=sub_id, training_condition=1)
"""
    training_conditions:    1 -> nearest_speaker = 0, farthest_speaker = 10
                            2 -> nearest_speaker = 1, farthest_speaker = 6
"""
# test conditions
distance_anchoring.start_test(sub_id=sub_id, test_condition=1)
"""
    test_conditions:        1 -> nearest_speaker = 0, farthest_speaker = 10
                            2 -> nearest_speaker = 1, farthest_speaker = 6
                            3 -> manipulation of reverb
"""
'''
