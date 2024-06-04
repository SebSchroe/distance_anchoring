# load necessary modules
import freefield
import slab
import pathlib
import os
import time
import random
import numpy as np
import pandas as pd

# global variables
normalization_method = None
DIR = pathlib.Path(os.curdir)

# def get_precomputed_USOs():
USO_file_folder = DIR / 'data' / 'cutted_USOs' / 'uso_300ms'
USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / 'uso_300ms')
precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names]) # precompute sound files out of USO name list

# new equalization method (universally applicable)
def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def logarithmic_func(x, a, b, c):
    return a * np.log(b * x) + c

def get_log_parameters(distance):
    parameters_file = DIR / "data" / "mgb_equalization_parameters" / "logarithmic_function_parameters.csv"
    parameters_df = pd.read_csv(parameters_file)
    params = parameters_df[parameters_df['speaker_distance'] == distance]
    a, b, c = params.iloc[0][['a', 'b', 'c']]
    return a, b, c

def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal

def get_speaker_normalization_level(speaker):
    a, b, c = get_log_parameters(speaker.distance)
    return logarithmic_func(x=30, a=a, b=b, c=c)

# initialize setup to connect to the processors
def initialize_setup(normalization_algorithm="rms", normalization_sound_type="syllable"): # change normalization setup for my purpose
    global normalization_method
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"], # should stay the same
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]] # has to be adjusted to the slider/rotating wheel/joystick/etc.
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    '''
    normalization_file = DIR / "data" / "calibration" / f"calibration_cathedral_{normalization_sound_type}_{normalization_algorithm}.pkl" # has to be adjusted
    freefield.load_equalization(file=pathlib.Path(normalization_file), frequency=False)
    normalization_method = f"{normalization_sound_type}_{normalization_algorithm}"
    '''
    normalization_method = "mgb_normalization"
    freefield.set_logger("DEBUG")

# def training():

# main code for running the experiments test phase
def test_1(sub_id, test_condition, n_reps, environment='laptop'):

    # select condition of the block
    if test_condition == 1: # n_reps = 5 for 55 trials
        nearest_speaker = 0
        farthest_speaker = 10
    elif test_condition == 2: # n_reps = 9 for 54 trials
        nearest_speaker = 1
        farthest_speaker = 6
    else: # n_reps = 5 for 55 trials
        nearest_speaker = 0
        farthest_speaker = 10
        frequency = 500 # TODO: think about third condition (maybe filter would be best)

    # initialize sequence with corresponding conditions
    speakers = list(range(nearest_speaker, farthest_speaker + 1))
    seq = slab.Trialsequence(conditions=speakers, n_reps=n_reps)
    for speaker in seq:
        # get USO sound
        random_index = random.randint(0, len(precomputed_USOs) - 1)
        USO = slab.Sound(precomputed_USOs[random_index])
        stim_id = USO_file_names[random_index]
        # prepare and playing USO for cathedral or on laptop
        if environment == 'laptop':
            USO.play()
        else:
            USO = apply_mgb_equalization(signal=USO, speaker=speaker)
            freefield.set_signal_and_speaker(signal=USO, speaker=speaker, equalize=False)
            freefield.play(kind=1, proc='RX81')
            freefield.flush_buffers(processor='RX81')

        # wait for response and read it
        time_before = time.time()
        response = input("Enter estimated distance (in m) here: ") # TODO: get correct response method here
        time_after = time.time()
        response_time = time_after - time_before

        # finish this trial
        event_id = seq.this_n + 1 # Q: event_id = seq.this_n? start counting at 0 or 1?
        print('Trial:', event_id)
        time.sleep(USO.duration)

        # save data event by event
        save_results(event_id=event_id, sub_id=sub_id, stage='test',
                     task_id=test_condition, stim_id=stim_id, speaker_id=speaker, response=response,
                     response_time=response_time, normalization_method='normalization_method')
    print("Done with training")


def save_results(event_id, sub_id, stage, task_id, stim_id, speaker_id, response, response_time, normalization_method): # TODO: think about datastructure

    # create file name
    file_name = DIR / 'results' / f'results_{sub_id}_{stage}_{task_id}.csv'
    if file_name.exists():
        df_curr_results = pd.read_csv(file_name)
    else:
        df_curr_results = pd.DataFrame()

    # convert values in desired data types
    event_id = int(event_id)
    sub_id = int(sub_id)
    stage = str(stage)
    task_id = int(task_id) # condition 1, 2 or 3
    stim_id = str(stim_id)
    speaker_id = str(speaker_id)
    response = float(response)
    response_time = float(response_time)
    normalization_method = str(normalization_method)

    # building current data structure
    new_row = {'event_id' : event_id,
        'sub_id' : sub_id,
        'stage' : stage,
        'task_id' : task_id,
        'stim_id' : stim_id,
        'speaker_id' : speaker_id,
        'response' : response,
        'response_time' : response_time,
        'normalization_method' : normalization_method}

    # add row to df
    df_curr_results = df_curr_results._append(new_row, ignore_index=True)
    df_curr_results.to_csv(file_name, mode='w', header=True, index=False)

# def controller_input():
