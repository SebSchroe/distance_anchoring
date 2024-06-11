# load necessary modules
import freefield
import slab
import pathlib
import os
import time
import random
import serial
import numpy as np
import pandas as pd

# global variables
DIR = pathlib.Path(os.curdir)
normalization_method = None
port = "COM5"
slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)
speaker_dict = {0: 2.10,
                1: 2.96,
                2: 3.84,
                3: 4.74,
                4: 5.67,
                5: 6.62,
                6: 7.60,
                7: 8.80,
                8: 9.64,
                9: 10.70,
                10: 11.78} # TODO: adjust values to current setup

#def get_precomputed_USOs (ms):
#    USO_file_folder = DIR / 'data' / 'cutted_USOs' / f'uso_{ms}ms'
 #   USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / f'uso_{ms}ms')
  #  precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names]) # precompute sound files out of USO name list
   # return precomputed_USOs, USO_file_names

USO_file_folder = DIR / 'data' / 'cutted_USOs' / f'uso_300ms'
USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / f'uso_300ms')
precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names])

# initialize setup to connect to the processors
def initialize_setup(normalization_algorithm="rms", normalization_sound_type="syllable"): # TODO: attributes can be removed
    global normalization_method
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]] # is redundant
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

# main code for running the experiments training phase
def training(sub_id, block_id, task_id, training_duration=90, isi=2):

    # set condition for this block
    if task_id == 1:
        speaker_dic = speaker_dict # use all speakers
    elif task_id == 2:
        speaker_dic = {k: v for k, v in speaker_dict.items() if 1 <= k <= 6} # only use speakers with index 1 to 6
    else:
        return print('You can only set task_id to 1 or 2')

    # initialize sequence with corresponding conditions
    # precomputed_USOs = get_precomputed_USOs(ms=300)
    seq = slab.Trialsequence(conditions=int(training_duration / isi))
    for trial in seq:
        # get random USO sound
        random_index = random.randint(0, len(precomputed_USOs) - 1)
        USO = slab.Sound(precomputed_USOs[random_index])
        stim_id = USO_file_names[random_index]

        # read slider value and convert it to speaker value
        slider_value = get_slider_value() # button has to be pressed all the time
        # slider_value = seq.this_n
        closest_speaker = min(speaker_dic, key=lambda k: abs(speaker_dic[k] - slider_value)) # calculates speaker which is closest to distance of the slider value
        # play USO from speaker corresponding to to slider value
        USO = apply_mgb_equalization(signal=USO, speaker=freefield.pick_speakers(closest_speaker)[0])
        freefield.set_signal_and_speaker(signal=USO, speaker=closest_speaker, equalize=False)
        freefield.play(kind=1, proc='RX81')

        # finish this trial
        event_id = seq.this_n + 1 # TODO: event_id = seq.this_n? start counting at 0 or 1?
        print('Trial:', event_id)
        print('Slider value', slider_value)
        print('Closest speaker', closest_speaker)
        freefield.flush_buffers(processor='RX81')
        time.sleep(0) # used to be isi
        # save results
        save_results(event_id=event_id, sub_id=sub_id, block_id=block_id, task_id=task_id,
                     stim_id=stim_id, speaker_id=closest_speaker, response_time=1,
                     response=slider_value, training_duration=training_duration,
                     isi=isi, normalization_method='normalization_method')
    print("Done with training")

# main code for running the experiments test phase
def test(sub_id, block_id, task_id, n_reps, play_via='cathedral'):

    # set condition for this block
    if task_id == 1: # n_reps = 5 for 55 trials
        nearest_speaker = 0
        farthest_speaker = 10
    elif task_id == 2: # n_reps = 9 for 54 trials
        nearest_speaker = 1
        farthest_speaker = 6
    elif task_id == 3:# n_reps = 5 for 55 trials
        nearest_speaker = 0
        farthest_speaker = 10
        frequency = 500 # TODO: think about third condition (maybe filter would be best)
    else:
        return print('You can only set task_id to 1, 2 or 3')

    # initialize sequence with corresponding conditions
    speakers = list(range(nearest_speaker, farthest_speaker + 1))
    #precomputed_USOs = get_precomputed_USOs(ms=300)
    seq = slab.Trialsequence(conditions=speakers, n_reps=n_reps)
    for speaker in seq:
        # get random USO sound
        random_index = random.randint(0, len(precomputed_USOs) - 1)
        USO = slab.Sound(precomputed_USOs[random_index])
        stim_id = USO_file_names[random_index]
        # prepare and playing USO on laptop or in cathedral
        if play_via == 'cathedral':
            USO = apply_mgb_equalization(signal=USO, speaker=freefield.pick_speakers(speaker)[0])
            freefield.set_signal_and_speaker(signal=USO, speaker=speaker, equalize=False)
            freefield.play(kind=1, proc='RX81')
        else:
            USO.play()

        # wait for response and read it
        time_before = time.time()
        if play_via == 'cathedral':
            response = get_slider_value()
        else:
            response = input("Enter estimated distance (in m) here: ")
        time_after = time.time()

        # finish this trial
        response_time = time_after - time_before
        event_id = seq.this_n + 1 # Q: event_id = seq.this_n? start counting at 0 or 1?
        print('Trial:', event_id)
        print('speaker_id', speaker)
        print('response', response)
        if play_via == 'cathedral':
            freefield.flush_buffers(processor='RX81')
        time.sleep(USO.duration)

        # save data event by event
        save_results(event_id=event_id, sub_id=sub_id, block_id=block_id, task_id=task_id,
                     stim_id=stim_id, speaker_id=speaker, response=response,
                     response_time=response_time, training_duration=5, isi=5, normalization_method='normalization_method')
    print("Done with test")


def save_results(event_id, sub_id, block_id, task_id, stim_id, speaker_id, response, response_time, training_duration, isi, normalization_method): # TODO: think about datastructure

    # create file name
    file_name = DIR / 'results' / f'results_sub-{sub_id}_block-{block_id}_task-{task_id}.csv'
    if file_name.exists():
        df_curr_results = pd.read_csv(file_name)
    else:
        df_curr_results = pd.DataFrame()

    # convert values in desired data types
    event_id = int(event_id)
    sub_id = int(sub_id)
    block_id = str(block_id)
    task_id = int(task_id) # condition 1, 2 or 3
    stim_id = str(stim_id)
    speaker_id = str(speaker_id)
    response = int(response)
    response_time = float(response_time)
    training_duration = int(training_duration)
    isi = int(isi)
    normalization_method = str(normalization_method)

    # building current data structure
    new_row = {'event_id' : event_id,
        'sub_id' : sub_id,
        'block_id' : block_id,
        'task_id' : task_id,
        'stim_id' : stim_id,
        'speaker_id' : speaker_id,
        'response' : response,
        'response_time' : response_time,
        'training_duration' : training_duration,
        'isi' : isi,
        'normalization_method' : normalization_method}

    # add row to df
    df_curr_results = df_curr_results._append(new_row, ignore_index=True)
    df_curr_results.to_csv(file_name, mode='w', header=True, index=False)

# read and return slider value
def get_slider_value(serial_port=slider, in_metres=True):
    serial_port.flushInput()
    buffer_string = ''
    while True:
        buffer_string = buffer_string + serial_port.read(serial_port.inWaiting()).decode("ascii")
        if '\n' in buffer_string:
            lines = buffer_string.split('\n')  # Guaranteed to have at least 2 entries
            last_received = lines[-2].rstrip()
            if last_received:
                last_received = int(last_received)
                if in_metres:
                    last_received = np.interp(last_received, xp=[0, 1023], fp=[0, 15]) - 1.5
                return last_received

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
