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
normalization_method = "mgb_normalization"
port = "COM5"
slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)
speaker_dict = {0: 2.00,
                1: 3.00,
                2: 4.00,
                3: 5.00,
                4: 6.00,
                5: 7.00,
                6: 8.00,
                7: 9.00,
                8: 10.00,
                9: 11.00,
                10: 11.80} # TODO: adjust values to current setup

# TODO: add function to test speaker
# TODO: adjust code to new logging-thing
# TODO: add variable for condition
# TODO: smooth out training() and test() with more functions

#def get_precomputed_USOs (ms):
#    USO_file_folder = DIR / 'data' / 'cutted_USOs' / f'uso_{ms}ms'
 #   USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / f'uso_{ms}ms')
  #  precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names]) # precompute sound files out of USO name list
   # return precomputed_USOs, USO_file_names

USO_file_folder = DIR / 'data' / 'cutted_USOs' / 'uso_300ms'
USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / 'uso_300ms')
precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names])

# initialize setup to connect to the processors
def initialize_setup():
    procs = ["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG") #TODO: Lukas is using 'INFO' now.

# main code for running the experiments training phase
def training(cond_id, sub_id, block_id, task_id, n_reps=1, isi=2):

    # set condition for this block
    if task_id == 1:
        speaker_dic = speaker_dict # use all speakers
    elif task_id == 2:
        speaker_dic = {k: v for k, v in speaker_dict.items() if 1 <= k <= 6} # only use speakers with index 1 to 6
    else:
        return print('You can only set task_id to 1 or 2')

    # initialize sequence with corresponding conditions
    seq = slab.Trialsequence(conditions=1, n_reps=n_reps)
    for trial in seq:

        # get random USO sound
        random_index = random.randint(0, len(precomputed_USOs) - 1)
        USO = slab.Sound(precomputed_USOs[random_index])
        stim_id = USO_file_names[random_index]

        # read slider value and convert it to speaker value
        slider_value = get_slider_value()
        closest_speaker = min(speaker_dic, key=lambda k: abs(speaker_dic[k] - slider_value)) # calculates speaker which is closest to distance of the slider value

        # play USO from speaker corresponding to to slider value
        USO = apply_mgb_equalization(signal=USO, speaker=freefield.pick_speakers(closest_speaker)[0])
        freefield.set_signal_and_speaker(signal=USO, speaker=closest_speaker, equalize=False)
        freefield.play(kind=1, proc='RX81')

        # finish this trial
        event_id = seq.this_n + 1
        print(f'Trial: {event_id}')
        print(f'Slider value: {slider_value}')
        print(f'Closest speaker: {closest_speaker}')
        # freefield.flush_buffers(processor='RX81')
        # freefield.write(tag='data0', value=0, processors='RX81') # TODO: is it enough or do I have to send all other buffer to 99 or do I dont have to flush the buffer. Only fush buffer after experiment
        time.sleep(isi)

        # save results
        save_results(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id,
                     event_id=event_id, stim_id=stim_id, speaker_id=closest_speaker, response=slider_value,
                     response_time=np.nan, n_reps=n_reps, isi=isi)
    print("Done with training")

# main code for running the experiments test phase
def test(cond_id, sub_id, block_id, task_id, n_reps=1, isi=0): # TODO: think about isi

    # set condition for this block
    if task_id == 1: # n_reps = 8 for 88 trials
        nearest_speaker = 0
        farthest_speaker = 10
    elif task_id == 2: # n_reps = 15 for 90 trials
        nearest_speaker = 1
        farthest_speaker = 6
    elif task_id == 3:# n_reps = 8 for 88 trials
        nearest_speaker = 4
        farthest_speaker = 9
    else:
        return print('You can only set task_id to 1, 2 or 3')

    # initialize sequence with corresponding conditions
    speakers = list(range(nearest_speaker, farthest_speaker + 1))
    seq = slab.Trialsequence(conditions=speakers, n_reps=n_reps)
    for speaker in seq:

        # get random USO sound
        random_index = random.randint(0, len(precomputed_USOs) - 1)
        USO = slab.Sound(precomputed_USOs[random_index])
        stim_id = USO_file_names[random_index]

        # prepare and playing USO
        USO = apply_mgb_equalization(signal=USO, speaker=freefield.pick_speakers(speaker)[0])
        freefield.set_signal_and_speaker(signal=USO, speaker=speaker, equalize=False)
        freefield.play(kind=1, proc='RX81')

        # wait for response and read it
        time_before = time.time()
        slider_value = get_slider_value()
        time_after = time.time()

        # finish this trial
        response_time = time_after - time_before
        event_id = seq.this_n + 1
        print('Trial:', event_id)
        print('speaker_id', speaker)
        print('response', response)
        # freefield.flush_buffers(processor='RX81')
        # freefield.write(tag='data0', value=0, processors='RX81') # clear buffer
        time.sleep(isi)

        # save data event by event
        save_results(cond_id=cond_id, sub_id=sub_id, block_id=block_id, task_id=task_id,
                     event_id=event_id, stim_id=stim_id, speaker_id=speaker, response=slider_value, response_time=response_time,
                     n_reps=n_reps, isi=isi)
    print("Done with test")


def save_results(cond_id, sub_id, block_id, task_id, event_id, stim_id, speaker_id, response, response_time, n_reps, isi):

    # create file name
    file_name = DIR / 'results' / f'results_cond-{cond_id}_sub-{sub_id}_block-{block_id}_task-{task_id}.csv'
    if file_name.exists():
        df_curr_results = pd.read_csv(file_name)
    else:
        df_curr_results = pd.DataFrame()

    # convert values in desired data types
    cond_id = int(cond_id)
    sub_id = int(sub_id)
    block_id = int(block_id)
    task_id = int(task_id)
    event_id = int(event_id)
    stim_id = str(stim_id)
    speaker_id = int(speaker_id)
    response = float(response)
    response_time = float(response_time)
    n_reps = int(n_reps)
    isi = float(isi)

    # building current data structure
    new_row = {'cond_id' : cond_id,
        'sub_id' : sub_id,
        'block_id' : block_id,
        'task_id' : task_id,
        'event_id' : event_id,
        'stim_id' : stim_id,
        'speaker_id' : speaker_id,
        'response' : response,
        'response_time' : response_time,
        'n_reps' : n_reps,
        'isi' : isi}

    # add row to df and add df to csv file
    df_curr_results = df_curr_results._append(new_row, ignore_index=True)
    df_curr_results.to_csv(file_name, mode='w', header=True, index=False)

# read and return slider value
def get_slider_value(serial_port=slider, in_metres=True):
    serial_port.flushInput()
    buffer_string = ''
    while True:
        while serial_port.inWaiting() == 0: # added waiting loop until new values are in buffer
            time.sleep(0.05)
        buffer_string += serial_port.read(serial_port.inWaiting()).decode("ascii")
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

def set_ids(cond_id, block_id):
    if cond_id == 1:
        if block_id == 6:
            task_id = 3
        elif block_id == 4:
            task_id = 2
        else:
            task_id = 1
        return block_id, task_id
    if cond_id == 2:
        if block_id == 6:
            task_id = 1
        else:
            task_id = 2
    return block_id, task_id
