# load necessary modules
import freefield
import slab
import pathlib
import os
import time
import random
import numpy as np
import pandas as pd
import LedControl

# TODO: make code more stable with try/except

# global variables
DIR = pathlib.Path(os.curdir)
normalization_method = "mgb_normalization"
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
                10: 12.00}

# assign results folder
results_folder = DIR / 'results'

# define columns for results saving
header = ['sub_id', 'cond_id', 'block_id', 'task_id', 'event_id', 'stim_id', 'speaker_id', 'led_id', 'led_distance', 'response_time', 'n_reps', 'isi']

# precompute USOs
USO_file_folder = DIR / 'data' / 'cutted_USOs' / 'uso_300ms_new'
USO_file_names = os.listdir(DIR / 'data' / 'cutted_USOs' / 'uso_300ms_new')
precomputed_USOs = slab.Precomputed([slab.Sound(os.path.join(USO_file_folder, f)) for f in USO_file_names])

# initialize setup to connect to the processors
def initialize_setup():
    freefield.freefield.initialize(setup="cathedral", default="play_birec", connection="USB", zbus=False)
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG")

# main code to execute each block
def start_block(sub_id, cond_id, block_id):
    def execute_procedure(procedure, task_id, n_reps, isi):
        procedure(sub_id=sub_id, cond_id=cond_id, block_id=block_id, task_id=task_id, n_reps=n_reps, isi=isi)

    if sub_id == 'test_run':
        if cond_id == 1:
            if block_id in [1, 2, 4]:
                execute_procedure(test, 2, 1, 0.3)
            elif block_id in [3, 5]:
                execute_procedure(training, 2, 11, 2)
            elif block_id == 6:
                execute_procedure(test, 1, 1, 0.3)
            else:
                print('block_id can only be 1 to 6')

        elif cond_id == 2:
            if block_id in [1, 2, 4]:
                execute_procedure(test, 3, 1, 0.3)
            elif block_id in [3, 5]:
                execute_procedure(training, 3, 11, 2)
            elif block_id == 6:
                execute_procedure(test, 1, 1, 0.3)
            else:
                print('block_id can only be 1 to 6')
        else:
            print('cond_id can only be 1 or 2')

    else:
        if cond_id == 1:
            if block_id in [1, 2, 4]:
                execute_procedure(test, 2, 15, 0.3)
            elif block_id in [3, 5]:
                execute_procedure(training, 2, 90, 2)
            elif block_id == 6:
                execute_procedure(test, 1, 15, 0.3)
            else:
                print('block_id can only be 1 to 6')

        elif cond_id == 2:
            if block_id in [1, 2, 4]:
                execute_procedure(test, 3, 15, 0.3)
            elif block_id in [3, 5]:
                execute_procedure(training, 3, 90, 2)
            elif block_id == 6:
                execute_procedure(test, 1, 15, 0.3)
            else:
                print('block_id can only be 1 to 6')
        else:
            print('cond_id can only be 1 or 2')

# main code for executing training block
def training(sub_id, cond_id, block_id, task_id, n_reps, isi):

    # set speaker set depending on task_id
    speaker_set = get_speaker_set(task_id)

    # filter speaker_dict depending on speaker set
    speaker_dic = {key: speaker_dict[key] for key in speaker_set if key in speaker_dict}

    # prepare results table
    table = slab.ResultsTable(columns=header, subject=f'sub-{sub_id}', folder=results_folder, filename=f'cond-{cond_id}_block-{block_id}')

    # initialize sequence with corresponding conditions
    seq = slab.Trialsequence(conditions=1, n_reps=n_reps)
    for trial in seq:

        # get random USO sound
        USO, stim_id = get_random_USO()

        # read slider value and convert it to speaker value
        led_id = LedControl.CURR_LED
        led_distance = LedControl.led_to_meter(curr_led=led_id)
        closest_speaker_id = min(speaker_dic, key=lambda k: abs(speaker_dic[k] - led_distance)) # calculates speaker which is closest to distance of the slider value

        # equalize USO und play from speaker
        play_USO_from_speaker(USO, closest_speaker_id)

        # finish this trial
        event_id = seq.this_n + 1
        print(f'Trial: {event_id}')
        print(f'led_id: {led_id}')
        print(f'LED distance: {led_distance:.2f}')
        print(f'Closest speaker: {closest_speaker_id}')
        time.sleep(isi)

        # save results trial by trial
        row = table.Row(sub_id=sub_id, cond_id=cond_id, block_id=block_id, task_id=task_id,
                        event_id=event_id, stim_id=stim_id, speaker_id=closest_speaker_id,
                        led_id=led_id, led_distance=led_distance, response_time=np.nan, n_reps=n_reps, isi=isi)
        table.write(row)

    print("Done with training")

# main code for executing test block
def test(sub_id, cond_id, block_id, task_id, n_reps, isi):

    # set speaker set depending on task_id
    speaker_set = get_speaker_set(task_id)

    # prepare results table
    table = slab.ResultsTable(columns=header, subject=f'sub-{sub_id}', folder=results_folder, filename=f'cond-{cond_id}_block-{block_id}')

    # initialize sequence
    seq = slab.Trialsequence(conditions=speaker_set, n_reps=n_reps)
    for speaker_id in seq:

        # get random USO sound
        USO, stim_id = get_random_USO()

        # equalize USO und play from speaker
        play_USO_from_speaker(USO, speaker_id)

        # wait for response and read it
        time_before = time.time()
        led_id = LedControl.get_led()
        led_distance = LedControl.led_to_meter(curr_led=led_id)
        time_after = time.time()
        response_time = time_after - time_before

        # finish this trial
        event_id = seq.this_n + 1
        print(f'Trial: {event_id}')
        print(f'speaker_id: {speaker_id}')
        print(f'led_id: {led_id}')
        print(f'LED distance: {led_distance}')

        time.sleep(isi)

        # save results trial by trial
        row = table.Row(sub_id=sub_id, cond_id=cond_id, block_id=block_id, task_id=task_id,
                        event_id=event_id, stim_id=stim_id, speaker_id=speaker_id, led_id=led_id,
                        led_distance=led_distance, response_time=response_time, n_reps=n_reps, isi=isi)
        table.write(row)

    print("Done with test")


def get_speaker_set(task_id):

    # define speaker set depending on task_id
    if task_id == 1:
        speaker_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif task_id == 2:
        speaker_set = [1, 2, 3, 4, 5]
    elif task_id == 3:
        speaker_set = [5, 6, 7, 8, 9]
    else:
        return print('You can only set task_id to 1, 2 or 3')

    return speaker_set

def get_random_USO():

    # pick a random USO out of the precomputed USOs and return it together with the file name
    random_index = random.randint(0, len(precomputed_USOs) - 1)
    USO = slab.Sound(precomputed_USOs[random_index])
    stim_id = USO_file_names[random_index]
    return USO, stim_id

def play_USO_from_speaker(USO, speaker):

    # equalize loudness
    USO = apply_mgb_equalization(signal=USO, speaker=freefield.pick_speakers(speaker)[0])

    # play USO from speaker
    freefield.set_signal_and_speaker(signal=USO, speaker=speaker, equalize=False)
    freefield.play(kind=1, proc='RX81')

# new equalization method (universally applicable)
def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal

def get_log_parameters(distance):
    parameters_file = DIR / "data" / "mgb_equalization_parameters" / "logarithmic_function_parameters.csv"
    parameters_df = pd.read_csv(parameters_file)
    params = parameters_df[parameters_df['speaker_distance'] == distance]
    a, b, c = params.iloc[0][['a', 'b', 'c']]
    return a, b, c

def logarithmic_func(x, a, b, c):
    return a * np.log(b * x) + c

def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def get_speaker_normalization_level(speaker):
    a, b, c = get_log_parameters(speaker.distance)
    return logarithmic_func(x=30, a=a, b=b, c=c)
