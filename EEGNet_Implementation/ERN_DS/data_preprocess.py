"""
@Author: Marc Tunnell

Load, Process, Filter, and down-sample .csv EEG data. An implementation to filter data from BCI challenge
    https://kaggle.com/c/inria-bci-challenge/

Some or all of this code has been based on or adapted from the following by Yundong Wang, Zimu Li
    https://github.com/YundongWang/BCI_Challenge/blob/master/preprocess.py
and https://github.com/YundongWang/BCI_Challenge/blob/master/generate_epoch.py
"""

import numpy as np
import os
import mne
import pandas as pd

training = 16
testing = 10
presentation = (0, 1250)
total = abs(presentation[0]) + abs(presentation[1])
stimulus = 60, 60, 60, 60, 100
total_stimulus = sum(stimulus)

trials = 5

starting_hz = 200
target_hz = 128

l_freq = 1
h_freq = 40

nodes = 56
target_dimension = int(total * (target_hz / 1000))

def mne_implementation(data, min_freq, max_freq, fs):
    return mne.filter.filter_data(data=data, method='fir',
                                  l_freq=min_freq, h_freq=max_freq,
                                  sfreq=fs, n_jobs=40)


def mne_ds(data, start, end):
    return mne.filter.resample(x=data, down=float(start / end))




if __name__ == "__main__":




    train_dir = './BCI Challenge/train/'
    test_dir = './BCI Challenge/test/'

    training_data = np.reshape(sorted(os.listdir(train_dir)), newshape=(training, trials))
    testing_data = np.reshape(sorted(os.listdir(test_dir)), newshape=(testing, trials))





    sets = [
        [training_data, train_dir, "Training"], [testing_data, test_dir, "Testing"]
    ]

    """
    The following section filters the code column by column,
    then splits the the columns into the presentation of ERN
    stimuli based on the FeedBackEvent column, then down samples
    the presentation following the onset of the stimulus from
    200 hz to 128 hz.
    """

    for data_set in sets:
        data_list = np.empty(
            (0, total_stimulus, nodes, target_dimension), float)
        for user in data_set[0]:
            tmp = np.empty((0, nodes, target_dimension), float)
            for trial in user:
                load = pd.read_csv(data_set[1] + trial)
                columns = load.columns[1:-2]
                relevant_data = load[load['FeedBackEvent'] == 1].index.to_numpy(dtype='int32')
                total_cols = np.empty((len(relevant_data), target_dimension, 0), float)
                for item in columns:
                    filtered_data = mne_implementation(load[item], l_freq, h_freq, starting_hz)
                    col = np.empty(shape=(len(relevant_data), target_dimension))
                    for k, thing in enumerate(relevant_data):
                        col[k, :] = mne_ds(filtered_data[thing: thing + int(total * (starting_hz / 1000))], starting_hz, target_hz)
                    total_cols = np.dstack((total_cols, col))
                total_cols = np.swapaxes(total_cols, 1, 2)
                tmp = np.vstack((tmp, total_cols))
            tmp = np.reshape(tmp, (1, total_stimulus, nodes, target_dimension))
            data_list = np.vstack((data_list, tmp))
        np.save('./BCI Challenge/{}.npy'.format(data_set[2]), data_list)

