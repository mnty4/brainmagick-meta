# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Force speech alignement manually performed on privately shared audio
files, using Gentle. There seems to a significant discrepency on the
result of run_15 around 170 s.
"""
import json
import typing as tp
from urllib.request import urlretrieve
from zipfile import ZipFile

import mne
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import spacy

import matplotlib.pyplot as plt

def test_bodrick_eeg():
    mat = loadmat('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/broderick2019/download/Natural Speech/EEG/Subject1/Subject1_Run1.mat')
    assert mat["fs"][0][0] == 128
    ch_types = ["eeg"] * 128
    # FIXME montage?
    montage = mne.channels.make_standard_montage("biosemi128")
    info = mne.create_info(montage.ch_names, 128.0, ch_types)
    eeg_preprocessed = mat["eegData"]

    raw = mne.io.RawArray(eeg_preprocessed.T, info)        
    raw.set_montage(montage)
    # do a 1-50 Hz bandpass filter
    raw.filter(1, 50)
    # do a 50 Hz notch filter
    raw.notch_filter(50)

    # raw.plot_sensors(show_names=True)
    # plot
    raw.plot(duration=60, n_channels=30, scalings="auto")
    raw.plot_psd(fmax=50)
    plt.show()
# if main
if __name__ == "__main__":
    test_bodrick_eeg()