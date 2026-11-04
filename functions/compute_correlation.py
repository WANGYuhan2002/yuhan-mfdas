# === dasmf path bootstrap (auto-added; do not remove) ===
import os as _dasmf_os, sys as _dasmf_sys
_dasmf_p = _dasmf_os.path.abspath(__file__)
while _dasmf_p != _dasmf_os.path.dirname(_dasmf_p) and not _dasmf_os.path.exists(_dasmf_os.path.join(_dasmf_p, '.dasmf_root')):
    _dasmf_p = _dasmf_os.path.dirname(_dasmf_p)
PROJECT_ROOT = _dasmf_p
if _dasmf_os.path.join(PROJECT_ROOT, 'functions') not in _dasmf_sys.path:
    _dasmf_sys.path.insert(0, _dasmf_os.path.join(PROJECT_ROOT, 'functions'))
if PROJECT_ROOT not in _dasmf_sys.path:
    _dasmf_sys.path.insert(0, PROJECT_ROOT)
# === end bootstrap ===
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: salvus
#     language: python
#     name: python3
# ---

#
# +
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
#-- To import a function on a relative path:
import sys
sys.path.append(f"{PROJECT_ROOT}/functions/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

import os
import gc
from IPython.display import clear_output
import pickle
import re
import glob

import h5py, json
from tqdm import tqdm
import time
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count

# -

def moving_average_2d_axis0(data, size):
    """
    Perform a moving average along axis=0 (row direction) on a 2D array,
    keeping other dimensions unchanged.

    Parameters:
    - data: Input 2D array (m, n)
    - size: Window size for the moving average

    Returns:
    - result: 2D array after applying the moving average, with shape (m - size + 1, n)
    """
    # Ensure the input is a 2D array
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    m, n = data.shape

    # Construct the sliding window view along axis=0
    shape = (m - size + 1, size, n)  # (number of windows, window size, number of columns)
    strides = (data.strides[0], data.strides[0], data.strides[1])  # (row step, row step, column step)

    # Create the sliding window view
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # Calculate the mean for each window, resulting in shape (m - size + 1, n)
    return windows.mean(axis=1)


def CCs_comp(data, temp):
    """
    correlation coefficient by every time point.
    template moves along the data.
    
    <START>
    [---temp---] ==> 
    [--------------------------data--------------------------------]
    ...
    ...
    <END>
                                                        [---temp---] 
    [--------------------------data--------------------------------]
    [----------------correlation length-----------------]
    
    INPUT:  2D numpy array, data(i,j): i is time, j is channel number.
            2D numpy array, temp(i,j): i is time, j is channel number.
    
    OUTPUT: 2D numpy array, CCs_f(i,j) and RAWCCs_f(i,j)
            shape = (data.shape[0]-temp.shape[0]+1, data.shape[1])
    """
    if data.shape[1] != temp.shape[1]:
        raise ValueError("data and template should have the same channels")
    if data.shape[0] < temp.shape[0]:
        raise ValueError("data should have more time steps than template")
    
    ch = data.shape[1]
    ld = data.shape[0]
    lt = temp.shape[0]
    
    RAWCCs_f_pad = np.zeros((ld+lt-1,ch))
    RAWCCs_f_idx = np.arange(lt-1,ld,1)
    RAWCCs_f_pad = signal.fftconvolve(data, np.flip(temp, axis=0), mode='full', axes=0)
    RAWCCs_f = RAWCCs_f_pad[RAWCCs_f_idx,:]
    
    meanData_f = moving_average_2d_axis0(data, size=lt)
    mean2Data_f = moving_average_2d_axis0(data**2, size=lt)
    devData_f = np.sqrt(mean2Data_f - meanData_f**2)
    devTemp = np.sqrt(np.var(temp, axis=0))
    
    CCs_f = (RAWCCs_f) / (devData_f * devTemp) / lt
    
    return CCs_f,RAWCCs_f


def preprocess(data,fs,f1,f2):
    """
    INPUT: 2D numpy array, data(i,j): i is time, j is channel number.
    
    OUTPUT: 2D numpy array, data_pre(i,j), pre-processed, same shape
    
    Preprocess the data including:
    1) remove the spatial mean 
    2) remove linear trend, remove temporal mean
    3) apply a bandpass filter
    
    """
    # Remove the spatial mean
    data -= np.median(data, axis=1)[:, np.newaxis]
    # Remove the linear trend and temporal mean
    
    # Filter

    
    data = block_cleaning.detrend(data, type='linear')

    data = block_filters.block_bandpass(data, f1, f2, fs, zerophase=False, taper=0.02)

    data -= np.mean(data, axis=0)
    return data
