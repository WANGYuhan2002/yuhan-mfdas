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


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
#-- To import a function on a relative path:
import sys
sys.path.append(f"{PROJECT_ROOT}/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

import os
import gc
from IPython.display import clear_output
import pickle

import h5py, json
import numpy as np
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


#!__________________________functions_____________________________
sys.path.append("./")
from functions.readers import Read_CCs_stack_h5
from functions.readers import write_pkl, load_pkl, write_h5, load_h5
from functions.plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, generate_gs_grid
from functions.trigger import Trigger_with_MADarray
from functions.readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS


def simple_corr_coef_1D(a,b):
    # a and b are 1D array
    if len(a) != len(b):
        print('The length of a and b are not the same')
        return
    a = a - np.mean(a)
    b = b - np.mean(b)
    corr = np.sum(a*b) / np.sqrt(np.sum(a**2)*np.sum(b**2))
    return corr
def simple_corr_coef_2D(A,B):
    # A and B are 2D array
    if A.shape != B.shape:
        print('The shape of A and B are not the same')
        return
    A = A - np.mean(A,axis=0)
    B = B - np.mean(B,axis=0)
    corr_chans = np.sum(A*B,axis=0) / np.sqrt(np.sum(A**2,axis=0)*np.sum(B**2,axis=0))
    corr_av = np.mean(corr_chans)
    corr_md = np.median(corr_chans)
    return corr_av

# read the DAS data at peak_time: [peak_time-t_span, peak_time+t_span]
def read_DAS_span(tc, t_left, t_right, freq_band, input_dir,dist_start,dist_end,sel_dist=np.array([0])):
    # t_start = tc - t_span
    # t_end = tc + t_span
    t_start = tc - t_left
    t_end = tc + t_right
    if input_dir == f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/':
        data, headers_data, axis_data = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=input_dir)
        fs = 80
        dx = 16*1.02095198631286
    else:
        print('input dir not considered')
    #-- remove spatial median
    data = data - np.median(data,axis=1)[:,np.newaxis]
    #-- trim
    if len(sel_dist) == 1:
        chan_start = np.ceil(dist_start/dx)
        chan_end = np.floor(dist_end/dx)
        sel_chan = np.arange(chan_start, chan_end,1)
        sel_chan = sel_chan.astype(int)
    else:
        sel_chan = np.array([int(sel_dist[i]/dx)   for i in range(len(sel_dist))])
    data = data[:,sel_chan]
    #-- detrend
    data_detrended = block_cleaning.detrend(data, type='linear')
    data_filtered = block_filters.block_bandpass(data_detrended, freq_band[0], freq_band[1], fs, zerophase=False, taper=0.02)
    
    return data_filtered, fs, dx

#* a function from DATA_tc to DATA_tc_normed
def norm_the_flattened_DATA(DATA_tc,stream_shape,idx_t0=0,idx_t1=None):
    # stream is the data block throwed into each column of DATA_tc
    if idx_t1 is None:
        idx_t1 = stream_shape[0] - 1
    print(f'one stream {stream_shape} as one column of DATA_tc {DATA_tc.shape}')
    print(f'norm the stream\'s every column {stream_shape[0]}:  from {idx_t0} to {idx_t1}')
    DATA_tc_normed = np.zeros(DATA_tc.shape)
    for j in range(stream_shape[1]):
        i0 = j*stream_shape[0] + idx_t0
        i1 = j*stream_shape[0] + idx_t1
        DATA_tc_normed[i0:i1,:] = (DATA_tc[i0:i1,:] - np.mean(DATA_tc[i0:i1,:],axis=0)) / np.std(DATA_tc[i0:i1,:],axis=0) / np.sqrt(i1-i0+1)
    return DATA_tc_normed
