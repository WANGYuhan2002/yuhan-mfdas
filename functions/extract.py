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



def Record_Shift_Bychan(Nt_trig_s, CCs_f, RAWCCs_f, Record):
    """
    Record the maximum CCs value and its time shift to t_trig_s.
    
    INPUT:  1D numpy array, Nt_trig_s(i): time index of the peaks. 
            2D numpy array, CCs_f(i,j): i is time, j is channel number.
            2D numpy array, RAWCCs_f(i,j): i is time, j is channel number.
            dict(), Record = {'Search_win_CCs':        , (dimensionless, time npts)
                              'Search_win_datamax':    , (dimensionless, time npts)
                              }
                              
    OUPUT:  2D numpy array, Nt_CCsmax_f(i,j): time index of the maximum CCs value. 
            2D numpy array, CCsmax_f(i,j): maximum absolute CCs value.
            2D numpy array, RAWCCsmax_f(i,j): absolute RAWCCs value at the maximum absolute CCs value.
            ;i is event, j is channel number.
    """
    search_win = Record['Search_win_CCs']
    
    Ch = CCs_f.shape[1]
    Nps = len(Nt_trig_s)
    
    Nt_CCsmax_f = np.zeros((Nps,Ch))
    CCsmax_f = np.zeros((Nps,Ch))
    RAWCCsmax_f = np.zeros((Nps,Ch))
    
    # loop for each peak
    for n in range(Nps):
        Nt_tmp = Nt_trig_s[n]
        CCs_f_win = np.zeros((search_win*2+1, Ch))
        # 计算当前通道的窗口起始和结束索引
        start = max(0, Nt_tmp - search_win)   # 防止起点越界
        end = min(CCs_f.shape[0], Nt_tmp + search_win + 1)  # 防止终点越界
        # 确保写入目标数组的索引范围一致
        win_len = end - start
        CCs_f_win[:win_len, :] = CCs_f[start:end, :]
        
        CCsmax_f[n, :] = np.max(abs(CCs_f_win), axis=0)
        Nt_CCsmax_f[n, :] = start + np.argmax(abs(CCs_f_win), axis=0)
        # RAWCCsmax_f[n, :] = abs(RAWCCs_f[Nt_CCsmax_f[n, :], :])
        # RAWCCsmax_f[n, :] = abs(RAWCCs_f[Nt_CCsmax_f[n, :].astype(int)])
        RAWCCsmax_f[n, :] = [RAWCCs_f[int(Nt_CCsmax_f[n, j]), j] for j in range(Ch)]
    
    return Nt_CCsmax_f, CCsmax_f, RAWCCsmax_f



def Record_DataMax_Bychan(Nt_trig_s, data, Record):
    """
    Record the maximum CCs value and its time shift to t_trig_s.
    
    INPUT:  1D numpy array, Nt_trig_s(i): time index of the peaks. 
            2D numpy array, data(i,j): i is time, j is channel number.
            dict(), Record = {'Search_win_CCs':        , (dimensionless, time npts)
                              'Search_win_datamax':    , (dimensionless, time npts)
                              }
                              
    OUPUT:  2D numpy array, Nt_datamax_f(i,j): time index of the maximum absolute data value. 
            2D numpy array, datamax_f(i,j): maximum absolute data value.
            ;i is event, j is channel number.
    """
    search_win = Record['Search_win_datamax']    
    Ch = data.shape[1]
    Nps = len(Nt_trig_s)
    
    Nt_datamax_f = np.zeros((Nps,Ch))
    datamax_f = np.zeros((Nps,Ch))
    
    # loop for each peak
    for n in range(Nps):
        Nt_tmp = Nt_trig_s[n]
        data_win = np.zeros((search_win*2+1, Ch))
        # 计算当前通道的窗口起始和结束索引
        start = max(0, Nt_tmp - search_win)   # 防止起点越界
        end = min(data.shape[0], Nt_tmp + search_win + 1)  # 防止终点越界
        # 确保写入目标数组的索引范围一致
        win_len = end - start
        data_win[:win_len, :] = data[start:end, :]
        
        datamax_f[n, :] = np.max(abs(data_win), axis=0)
        Nt_datamax_f[n, :] = start + np.argmax(abs(data_win), axis=0)    
    return Nt_datamax_f, datamax_f


