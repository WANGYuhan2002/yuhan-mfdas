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


# 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd

#-- To import a function on a relative path:
import sys
# 注：原指向外部 stalta_clustering/ 的 sys.path 已移除（本发布版与该外部工程零耦合，
#     STA/LTA 目录改由 IN_STALTA_cata/ 提供；本模块的 STA/LTA 检测函数在本发布版中未被调用）。
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

sys.path.append("./")
from functions.plot import Ticks_xaxis_day_or_hour, waterfall_date

import os
import gc
from matplotlib import ticker
import matplotlib.dates as mdates


def plot_if_really_a_event_STALTA(event_list_tmp,evt_NO_inlist,ff,input_dir,fig):
    dx = 16 * 1.02095198631286
    fs = 80
#     t_evt = event_list_tmp['peak_time'][evt_NO_inlist]
#     A_evt = event_list_tmp['peak_amp'][evt_NO_inlist]
    
#     t_start = t_evt - timedelta(seconds=15)
#     t_end = t_evt + timedelta(seconds=15)
#     #####--1--############ Read plot data #####################
#     # data, headers_data, axis_data = load_das_h5.load_das_custom(t_start+data_plot_offset, t_end+data_plot_offset, input_dir='../2024_Greece_test_tmp_40Hz/', verbose=False, convert=True,ichan=headers['selected_channels'])
#     data, headers_data, axis_data = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=input_dir)
#     #-- remove spatial median
#     data = data - np.median(data,axis=1)[:,np.newaxis]
#     #-- detrend
#     data_detrended = block_cleaning.detrend(data, type='linear')
#     #-- filter
#     f1 = ff[0]
#     f2 = ff[1]
#     data_filtered = block_filters.block_bandpass(data_detrended, f1, f2, fs, zerophase=False, taper=0.02)
    
#     ##########################################################################
#     ax = fig.subplots(1,1,figsize=(15, 10))
#     #- data:  t_start~t_end
#     # waterfall_simple(data_filtered,ax[1],0.0,fs,dx,t_start,'(nm/m)/s')
#     waterfall_date(ax, data, fs, t_start, dx, d_start, unit, vlim=0)
#     ax.set_title('<data> on {0} GMT {1}'.format((t_start).strftime('%Y.%m.%d')))    
#     return ax



# +


def sta_lta_centred_unnorm(signal, nsta, nlta):
    """
    Calculates the ratio of the average of a^2 in a short-term (signal) window to a
    preceding long-term (noise) window. STA/LTA value is assigned to the end of the LTA
    /one sample before the start of the STA.

    Parameters
    ----------
    signal : array-like
        Signal array
    nsta : int
        Number of samples in short-term window
    nlta : int
        Number of samples in long-term window

    Returns
    -------
    sta / lta : array-like
        Ratio of a^2 in a short term average window to a preceding long term average
        window. STA/LTA value is assigned to end of LTA window / one sample before the
        start of STA window -- "centred".

    """
    # Cumulative sum to calculate moving average
    sta = np.cumsum(signal**2)
    sta = np.require(sta, dtype=float)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta * 2 :]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    sta[: (nlta - 1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(float).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = 0.0

    return sta / lta



def sta_lta_centred(signal, nsta, nlta):
    """
    Calculates the ratio of the average of a^2 in a short-term (signal) window to a
    preceding long-term (noise) window. STA/LTA value is assigned to the end of the LTA
    /one sample before the start of the STA.

    Parameters
    ----------
    signal : array-like
        Signal array
    nsta : int
        Number of samples in short-term window
    nlta : int
        Number of samples in long-term window

    Returns
    -------
    sta / lta : array-like
        Ratio of a^2 in a short term average window to a preceding long term average
        window. STA/LTA value is assigned to end of LTA window / one sample before the
        start of STA window -- "centred".

    """
    # Cumulative sum to calculate moving average
    sta = np.cumsum(signal**2)
    sta = np.require(sta, dtype=float)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta * 2 :]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    sta[: (nlta - 1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(float).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = 0.0

    # normalize the sta/lta by the max or MAD
    norm = 'MAD'
    if norm == 'MAD':
        mad = np.median(np.abs(sta/lta - np.median(sta/lta)))
        stalta_normed = sta / lta /mad
    if norm == 'MAX':
        stalta_normed = sta / lta /np.max(sta/lta) *2
    return stalta_normed


def sta_lta_centred_multichan(stream, nsta, nlta):
    csl_waterfall = np.zeros((stream.shape[0],stream.shape[1]))
    for idx_channel in range(0,stream.shape[1]):
        # csl = classic_sta_lta(trace.data, int(t_sta * df), int(t_lta * df))
        # calculate centered STA/LTA value, normalized by MAD, for each channel
        csl = sta_lta_centred(stream[:,idx_channel], nsta, nlta)
        csl_waterfall[:,idx_channel] = csl
    return csl_waterfall
# +
def trig_onechan_and_outindex(csl_onechan,threshold,minimal_event_interval,fs_das):
    csl = csl_onechan
    csl_01 = np.zeros(csl.shape)
    csl_01[csl > threshold] = 1
    index_1 = np.where(csl_01 == 1)
    # check if there is no event
    if len(index_1[0]) == 0:
        return np.array([])
    elif len(index_1[0]) > 0:
        # cluster the events with minimal_event_interval
        # index_2 saves the index of the first peak in each cluster
        index_2 = [] 
        index_2.append(index_1[0][0])
        for i in range(1, len(index_1[0])):
            if index_1[0][i] - index_1[0][i-1] > minimal_event_interval*fs_das:
                index_2.append(index_1[0][i])
        # print('start index',index_2)

        # index_3 saves the index of the last peak in each cluster
        index_3 = []
        index_3.append(index_1[0][-1])
        for i in range(len(index_1[0])-2, -1, -1):
            if index_1[0][i+1] - index_1[0][i] > minimal_event_interval*fs_das:
                index_3.append(index_1[0][i])
        index_3 = index_3[::-1]
        # print('end index' ,index_3)

        # pick the maximum in each cluster
        index_arg_max = []
        index_2 = np.array(index_2)
        index_3 = np.array(index_3)
        for i in range(len(index_2)):
            if index_2[i] == index_3[i]:
                index_arg_max.append(index_2[i])
            else:
                index_arg_max.append(index_2[i] + np.argmax(abs(csl[index_2[i]:index_3[i]])))
        return np.array(index_arg_max)
        
def trig_allchan_and_collapsed_index(csl,threshold,minimal_event_interval,fs_das):
    # in index_t_trig, it is already sure that the (idx2-idx1)*fs_das>minimal_event_interval
    # then in index_arg_max, collapse the events that are too close to each other, and count how many are triggered
    #!!!!!! need to rewrite the loop if you want to not only copy the score but also the index of channels triggered
    index_arg_max = np.array([])
    score = np.array([])
    empty_no = 0
    csl_value = np.array([])
    #! 以下输出的trig time是基于迭代的，因此具有一定的浮动性，如果想要更稳定的trig time，需要取平均值。更稳定的 trig_tim 会带来更好的与NOA吻合，以及approximate corr matrix
    for ichan in range(csl.shape[1]):
        index_t_trig = trig_onechan_and_outindex(csl[:,ichan],threshold,minimal_event_interval,fs_das)
        
        if len(index_t_trig) == 0:
            empty_no = empty_no + 1
            # print('------- empty in channel dist =',sel_dist[ichan])
        elif len(index_t_trig) > 0:
            if len(index_arg_max) == 0: # means still no initialization
                index_arg_max = index_t_trig
                #add a score to each index
                score = np.ones(len(index_t_trig))
                csl_value = csl[index_t_trig,ichan]
            else:
                for index_t in index_t_trig:
                    if np.min(abs(index_t - np.array(index_arg_max))) > minimal_event_interval*fs_das:
                        # print('new event','index_t',index_t)
                        index_arg_max = np.append(index_arg_max,index_t) # add a new index
                        score = np.append(score,1)
                        csl_value = np.append(csl_value,csl[index_t,ichan])
                    elif np.min(abs(index_t - np.array(index_arg_max))) < minimal_event_interval*fs_das:
                        idx_overlap = np.argmin(abs(index_t - np.array(index_arg_max)))
                        # print('overlap event','index_t',index_t)
                        # index_arg_max[idx_overlap] = index_t
                        score[idx_overlap] += 1
                        csl_value[idx_overlap] = csl_value[idx_overlap] + csl[index_t,ichan]
            # print(ichan,'index_arg_max',index_arg_max)
    if len(csl_value) == 0:
        return np.array([]), np.array([]), np.array([]), empty_no
    
    return index_arg_max, score, csl_value/score, empty_no
    
