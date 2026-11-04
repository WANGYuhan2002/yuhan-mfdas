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

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from datetime import datetime, timedelta
import glob
import pandas as pd

import os
from re import split
#-- To import a function on a relative path:
import sys
sys.path.append("./")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning

import scipy.io as scio
import scipy.signal as signal

import load_das_h5_PE2024 as load_das_h5_PE2024
from matplotlib import ticker
import matplotlib.dates as mdates

#-- Define a plotting function

#-- (Easier to define such a function in-notebook rather than in a loaded function, 
#--    since updating labels or colorbars is easier)
plt.rcParams.update({'font.size': 14})
def waterfall_simple(data,ax,vlim,fs,dx,t_start,unit):
    #-- imshow data
    if(vlim==0):
        vlim = np.mean(np.abs(data))*3
    else:
        vlim = np.mean(np.abs(data))*vlim

    im = ax.imshow(data.T, aspect="auto", cmap='seismic', interpolation='antialiased', 
            extent=(0,data.shape[0]/fs, data.shape[1]*dx/1000,0),
            vmin=-vlim, vmax=vlim)
    
    #-- y-axis
    ax.set_ylabel("Distance [km]")

    #-- Define the x-tick marks we want, format the labels
    x_timestamp_interval = np.ceil(data.shape[0]/fs/6) # seconds
    x_positions = np.arange(0, data.shape[0]/fs, x_timestamp_interval)  # time-samples
    x_labels_times = [None]*len(x_positions)
    for i,t in enumerate(x_positions):
        x_labels_times[i] = (t_start + timedelta(seconds=t)).strftime('%H:%M:%S')
    ax.set_xticks(x_positions, x_labels_times)
    # ax.set_xlabel("Time (UTC)")

    plt.colorbar(im,label=unit,ax=ax)
 
    ax.set_title('Data on {0} GMT'.format(t_start.strftime('%Y.%m.%d')))
    ax.invert_yaxis()
    ax.grid(linestyle='--', linewidth=0.5)
    return ax

def plot_xdate(data,ax,fs,t_start,label,color='k'):
    # t_sec = np.arange(0,len(data)/fs, 1/fs)# seconds
    t_sec = np.linspace(0,len(data)/fs,len(data))# seconds
    # print(t_sec.shape)
    # arrange a t_sec that have the same length as data starting with t_start
    # print('t_sec',t_sec.shape)
    t_date = t_start + np.array([timedelta(seconds=t) for t in t_sec])
    ax.plot(t_date,data,label=label,color=color)
    # ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("")
    ax.legend()
    ax.set_xlim([t_date[0],t_date[-1]])
    ax.grid(linestyle='--', linewidth=0.5) 
    
    #add xticks ********************************************************
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=int(len(data)/fs/6)))
    ax.xaxis.set_minor_locator(mdates.SecondLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.grid(True)
        # set rotation
    ax.xaxis.set_tick_params(rotation=0)

def plot_together(das_v_sum_int,mffi_d_filtered,ax,fs_das,fs_mffi,t_start,color1='k',color2='r',y_labeling='Intensity',label1='DAS sum',label2='MFFI'):
    # t_sec_das = np.arange(0,len(das_v_sum_int)/fs_das, 1/fs_das)# seconds
    t_sec_das = np.linspace(0,len(das_v_sum_int)/fs_das,len(das_v_sum_int))# seconds
    t_date_das = t_start + np.array([timedelta(seconds=t) for t in t_sec_das])
    t_sec_mffi = np.arange(0,len(mffi_d_filtered)/fs_mffi, 1/fs_mffi)# seconds
    t_date_mffi = t_start + np.array([timedelta(seconds=t) for t in t_sec_mffi])
    
    # # unify the scales
    # mffi_d_filtered = mffi_d_filtered/np.max(np.abs(mffi_d_filtered))
    # das_v_sum_int = das_v_sum_int/np.max(np.abs(das_v_sum_int))
    
    ax.plot(t_date_mffi,mffi_d_filtered,label=label2,color=color2,linewidth=1)
    ax.plot(t_date_das,das_v_sum_int,label=label1,color=color1,linewidth=1,alpha=0.7)
   
    # ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(y_labeling)
    ax.legend()
    ax.set_xlim([t_date_das[0],t_date_das[-1]])
    ax.grid(linestyle='--', linewidth=0.5)
    
        #add xticks ********************************************************
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.SecondLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.grid(True)
        # set rotation
    ax.xaxis.set_tick_params(rotation=0)

def plot_multiple_band(ff_bands, mffi, fs_mffi, das_full,fs_das, fig, suptitle_name):
    gs = plt.GridSpec(5,2,figure=fig,height_ratios=[1,1,1,1,1],width_ratios=[7.3,1])
    fig.suptitle((suptitle_name), fontsize=32, position=(0.2,0.92))
    ax = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:]),fig.add_subplot(gs[2,:]),fig.add_subplot(gs[3,:]),fig.add_subplot(gs[4,:])]
    no_allchan = das_full.shape[1]
    
    for i in np.array([0,1,2,3,4]):
        
        if i == 0:
            # #--0-- plot the MFFI and DAS data together
            plot_xdate(mffi,ax[i],fs_mffi,t_start,'MFFI ',color='r')
            das_sum_tmp = np.sum(das_full, axis=1) / no_allchan # sum up all the channels
            plot_xdate(das_sum_tmp,ax[i],fs_das,t_start,'DAS sum ',color='k') 
            ax[i].set_title('RAW data')
        
        else:
            f1 = ff_bands[i-1,0]
            f2 = ff_bands[i-1,1]
            # control the str(1/f1) to be 2 digits
            if i==1 or i==2:
                filter_label = 'filtered ' + str(round(1/f2,2)) + ' - ' + str(round(1/f1,2)) + ' sec'
            if i==3 or i==4:
                filter_label = 'filtered ' + str(f1) + ' - ' + str(f2) + ' Hz'

            #--1-- MFFI
            mffi_detrend = block_cleaning.detrend(mffi, type='linear')
            mffi_filtered = block_filters.block_bandpass(mffi_detrend, f1, f2, fs_mffi, zerophase=False, taper=0.02)
            # mffi_filtered = mffi_filtered/np.max(np.abs(mffi_filtered))
            plot_xdate(mffi_filtered,ax[i],fs_mffi,t_start,'MFFI ',color='r')

            #--2-- DAS sum
            das_full_detrend = block_cleaning.detrend(das_full, type='linear')
            das_full_filtered_tmp = block_filters.block_bandpass(das_full_detrend, f1, f2, fs_das, zerophase=False, taper=0.02)
            das_sum_tmp = np.sum(das_full_filtered_tmp, axis=1) / no_allchan # sum up all the channels
            # das_sum_tmp = das_sum_tmp/np.max(np.abs(das_sum_tmp))
            plot_xdate(das_sum_tmp,ax[i],fs_das,t_start,'DAS sum ',color='k')
            ax[i].set_title(filter_label)
            
            # if i == 3:
            #     t_left = t_start + timedelta(seconds=2)
            #     t_right = t_start + timedelta(seconds=5)
            #     ax[i].set_xlim([t_left,t_right])
            # if i == 4:
            #     t_left = t_start + timedelta(seconds=3)
            #     t_right = t_start + timedelta(seconds=4)
            #     ax[i].set_xlim([t_left,t_right])
            
    return ax

def plot_waterfall_dassum_mffi(das_full,das_sum,mffi,fs_das,fs_mffi,t_start,dx,fig,unit_name,title1,title2,title3):
    
    #--5.1-- plot strain [waterfall + DAS sum + mffi data] 
    gs = plt.GridSpec(4,2,figure=fig,height_ratios=[2,1,1,1],width_ratios=[7.3,1])
    fig.suptitle('<'+unit_name+'> data on {0} GMT'.format(t_start.strftime('%Y.%m.%d')), fontsize=20, position=(0.2,0.92))
    ax = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[3,0])]

    #--1-- das full
    waterfall_simple(das_full,ax[0],vlim=0 ,fs=fs_das,dx=dx,t_start=t_start,unit=unit_name)
    ax[0].set_title(title1)

    #--2-- das sum
    plot_xdate(das_sum,ax[1],fs_das,t_start,title2,color='k')
    ax[1].set_title(title2)
    ax[1].set_ylabel(unit_name)

    #--3-- mffi data
    plot_xdate(mffi,ax[2],fs_mffi,t_start,title3,color='r')
    ax[2].set_title(title3)
    ax[2].set_ylabel(unit_name)

    #--4-- together
    plot_together(das_sum,mffi,ax[3],fs_das,fs_mffi,t_start,color1='k',color2='r',y_labeling='strain')
    return ax

def read_das_full_sum(t_start,t_end,dist_start,dist_end,input_dir,plot=False):
    #--1a-- read in DAS data with load_das_custom function with Pascal's header
    if input_dir == f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/' or input_dir == f'{PROJECT_ROOT}/IN_DAS_data/DATA_DEC_8Hz_16m_SORTED/':
        das_v_full, headers, axis = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=input_dir)
        dx = 16*1.02095198631286
    #--1b-- read in DAS data with load_das_custom function with Daniel's header
    if input_dir == f'{PROJECT_ROOT}/IN_DAS_data/DATA_RAW_400Hz_8m/':
        das_v_full, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=input_dir,verbose=False,convert=False)
        dx = 8*1.02095198631286
    fs_das = headers['fs']
    # print('das data shape',das_v_full.shape)
    # print('das data fs',fs_das,'\n')
    
    # trim in space ****************************************
    chan_start = int(dist_start/dx)
    chan_end = int(dist_end/dx)
    sel_chan_tosum = np.arange(chan_start, chan_end,1)
    no_allchan = das_v_full.shape[1]
    # print('===========Trim==========')
    # print('---- number of channels',no_allchan)
    # print('---- picked', len(sel_chan_tosum),'channels from',sel_chan_tosum[0], 'to',sel_chan_tosum[-1])

    das_v_full = das_v_full[:,sel_chan_tosum]
    no_allchan = len(sel_chan_tosum)
    # *******************************************************

    # convert to strain rate in unit of 1/s
    # print('\n =========Convert intensity to strain rate=========')
    if input_dir == f'{PROJECT_ROOT}/IN_DAS_data/DATA_RAW_400Hz_8m/':
        fs_orig = 400
        convert_intensity_to_strain_rate = 116 / 8192 * fs_orig/10 * 1e-9  
        # print('---- convert_intensity_to_strain_rate in m/m/s for Daniel\'s header =',convert_intensity_to_strain_rate)
    if input_dir == f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/' or input_dir == f'{PROJECT_ROOT}/IN_DAS_data/DATA_DEC_8Hz_16m_SORTED/':
        convert_intensity_to_strain_rate = 1
        # print('---- das data convert_to_strain_rate in m/m/s for Pascal\'s header =',1e-7)
    
    das_v_full = das_v_full * convert_intensity_to_strain_rate # convert intensity to strain rate in unit of 1/s

    # remove spatial median
    das_v_full = das_v_full - np.median(das_v_full,axis=1)[:,np.newaxis]
    # print('das_v_full',das_v_full.shape)

    # convert to strain acceleration in unit of 1/s^2 and strain in unit of 1
    das_a_full = np.diff(das_v_full,axis=0) * fs_das
    # print('das_a_full = das_v_full \'s derivative',das_a_full.shape)
    das_d_full = np.cumsum(das_v_full,axis=0) / fs_das
    # print('das_d_full = das_v_full \'s integral',das_d_full.shape)

    #--- sum up all the channels and devided by the number of all channels -----------------
    das_d_sum = np.sum(das_d_full, axis=1) / no_allchan 
    das_v_sum = np.sum(das_v_full, axis=1) / no_allchan
    das_a_sum = np.sum(das_a_full, axis=1) / no_allchan

    #-- imshow data
    if plot:
        #--5.1-- plot strain rate [waterfall + DAS sum + mffi data] 
        fig = plt.figure(figsize=(25,25))
        plt.rcParams.update({'font.size': 22})
        gs = plt.GridSpec(2,2,figure=fig,height_ratios=[2,1],width_ratios=[7.3,1])
        fig.suptitle('<Strain> data on {0} GMT'.format(t_start.strftime('%Y.%m.%d')), fontsize=20, position=(0.2,0.92))
        ax = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0])]

        #--1-- das full
        waterfall_simple(das_v_full,ax[0],vlim=0 ,fs=fs_das,dx=dx,t_start=t_start,unit='strain rate [1/s]')
        ax[0].set_title('DAS full data raw')

        #--2-- das sum
        plot_xdate(das_v_sum,ax[1],fs_das,t_start,'DAS sum',color='k')
        ax[1].set_title('DAS sum raw')
        ax[1].set_ylabel("strain rate ")
        
    
    # construct the dictionary of das_full and das_sum
    das_full = {'d':das_d_full,'v':das_v_full,'a':das_a_full}
    das_sum = {'d':das_d_sum,'v':das_v_sum,'a':das_a_sum}
    return fs_das, dx, das_full, no_allchan, das_sum

def das_full_sum_filter_all_in_one(f1,f2,fs_das,das_full,no_allchan):    
    das_d_full = das_full['d']
    das_v_full = das_full['v']
    das_a_full = das_full['a']
    
    das_d_full_detrend = block_cleaning.detrend(das_d_full, type='linear')
    das_d_full_filtered = block_filters.block_bandpass(das_d_full_detrend, f1, f2, fs_das, zerophase=False, taper=0.02)
    das_d_sum_filtered = np.sum(das_d_full_filtered, axis=1) / no_allchan # sum up all the channels
    
    das_v_full_detrend = block_cleaning.detrend(das_v_full, type='linear')
    das_v_full_filtered = block_filters.block_bandpass(das_v_full_detrend, f1, f2, fs_das, zerophase=False, taper=0.02)
    das_v_sum_filtered = np.sum(das_v_full_filtered, axis=1) / no_allchan # sum up all the channels

    das_a_full_detrend = block_cleaning.detrend(das_a_full, type='linear')
    das_a_full_filtered = block_filters.block_bandpass(das_a_full_detrend, f1, f2, fs_das, zerophase=False, taper=0.02)
    das_a_sum_filtered = np.sum(das_a_full_filtered, axis=1) / no_allchan # sum up all the channels

    # construct the dictionary of das_full_filtered and das_sum_filtered
    das_full_filtered = {'d':das_d_full_filtered,'v':das_v_full_filtered,'a':das_a_full_filtered}
    das_sum_filtered = {'d':das_d_sum_filtered,'v':das_v_sum_filtered,'a':das_a_sum_filtered}
    return das_full_filtered, das_sum_filtered

def mffi_filter_all_in_one(f1,f2,fs_mffi,mffi):
    mffi_d,mffi_v,mffi_a = mffi['d'],mffi['v'],mffi['a']
    mffi_d_detrend = block_cleaning.detrend(mffi_d, type='linear')
    mffi_d_filtered = block_filters.block_bandpass(mffi_d_detrend, f1, f2, fs_mffi, zerophase=False, taper=0.02)      
    mffi_v_detrend = block_cleaning.detrend(mffi_v, type='linear')
    mffi_v_filtered = block_filters.block_bandpass(mffi_v_detrend, f1, f2, fs_mffi, zerophase=False, taper=0.02)
    mffi_a_detrend = block_cleaning.detrend(mffi_a, type='linear')
    mffi_a_filtered = block_filters.block_bandpass(mffi_a_detrend, f1, f2, fs_mffi, zerophase=False, taper=0.02)
    # construct the dictionary of mffi_filtered
    mffi_filtered = {'d':mffi_d_filtered,'v':mffi_v_filtered,'a':mffi_a_filtered}
    return mffi_filtered

def get_noise_dassum(noise_name_list,noise_dist_list,t_start,t_end,input_dir):
    # store the noise das_v_sum, das_v_sum_filtered and d, a in a noise signal list
    noise_das_d_sum = []
    noise_das_d_sum_filtered = []
    noise_das_v_sum = []
    noise_das_v_sum_filtered = []
    noise_das_a_sum = []
    noise_das_a_sum_filtered = []
    
    noise_das_sum = dict()
    noise_das_sum_filtered = dict()
    
    for i in np.arange(0,noise_dist_list.shape[0],1):
        ds_tmp = noise_dist_list[i,0]
        de_tmp = noise_dist_list[i,1]
        # read the 'noise' das data from selected distance
        fs_das, dx, das_full_tmp, no_allchan, das_sum_tmp \
            = read_das_full_sum(t_start,t_end,ds_tmp,de_tmp,input_dir,plot=False)
        # filter the 'noise' das data in [f1,f2]
        das_full_filtered_tmp,das_sum_filter_tmp \
            = das_full_sum_filter_all_in_one(f1,f2,fs_das,das_full_tmp,no_allchan)
        
        # add the das_sum_tmp and das_sum_filter_tmp to the dictionary of 
        #       noise_das_sum and noise_das_sum_filtered
        # with the key of noise_name_list[i]
        noise_das_sum[noise_name_list[i]] = das_sum_tmp
        noise_das_sum_filtered[noise_name_list[i]] = das_sum_filter_tmp
        

    return noise_das_sum, noise_das_sum_filtered

def read_mffi_any(t_start,t_end,reverse=False):
    if t_start < datetime(2024,4,24,16,0,0) or t_end > datetime(2024,6,23,10,00,00):
        print('MFFI data is not available in the date specified')
        return
    if t_start.date() != t_end.date():
        print('the function is now only handling the data read in the same day')
        return
    mffi_80Hz_dir = f'{PROJECT_ROOT}/IN_MFFI_data/kefalonia_ts_h5_80Hz/'
    ####choose the .h5 file with the date string
    file_h5_name = sorted(glob.glob(mffi_80Hz_dir+'kefalonia_ts_UTC_'+ t_start.strftime('%Y%m%d') + '*.h5'))
    file_h5_name = file_h5_name[0]
    
    # if os.path.exists(file_h5_name):
    #     print('read file: ',file_h5_name)
    # else:
    #     print('no file found')
        
    ####read the h5 file
    with h5py.File(file_h5_name, 'r') as f:
        # print(f.keys())
        # print(f['mffi_dsp'].keys())
        # print(f['mffi_dsp/strain'])
        strain_intensity = f['mffi_dsp/strain'][()]
        # print('strain intensity',type(strain_intensity),strain_intensity.shape)
        datetime_0 = f['mffi_dsp/strain'].attrs['datetime_0']
        datetime_0 = datetime.strptime(datetime_0, '%Y-%m-%d %H:%M:%S.%f') 
        datetime_1 = f['mffi_dsp/strain'].attrs['datetime_1']
        datetime_1 = datetime.strptime(datetime_1, '%Y-%m-%d %H:%M:%S.%f')
        npts = f['mffi_dsp/strain'].attrs['npts']
        npts = int(npts)
        fs_mffi = f['mffi_dsp/strain'].attrs['fs']
        fs_mffi = float(fs_mffi)
        
        #because the datetime_1 is 59:59, not very precise!!!
        # datetime_1 = datetime_0 + timedelta(seconds=npts/fs_mffi)
    # check if datetime_0 < t_start < t_end < datetime_1
    if t_start < datetime_0 or t_end > datetime_1+timedelta(seconds=1):
        print('the time range is out of the range of the MFFI data')
        return
    # ---------------- convert the intensity to strain --------------------
    votage_to_phase = np.pi/180*5.625/2500
    # print('votage_to_phase',votage_to_phase)
    L = 2*14976  # 2*length of the fiber in meters
    phase_to_strain = 2e8/(2*np.pi*0.78*10e9*L) # 2e8 is light speed, 10e9 means 10GHz
    mffi_convert = votage_to_phase * phase_to_strain 
    # print('mffi_convert',mffi_convert)
    if reverse:
        strain = strain_intensity * mffi_convert * -1
        print('reverse the mffi data')
    else:
        strain = strain_intensity * mffi_convert * 1
    # ----------------------------------------------------------------------
    # print('====reading MFFI data from',t_start,'to',t_end)
    i_ts = int((t_start-datetime_0).total_seconds()*fs_mffi)
    i_te = int((t_end-datetime_0).total_seconds()*fs_mffi)
    mffi_d = strain[i_ts:i_te+1]
    # print('mffi strain',mffi_d.shape)

    # mffi_v the strain-rate and mffi_a the strain-acceleration
    mffi_v = np.diff(mffi_d,axis=0) * fs_mffi
    mffi_a = np.diff(mffi_v,axis=0) * fs_mffi
    # print('mffi strain_rate',mffi_v.shape)
    # print('mffi strain_acceleration',mffi_a.shape)
     
    # construct the dictionary of mffi data
    mffi = {'d':mffi_d,'v':mffi_v,'a':mffi_a}
    return fs_mffi, mffi


def read_mffi_any_v2(t_start,t_end,reverse=False):
    if t_start < datetime(2024,4,24,16,0,0) or t_end > datetime(2024,6,23,10,00,00):
        print('指定日期内MFFI数据不可用')
        return None, None
    
    mffi_80Hz_dir = f'{PROJECT_ROOT}/IN_MFFI_data/kefalonia_ts_h5_80Hz/'
    
    votage_to_phase = np.pi/180*5.625/2500
    L = 2*14976
    phase_to_strain = 2e8/(2*np.pi*0.78*10e9*L)
    mffi_convert = votage_to_phase * phase_to_strain 
    
    if reverse:
        mffi_convert *= -1
        print('反转mffi数据')

    # --- Stage 1: Determine fs_mffi_overall and collect available file metadata ---
    fs_mffi_overall = None
    file_metadata_list = [] 

    dates_to_process = []
    current_date_iter = t_start.date()
    while current_date_iter <= t_end.date():
        dates_to_process.append(current_date_iter)
        current_date_iter += timedelta(days=1)

    for date_obj in dates_to_process:
        file_h5_search_pattern = mffi_80Hz_dir + 'kefalonia_ts_UTC_' + date_obj.strftime('%Y%m%d') + '*.h5'
        file_h5_list = sorted(glob.glob(file_h5_search_pattern))
        
        if file_h5_list:
            file_h5_name = file_h5_list[0]
            try:
                with h5py.File(file_h5_name, 'r') as f:
                    dt0_str = f['mffi_dsp/strain'].attrs['datetime_0']
                    fs_file = float(f['mffi_dsp/strain'].attrs['fs'])
                    npts_file = int(f['mffi_dsp/strain'].attrs['npts'])
                    
                    dt0 = datetime.strptime(dt0_str, '%Y-%m-%d %H:%M:%S.%f')
                    
                    file_metadata_list.append({
                        "path": file_h5_name, 
                        "dt0": dt0, 
                        "fs": fs_file,
                        "npts": npts_file
                    })

                    if fs_mffi_overall is None:
                        fs_mffi_overall = fs_file
                    elif fs_mffi_overall != fs_file:
                        print(f"错误：文件 {file_h5_name} 的采样频率 {fs_file}Hz 与已确定的全局采样频率 {fs_mffi_overall}Hz 不匹配。")
                        return None, None 
            except Exception as e:
                print(f"警告：预处理文件 {file_h5_name} 时出错: {e}。该文件将被跳过。")

    if fs_mffi_overall is None:
        # 如果查询时长为零或负，依然可以返回空数据和None的fs
        if t_start >= t_end:
            mffi_empty = {'d': np.array([]), 'v': np.array([]), 'a': np.array([])}
            return None, mffi_empty # fs 未知，数据为空
        print('在指定范围内未找到任何可读的MFFI数据文件，或无法确定一致的采样频率。无法填充零值。')
        return None, None

    file_metadata_list.sort(key=lambda x: x["dt0"])

    # --- Stage 2: Construct the data array, filling gaps with zeros ---
    all_segments = []
    current_time = t_start
    file_meta_idx = 0 

    while current_time < t_end:
        next_event_time_for_gap_calc = t_end # 默认情况下，空隙一直填充到查询结束时间 t_end
        
        data_read_successfully_in_iteration = False

        if file_meta_idx < len(file_metadata_list):
            meta = file_metadata_list[file_meta_idx]
            file_dt0 = meta["dt0"]
            # 文件中数据的实际结束时间 = 文件开始时间 + 点数 / 采样率
            file_effective_end_dt = meta["dt0"] + timedelta(seconds=meta["npts"] / meta["fs"])

            if file_dt0 >= t_end: 
                # 此文件及后续文件完全超出了查询范围的 t_end
                # next_event_time_for_gap_calc 保持为 t_end
                pass
            elif current_time >= file_effective_end_dt: 
                # 此文件完全在 current_time 之前，已过时
                file_meta_idx += 1 
                continue # 重新进入while循环，用下一个 file_meta_idx 或填充空隙
            elif current_time >= file_dt0: 
                # current_time 落在此文件的数据段内 [file_dt0, file_effective_end_dt)
                read_start = current_time
                read_end = min(t_end, file_effective_end_dt)
                
                try:
                    with h5py.File(meta["path"], 'r') as f_data:
                        # 确保读取整个文件的数据以进行切片，或者只读取所需部分（如果h5py支持高效部分读取）
                        # 为简单起见，先读取全部，然后切片
                        strain_intensity_full = f_data['mffi_dsp/strain'][()]
                    
                    idx_s = round((read_start - file_dt0).total_seconds() * meta["fs"])
                    idx_e = round((read_end - file_dt0).total_seconds() * meta["fs"]) # 索引应为开区间上限
                    
                    idx_s = max(0, idx_s)
                    # 确保 idx_e 不超过实际数据点数
                    idx_e = min(meta["npts"], idx_e)


                    if idx_s < idx_e:
                        segment_intensity = strain_intensity_full[idx_s:idx_e]
                        data_segment = segment_intensity * mffi_convert
                        all_segments.append(data_segment)
                    
                    current_time = read_end 
                    data_read_successfully_in_iteration = True
                    
                    if current_time >= file_effective_end_dt: 
                        file_meta_idx += 1
                    # continue # 返回循环顶部处理下一个时间点

                except Exception as e_read:
                    print(f"警告: 读取文件 {meta['path']} (尝试时段 {read_start} 到 {read_end}) 时出错: {e_read}。此段将补零。")
                    # 读取失败，将这段视为一个空隙，空隙的结束点是 read_end
                    next_event_time_for_gap_calc = read_end
                    file_meta_idx += 1 # 重要：跳过这个损坏的文件，避免无限循环
                    # 让下面的空隙填充逻辑处理从 current_time 到 next_event_time_for_gap_calc
                
            else: # current_time < file_dt0: 在此文件开始前存在空隙
                next_event_time_for_gap_calc = file_dt0 
                # 让下面的空隙填充逻辑处理
        
        # 如果成功读取了数据，则跳过空隙填充，直接进入下一次while循环
        if data_read_successfully_in_iteration:
            continue

        # --- 填充空隙 ---
        # 空隙从 current_time 填充到 min(t_end, next_event_time_for_gap_calc)
        fill_until = min(t_end, next_event_time_for_gap_calc)
        if current_time < fill_until:
            gap_s = (fill_until - current_time).total_seconds()
            # 仅当空隙持续时间大于一个非常小的值时才填充 (避免浮点精度问题导致的小空隙)
            if gap_s > 1e-9: # 1纳秒
                num_z = round(gap_s * fs_mffi_overall)
                if num_z > 0:
                    all_segments.append(np.zeros(num_z))
            current_time = fill_until
        elif file_meta_idx >= len(file_metadata_list) and current_time < t_end : # 所有文件处理完，但未到t_end
             # 确保current_time最终能达到t_end以结束循环
            if current_time < t_end: # Final gap if any
                gap_s = (t_end - current_time).total_seconds()
                if gap_s > 1e-9:
                    num_z = round(gap_s * fs_mffi_overall)
                    if num_z > 0:
                        all_segments.append(np.zeros(num_z))
                current_time = t_end


    # --- Stage 3: Finalize data ---
    if not all_segments:
        mffi_d = np.array([])
    else:
        try:
            mffi_d = np.concatenate(all_segments)
        except ValueError as e:
            print(f"错误：拼接数据段时发生错误（可能由于段为空或形状不兼容）：{e}")
            mffi_d = np.array([]) # 返回空数组

    # (可选) 长度校验
    expected_total_seconds = (t_end - t_start).total_seconds()
    if expected_total_seconds < 0: expected_total_seconds = 0 # 处理 t_start > t_end 的情况
    
    # 仅当 fs_mffi_overall 有效时才计算期望样本数
    expected_total_samples = 0
    if fs_mffi_overall and fs_mffi_overall > 0:
        expected_total_samples = round(expected_total_seconds * fs_mffi_overall)

    if len(mffi_d) != expected_total_samples and expected_total_samples > 0 :
        print(f"警告：最终数据长度 {len(mffi_d)} 与期望长度 {expected_total_samples} ({expected_total_seconds:.3f}s @ {fs_mffi_overall}Hz) 不完全匹配。"
              f"可能是由于时间戳和采样率的舍入误差。")

    # 计算应变率和应变加速度
    if len(mffi_d) > 1 and fs_mffi_overall :
        mffi_v = np.diff(mffi_d) * fs_mffi_overall
        if len(mffi_v) > 1:
            mffi_a = np.diff(mffi_v) * fs_mffi_overall
        else:
            mffi_a = np.array([])
    else:
        mffi_v = np.array([])
        mffi_a = np.array([])
    
    mffi = {'d':mffi_d,'v':mffi_v,'a':mffi_a}
    return fs_mffi_overall, mffi
