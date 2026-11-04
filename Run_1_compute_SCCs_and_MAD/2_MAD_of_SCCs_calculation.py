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
import os
import gc
from IPython.display import clear_output
import pickle

import h5py, json
import numpy as np


#!_________newly defined functions_________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl
from plot import Ticks_xaxis_day_or_hour
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

def mad(data):
    """
    only for 1D array
    """
    return np.median(np.abs(data - np.median(data)))
#!______________________________________
#!______________________________________
#!______________________________________
# prepare MAD_array for each template

# 1st loop: t_array loop (day by day)
# 2nd loop: Temp_name_list loop


# --1-- read CCs_s (stacked) in h5 format
# --2-- compute MAD_array according to t_array, 
# --3-- saved in MAD_array_dict in pkl format

# --4-- accompanied with each trigger file day by day, generate a plot: CCs, MAD


#!__________________________PARAS_____________________________
Temp_name_list = [
    'temp_0',   'temp_1',   'temp_2',  
    'temp_3',   'temp_4',   'temp_5',
    'temp_6',   'temp_7',   'temp_8',   
    'temp_9',   'temp_10',  'temp_11',
    'temp_12'
]
# nt delta in minutes, the time window for MAD calculation
for nt_delta in [240]:#[240,300,360,480,600,720]:#[5, 10, 20, 30, 40, 50, 60, 120, 180]:
    t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
    t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
    t_delta = timedelta(minutes=nt_delta)
    t_array = np.arange(t_all_start, t_all_end, t_delta)
    # t_all_end > t_array[-1] must be True
    t_array = np.append(t_array, t_all_end)

    print('t_array', t_array, '\n')
    print(f'********* calculate MAD array from {t_all_start} to {t_all_end} with time interval={t_delta}\n')

    IN_CCs_stack_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/'

    t_str = t_all_start.strftime('%Y%m%d_%H%M%S') + '_to_' + t_all_end.strftime('%Y%m%d_%H%M%S') + '/'
    OUTPUT_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/' + 'MAD_arrays/' + t_str
    OUTPUT_dir = OUTPUT_dir + t_delta.__str__() + '/'
    if not os.path.exists(OUTPUT_dir):
        os.makedirs(OUTPUT_dir)


#!________________________Heavy Reading Load here, the real computing part____________________________________
    # MAD_array_dict = {}
    # for Template in Temp_name_list:  
    #     print(f'\n=====> Template: {Template}')
    #     tmpMAD_array = np.zeros(len(t_array)-1)
    #     for ii in range(len(t_array)-1):
    #         t_start = t_array[ii]  
    #         t_end = t_array[ii+1]
    #         print(f'>> {Template}:  {ii + 1} / {len(t_array)} Calculate from {t_start} to {t_end}')
    #         CCs_s, RAWCCs_s = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end)
    #         tmpMAD_array[ii] = mad(CCs_s)
    #         MAD_array_dict[Template] = tmpMAD_array
            
    # #? saving the completed MAD_array_dict
    # MAD_array_dict['t_array'] = t_array
    # # save MAD_array_dict in the directory of 2, and the subdir only for MAD_arrays
    # filename = 'MAD_array_dict.pkl'
    # write_pkl(MAD_array_dict, OUTPUT_dir + filename)
    # print('Saved: ', OUTPUT_dir + filename)
    
    
    # #? saving the average MAD for each template, t_array take the first and last time
    # MAD_Average_dict = {}
    # MAD_Average_dict['t_array'] = np.array([t_array[0], t_array[-1]])
    # for Template in Temp_name_list:
    #     MAD_Average_dict[Template] = np.array([np.mean(MAD_array_dict[Template])])
    # filename = 'MAD_Average_dict.pkl'
    # write_pkl(MAD_Average_dict, OUTPUT_dir + filename)
    # print('Saved: ', OUTPUT_dir + filename)

# #!_________________________________________________________________________________

    # If just want to read the dict to plot
    #* read MAD_array_dict from pkl

    file_path = OUTPUT_dir + 'MAD_array_dict.pkl'
    MAD_array_dict = load_pkl(file_path)
    t_array = MAD_array_dict['t_array']
    print('Loaded: ', file_path)
    
    
    #? saving the average MAD for each template, t_array take the first and last time
    MAD_Average_dict = {}
    MAD_Average_dict['t_array'] = np.array([t_array[0], t_array[-1]])
    for Template in Temp_name_list:
        MAD_Average_dict[Template] = np.array([np.mean(MAD_array_dict[Template])])
    filename = 'MAD_Average_dict.pkl'
    write_pkl(MAD_Average_dict, OUTPUT_dir + filename)
    print('Saved: ', OUTPUT_dir + filename)


    
    #? plot MAD_array for each template in seperate figures
    no = len(Temp_name_list)
    fig, axes = plt.subplots(no, 1, figsize=(12, 20), sharex=True)

    for ii, Template in enumerate(Temp_name_list):
        ax = axes[ii]

        ax.scatter(t_array[:-1], MAD_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')

        avg_MAD = np.mean(MAD_array_dict[Template])
        ax.axhline(avg_MAD, color='r', linestyle='--', label=f'Avg MAD: {avg_MAD:.4f}')
        
        ax.set_title(f'SCCs MAD every {t_delta} for {Template}')
        ax.set_xlabel('Time')
        ax.set_ylabel('MAD of SCCs')    
        Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0)
        ax.legend()
    plt.show()
    fig.savefig(OUTPUT_dir + f'SCCs_MAD_array_AllTemp.png')
    print('Saved: ', OUTPUT_dir + f'SCCs_MAD_array_AllTemp.png')
    fig.clf()
    plt.close(fig)


    #? plot MAD_array of all templates in one figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for ii, Template in enumerate(Temp_name_list):
        ax.scatter(t_array[:-1], MAD_array_dict[Template], label=Template, s=10, alpha =0.5, color=f'C{ii}')
    ax.set_xlabel('Time')
    ax.set_ylabel('MAD')
    ax.set_title(f'MAD every {t_delta} for all templates')
    Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Templates')
    plt.show()
    fig.savefig(OUTPUT_dir + 'SCCs_MAD_array_AllTemp_inOne.png')
    print('Saved: ', OUTPUT_dir + 'SCCs_MAD_array_AllTemp_inOne.png')
    fig.clf()
    plt.close(fig)
    
    del MAD_array_dict 




# #!___________________________________ use existing data downwards____________________________________

# file_path = './Output_2_Trigger/MAD_arrays/'+ '0:20:00/' + 'MAD_array_dict.pkl'
# MAD_array_dict = load_pkl(file_path)
# t_array = MAD_array_dict['t_array']
# print('Loaded: ', file_path)


# #!___________FIG FOR SSA 1______________
# def _get_STALTA_catalog(t_start, t_end):
#     cata_path = f'{PROJECT_ROOT}/Ref_STALTA_NOA/' +  '20240424_101500_to_20240622_161500.csv'
#     DF_stalta = pd.read_csv(cata_path)
#     peak_time = DF_stalta['peak_time'].to_numpy()
#         #convert each element from str to datetime 
#     peak_time = np.array([datetime.strptime(peak_time[i], '%Y-%m-%d %H:%M:%S.%f') for i in range(len(peak_time))])
#     peak_amp = DF_stalta['peak_amp'].to_numpy()
#     peak_score = DF_stalta['peak_score'].to_numpy()
    
#     # cut the time window
#     idx = np.where((peak_time >= t_start) & (peak_time <= t_end))[0]
#     print(f'===>read STALTA: {len(idx)} events from {t_start} to {t_end}')
#     return peak_time[idx], peak_amp[idx], peak_score[idx]
    
# peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_all_start, t_all_end)
# dt = timedelta(hours=1, minutes=00, seconds=0)
# t_edges = np.arange(t_all_start, t_all_end+dt, dt)


# Template = 'template_0423_80Hz'
# plt.rcParams['font.size'] = 14
# fig, ax = plt.subplots(2, 1, figsize=(12, 6))

# ax[0].scatter(t_array[:-1], MAD_array_dict[Template], label='Stacked corr. coeff. MAD value', s=4, alpha =0.4, color='k')
# ax[0].set_title(f'Corr. Coeff. MAD every 20min of {Template}')
# ax[0].set_ylabel('Median Absolute Deviation')    
# ax[0].set_xlim([t_all_start, t_all_end])
# Ticks_xaxis_day_or_hour(ax[0], Day_tick=5, Hour_tick=0)
# ax[0].legend(loc='upper right')


# ax[1].hist(peak_time,bins=t_edges, color='k',label=f'STALTA events: #{len(peak_time)}',alpha=0.7)
# ax[1].set_xlabel('Time')
# ax[1].tick_params(axis='y', labelsize=14)
# ax[1].tick_params(axis='x', labelsize=14)
# ax[1].set_ylabel(f'Counts per hour', fontsize=14)
# ax[1].set_title(f'STALTA Earthquake Temporal Distribution', fontsize=14)
# ax[1].set_xlim([t_all_start, t_all_end])
# Ticks_xaxis_day_or_hour(ax[1], Day_tick=5, Hour_tick=0, font=14)
# # Ticks_xaxis_minute_or_second(ax[1], Minute_tick=00, Second_tick=10, rotation=45)
# ax[1].legend(loc='upper right')
# ax[1].grid('--')

# fig.tight_layout()
# plt.show()
# OUTPUT_dir = f'{PROJECT_ROOT}/Output_2_Trigger/MAD_arrays/0:20:00/'
# fig.savefig(OUTPUT_dir + f'MAD_array_{Template}_stalta.png')
# print('Saved: ', OUTPUT_dir + f'MAD_array_{Template}_stalta.png')
# fig.clf()
# plt.close(fig)


# #!___________FIG FOR SSA 2______________


# plt.rcParams['font.size'] = 14
# fig, ax = plt.subplots(1, 1, figsize=(10, 4))
# # name_list = [
# #     'template_10_80Hz',
# #     'template_0423_80Hz',
# #     'template_8_80Hz',
# #     'template_5_80Hz',
# #     'template_11_80Hz',
    
    
# #     'template_20_80Hz',
# #     'template_0425_80Hz',
# # ]
# name_list = [
#     # 'template_0423_80Hz',
#     # 'template_0425_80Hz',
#     'template_8_80Hz',
#     # 'template_10_80Hz',
#     'template_11_80Hz',
#     'template_18_80Hz',
# ]

# colors = ['r', 'b', 'g']
# for ii,Template in enumerate(name_list):
#     ax.scatter(t_array[:-1], MAD_array_dict[Template], label=f'Template {ii}', s=3, alpha =0.4, color=colors[ii])
# ax.set_title(f'Corr. Coeff. MAD every 20min of 3 templates')
# ax.set_ylabel('Median Absolute Deviation')    
# ax.set_xlim([t_all_start, t_all_end])
# Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0)
# ax.legend(loc='upper right')

# fig.tight_layout()
# plt.show()
# OUTPUT_dir = f'{PROJECT_ROOT}/Output_2_Trigger/MAD_arrays/0:20:00/'
# fig.savefig(OUTPUT_dir + f'MAD_array_allTemp.png')
# print('Saved: ', OUTPUT_dir + f'MAD_array_allTemp.png')
# fig.clf()
# plt.close(fig)