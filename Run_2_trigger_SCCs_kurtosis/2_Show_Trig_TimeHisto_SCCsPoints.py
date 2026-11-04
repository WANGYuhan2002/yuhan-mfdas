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
import matplotlib.patches as patches
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
import matplotlib.dates as mdates


# 1st loop: Temp_name_list loop
# 2nd loop: t_array loop (day by day)


# --1-- read CCs_s (stacked) in h5 format
# --2-- do dynamic triggering for certain time interval, using MAD_array_dict and threshold_level
# --3-- save the results in h5 format

# --4-- accompanied with each trigger file day by day, generate a plot: CCs, triggered point, MAD, threshold

#!__________________________functions_____________________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, generate_gs_grid
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024


def _get_STALTA_catalog(t_start, t_end):
    cata_path = f"{PROJECT_ROOT}/IN_STALTA_cata/sta_lta_catalog.csv"  # 本发布版自带的 STA/LTA 目录副本
    DF_stalta = pd.read_csv(cata_path)
    peak_time = DF_stalta['peak_time'].to_numpy()
        #convert each element from str to datetime 
    peak_time = np.array([datetime.strptime(peak_time[i], '%Y-%m-%d %H:%M:%S.%f') for i in range(len(peak_time))])
    peak_amp = DF_stalta['peak_amp'].to_numpy()
    peak_score = DF_stalta['peak_score'].to_numpy()
    
    # cut the time window
    idx = np.where((peak_time >= t_start) & (peak_time <= t_end))[0]
    print(f'===>read STALTA: {len(idx)} events from {t_start} to {t_end}')
    return peak_time[idx], peak_amp[idx], peak_score[idx]
    


#!__________________________PARAS_____________________________
fs = 80

Check_Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]

IN_Temp_lib_csv = f"{PROJECT_ROOT}/Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv,sep=',')

# trim to the templates we want to check
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Check_Temp_name_list)]
# rank in plot order
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)



IN_CCs_stack_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/'

MAD_Avg_path = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/'+'MAD_Average_dict.pkl'  # MAD_array_dict.pkl

MAD_array_dict = load_pkl(MAD_Avg_path)  # load the MAD_array_dict, keys are template names, values are MAD_array

#?++++++++++++++++++++++++++++
Threshold_level = 40
subdir = 'level_' + str(Threshold_level) + '/'  # subdir for triggered SCCs points

IN_CCs_trig_dir = f"{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/" + subdir  # Triggered SCCs points

OUTPUT_dir = f'{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/' + 'show_trigger/' + subdir  # Output directory for triggered SCCs points plots
if not os.path.exists(OUTPUT_dir):
    os.makedirs(OUTPUT_dir)
    




#!____________________________________________________________

#* plot for each template: 
#* 1) Continuous CCs_s with threshold cut
#* 2) Triggered CCs_s_trig histogram
# t_start = datetime.strptime('2024-04-25 15:10:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
# t_end = datetime.strptime('2024-04-25 15:20:00', '%Y-%m-%d %H:%M:%S')
t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')

t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')

OUTPUT_dir_byTime = OUTPUT_dir + t_str + '/'
if not os.path.exists(OUTPUT_dir_byTime):
    os.makedirs(OUTPUT_dir_byTime)
    
#? TWO CHECK
#? Ticks_xaxis_day_or_hour
#? dt = timedelta

# #!______________________1. each Temp. Triggered points______________________________________
# #ADD UP STALTA TRIGGERED EVENTS as reference
# peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)

# # read
# for Template in Temp_name_list:
#     print(f'\n===>Plotting {Template} from {t_start} to {t_end}')
    
#     CCs_s, RAWCCs_s = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end)
    
#     t_s = np.array([t_start + timedelta(seconds=i/fs) for i in range(len(CCs_s))])
#     print('t_s', t_s.shape)#, t_s[0], t_s[-2], t_s[-1])
#     print('CCs_s:', CCs_s.shape, type(CCs_s))
    
    
#     CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
    
#     Threshold = MAD_array_dict[Template][0] * Threshold_level
#     print('CCs_s_trig: ', CCs_s_trig.shape)# CCs_s_trig[0],CCs_s_trig[-1])
#     print('t_trig: ', t_trig.shape)# t_trig[0],t_trig[-1])
#     #
#     fig = plt.figure(figsize=(40, 30))
#     gs = plt.GridSpec(2,2,figure=fig,height_ratios=[3,1],width_ratios=[7.3,1])
#     axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:])]
#     plt.rcParams['font.size'] = 18
    
#     #* plot CCs_s w
#     axes[0].plot(t_s, abs(CCs_s), 'k', alpha=0.6, linewidth=0.4, label='CCs')
#     #* plot CCs_s_trig, triggered points
#     axes[0].scatter(t_trig, abs(CCs_s_trig), c='r', s=15, alpha=0.9, marker='*', label='Triggered events')

#     # 遍历每个矩形
#     for t1, t2, h in zip(t_left, t_right, abs(CCs_s_trig)):
#         rect = patches.Rectangle(
#             (t1, 0),         # 左下角 (t1, y=0)
#             t2 - t1,         # 宽度 (t2 - t1)
#             h,               # 高度 (h)
#             color='red',     # 颜色
#             alpha=0.3        # 透明度
#         )
#         axes[0].add_patch(rect)  # 添加矩形到图上
    
#     axes[0].axhline(y=Threshold, color='r', linestyle='--', label=f'Threshold={Threshold:.3e}', alpha=0.8)
#     # axes[0].axhline(y=-Threshold, color='r', linestyle='--', alpha=0.8)
#     axes[0].set_title(f'Stacked Corre. Coeff. of {Template} with threshold cut')
#     axes[0].set_ylabel('Stacked Corre. Coeff.')
#     axes[0].set_xlim(t_start, t_end)
#     axes[0].legend(loc='upper right')
#     axes[0].grid('--')
    
#     #* triggered events histogram
#     dt = timedelta(hours=1, minutes=10, seconds=0)
#     t_edges = np.arange(t_start, t_end+dt, dt)
#     axes[1].hist(t_trig,bins=t_edges, color='k')
#     axes[1].set_ylabel(f'Counts per {dt}')
#     axes[1].set_xlim([t_start, t_end])

#     for ax in axes:
#         Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0)
#         # Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=10, rotation=45)

    
#     #* reference STALTA triggered events histogram
#     axes[1].hist(peak_time,bins=t_edges, color='r',label='STALTA',alpha=0.3)
#     axes[1].legend(loc='upper right')

#     png_name = OUTPUT_dir_byTime+f"Trig_{Template}_{t_str}.png"
#     fig.savefig(png_name)
#     print(f'>>>>>>>>Saved in: {png_name}')
#     fig.clf()
#     plt.close(fig)

#     del CCs_s, RAWCCs_s, t_s, CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, fig, gs, axes


#!______________________2. All Temp Histo in one____________________________________

# #* plot the histogram of triggered events of all Templates together
# t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
# t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
# t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')

# adapt the axes according to the number of templates

# fig = plt.figure(figsize=(15, 20))
# gs = plt.GridSpec(6,2,figure=fig,height_ratios=[1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:])] 

# fig = plt.figure(figsize=(15, 25))
# gs = plt.GridSpec(9,2,figure=fig,height_ratios=[1,1,1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:]), fig.add_subplot(gs[7,:]), fig.add_subplot(gs[8,:])] 

fig = plt.figure(figsize=(20, 4*(len(check_lib)+1)))
axes = generate_gs_grid(fig,len(check_lib)+1)

plt.rcParams['font.size'] = 20

#ADD UP STALTA TRIGGERED EVENTS as reference
peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)

count = 0
for tt, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
    print(f'====> Read {Template} triggered events: #{len(t_trig)}, display name: {disp_Template}')
    count += len(t_trig)
    
    dt = timedelta(hours=1, minutes=00, seconds=0)
    t_edges = np.arange(t_start, t_end+dt, dt)
    axes[tt].hist(t_trig,bins=t_edges, color='k',label=f"{disp_Template}=#{len(t_trig)}")
    axes[tt].set_ylabel(f'Counts per {dt}')
    axes[tt].set_xlim([t_start, t_end])
    
    Ticks_xaxis_day_or_hour(axes[tt], Day_tick=5, Hour_tick=0)
    # Ticks_xaxis_minute_or_second(axes[tt], Minute_tick=0, Second_tick=10, rotation=45)
    
    axes[tt].legend(loc='upper right')
    axes[tt].grid('--')
    del CCs_s_trig, RAWCCs_s_trig, t_trig
    
axes[0].set_title('Temporal Distribution of Triggered Events = #'+str(count))
axes[-1].set_xlabel('Time')

axes[-1].hist(peak_time,bins=t_edges, color='r',label=f'STALTA=#{len(peak_time)}',alpha=0.3)
axes[-1].set_ylabel(f'Counts per {dt}')
axes[-1].set_xlim([t_start, t_end])
axes[-1].legend(loc='upper right')
axes[-1].grid('--')


# axes[-1].hist(peak_time,bins=t_edges, color='r',label='STALTA')
# axes[-1].legend(loc='upper right')
png_name = OUTPUT_dir_byTime+f"Trig_allTemp_HIST_{t_str}.png"
fig.savefig(png_name)
print(f'>>>>>>>>Saved in: {png_name}')
plt.close(fig)




# #!__________________3. All Temp triggered points in One__________________________________________

# #* together plot the CCs_s and triggered points of all Templates 
# t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
# t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
# t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')

# adapt the axes according to the number of templates
# fig = plt.figure(figsize=(30, 40))
# gs = plt.GridSpec(7,2,figure=fig,height_ratios=[1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:])] 

# fig = plt.figure(figsize=(30, 45))
# gs = plt.GridSpec(9,2,figure=fig,height_ratios=[1,1,1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:]), fig.add_subplot(gs[7,:]), fig.add_subplot(gs[8,:])] 

fig = plt.figure(figsize=(30, 5*(len(check_lib)+1)))
axes = generate_gs_grid(fig,len(check_lib)+1)

plt.rcParams['font.size'] = 20

for tt, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    
    #* read CCs_s_trig, triggered points
    CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
    print(f'====> Read {Template} triggered events: #{len(t_trig)}, display name: {disp_Template}')
    
    #* read CCs_s
    # CCs_s, RAWCCs_s = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end)
    # t_s = np.array([t_start + timedelta(seconds=i/fs) for i in range(len(CCs_s))])
    
    
    # axes[tt].plot(t_s, abs(CCs_s), 'k', alpha=0.6, linewidth=0.4, label='CCs')
    #* plot CCs_s_trig, triggered points
    axes[tt].scatter(t_trig, abs(CCs_s_trig), c='r', s=15, alpha=0.3, marker='*', label=f"{disp_Template}=#{len(t_trig)}")

    # 遍历每个矩形
    for t1, t2, h in zip(t_left, t_right, abs(CCs_s_trig)):
        rect = patches.Rectangle(
            (t1, 0),         # 左下角 (t1, y=0)
            t2 - t1,         # 宽度 (t2 - t1)
            h,               # 高度 (h)
            color='k',     # 颜色
            alpha=0.12        # 透明度
        )
        axes[tt].add_patch(rect)  # 添加矩形到图上
    
    #* limit the y axis scale is very important
    axes[tt].set_ylim([0, 0.5])
    
    Threshold = MAD_array_dict[Template][0] * Threshold_level
    axes[tt].axhline(y=Threshold, color='r', linestyle='--', label=f'Threshold={Threshold:.3e}', alpha=0.8)
    # axes[0].axhline(y=-Threshold, color='r', linestyle='--', alpha=0.8)
    axes[tt].set_ylabel('Stacked Corre. Coeff.')
    axes[tt].set_xlim(t_start, t_end)
    axes[tt].legend(loc='upper right')
    axes[tt].grid('--')
    
    Ticks_xaxis_day_or_hour(axes[tt], Day_tick=5, Hour_tick=0)
    # Ticks_xaxis_minute_or_second(axes[tt], Minute_tick=0, Second_tick=10, rotation=45)
    
    del CCs_s_trig, RAWCCs_s_trig, t_trig, t_left, t_right
    # del CCs_s, RAWCCs_s, t_s
    gc.collect()
    
axes[0].set_title('Temporal Distribution of Triggered Events =#' +str(count))


#ADD UP STALTA TRIGGERED EVENTS as reference
peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)
dt = timedelta(hours=1, minutes=0, seconds=0)
t_edges = np.arange(t_start, t_end+dt, dt)
axes[-1].hist(peak_time,bins=t_edges, color='r',label=f'STALTA=#{len(peak_time)}',alpha=0.3)
axes[-1].set_ylabel(f'Counts per {dt}')
axes[-1].set_xlim([t_start, t_end])
Ticks_xaxis_day_or_hour(axes[-1], Day_tick=5, Hour_tick=0)
# Ticks_xaxis_minute_or_second(axes[-1], Minute_tick=00, Second_tick=10, rotation=45)
axes[-1].legend(loc='upper right')
axes[-1].grid('--')
axes[-1].set_xlabel('Time')


# axes[-1].hist(peak_time,bins=t_edges, color='r',label='STALTA')
# axes[-1].legend(loc='upper right')
png_name = OUTPUT_dir_byTime+f"Trig_allTemp_TRIG_POINTS_{t_str}.png"
fig.savefig(png_name)
print(f'>>>>>>>>Saved in: {png_name}')
plt.close(fig)



