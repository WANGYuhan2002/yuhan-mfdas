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
## ---
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

HOME_dir = f'{PROJECT_ROOT}/'
#!__________________________functions_____________________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, generate_gs_grid
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
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
    
    
#!_______________________DATA IN________________________________
fs = 80

# Check_Temp_name_list = [
#             'temp_0',   'temp_1',   
#             'temp_2',   'temp_3',   
#             'temp_4',   'temp_5',
#             'temp_6',   'temp_7',   
#             'temp_8',   'temp_9',   
#             'temp_10',  'temp_11',
#             'temp_12'
#         ]

# Check_Temp_name_list = [
#             'temp_0',   'temp_1',   
#             'temp_2',   #'temp_3',   
#             'temp_4',   #'temp_5',
#             # 'temp_6',   
#             'temp_7',   
#             'temp_8',   'temp_9',   
#             'temp_10',  'temp_11',
#             'temp_12'
#         ]

Check_Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]

IN_Temp_lib_csv = f"{HOME_dir}Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv,sep=',')
# trim to the templates we want to check
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Check_Temp_name_list)]
# rank in plot order
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)



subdir = 'level_20/'
IN_CCs_Classification_dir = f'{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/' + subdir

#!_______________________DATA OUT________________________________
OUTPUT_dir =  f'{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/' + 'Show_Classification_byCCs/' + subdir


#!__________________________PARAS__________________________________
t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')

OUTPUT_dir_byTime = OUTPUT_dir + t_str + '/'
if not os.path.exists(OUTPUT_dir_byTime):
    os.makedirs(OUTPUT_dir_byTime)
    
#!__________________________PLOTs_______________________________________
Events_classified = pd.read_csv(IN_CCs_Classification_dir + 'Events_classified_AllTemp.csv', sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])

Events_classified = Events_classified[ (Events_classified['Rt_trig'] >= t_start) & (Events_classified['Rt_trig'] <= t_end) ]
print(f'====> Events_classified cut from {t_start} to {t_end}:\n {Events_classified}')


#*_______1st Plot_______ Histogram of Events_classified, in Each template, all plot in one gs
#---> background: unclassified case, in blue
#---> reference: STALTA histogram

# adapt the axes according to the number of templates
# fig = plt.figure(figsize=(30, 40))
# gs = plt.GridSpec(7,2,figure=fig,height_ratios=[1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:])] 

# fig = plt.figure(figsize=(30, 45))
# gs = plt.GridSpec(9,2,figure=fig,height_ratios=[1,1,1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:]), fig.add_subplot(gs[7,:]), fig.add_subplot(gs[8,:])] 

fig = plt.figure(figsize=(40, 5*(len(check_lib)+1)))
axes = generate_gs_grid(fig,len(check_lib)+1)

plt.rcParams['font.size'] = 32

dt = timedelta(hours=1, minutes=00, seconds=0)
t_edges = np.arange(t_start, t_end+dt, dt)
EQ_sum = 0
for tt, row in check_lib.iterrows():
    Template = row['infile_name'] # name in file 
    disp_Template = row['plot_name'] # name we want to display in the figure
    
    # #* plot histogram of unclassified events
    # Rt_trig_unclassified = Events_classified[Events_classified['Template'] == Template] ['Rt_trig']
    # axes[tt].hist(Rt_trig_unclassified,bins=t_edges, color='b', alpha=0.2, label='Unclassified')
    
    #* plot histogram of classified events
    Rt_trig_classified = Events_classified[(Events_classified['Template'] == Template) & (Events_classified['Belong_or_not'] == 1)]['Rt_trig']    
    axes[tt].hist(Rt_trig_classified,bins=t_edges, color='k', alpha=0.7, label=f'{disp_Template}: #{len(Rt_trig_classified)}')
    # axes[tt].hist(Rt_trig_classified,bins=t_edges, color='k', alpha=0.7, label=f'{Template}: #{len(Rt_trig_classified)}')

    
    print(f"====> Read {Template}: #{len(Rt_trig_classified)} events, display as {disp_Template}")
    EQ_sum += len(Rt_trig_classified)
    
axes[0].set_title(f'Temporal Distribution of Classified Events.')# Total = {EQ_sum} events')
print('====> Total number of classified events:', EQ_sum)

#ADD UP STALTA TRIGGERED EVENTS as reference
peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)

#*----- STA/LTA 
axes[-1].hist(peak_time,bins=t_edges, color='r',label=f'STA/LTA: #{len(peak_time)}',alpha=0.5)
axes[-1].set_xlabel('Real Time')
axes[-1].set_ylim([0,100])

#*---—— ADD UP matched filter sum-up results to the last plot of STALTA
All_Rt_trig = Events_classified[(Events_classified['Belong_or_not'] == 1)]['Rt_trig']  
axes[-1].hist(All_Rt_trig,bins=t_edges, color='k',label=f'All template sum-up:\n #{EQ_sum}',alpha=0.4)

for ax in axes:
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    # ax.set_ylabel(f'Counts per {dt}', fontsize=22)
    ax.set_xlim([t_start, t_end])
    Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=30)
    # Ticks_xaxis_minute_or_second(ax, Minute_tick=00, Second_tick=10, rotation=45)
    ax.legend(loc='upper right',fontsize=36)
    ax.grid('--')
    ax.tick_params(axis='y', labelsize=36)
    
axes[0].set_ylabel(f'Counts per hour', fontsize=36)
    

png_name = OUTPUT_dir_byTime+f"Classified_allTemp_HIST_{t_str}.png"
fig.savefig(png_name)
print(f'>>>>>>>>Saved in: {png_name}')
plt.close(fig)


#*_______2nd Plot_______ Mark triggered points and height as CCs_s value,  in Each template, all plot in one gs
#---> background: unclassified case, in red

# adapt the axes according to the number of templates
# fig = plt.figure(figsize=(30, 40))
# gs = plt.GridSpec(7,2,figure=fig,height_ratios=[1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
# axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:])] 

fig = plt.figure(figsize=(30, 5*(len(check_lib)+1)))
axes = generate_gs_grid(fig,len(check_lib)+1)


plt.rcParams['font.size'] = 27


for tt, row in check_lib.iterrows():
    Template = row['infile_name'] # name in file 
    disp_Template = row['plot_name'] # name we want to display in the figure
    
    def mini_plot_abs(pd_events, ax, color, alpha, label):
        Rt_trig = pd_events['Rt_trig']
        Rt_left = pd_events['Rt_left']
        Rt_right = pd_events['Rt_right']
        CCs_s_trig = pd_events['CCs_s_trig']
        #* plot CCs_s_trig, triggered points
        ax.scatter(Rt_trig, abs(CCs_s_trig), c=color, s=15, alpha=alpha, marker='*', label=label)
        # 遍历每个矩形
        for t1, t2, h in zip(Rt_left, Rt_right, abs(CCs_s_trig)):
            rect = patches.Rectangle(
                (t1, 0),         # 左下角 (t1, y=0)
                t2 - t1,         # 宽度 (t2 - t1)
                h,               # 高度 (h)
                color=color,     # 颜色
                alpha=0.07        # 透明度
            )
            ax.add_patch(rect)
    
    # mini_plot(Events_classified[ (Events_classified['Template'] == Template) ], 
    #           axes[tt], 'b', 0.3, 'Unclassified')
    mini_plot_abs(Events_classified[ (Events_classified['Template'] == Template) & (Events_classified['Belong_or_not'] == 1) ], 
              axes[tt], 'k', 0.2, label=f'{disp_Template}: #{len(Rt_trig_classified)}')

    #* limit the y axis scale is very important
    axes[tt].set_ylim([0, 0.5])
    axes[tt].set_ylabel('Stacked Corre. Coeff.', fontsize=22)
    
axes[0].set_title(f'Trigger and Classified Points of Classified Events: Total = {EQ_sum} events')


#ADD UP STALTA TRIGGERED EVENTS as reference
peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)
dt = timedelta(hours=1, minutes=0, seconds=0)
t_edges = np.arange(t_start, t_end+dt, dt)
axes[-1].hist(peak_time,bins=t_edges, color='r',label=f'STALTA: #{len(peak_time)}',alpha=0.3)
axes[-1].set_ylabel(f'Counts per {dt}', fontsize=22)
axes[-1].set_xlabel('Time')

for ax in axes:
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.set_xlim([t_start, t_end])
    Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=22)
    # Ticks_xaxis_minute_or_second(ax, Minute_tick=00, Second_tick=10, rotation=45)
    ax.legend(loc='upper right')
    ax.grid('--')

    
# axes[-1].hist(peak_time,bins=t_edges, color='r',label='STALTA')
# axes[-1].legend(loc='upper right')
png_name = OUTPUT_dir_byTime+f"Classified_allTemp_TRIG_POINTS_abs_{t_str}.png"
fig.savefig(png_name)
print(f'>>>>>>>>Saved in: {png_name}')
plt.close(fig)


#*_______3rd Plot_______ 



fig = plt.figure(figsize=(30, 5*(len(check_lib)+1)))
axes = generate_gs_grid(fig,len(check_lib)+1)


plt.rcParams['font.size'] = 27


for tt, row in check_lib.iterrows():
    Template = row['infile_name'] # name in file 
    disp_Template = row['plot_name'] # name we want to display in the figure
    
    def mini_plot(pd_events, ax, color, alpha, label):
        Rt_trig = pd_events['Rt_trig']
        Rt_left = pd_events['Rt_left']
        Rt_right = pd_events['Rt_right']
        CCs_s_trig = pd_events['CCs_s_trig']
        #* plot CCs_s_trig, triggered points
        ax.scatter(Rt_trig, (CCs_s_trig), c=color, s=15, alpha=alpha, marker='*', label=label)
        # 遍历每个矩形
        for t1, t2, h in zip(Rt_left, Rt_right, (CCs_s_trig)):
            rect = patches.Rectangle(
                (t1, 0),         # 左下角 (t1, y=0)
                t2 - t1,         # 宽度 (t2 - t1)
                h,               # 高度 (h)
                color=color,     # 颜色
                alpha=0.07        # 透明度
            )
            ax.add_patch(rect)
    
    # mini_plot(Events_classified[ (Events_classified['Template'] == Template) ], 
    #           axes[tt], 'b', 0.3, 'Unclassified')
    mini_plot(Events_classified[ (Events_classified['Template'] == Template) & (Events_classified['Belong_or_not'] == 1) ], 
              axes[tt], 'k', 0.2, label=f'{disp_Template}: #{len(Rt_trig_classified)}')

    #* limit the y axis scale is very important
    axes[tt].set_ylim([-0.5, 0.5])
    axes[tt].set_ylabel('Stacked Corre. Coeff.', fontsize=22)
    
axes[0].set_title(f'Trigger and Classified Points of Classified Events: Total = {EQ_sum} events')


#ADD UP STALTA TRIGGERED EVENTS as reference
peak_time, peak_amp, peak_score = _get_STALTA_catalog(t_start, t_end)
dt = timedelta(hours=1, minutes=0, seconds=0)
t_edges = np.arange(t_start, t_end+dt, dt)
axes[-1].hist(peak_time,bins=t_edges, color='r',label=f'STALTA: #{len(peak_time)}',alpha=0.3)
axes[-1].set_ylabel(f'Counts per {dt}', fontsize=22)
axes[-1].set_xlabel('Time')

for ax in axes:
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.set_xlim([t_start, t_end])
    Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=22)
    # Ticks_xaxis_minute_or_second(ax, Minute_tick=00, Second_tick=10, rotation=45)
    ax.legend(loc='upper right')
    ax.grid('--')

    
# axes[-1].hist(peak_time,bins=t_edges, color='r',label='STALTA')
# axes[-1].legend(loc='upper right')
png_name = OUTPUT_dir_byTime+f"Classified_allTemp_TRIG_POINTS_{t_str}.png"
fig.savefig(png_name)
print(f'>>>>>>>>Saved in: {png_name}')
plt.close(fig)




# #*_______________________COLLECT Collapse_group from all templates_______________________________
# Collapse_group = pd.DataFrame()
# for tt,Template in enumerate(Temp_name_list):
#     #? Read CCs_s_trig
#     CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right \
#         = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
#     DF_trig = pd.DataFrame({
#         'CCs_s_trig': CCs_s_trig,
#         'RAWCCs_s_trig': RAWCCs_s_trig,
#         't_trig': t_trig,
#         'width_trig': width_trig,
#         't_left': t_left,
#         't_right': t_right
#     })
#     # DF_trig = Read_CCs_trig_pkl_toPANDAS(IN_CCs_trig_dir, Template, t_start, t_end)
    
#     #? ADD Template time offset to the DF_trig
#     DF_trig.loc[:, 'Rt_trig'] = Remove_Temp_Offest_forSeries( DF_trig['t_trig'], Template, DF_Temp_Offset, offset_Type = 'S')
#     DF_trig.loc[:, 'Rt_left'] = Remove_Temp_Offest_forSeries( DF_trig['t_left'], Template, DF_Temp_Offset, offset_Type = 'S' )
#     DF_trig.loc[:, 'Rt_right'] = Remove_Temp_Offest_forSeries( DF_trig['t_right'], Template, DF_Temp_Offset, offset_Type = 'S' )
#     #? ADD MAD value normalization
#     DF_trig.loc[:, 'CCs_s_trig_ofMAD'] = DF_trig['CCs_s_trig'].astype('float64') / MAD_array_dict[Template][0]
#     #? ADD template name
#     DF_trig.loc[:, 'Template'] = Template
#     # print(Template,'==',DF_trig)
    
#     Collapse_group = pd.concat([Collapse_group, DF_trig], axis=0)
    
# Collapse_group.reset_index(drop=True, inplace=True)
# print('====>Collapse_group:', Collapse_group)
# Collapse_group = Collapse_group.sort_values(by=['Rt_trig'], ascending=True, ignore_index=True)
# print('====>Collapse_group by time',Collapse_group)


# #*____________________PROCESS the Collapse_group________________________________________________
# # use pd functions to collapse the overlapped multiple-template events, use parameter: IN_one_event_interval
# # Collapse_group['Rt_trig'] in order
# # mark the row with highest abs(CCs_s_trig) in the group, --> mark the 'Belong_or_not' = 1
# # 初始化标记列

# Events_classified = Collapse_with_MAX_CCs(Collapse_group, IN_one_event_interval)
# Events_classified.to_csv(OUTPUT_dir_byTime + 'Events_classified_AllTemp.csv', sep='\t', index=False)
# # 统计'Belong_or_not' = 1 的行数， 这也等于group的最大值
# print(f'*********Event number before collapsed:', len(Events_classified))
# print(f'*********Event number after collapsed:', Events_classified['Belong_or_not'].sum())
# print(f'*********Event number after collapsed:', Events_classified['group'].max())

# # this is done!!!!
