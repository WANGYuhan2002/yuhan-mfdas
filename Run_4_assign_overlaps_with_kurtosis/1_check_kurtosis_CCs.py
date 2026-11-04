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

HOME_dir = f'{PROJECT_ROOT}/'

sys.path.append(f"{HOME_dir}functions/")
from readers import Read_DAS_data, Read_template, Read_CCs_stack_h5, load_pkl, write_pkl
from plot import waterfall_date
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024


def event_row_IN_fig_OUTput(Event_row, IN_CCs_stack_dir, png_path, Threshold):
    """
    Check if the trigger point is really triggered
    """
    #*______________GET attributes from events___________________________
    t_trig = Event_row.t_trig.to_pydatetime()
    t_right = Event_row.t_right.to_pydatetime()
    t_left = Event_row.t_left.to_pydatetime()
    CCs_s_trig = Event_row.CCs_s_trig
    Template = Event_row.Template

    Rt_trig = Event_row.Rt_trig.to_pydatetime()
    Rt_left = Event_row.Rt_left.to_pydatetime()
    Rt_right = Event_row.Rt_right.to_pydatetime()
    
    Offset = Rt_trig - t_trig
    
    #*___________________Read data and Templates____________________
    pad_pre_sec = 15  # seconds before the trigger time to read data
    pad_post_sec = 15  # seconds after the trigger time to read data
    t_start = t_trig - timedelta(seconds=pad_pre_sec)
    t_end = t_trig + timedelta(seconds=pad_post_sec)
    
    ########################################################################################
    # use the correlation range we used to read the DAS data
    fs,f1,f2 = 80, 8, 20
    dx = 16 * 1.02095199
    d1 = 8300
    d2 = 14000

    filter_str = 'filtered ' + str(f1) + ' - ' + str(f2) + ' Hz'
    ########################################################################################
    
    data = Read_DAS_data(t_start, t_end, fs,f1,f2, dx,d1,d2)
    print('--read DAS data from',t_start,'to',t_end)
    temp = Read_template(Template, fs,f1,f2, dx,d1,d2,f'{HOME_dir}Output_0_Template_library/Template_npy_files/')

    #? the data range from t_start to t_end (real time) 
    CCs_s, RAWCCs_s = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end)
    t_s = np.array([t_start + timedelta(seconds=ii/fs) for ii in range(len(CCs_s))])
    
    # modifying before plot ------
    npts_diff = data.shape[0] - temp.shape[0]

    pad_pre = int(pad_pre_sec * fs)

    pad_post = npts_diff - pad_pre
    if npts_diff > 0:
        if pad_post < 0:
            print('Case 1: pad_post < 0')
            temp_padded = np.pad(temp, ((pad_pre, 0), (0,0)), 'constant')
            temp_padded = temp_padded[:pad_post]
        else:
            print('Case 2: pad_post > 0')
            temp_padded = np.pad(temp, ((pad_pre, pad_post), (0,0)), 'constant')
    elif npts_diff <= 0:
        print('Case 3: npts_diff < 0')
        temp_padded = np.pad(temp, ((pad_pre, 0), (0,0)), 'constant')
        temp_padded = temp_padded[:pad_post]
    print('temp_padded shape:', temp_padded.shape)
    print('data shape:', data.shape)
    # #*_________________________________
    # fig = plt.figure(figsize=(22,22))
    # fig.suptitle('Template: {0}; Filter Band: {1}; Corr: {2}'.format(Template, [f1,f2],CCs_s_trig))
    # gs = plt.GridSpec(3,2,figure=fig,height_ratios=[3,3,2],width_ratios=[7.3,1])
    # ax = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:]),fig.add_subplot(gs[2,0])]
    # unit = 'Strain rate (nm/m)/s'
    
    #*_________________________________
    fig = plt.figure(figsize=(22,22))
    fig.suptitle('Template: {0}; Filter Band: {1}; Corr: {2}'.format(Template, [f1,f2],CCs_s_trig))
    gs = plt.GridSpec(4,2,figure=fig,height_ratios=[3,3,2,2],width_ratios=[7.3,1])
    ax = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[3,0])]
    unit = 'Strain rate (nm/m)/s'


    #?--1-- template_padded
    # waterfall_simple(template_padded,ax[0],0.0,fs,dx,template_t0,'(nm/m)/s')
    # waterfall_date(ax, data, fs, t_start, dx, d1, unit, vlim=0)
    waterfall_date(ax[0], temp_padded, fs, t_start, dx, d1/1e3, unit, vlim=0)
    ax[0].set_title(f'{Template} used in matched filter; S offset = {Offset} sec')
    ax[0].grid()

    #?--2-- data:  t_start+data_plot_offset~t_end+data_plot_offset
    # waterfall_simple(data_filtered,ax[1],0.0,fs,dx,t_start+data_plot_offset,'(nm/m)/s')
    waterfall_date(ax[1], data, fs, t_start, dx, d1/1e3, unit, vlim=0)
    ax[1].set_xlabel('Time of DAS data')

    # mark the triggered points
    ax[0].axvline(x=t_trig, color='k', linestyle='--')
    ax[0].axvline(x=Rt_trig, color='r', linestyle='--')
    
    ax[1].axvline(x=Rt_trig, color='r', linestyle='--', label='Expected EQ time')
    ax[1].axvline(x=t_trig, color='k', linestyle='--')
    ax[1].legend(loc='upper right')
    
    
    # ax[2].axvline(x=t_trig, color='r', linestyle='--', label='Triggered points')
    # ax[2].legend(loc='upper right')
    
    #?--3-- corr stacked: t_start~t_end
    ax[2].set_xlim([t_start, t_end])
    ax[2].set_ylabel('Stacked Corr. Coeff.')
    ax[2].set_xlabel('Time of Corr. Coeff.')
    ax[2].set_title('Corr stacked on {0} GMT'.format(t_start.strftime('%Y.%m.%d')))
    
    ax[2].scatter(t_trig, (CCs_s_trig), c='r', s=15, alpha=0.7, marker='*', label='Trigger points')
        # 遍历每个矩形
    t1, t2, h = t_left, t_right, abs(CCs_s_trig)
    rect = patches.Rectangle(
        (t1, -h),         # 左下角 (t1, y=0)
        t2 - t1,         # 宽度 (t2 - t1)
        2*h,               # 高度 (h)
        color='r',     # 颜色
        alpha=0.1        # 透明度
    )
    ax[2].add_patch(rect)
    
    ax[2].plot(t_s,CCs_s, color='k', alpha=0.7,linewidth=0.7)
    
    ax[2].axhline(y=Threshold, color='b', linestyle='--', alpha=0.5, label=f'Threshold={Threshold:.3f}')
    # ax[2].axhline(y=Threshold, color='b', linestyle='--', alpha=0.5)
    ax[2].axhline(y=-Threshold, color='b', linestyle='--', alpha=0.5)
    ax[2].legend(loc='upper right')
    
    
    
    #?--4-- RAWCCs stacked: t_start~t_end
    ax[3].set_xlim([t_start, t_end])
    ax[3].set_ylabel('Stacked RAW. CCs.')
    ax[3].set_xlabel('Time of RAW. CCs.')
    ax[3].set_title('RAW. stacked CCs on {0} GMT'.format(t_start.strftime('%Y.%m.%d')))


    win_indices = np.where((t_s >= t_left) & (t_s <= t_right))[0]
    win_RAWCCs = RAWCCs_s[win_indices]

    i_arg_max = np.argmax(np.abs(win_RAWCCs))
    max_RAWCCs = win_RAWCCs[i_arg_max]
    t_arg_max = t_s[win_indices][i_arg_max]

    ax[3].scatter(t_arg_max, max_RAWCCs, c='r', s=15, alpha=0.7, marker='*', label='max RAWCCs')
        # 遍历每个矩形
    t1, t2, h = t_left, t_right, max(abs(RAWCCs_s))
    rect = patches.Rectangle(
        (t1, -h),         # 左下角 (t1, y=0)
        t2 - t1,         # 宽度 (t2 - t1)
        2*h,               # 高度 (h)
        color='r',     # 颜色
        alpha=0.1        # 透明度
    )
    ax[3].add_patch(rect)
    
    ax[3].plot(t_s,RAWCCs_s, color='k', alpha=0.7,linewidth=0.7)
    
    ax[3].legend(loc='upper right')
    
    
    
    
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    print(f'-----> {png_path}')

    return ax



#!_______________________DATA IN________________________________
fs = 80

Temp_name_list = [
    'template_0423_80Hz',
    'template_0425_80Hz',
    'template_10_80Hz',
    'template_8_80Hz',
    'template_20_80Hz',
]
        
IN_Temp_lib_csv = f"{HOME_dir}Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv,sep=',')
# trim to the templates we want to check
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Temp_name_list)]
# rank in plot order
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)


for Threshold_level in [20]:
    subdir = f'level_{Threshold_level}/'
    IN_CCs_Classification_dir = f'{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withMagRegressed/' + subdir

    IN_CCs_stack_dir = f'{HOME_dir}Output_1_SCCs_and_MAD/SCCs_h5/'

    IN_MAD_Avg_path = f'{HOME_dir}Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/' + 'MAD_Average_dict.pkl'
    MAD_Avg_dict = load_pkl(IN_MAD_Avg_path)  # load the MAD_array_dict, keys are template names, values are MAD_array
    #!_______________________DATA OUT_______________________________
        


    #!_______________________DATA OUT________________________________

    t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
    t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
    t_str = '_' + t_all_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_all_end.strftime('%Y-%m-%d %H:%M:%S')

    Le_0 = Threshold_level + 1
    Le_1 = Threshold_level + 10


    OUTPUT_dir = f'{HOME_dir}Output_6_Classified_bySCCs_cleaned_overlaps_withRAWCCs/1_Check_RAWCCs_ok_wtf_snapshots/' + subdir
    if not os.path.exists(OUTPUT_dir):
        os.makedirs(OUTPUT_dir)
        
    OUTPUT_dir_byTimeLevel = OUTPUT_dir + f'ThreLevel_{Le_0}_to_{Le_1}' + t_str + '/'
    if not os.path.exists(OUTPUT_dir_byTimeLevel):
        os.makedirs(OUTPUT_dir_byTimeLevel)
        
    #*_________________Events_classified PREPROCESSING AND TRIM_____________________________________

    #! option 1 ——————Read in triggered points and make restrictions on the CCs_s_trig value and time
    Events_classified = pd.read_csv(IN_CCs_Classification_dir + 'Events_classified_AllTemp.csv', sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])

    mask =  (Events_classified['Template'].isin(Temp_name_list)) & \
            (Events_classified['Rt_trig'] >= t_all_start) & (Events_classified['Rt_trig'] <= t_all_end) & \
            (Events_classified['Belong_or_not'] == 1) & \
            (Events_classified['CCs_s_trig_ofMAD'].abs() >= Le_0) & (Events_classified['CCs_s_trig_ofMAD'].abs() <= Le_1)
            
            
    Events_classified = Events_classified[mask]  # 直接保留满足条件的行
    print(f'====> Events_classified cut from threhshold level {Le_0} to {Le_1} and time range from \
        {t_all_start} to {t_all_end}:\n {Events_classified}')


    print(f'====> from {t_all_start} to {t_all_end}\n\
        Threshold level: {Le_0} to {Le_1}\n\
        Get {len(Events_classified)} events.\n')

    for Template in Temp_name_list:
        print(f'    {Template}: {len(Events_classified[Events_classified["Template"] == Template])} events')
    print(f'Events_classified :\n {Events_classified}')

    #*______________________________________________________

    # for Template in Temp_name_list:
    #     if not os.path.exists(OUTPUT_dir_byTimeLevel + Template + '/'):
    #         os.makedirs(OUTPUT_dir_byTimeLevel + Template + '/')
    # if not os.path.exists(OUTPUT_dir_byTimeLevel + 'rank_byTime/'):
    #     os.makedirs(OUTPUT_dir_byTimeLevel + 'rank_byTime/')

    #*_________________________OUTPUT PNG____________________________
    #++++++++++++++++DECIDE HOW MANY PNG for EACH TEMPLATE we WANT TO OUTPUT++++++++++
    PNG_num = 5
    Temp_png_count = {Template: 0 for Template in Temp_name_list}

    for Event_row in Events_classified.itertuples(index=False):  # index=False 可以省略索引
        
        # count the number of events plot for each template
        Temp_png_count[Event_row.Template] += 1
        if Temp_png_count[Event_row.Template] > PNG_num:
            continue
        print('\n')
        print(f'====> plot for {Event_row.Template} the {Temp_png_count[Event_row.Template]}th figure')
        
        Threshold = Threshold_level * MAD_Avg_dict[Event_row.Template]
        Threshold = Threshold[0]
        print(f'    Threshold level: {Threshold_level} * MAD = {Threshold:.4f}')
        
        #*____output png by template
        disp_Template = check_lib[check_lib['infile_name'] == Event_row.Template]['plot_name'].values[0]
        
        png_fname = f"{disp_Template}_{abs(Event_row.CCs_s_trig):.4f}_{Event_row.CCs_s_trig_ofMAD:.1f}_{Event_row.t_trig}.png"
        png_path = OUTPUT_dir_byTimeLevel + png_fname
        
        #*____output png by time level
        # png_fname = f'{Event_row.t_trig}_{Event_row.Template}_CCs_{abs(Event_row.CCs_s_trig):.4f}.png'
        # png_path = OUTPUT_dir_byTimeLevel  + 'rank_byTime/' + png_fname
        ax = event_row_IN_fig_OUTput(Event_row, IN_CCs_stack_dir, png_path, Threshold)
