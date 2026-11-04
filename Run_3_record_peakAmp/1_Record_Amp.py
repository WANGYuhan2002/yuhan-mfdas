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

#! FUNCTIONS:
#! ---> used to  trace back the Matched Filtered detected events to the original data
#! WHY?:  Because save the complete data and CCs matrix is too heavy, we save only stacked CCs,
#! But with the final results of those events, we can go back with very limited cost, re-computing or re-save


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


#!__________________________functions_____________________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_CCs_stack_h5, Read_DAS_data
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from classify import Remove_Temp_Offest_forSeries, Collapse_with_MAX_CCs, Read_Collapse_group_df


#!_______________________DATA IN________________________________
fs = 80

Temp_name_list = [
            'temp_0',   'temp_1',   
            'temp_2',   'temp_3',   
            'temp_4',   'temp_5',
            'temp_6',   'temp_7',   
            'temp_8',   'temp_9',   
            'temp_10',  'temp_11',
            'temp_12'
        ]

IN_Temp_lib_csv = f"{PROJECT_ROOT}/Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv,sep=',')
# trim to the templates we want to check
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Temp_name_list)]
# rank in plot order
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)


#* 创建补充的DF_Temp_Offset, 从 template library中
# template_name	P_offset	S_offset
temp_lib_df = pd.read_csv(config.temp_lib_csv, sep=',')
#* 从 template_lib_df 中提取所需的列，t_P_sec,t_S_sec，还有f'temp_{index}'
DF_Temp_Offset = pd.DataFrame({
    'template_name': temp_lib_df['infile_name'],#[f"temp_{i}" for i in range(len(temp_lib_df))],
    'P_offset': temp_lib_df['t_P_sec'],
    'S_offset': temp_lib_df['t_S_sec']
})
print('====>DF_Temp_Offset:', DF_Temp_Offset)


MAD_Avg_path = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/4:00:00/' + 'MAD_Average_dict.pkl'
MAD_Avg_dict = load_pkl(MAD_Avg_path)  # load the MAD_array_dict



subdir = 'level_25/'
IN_CCs_Classification_dir = f'{PROJECT_ROOT}/Output_3_Classified_bySCCs_cleaned_overlaps/' + subdir
IN_CCs_trig_dir = f'{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/' + subdir

#!_______________________DATA OUT________________________________
OUTPUT_dir =  f'{PROJECT_ROOT}/Output_4_PeakAmp_Mag_regression/' + 'RecordPeakMag/' + subdir

#!_______________________DATA OUT________________________________

t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
t_str = '_' + t_all_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_all_end.strftime('%Y-%m-%d %H:%M:%S')


for Template in Temp_name_list:
    OUTPUT_dir_byTemp = OUTPUT_dir + Template + '/'
    if not os.path.exists(OUTPUT_dir_byTemp):
        os.makedirs(OUTPUT_dir_byTemp)

# file saving logic:
# read in classifiesd events of different templates, each row is a event from certain template
# extract the maximum CCs_s near the triggered time, of each channel, and the corresponding time
# save the results in a new file, within the subdirectory of the template, named by the Rt_trig time. saved in pickle (dict)

#!_________________________PARAS_____________________________
Search_win_right = 2.5  # seconds, searched near triggered time: (Rt_trig -Search_win ~ Rt_trig + Search_win)
Search_win_left = 1.5  # seconds, searched near triggered time: (Rt_trig -Search_win ~ Rt_trig + Search_win)

#*______________________________________________________
#*_______________________COLLECT Collapse_group from all templates_______________________________
Collapse_group = Read_Collapse_group_df(IN_CCs_trig_dir, Temp_name_list, DF_Temp_Offset, MAD_Avg_dict, t_all_start, t_all_end)
print('====>Collapse_group by time',Collapse_group)
#* keys: ['Template', 'Rt_trig', 'Rt_left', 'Rt_right', 't_trig', 't_left', 't_right', 'width_trig', 'CCs_s_trig', 'RAWCCs_s_trig', 'CCs_s_trig_ofMAD']


#*______________________________________________________
Trig_MAX_amp_df = Collapse_group.copy()  # make a copy of Collapse_group to store the results

for ii, row in Collapse_group.iterrows():
    print(f"=====> Processing <{ii}/{len(Collapse_group)}>: {row.Template} at {row.Rt_trig.strftime('%Y-%m-%d %H:%M:%S')}")
    # read in the DAS data
    fs,f1,f2 = 80,5,20
    dx = 16 * 1.02095199
    d1 = 8300
    d2 = 14000
    
    t_left = row.Rt_left - timedelta(seconds=Search_win_left)
    t_right = row.Rt_right + timedelta(seconds=Search_win_right)
    data = Read_DAS_data(t_left, t_right, fs,f1,f2, dx,d1,d2)
    
    # extract the maximum absolute value of each channel
    amp_max = np.max(np.abs(data), axis=0)
    # find the index of the maximum value
    index_max = np.argmax(np.abs(data), axis=0)
    # find the time of the maximum value
    # t_max = [None] * len(index_max)
    # for i in range(len(index_max)):
    #     t_max[i] = t_left + timedelta(seconds=index_max[i] / fs)
    t_max = np.array([t_left + timedelta(seconds=index_max[i] / fs) for i in range(len(index_max))])
    
    # append the amp_max_mean to Dataframe 
    Trig_MAX_amp_df.loc[ii, 'amp_max_mean'] = np.mean(amp_max)
    Trig_MAX_amp_df.loc[ii, 'amp_log_mean'] = np.mean(np.log10(abs(amp_max)))  # add a small value to avoid log(0)
    
    # Trig_MAX_amp_df.loc[ii, ['Template', 'CCs_s_trig', 'RAWCCs_s_trig', 'CCs_s_trig_ofMAD']] = \
    #     row[['Template', 'CCs_s_trig', 'RAWCCs_s_trig', 'CCs_s_trig_ofMAD']]
    # Trig_MAX_amp_df.loc[ii, ['Rt_trig', 'Rt_left', 'Rt_right', 't_trig', 't_left', 't_right', 'width_trig']] = \
    #     row[['Rt_trig', 'Rt_left', 'Rt_right', 't_trig', 't_left', 't_right', 'width_trig']]

    # create a dictionary to save the results
    result_dict = {
        'Rt_trig': row.Rt_trig,
        'Rt_left': row.Rt_left,
        'Rt_right': row.Rt_right,
        't_trig': row.t_trig,
        't_left': t_left,
        't_right': t_right,
        'width_trig': row.width_trig,

        'Template': row.Template,
        'group': row.group,
        'CCs_s_trig': row.CCs_s_trig,
        'RAWCCs_s_trig': row.RAWCCs_s_trig,
        'CCs_s_trig_ofMAD': row.CCs_s_trig_ofMAD,

        'amp_max_mean': np.mean(amp_max),
        'amp_max': amp_max,
        't_max': t_max,
        
        'amp_log_mean': np.mean(np.log10(abs(amp_max))),  # add a small value to avoid log(0)
    }
    
    OUTPUT_dir_byTemp = OUTPUT_dir + row.Template + '/'
    # save the results in a pickle file
    output_fpath = OUTPUT_dir_byTemp + f'amp_max_at_Rt_trig_{row.Rt_trig.strftime("%Y-%m-%d %H:%M:%S")}.pkl'
    with open(output_fpath, 'wb') as f:
        pickle.dump(result_dict, f)
    

# can save the Event_classified with the new columns
Trig_MAX_amp_df.to_csv(OUTPUT_dir + 'Trig_MAX_amp_df.csv', sep='\t', index=False)
    
