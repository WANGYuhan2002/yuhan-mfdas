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



# 1st loop: Temp_name_list loop
# 2nd loop: t_array loop (day by day)


# --1-- read CCs_s (stacked) in h5 format
# --2-- do dynamic triggering for certain time interval, using MAD_array_dict and threshold_level
# --3-- save the results in h5 format

# --4-- accompanied with each trigger file day by day, generate a plot: CCs, triggered point, MAD, threshold

#!__________________________functions_____________________________
sys.path.append(f"{PROJECT_ROOT}/functions")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour
from trigger import Trigger_with_MADarray
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

# def _average_MAD_array_dict(MAD_array_dict, Temp_name_list):
#     """
#     Args:
#         MAD_array_dict: dict, keys are template names, values are MAD_array
#         Temp_name_list: list of template names
#     Returns:
#         MAD_array_dict: dict, keys are template names, values are average MAD_array
#     """
#     MAD_array_dict_new = {}
#     for Template in Temp_name_list:
#         MAD_array_dict_new[Template] = np.array([np.mean(MAD_array_dict[Template])])
#         MAD_array_dict_new['t_array'] = np.array([MAD_array_dict['t_array'][0], MAD_array_dict['t_array'][-1]])
#     return MAD_array_dict_new

#!__________________________config_____________________________

class Config:
    def __init__(self):
        
        #?++++++++++++++++trigger 关键 parameters+++++++++++++++++++++++++++++
        self.Threshold_level = 40 # 触发阈值，单位为该模板的MAD值
        self.Minimal_event_interval = 3 # in seconds
        #?+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        #* 输入路径
        self.temp_npy_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"
        
        self.IN_CCs_stack_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/'
        self.MAD_Avg_path = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/4:00:00/' + 'MAD_Average_dict.pkl'  # MAD_array_dict.pkl
        self.MAD_Avg_dict = load_pkl(self.MAD_Avg_path)  # load the MAD_array_dict, keys are template names, values are MAD_array


        #* 输出路径
        self.subdir = f"level_{self.Threshold_level}/"
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/" + self.subdir

        
        
        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [5, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 8e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286

        
        self.Temp_name_list = [
            'temp_0',   'temp_1',   
            'temp_2',   'temp_3',   
            'temp_4',   'temp_5',
            'temp_6',   'temp_7',   
            'temp_8',   'temp_9',   
            'temp_10',  'temp_11',
            'temp_12'
        ]
        
        self.t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        self.t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_delta = timedelta(minutes=10)
        self.t_array = np.arange(self.t_all_start, self.t_all_end, self.t_delta)
        self.t_array = np.append(self.t_array, self.t_all_end)

    def _save_json(self, path):
        def is_json_serializable(value):
            try:
                json.dumps(value)
                return True
            except:
                return False

        clean_dict = {k: v for k, v in self.__dict__.items() if is_json_serializable(v)}

        with open(path, 'w') as f:
            json.dump(clean_dict, f, indent=4, ensure_ascii=False)


def main():
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir, 'config.json'))


    print('t_array', config.t_array, '\n')
    print(f'****** Triggering from {config.t_all_start} to {config.t_all_end} with time interval={config.t_delta}\n')

    print(f"Template names: {config.Temp_name_list}\n")
    print(f"Threshold level: {config.Threshold_level}\n")
    print(f"Minimal event interval: {config.Minimal_event_interval} seconds\n")
    print('MAD_array_dict:', config.MAD_Avg_dict, '\n')


    #!____________________________________________________________
    for tt,Template in enumerate(config.Temp_name_list):  
        OUTPUT_dir_byTemp = config.OUTPUT_dir + Template + '/'
        if not os.path.exists(OUTPUT_dir_byTemp):
            os.makedirs(OUTPUT_dir_byTemp)
        print(f'\n====> Trigger for {Template}. {tt} of {len(config.Temp_name_list)}')      

        NOtrig_Temp = 0
        for ii in range(len(config.t_array)-1):
            t_start = config.t_array[ii]  
            t_end = config.t_array[ii+1]
            print(f'\n>>>>> \033[33m{ii + 1} / {len(config.t_array)}\033[0m Triggering CCs of \033[34m{Template}\033[0m from {t_start} to {t_end}')

            filename = f'CCs_trig_{Template}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.pkl'
            if os.path.exists(OUTPUT_dir_byTemp + filename):
                print(f'File {filename} already exists, skip')
                continue
            
            CCs_s, RAWCCs_s = Read_CCs_stack_h5(config.IN_CCs_stack_dir, Template, t_start, t_end)

            CCs_s_trig, t_trig, t_left, t_right, width_trig = Trigger_with_MADarray(t_start, t_end, config.fs, CCs_s, config.MAD_Avg_dict['t_array'], config.MAD_Avg_dict[Template], config.Threshold_level, config.Minimal_event_interval)
            # decide if the trigger is valid or empty
            NOtrig_Temp += len(t_trig)
            print(f'\033[32m\-->Found #{len(t_trig)} events, --> Accumulated #{NOtrig_Temp} events.\033[0m')
            if len(t_trig) == 0:
                continue 
            
            #?小心这个估计可能不稳定，会崩掉
            idx_t_trig = np.array([round( (t-t_start).total_seconds() *config.fs ) for t in t_trig], dtype=int)
            RAWCCs_s_trig = RAWCCs_s[idx_t_trig] #! 数量级在e-14左右，担心加减法会有很大的相对误差，float64的精度只有e-16
            
            
            #!之后再改
            # t_s = np.array([t_start + timedelta(seconds=ii/fs) for ii in range(len(CCs_s))])
            # win_indices = np.where((t_s >= t_left) & (t_s <= t_right))[0]
            # win_RAWCCs = RAWCCs_s[win_indices]

            # i_arg_max = np.argmax(np.abs(win_RAWCCs))
            # max_RAWCCs = win_RAWCCs[i_arg_max]
            # t_arg_max = t_s[win_indices][i_arg_max]
            
            CCs_s_trig_ = CCs_s[idx_t_trig]
            if CCs_s_trig_.all() == CCs_s_trig.all():
                print("OKOKOKOKOK")
            
            #* save to pkl (only python)
            OUT_dict = {'t_trig': t_trig, 'CCs_s_trig': CCs_s_trig, 'RAWCCs_s_trig': RAWCCs_s_trig, 
                        'width_trig': width_trig, 't_left': t_left, 't_right': t_right,
                        'SCCs_of_MAD': CCs_s_trig/config.MAD_Avg_dict[Template],
                        # 't_arg_max': t_arg_max, 'max_RAWCCs': max_RAWCCs,
                        }
            filename = f'CCs_trig_{Template}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.pkl'
            write_pkl(OUT_dict, OUTPUT_dir_byTemp + filename)
            
            ##* save to h5 (broader use)
            ## if save to h5 file do not support for datetime64, so convert to string
            # t_trig_str = [t.strftime('%Y%m%d_%H%M%S.%f') for t in t_trig]
            # OUT_dict = {'t_trig': t_trig_str, 'CCs_s_trig': CCs_s_trig, 'RAWCCs_s_trig': RAWCCs_s_trig}
            # filename = f'CCs_trig_{Template}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.h5'
            # write_h5(OUTPUT_dir_byTemp + filename, OUT_dict)
            
            del CCs_s, RAWCCs_s
            del CCs_s_trig, t_trig, t_left, t_right, width_trig
            del RAWCCs_s_trig, idx_t_trig, CCs_s_trig_
            del OUT_dict

            if (ii + 1) % 50 == 0:  # 每处理 50 个模板进行一次垃圾回收
                gc.collect()
            


if __name__ == "__main__":
    main()
    print("\n\n\033[32mAll done!\033[0m")
    print("Please check the output directory:", Config().OUTPUT_dir)