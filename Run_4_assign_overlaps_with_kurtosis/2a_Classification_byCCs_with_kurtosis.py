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


#!__________________________functions_____________________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
from classify import Collapse_with_MAX_CCs, Read_Collapse_group_df,Remove_Temp_Offest_forSeries
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

#!__________________________config_____________________________

class Config:
    def __init__(self):
        
        #?++++++++++++++++trigger 关键 parameters+++++++++++++++++++++++++++++
        self.Threshold_level = 20 # 触发阈值，单位为该模板的MAD值
        self.Minimal_event_interval = 3 # in seconds
        
        self.max_stalta_threshold = 10
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        #     print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        

        self.t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
        self.t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_str = '_' + self.t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + self.t_end.strftime('%Y-%m-%d %H:%M:%S')


        #* 输入路径
        self.HOME_dir = f'{PROJECT_ROOT}/'

        self.temp_lib_csv = f"{self.HOME_dir}Output_0_Template_library/Template_library.csv"
        self.temp_npy_dir = f"{self.HOME_dir}Output_0_Template_library/Template_npy_files/"

        self.IN_CCs_stack_dir = f"{self.HOME_dir}Output_1_SCCs_and_MAD/SCCs_h5/"
        # Path to the MAD average dictionary
        self.MAD_Avg_path = f"{self.HOME_dir}Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/MAD_Average_dict.pkl"
        self.MAD_Avg_dict = load_pkl(self.MAD_Avg_path)  # load the MAD_array_dict
        
        self.subdir = f"level_{self.Threshold_level}/"
        self.IN_CCs_trig_dir = f"{self.HOME_dir}Output_2_Triggered_SCCs_points/{self.subdir}"

        #* 输出路径
        self.OUTPUT_dir = f"{self.HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/{self.subdir}"

        # self.OUTPUT_dir_byTime = self.OUTPUT_dir + 'classified' + self.t_str + '/'

        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286
        
        # 选择继续进入classification的模板列表
        # self.Temp_name_list = [
        #     'temp_0',   'temp_1',   
        #     'temp_2',   #'temp_3',   
        #     'temp_4',   #'temp_5',
        #     # 'temp_6',   
        #     'temp_7',   
        #     'temp_8',   'temp_9',   
        #     'temp_10',  'temp_11',
        #     'temp_12'
        # ]
    
        self.Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]
        
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

#!_______________________DATA IN________________________________

def main():
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir, 'config.json'))

    
    Temp_lib_df = pd.read_csv(config.temp_lib_csv,sep=',')
    # trim to the templates we want to check
    check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(config.Temp_name_list)]
    # rank in plot order
    check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)


# offset_fname = f'{PROJECT_ROOT}/Input_template_set/template_offset_kefa.csv'
# DF_Temp_Offset = pd.read_csv(offset_fname, sep='\t', header=0)
# print('====>DF_Temp_Offset:', DF_Temp_Offset)

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



    #*_______________________COLLECT Collapse_group from all templates_______________________________
    Collapse_group_raw = Read_Collapse_group_df(config.IN_CCs_trig_dir, config.Temp_name_list, DF_Temp_Offset, config.MAD_Avg_dict, config.t_start, config.t_end)
    print('====>Collapse_group_raw by time',Collapse_group_raw)
    #!+++++这里的 Read_Collapse_group_df 是修改过的，包含了新的 max_RAWCCs, t_arg_max 信息

    #!+++++这里的 Read_Collapse_group_df 是修改过的，包含了新的 max_stalta 信息
    mask = Collapse_group_raw['max_stalta'] > config.max_stalta_threshold
    #? 删除 不满足 max_stalta条件的触发事件
    Collapse_group = Collapse_group_raw[mask].copy()
    Collapse_group.reset_index(drop=True, inplace=True)
    print('====>Collapse_group after max_stalta threshold', len(Collapse_group))
    
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        print(f'\n********** {Template} disp as {disp_Template}**********')
        df_raw = Collapse_group_raw[Collapse_group_raw['Template'] == Template]
        df = Collapse_group[Collapse_group['Template'] == Template]
        print(f'-----Event number before clean:', len(df_raw))
        print(f'-----Event number after clean:', len(df))

    #*____________________PROCESS the Collapse_group________________________________________________
    # use pd functions to collapse the overlapped multiple-template events, use parameter: IN_one_event_interval
    # Collapse_group['Rt_trig'] in order
    # mark the row with highest abs(CCs_s_trig) in the group, --> mark the 'Belong_or_not' = 1
    # 初始化标记列

    #* assign overlaps by maximum CCs_s_trig
    Events_classified = Collapse_with_MAX_CCs(Collapse_group, config.IN_one_event_interval)
    Events_classified.to_csv(config.OUTPUT_dir + 'Events_classified_AllTemp.csv', sep='\t', index=False)


    #* add maximum RAWCCs_s between t_left and t_right to the list of Events_classified
    
    #!处理办法1----从连续CCs_s中读取 读取每个事件的最大 RAWCCs_s,   请在另一个脚本实现
    # for ii, row in Events_classified.iterrows():
    #     t_left = row['t_left']
    #     t_right = row['t_right']
    #     Template = row['Template']
        
    #     CCs_s, RAWCCs_s = Read_CCs_stack_h5(config.IN_CCs_stack_dir, Template, t_left, t_right)
        
    #     # 计算最大 RAWCCs_s
    #     i_arg_max = np.argmax(np.abs(RAWCCs_s))
    #     max_RAWCCs = RAWCCs_s[i_arg_max]
    #     t_arg_max = t_left + timedelta(seconds=i_arg_max / config.fs)

    #     Events_classified.at[ii, 'max_RAWCCs'] = max_RAWCCs
    #     Events_classified.at[ii, 't_arg_max'] = t_arg_max
    
    #! 处理办法2----如果 max_RAWCCs 已经被记录到了 读取每个事件的最大 RAWCCs_s
    # 读取触发的CCs_s数据
    # 使用新的函数读取最开始的collapsed触发事件数据
    
    
    
    
    Events_classified.to_csv(config.OUTPUT_dir + 'Events_classified_AllTemp_addRAWCCs.csv', sep='\t', index=False)
    
    
    # 统计'Belong_or_not' = 1 的行数， 这也等于group的最大值
    print("\n\n+++++++")
    print(f'*********Event number before collapsed:', len(Events_classified))
    print(f'*********Event number after collapsed:', Events_classified['Belong_or_not'].sum())
    print(f'*********Event number after collapsed:', Events_classified['group'].max())

    # this is done!!!!

    # initially triggered events of each template
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        print(f'\n********** {Template} disp as {disp_Template}**********')
        df = Events_classified[Events_classified['Template'] == Template]
        print(f'-----Event number before collapsed:', len(df))
        print(f'-----Event number after collapsed:', df['Belong_or_not'].sum())

if __name__ == "__main__":
    main()
    # clear_output(wait=True)
    print('Done!')