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

#!____________INPUT: EQs info in NOA_______________

#!____________OUTPUT: waterfall plot and .npy unfilterer template file______________________



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from datetime import datetime, timedelta
import glob
import pandas as pd
import time

import h5py
import os
from re import split
#-- To import a function on a relative path:
import sys

import scipy.io as scio
import scipy.signal as signal
import h5py
import gc

from matplotlib import ticker
import matplotlib.dates as mdates

import os
import gc
import sys
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import psutil
import time
from scipy.signal import correlate
from scipy.ndimage import uniform_filter1d
import multiprocessing

import json
import yaml
import inspect
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import geopandas
from matplotlib.lines import Line2D
import geodatasets
import contextily as cx
import cartopy.crs as ccrs # cartopy 坐标参考系统模块
import cartopy.feature as cfeature # cartopy 地理特征模块

#!_______________________FUNCTIONS_____________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, waterfall_date
from das_mffi_io import read_mffi_any, read_das_full_sum, das_full_sum_filter_all_in_one, mffi_filter_all_in_one, plot_together
import load_das_h5_PE2024 as load_das_h5_PE2024
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
from compute_correlation import preprocess
from readers import Read_DAS_data, Read_template

#!_______________________DATA IN________________________________import numpy as np

class Config:
    def __init__(self):
        
        # #* 输入路径 1, STA/LTA 目录
        # subdir = 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        # self.IN_cata_dir = f'{PROJECT_ROOT}/Run_0_preparing_templates_from_catalog/'
        # # self.IN_cata_dir += 'extract_threshold_300_trigNO_300_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'extract_threshold_100_trigNO_250_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'extract_threshold_300_trigNO_100_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'threshold_120_trigNO_240_sta0.3s_lta3s/'
        # # self.IN_cata_dir += 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        # # self.IN_cata_dir += 'threshold_50_trigNO_50_sta0.3s_lta3s/'
        # self.IN_cata_dir += subdir

        # self.catalog_path = f"{self.IN_cata_dir}/sta_lta_catalog.csv"
        
        # self.IN_SNRed_cata_dir = f"{self.IN_cata_dir}Output_1_Find_NOAeq_Mag2.0_10sec_SNR40/"
        # self.SNRed_catalog_path = f"{self.IN_SNRed_cata_dir}matched_df.csv"
        
        # self.IN_SNRed_corr_dir = f"{self.IN_SNRed_cata_dir}Output_2_Corr_movmax/"
        # self.SNRed_corr_path = f"{self.IN_SNRed_corr_dir}correlation_matrix.npy"
        
        # self.IN_duplicated_template_dir = f"{self.IN_SNRed_cata_dir}Output_3_Delete_similar_eqs/"
        # self.duplicated_template_path = f"{self.IN_duplicated_template_dir}dup_temp_df.csv"
        
        self.input_dir = das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        
        
        #* 唯一的输入文件
        self.IN_Template_library_path = f'{PROJECT_ROOT}/Output_0_Template_library/Template_library.csv'
        

        #* 输出路径---1---- 画图输出
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Check_again_corr_map_waterfall/"

        #* 输入路径---2---- 模板npy保存路径
        self.npy_output_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"

        # self.peak_time_left_pad = 8
        # self.peak_time_right_pad = 42


        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围.
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286

        self.save_as_temp_or_not = 1  # 是否保存为临时文件，1 为是，0 为否
        
        self.lat = [37.2, 39.4]
        self.lon = [20, 22.3]
        
        #?++++++++++++++++++++++++
        self.SNR_win_P = [0.2, 1.5]
        self.SNR_win_S = [0.2, 2.0]
        self.SNR_win_prepad = [30, 2]
    

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
    
    
    Temp_lib_df = pd.read_csv(config.IN_Template_library_path, sep=',', parse_dates=["peak_time","STA_arrival_time","EQ_time","t_start","t_end"])
    event_times = Temp_lib_df["t_start"].tolist()
    

    temp_lib_in_plot_order = Temp_lib_df.sort_values(by='plot_order', ascending=True).reset_index(drop=True)

    
    corr_mat_temp = np.load(config.OUTPUT_dir + 'correlation_matrix.npy')
    
    
    os.makedirs(config.npy_output_dir, exist_ok=True)
    for index, row in Temp_lib_df.iterrows():

        t_start = row['t_start']
        t_end = row['t_end']

        tP = t_start + pd.Timedelta(seconds=row['t_P_sec'])
        tS = t_start + pd.Timedelta(seconds=row['t_S_sec'])

        #!---读取数据---
        print(f"\n处理地震事件: {index}: 震级 {row['EQ_magnitude']}, 距离{(row['STA_distance']*111):.1f}, 时间范围: {t_start} 到 {t_end}")
        das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        DIR_das = das_h5_mnt_80Hz
        
        
        #?++++++++++ the window for SNR compute +++++++++++++++
        tS0 = tS - pd.Timedelta(seconds=config.SNR_win_S[0])
        tS1 = tS + pd.Timedelta(seconds=config.SNR_win_S[1])
        
        tP0 = tP - pd.Timedelta(seconds=config.SNR_win_P[0])
        tP1 = tP + pd.Timedelta(seconds=config.SNR_win_P[1])
        
        tpre0 = t_start - pd.Timedelta(seconds=config.SNR_win_prepad[0])
        tpre1 = t_start - pd.Timedelta(seconds=config.SNR_win_prepad[1])
        
        
        # dataS, _, _ = load_das_h5_PE2024.load_das_custom(tS0, tS1, input_dir=DIR_das)   
        # dataP, _, _ = load_das_h5_PE2024.load_das_custom(tP0, tP1, input_dir=DIR_das)
        # data_pre, _, _ = load_das_h5_PE2024.load_das_custom(tpre0, tpre1, input_dir=DIR_das)


        dataS = Read_DAS_data(tS0, tS1, config.fs,config.freq_band[0],config.freq_band[1], config.dx,config.das_dist_start,config.das_dist_end)
        dataP = Read_DAS_data(tP0, tP1, config.fs,config.freq_band[0],config.freq_band[1], config.dx,config.das_dist_start,config.das_dist_end)
        data_pre = Read_DAS_data(tpre0, tpre1, config.fs,config.freq_band[0],config.freq_band[1], config.dx,config.das_dist_start,config.das_dist_end)
        
        
        def rms_2d(data):
            """计算二维数据的均方根"""
            rms = np.sqrt(np.mean(data**2, axis=0))
            return np.mean(rms)
        
        S_rms = rms_2d(dataS)
        P_rms = rms_2d(dataP)
        pre_rms = rms_2d(data_pre)
        
        print(f"RMS for S window: {S_rms:3e}, P window: {P_rms:3e}, pre window: {pre_rms:3e}")
        
        #!计算SNR
        if S_rms > 0 and P_rms > 0 and pre_rms > 0:
            SNR_S = S_rms / pre_rms
            SNR_P = P_rms / pre_rms
        else:
            SNR_S = 0
            SNR_P = 0
            raise ValueError("RMS values are zero or negative, cannot compute SNR.")
        
        # 存到 DataFrame 中
        Temp_lib_df.at[index, 'SNR_S'] = SNR_S
        Temp_lib_df.at[index, 'SNR_P'] = SNR_P
        print(f"计算得到的 SNR_S: {SNR_S:.1f}, SNR_P: {SNR_P:.1f}")
        
        
        #?++++++++++++++++++ for energy compute ++++++++++++++++++++++++++++++
        # data, headers, axis = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=DIR_das)
        # temp = data[:, int(config.das_dist_start/config.dx):int(config.das_dist_end/config.dx)]
        # temp = preprocess(temp, config.fs, config.freq_band[0], config.freq_band[1])


        temp = Read_template(row['infile_name'], config.fs,config.freq_band[0],config.freq_band[1], config.dx,config.das_dist_start,config.das_dist_end,config.npy_output_dir)
        
        # temp = Read_DAS_data(t_start, t_end, config.fs,config.freq_band[0],config.freq_band[1], config.dx,config.das_dist_start,config.das_dist_end)
        
        print(f"temp: {temp.shape}")

        #!计算 self CC, unnormalized
        tempCC = temp * temp
        self_CC = np.mean(np.sum(tempCC,axis=0))
        
        Temp_lib_df.at[index, 'self_CC'] = self_CC
        print(f"计算得到的 self CC: {self_CC:3e}")

        #?++++++++++++++++++ for peak amp compute ++++++++++++++++++++++++++++++ 从 S 波 窗口中找最大值
        print(f"peak_amp: {dataS.shape}")
        peak_amp = np.max(np.abs(dataS),axis=0)
        amp_max_mean = np.mean(peak_amp)
        amp_log_mean = np.mean(np.log10(peak_amp))
        Temp_lib_df.at[index, 'amp_max_mean'] = amp_max_mean
        Temp_lib_df.at[index, 'amp_log_mean'] = amp_log_mean
        print(f"计算得到的峰值振幅: {amp_max_mean:3e}, 对数峰值振幅: {amp_log_mean:3e}")
        
        # ++++++++++++++++++ 将二级结果对 maxRAWCCs 转化为 peak amp 的缩放因子 fac_CC2amp 计算出来存储到 ++++++++++++++++++++++++++++++

        
        
        
        # #!---画图---
        # # data_plot = data[:,int(d1/dx):int(d2/dx)]
        # # data_plot = preprocess(data,fs,f1,f2)
        # data_filtered = preprocess(data[:, int(config.das_dist_start/config.dx):int(config.das_dist_end/config.dx)],\
        #                             config.fs, config.freq_band[0], config.freq_band[1])
         
        # fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        # vlim = np.mean(np.abs(data))*5
        # # plot grid
        # t_left = t_start
        # t_right = t_end
        # d_bottom = config.das_dist_start / 1000  # Convert to km
        # d_top = config.das_dist_end / 1000  # Convert to km
        # im = ax.imshow(data_filtered.T, aspect="auto", cmap='seismic', interpolation='none', 
        #         extent=(t_left,t_right, d_top, d_bottom),
        #         vmin=-vlim, vmax=vlim)
        
        
        # ax.axvline(tP, color='g', linestyle='--', label='P ')
        # ax.axvline(tS, color='g', linestyle='--', label='S ')

        # plt.colorbar(im,label='Strain rate (1/s)',ax=ax)
        
        # #-- y-axis
        # ax.set_ylabel("Distance [km]")
        # # ax.set_yticks(np.arange(d_bottom, d_top+1, 1))    
        # ax.set_yticks(np.arange(np.ceil(d_bottom), np.floor(d_top)+1, 1))    

        # #-- Define the x-tick marks we want, format the labels
        # Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=10, rotation=0,font=14)
        # ax.set_xlabel('Time [GMT] on {0}'.format(t_start.strftime('%Y-%m-%d')))

        # ax.set_title(f"template_{index}, ML={row['EQ_magnitude']}, distance to DAS={(row['STA_distance']*111):.1f} km\n SNR(RMS)={(np.sqrt(abs(row['peak_amp']))):.1f}, {config.filter_str}, {config.das_dist_start/1000:.1f} to {config.das_dist_end/1000:.1f} km", fontsize=14)
        # ax.invert_yaxis()
        # ax.grid(linestyle='--', linewidth=0.5)
        
        # png_fpath = os.path.join(config.OUTPUT_dir, f"{row['infile_name']}_ML{row['EQ_magnitude']}_dist{(row['STA_distance']*111):.1f}.png")
        # fig.savefig(png_fpath, dpi=300)
        # plt.close(fig)
    #? 保存结果
    Temp_lib_df.to_csv(config.IN_Template_library_path, index=False)

    #!==============画一张总图========
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")