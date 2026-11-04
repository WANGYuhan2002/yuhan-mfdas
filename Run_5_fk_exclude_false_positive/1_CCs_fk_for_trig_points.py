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
from readers import Read_CCs_stack_h5, Read_DAS_data, Read_template
from compute_correlation import CCs_comp
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, waterfall_date
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
from classify import Collapse_with_MAX_CCs, Read_Collapse_group_df,Remove_Temp_Offest_forSeries
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from stalta import sta_lta_centred,sta_lta_centred_unnorm

#!__________________________config_____________________________

class Config:
    def __init__(self):
        
        #?++++++++++++++++trigger 关键 parameters+++++++++++++++++++++++++++++
        self.Threshold_level = 20 # 触发阈值，单位为该模板的MAD值
        self.Minimal_event_interval = 3 # in seconds
        
        self.max_stalta_threshold = 14
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        #     print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?++++++++++++++++++ 筛选 max_stalta 和 CCs_s_trig 进入 CCs 的fk分析的范围+++++++++++++++++++++++++++++++++++++++++++++++++
        self.kurtosis_range =  [1, 12]
        self.C_trig_range = [0.1, 1]
        
        self.Search_win = [0.5, 0.5]  # seconds, [left, right] around t_trig
        
        self.max_plots_per_template = 5 # 每个模板最多保存的图片数量
        
        self.dir_byPara = f"kur{self.kurtosis_range[0]}_{self.kurtosis_range[1]}_C{self.C_trig_range[0]}_{self.C_trig_range[1]}_win{self.Search_win[0]}_{self.Search_win[1]}/"

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
        self.OUTPUT_dir = f"{self.HOME_dir}Output_5_CCs_fk/1_CCs_fk_spec/"
        self.OUTPUT_dir_byPara = os.path.join(self.OUTPUT_dir, self.dir_byPara)


        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286
        
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
    # 确保基础输出目录和参数子目录都存在
    os.makedirs(config.OUTPUT_dir_byPara, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir, 'config.json'))

    
    Temp_lib_df = pd.read_csv(config.temp_lib_csv,sep=',')
    check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(config.Temp_name_list)]
    check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)

    temp_lib_df = pd.read_csv(config.temp_lib_csv, sep=',')
    DF_Temp_Offset = pd.DataFrame({
        'template_name': temp_lib_df['infile_name'],
        'P_offset': temp_lib_df['t_P_sec'],
        'S_offset': temp_lib_df['t_S_sec']
    })
    print('====>DF_Temp_Offset:', DF_Temp_Offset)

    Collapse_group = Read_Collapse_group_df(config.IN_CCs_trig_dir, config.Temp_name_list, DF_Temp_Offset, config.MAD_Avg_dict, config.t_start, config.t_end)
    print('====>Collapse_group by time',Collapse_group)
    
    mask = Collapse_group['max_stalta'] > config.max_stalta_threshold
    good_Collapse_group = Collapse_group[mask]
    good_Collapse_group.reset_index(drop=True, inplace=True)
    print('====>Collapse_group after max_stalta threshold', len(good_Collapse_group))
    
    Events_classified = Collapse_with_MAX_CCs(good_Collapse_group, config.IN_one_event_interval)
    
    print(f'*********Event number before collapsed:', len(Events_classified))
    print(f'*********Event number after collapsed:', Events_classified['Belong_or_not'].sum())
    print(f'*********Event number after collapsed:', Events_classified['group'].max())

    
    for ii, row_check_lib in check_lib.iterrows():
        Template = row_check_lib['infile_name']
        disp_Template = row_check_lib['plot_name']
        print(f"\n\nProcessing Template: {Template} ({disp_Template})")
        
        mask = (Collapse_group['Template'] == Template)\
            & (Collapse_group['max_stalta'] > config.kurtosis_range[0])\
            & (Collapse_group['max_stalta'] < config.kurtosis_range[1])\
            & (Collapse_group['CCs_s_trig'].abs() > config.C_trig_range[0])\
            & (Collapse_group['CCs_s_trig'].abs() < config.C_trig_range[1])
            
        temp_group = Collapse_group[mask]
        print(f"has {len(temp_group)} events for {disp_Template}")
        
        PNG_count = 0
        for rr, row in temp_group.iterrows():
            if PNG_count >= config.max_plots_per_template:
                print(f"已达到每个模板 {config.max_plots_per_template} 张图片的保存上限，跳过后续处理。")
                break
        
            fs, f1, f2 = config.fs, config.freq_band[0], config.freq_band[1]
            dx, d1, d2 = config.dx, config.das_dist_start, config.das_dist_end
            
            temp = Read_template(row.Template, fs,f1,f2, dx,d1,d2, f'{config.HOME_dir}Output_0_Template_library/Template_npy_files/')
            
            t_left = row.t_trig - timedelta(seconds=config.Search_win[0])
            t_right = row.t_trig + timedelta(seconds=config.Search_win[1])
            t_right_extended = t_right + timedelta(seconds=(temp.shape[0]-1)/fs)
            
            if t_left < config.t_start or t_right_extended > config.t_end:
                continue
            
            try:
                print(f"read DAS data from {t_left} to {t_right_extended}")
                data = Read_DAS_data(t_left, t_right_extended, fs,f1,f2, dx,d1,d2)

                if data is None or data.size == 0:
                    print(f"  --> 警告: 对于此时间窗口没有返回数据，跳过此事件。")
                    continue

                print(f"computing CCs_f")
                CCs_f,RAWCCs_f = CCs_comp(data, temp)
                
                maxCCs_f = np.max(np.abs(CCs_f), axis=0)
                idx_maxCCs_f = np.argmax(np.abs(CCs_f), axis=0)
                
                from spectrum import Calculate_fk_amplitude_spectrum
                fk_results = Calculate_fk_amplitude_spectrum(CCs_f, 1/fs, dx)
                
                png_fname = f"{disp_Template}_{abs(float(row.CCs_s_trig)):.4f}_{float(row.CCs_s_trig_ofMAD):.1f}_{row.t_trig.strftime('%Y%m%d_%H%M%S_%f')}.png"
                
                png_path = os.path.join(config.OUTPUT_dir_byPara, png_fname)
                
                event_row_IN_fig_OUTput(row, CCs_f, fk_results, config, png_path)
                PNG_count += 1
            
            except UnboundLocalError as e:
                print(f"  --> 错误: 在处理位于 {row.t_trig} 的事件时捕获到 UnboundLocalError。")
                print(f"  --> 这很可能是因为缺少所请求时间窗口的DAS数据文件。")
                print(f"  --> 正在跳过此事件。错误详情: {str(e)}")
                continue 
            # except Exception as e:
            #     print(f"  --> 在处理位于 {row.t_trig} 的事件时发生意外错误: {str(e)}")
            #     print(f"  --> 正在跳过此事件。")
            #     continue

def event_row_IN_fig_OUTput(Event_row, CCs_f, fk_results, config, png_path):
    
    # ------------------------------------------------------------------
    # 1. 准备所有绘图所需的数据
    # ------------------------------------------------------------------
    freqs, wavenums, ampspec = fk_results
    IN_CCs_stack_dir = config.IN_CCs_stack_dir
    
    mad_value = config.MAD_Avg_dict.get(Event_row.Template)
    if mad_value is None:
        Threshold = 0.1
    else:
        if isinstance(mad_value, np.ndarray):
            mad_value = np.mean(mad_value)
        Threshold = config.Threshold_level * mad_value
        
    t_trig = Event_row.t_trig.to_pydatetime()
    t_left = Event_row.t_left.to_pydatetime()
    t_right = Event_row.t_right.to_pydatetime()
    Rt_trig = Event_row.Rt_trig.to_pydatetime()
    CCs_s_trig = Event_row.CCs_s_trig
    Template = Event_row.Template

    pad_pre_sec, pad_post_sec = 10, 15
    t_start_long = t_trig - timedelta(seconds=pad_pre_sec)
    t_end_long = t_trig + timedelta(seconds=pad_post_sec)
    
    t_start_zoom = Event_row.t_trig - timedelta(seconds=config.Search_win[0])
    t_end_zoom = Event_row.t_trig + timedelta(seconds=config.Search_win[1])
    
    fs, f1, f2, dx, d1, d2 = config.fs, config.freq_band[0], config.freq_band[1], config.dx, config.das_dist_start, config.das_dist_end

    try:
        data = Read_DAS_data(t_start_long, t_end_long, fs,f1,f2, dx,d1,d2)
        print('--read DAS data from', t_start_long,'to', t_end_long)
    except Exception as e:
        print(f"  --> 错误: 绘图函数读取DAS数据失败: {e}。")
        data = np.zeros((int((t_end_long-t_start_long).total_seconds() * fs), 1))
    
    temp = Read_template(Template, fs,f1,f2, dx,d1,d2,f'{config.HOME_dir}Output_0_Template_library/Template_npy_files/')
    CCs_s, _ = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start_long, t_end_long)
    t_s = np.array([t_start_long + timedelta(seconds=ii/fs) for ii in range(len(CCs_s))])
    
    sta_len, lta_len = 0.5, 1.5
    t_start_padded, t_end_padded = t_start_long - timedelta(seconds=lta_len), t_end_long + timedelta(seconds=sta_len)
    CCs_s_padded, _ = Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start_padded, t_end_padded)
    sl_unnorm = sta_lta_centred_unnorm(CCs_s_padded, int(sta_len * fs), int(lta_len * fs))
    sl_unnorm = sl_unnorm[int(lta_len * fs) : int(lta_len * fs) + len(CCs_s)]
    
    unit = 'Strain rate (nm/m)/s'

    # ------------------------------------------------------------------
    # 2. 定义双栏布局并使用 sharex 实现完美对齐
    # ------------------------------------------------------------------
    left_ratios = [2, 2, 1, 1]
    right_ratios = [2, 1, 2]
    fig_height = max(sum(left_ratios), sum(right_ratios)) * 2.2
    
    fig = plt.figure(figsize=(28, fig_height))
    fig.suptitle(f'Template: {Template}; Filter Band: {f1}-{f2} Hz; Corr: {float(CCs_s_trig):.4f}', fontsize=20)
    
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    # --- 左栏布局 ---
    gs_left = gs_main[0, 0].subgridspec(4, 2, height_ratios=left_ratios, width_ratios=[7, 1], hspace=0.5)
    axes_left = [fig.add_subplot(gs_left[0, :]),
                 fig.add_subplot(gs_left[1, :]),
                 fig.add_subplot(gs_left[2, 0]),
                 fig.add_subplot(gs_left[3, 0])
                ]

    # --- 右栏布局 (为 colorbar 创建专用列) ---
    gs_right = gs_main[0, 1].subgridspec(3, 2, height_ratios=right_ratios, width_ratios=[7, 1], hspace=0.5)
    axes_right = [  fig.add_subplot(gs_right[0, :]),
                    fig.add_subplot(gs_right[1, 0]),
                    fig.add_subplot(gs_right[2, :])
                ]
                     

    # ==================================================================
    # === 改动结束 ===
    # ==================================================================

    # ------------------------------------------------------------------
    # 3. 模块化绘图
    # ------------------------------------------------------------------

    # --- 左栏图 1: 模板波形 ---
    ax = axes_left[0]
    pad_pre = int(pad_pre_sec * fs)
    
    if pad_post_sec * fs > temp.shape[0]:
        pad_post = int(pad_post_sec * fs ) - temp.shape[0]
        temp_padded = np.pad(temp, ((pad_pre, pad_post), (0,0)), 'constant')
    else: 
        print(f"警告: pad_post_sec ({pad_post_sec}) * fs ({fs}) = {pad_post_sec * fs} 小于模板长度 {temp.shape[0]}，模版无法完全展示。")
        temp_padded = np.pad(temp, ((pad_pre, 0), (0,0)), 'constant')[:data.shape[0]]
        temp_padded = temp_padded[:data.shape[0], :]
    
    waterfall_date(ax, temp_padded, fs, t_start_long, dx, d1/1e3, unit, vlim=0)
    ax.set_title(f'Left [0]: Template waveform ({Template})')
    ax.axvline(x=t_trig, color='k', linestyle=':'); ax.axvline(x=Rt_trig, color='r', linestyle=':')
    ax.grid(linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=15, labelsize=10)

    # --- 左栏图 2: 原始数据波形 ---
    ax = axes_left[1]
    waterfall_date(ax, data, fs, t_start_long, dx, d1/1e3, unit, vlim=0)
    ax.set_title(f"Left [1]: Filtered DAS Data ({f1}-{f2} Hz)")
    ax.axvline(x=Rt_trig, color='r', linestyle='--', label='Expected EQ time'); ax.axvline(x=t_trig, color='k', linestyle='--')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=15, labelsize=10)

    # --- 左栏图 3: 叠加CCs与包络线 (长时程) ---
    ax = axes_left[2]
    ax.set_title('Left [2]: Stacked CCs Trace (Long window)')
    ax.set_ylabel('CCs')
    ax.scatter(t_trig, CCs_s_trig, c='r', s=80, alpha=0.9, marker='*', zorder=10, label='Trigger point')
    ax.plot(t_s, CCs_s, color='k', alpha=0.7, linewidth=0.7)
    ax.axhline(y=Threshold, color='b', linestyle='--', alpha=0.5, label=f'Thresh={float(Threshold):.3f}')
    ax.axhline(y=-Threshold, color='b', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    ax.set_xlim([t_start_long, t_end_long])
    rect = patches.Rectangle((t_left, -abs(CCs_s_trig)), t_right - t_left, 2*abs(CCs_s_trig), color='r', alpha=0.1, label='Triggered area')
    ax.add_patch(rect)

    # --- 左栏图 4: STA/LTA 曲线 ---
    ax = axes_left[3]
    ax.set_title('Left [3]: Centered STA/LTA (unnormed)')
    ax.set_ylabel('STA/LTA')
    ax.set_xlabel('Time') # X轴标题只在最下方的图显示
    sl_threshold = 14
    ax.plot(t_s, sl_unnorm, color='k', alpha=0.7, linewidth=0.7)
    max_sl_unnorm = max(sl_unnorm) if len(sl_unnorm) > 0 else 0
    if max_sl_unnorm > 0:
      ax.text(t_trig, max_sl_unnorm, f"{Event_row['max_stalta']:.2f}", color='b', fontsize=12, ha='left', va='bottom')
    ax.axhline(y=sl_threshold, color='b', linestyle='--', alpha=0.5, label=f'Kurtosis Thresh={sl_threshold}')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    ax.set_xlim([t_start_long, t_end_long])
    rect = patches.Rectangle((t_trig-timedelta(seconds=0.5), 0), timedelta(seconds=0.5+0.2), max(sl_unnorm), color='b', alpha=0.1)
    ax.add_patch(rect)


    # --- 右栏图 1: CCs矩阵波形 ---
    ax = axes_right[0]
    ax.set_title('Right [0]: Time-domain CCs Matrix (CCs_f)')
    waterfall_date(ax, CCs_f, fs, t_start_zoom, dx, d1/1e3, 'CCs', vlim=5)
    ax.axvline(x=t_trig, color='k', linestyle='--',alpha=0.3, label='Trigger Time')
    ax.legend(loc='upper right')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    
    
    # --- 右栏图 2: 叠加CCs (短时程/放大) ---
    ax = axes_right[1]
    ax.set_title('Right [1]: Stacked CCs Trace (Zoomed window)')
    ax.set_ylabel('CCs')
    ax.plot(t_s, CCs_s, color='k', alpha=0.7, linewidth=0.7)
    ax.set_xlim(t_start_zoom, t_end_zoom)
    ax.scatter(t_trig, CCs_s_trig, c='r', s=80, alpha=0.9, marker='*', zorder=10, label='Trigger point')
    ax.axhline(y=Threshold, color='b', linestyle='--', alpha=0.5, label=f'Thresh={float(Threshold):.3f}')
    ax.axhline(y=-Threshold, color='b', linestyle='--', alpha=0.5)
    ax.grid(linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    
    
    # #? --- 右栏图 3: f-k 谱 ---      choice 1       ---
    # ax = axes_right[2]
    # ax.set_title('Right [2]: 2D Amp. Spectrum (f-k Domain)')
    # ax.set_xlabel('Wavenumber (1/m)')
    # ax.set_ylabel('Frequency (Hz)')

    # # 计算 vmin 和 vmax
    # vmin_val, vmax_val = np.percentile(ampspec, 40), np.percentile(ampspec, 95)
    # safe_vmin = max(vmin_val, 1e-10)
    # print(f"vmin: {vmin_val} or {safe_vmin}; vmax: {vmax_val}")

    # # 使用修正后的 safe_vmin 进行绘图
    # from matplotlib import colors as mcolors
    # im = ax.pcolormesh(wavenums, freqs, ampspec, cmap='viridis', 
    #                 norm=mcolors.LogNorm(vmin=safe_vmin, vmax=vmax_val))
    # plt.colorbar(im,label='strain rate (m/m)/s',ax=ax)

    # # [次要改动] 稍微调整 set_ylim 的写法，让代码更稳健
    # # ax.set_ylim(bottom=0) # 这样设置可以避免一些不必要的警告
    # ax.axvline(0, color='white', linestyle=':', linewidth=0.5)
    # ax.set_ylim([config.freq_band[0], config.freq_band[1]])
    
    
    
    # #? --- 右栏图 3: f-k 谱 ---      choice 2     ---
    ax = axes_right[2]
    ax.set_title('Right [2]: 2D Amp. Spectrum (f-s Domain)') # <-- 修改: 标题更新
    ax.set_xlabel('Slowness (s/m)') # <-- 修改: x轴标签更新
    ax.set_ylabel('Frequency (Hz)')

    # 计算 vmin 和 vmax
    vmin_val, vmax_val = np.percentile(ampspec, 40), np.percentile(ampspec, 95)
    safe_vmin = max(vmin_val, 1e-10)
    print(f"vmin: {vmin_val} or {safe_vmin}; vmax: {vmax_val}")

    # 去掉 f=0 的行，避免 inf/nan 报错

    # 使用修正后的 safe_vmin 进行绘图
    from matplotlib import colors as mcolors
    im = ax.pcolormesh(wavenums, freqs, ampspec, cmap='viridis', 
                    norm=mcolors.LogNorm(vmin=safe_vmin, vmax=vmax_val),
                    shading='auto') # <-- 修改: 建议为非均匀网格添加 shading='auto'
    plt.colorbar(im,label='strain rate (m/m)/s',ax=ax)

    # [次要改动] 稍微调整 set_ylim 的写法，让代码更稳健
    # ax.set_ylim(bottom=0) # 这样设置可以避免一些不必要的警告
    ax.axvline(0, color='white', linestyle=':', linewidth=0.5)
    ax.set_ylim([config.freq_band[0], config.freq_band[1]])
    
    
    
    
    
    
    
    
    

    # ------------------------------------------------------------------
    # 4. 最终调整
    # ------------------------------------------------------------------
    # 由于 sharex 会自动隐藏非底部图的x-tick-labels，我们不再需要手动设置所有xlabel
    # 只需要为时间轴设置统一的刻度格式

    # 自动调整布局，为标题留出空间
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path)
    plt.close(fig)
    print(f'-----> 已保存图像: {png_path}')
    return


if __name__ == "__main__":
    main()
    print('Done!')