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

# ==================================================================
# === 改动开始: 导入新的群组相似度计算函数 ===
# ==================================================================
# 假设 group_similarity.py 文件与此脚本在同一目录下
from group_similarity import (
    compute_moving_similarity_metrics,
    calculate_top_outlier_masscenter_distance,
    calculate_loneliest_neighborhood_similarity,
    calculate_pairwise_similarity_quantile,
    calculate_average_pairwise_similarity
)
# ==================================================================
# === 改动结束 ===
# ==================================================================


#!__________________________config_____________________________

class Config:
    def __init__(self):
        #?+++++++++++++++++++++ DAS waterfall 范围 ++++++++++++++++++++++++++++++++++++++++++++++
        self.pad_pre_sec, self.pad_post_sec = 10, 15 # seconds, for the long time CCs calculation
        #?++++++++++++++++trigger 关键 parameters+++++++++++++++++++++++++++++
        self.Threshold_level = 20 # 触发阈值，单位为该模板的MAD值
        self.Minimal_event_interval = 3 # in seconds
        
        self.max_stalta_threshold = 10
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        #     print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?++++++++++++++++++ 筛选 max_stalta 和 CCs_s_trig 进入 CCs 的分析的范围+++++++++++++++++++++++++++++++++++++++++++++++++
        self.kurtosis_range =  [1, 10]
        self.C_trig_range = [0.1, 1]
        
        self.Search_win = [1.5, 1.5]  # seconds, [left, right] around t_trig
        
        # ==================================================================
        # === 改动开始: 为群组相似度计算添加新参数 ===
        # ==================================================================
        #?++++++++++++++++++ Group Similarity paras ++++++++++++++++++++++++++++++++++++++++++++++++++
        # Ztdt 指的是群组相似度指标
        self.Z_win = [0.15, 0.15]  # seconds, [left, right] for the moving window
        self.Z_p_ratio = 0.5     # p-ratio for top outlier and loneliest group methods
        self.Z_q_ratio = 0.5     # q-ratio for the quantile method
        self.zlcc_threshold = 0.0  # 仅为绘图示例，可调整
        
        self.Z_str = f"Z{self.Z_win[0]}_{self.Z_win[1]}_p{self.Z_p_ratio}_q{self.Z_q_ratio}"
        # ==================================================================
        # === 改动结束 ===
        # ==================================================================

        self.max_plots_per_template = 15 # 每个模板最多保存的图片数量
        
        self.dir_byPara = f"kur{self.kurtosis_range[0]}_{self.kurtosis_range[1]}_C{self.C_trig_range[0]}_{self.C_trig_range[1]}_win{self.Search_win[0]}_{self.Search_win[1]}_{self.Z_str}/"

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
        self.OUTPUT_dir = f"{self.HOME_dir}Output_5_CCs_fk/2_CCs_ZLCC/"
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
    config._save_json(os.path.join(config.OUTPUT_dir_byPara, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir_byPara, 'config.json'))

    
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
        print(f"为模板 {disp_Template} 找到 {len(temp_group)} 个待处理事件。")
        
        PNG_count = 0
        for rr, row in temp_group.iterrows():
            if PNG_count >= config.max_plots_per_template:
                print(f"已达到每个模板 {config.max_plots_per_template} 张图片的保存上限，跳过后续处理。")
                break
        
            fs, f1, f2 = config.fs, config.freq_band[0], config.freq_band[1]
            dx, d1, d2 = config.dx, config.das_dist_start, config.das_dist_end
            
            temp = Read_template(row.Template, fs,f1,f2, dx,d1,d2, f'{config.HOME_dir}Output_0_Template_library/Template_npy_files/')
            
            t_zoom_left = row.t_trig - timedelta(seconds=config.Search_win[0])
            t_zoom_right = row.t_trig + timedelta(seconds=config.Search_win[1])
            
            #?++++++++++ 为计算群组相似度，读取更长的数据 ++++++++++++++++++++++++++++++++++++++++++++++
            # ==================================================================
            # === 改动开始: 扩展读取时间窗口以进行移动计算 ===
            # ==================================================================
            # 基础窗口是Search_win，用于计算CCs_f进行可视化
            # 扩展窗口是Z_win，用于计算移动相似度
            t_vis_start = t_zoom_left
            t_vis_end = t_zoom_right + timedelta(seconds=(temp.shape[0]-1)/fs)

            # ==================================================================
            # === 改动结束 ===
            # ==================================================================

            try:
                #! 首先读取可视化所需的短时程数据
                print(f"1.1 : 读取可视化DAS数据，时间范围: {t_vis_start} to {t_vis_end}")
                data_vis = Read_DAS_data(t_vis_start, t_vis_end, fs,f1,f2, dx,d1,d2)

                if data_vis is None or data_vis.size == 0:
                    print(f"  --> 警告: 可视化时间窗口内无数据，跳过此事件。")
                    continue

                print(f"1.2 : 计算可视化CCs矩阵 (CCs_f)")
                CCs_f, RAWCCs_f = CCs_comp(data_vis, temp)
                
                
                #! 读取计算移动相似度所需的更长数据
                t_calc_start = row.t_trig - timedelta(seconds=config.pad_pre_sec) - timedelta(seconds=config.Z_win[0])
                t_calc_end = row.t_trig + timedelta(seconds=config.pad_post_sec) +  timedelta(seconds=(temp.shape[0]-1)/fs) + timedelta(seconds=config.Z_win[1])
            
                if t_calc_start < config.t_start or t_calc_end > config.t_end:
                    print(f"  --> 警告: 事件 {row.t_trig} 的计算时间窗口超出数据范围，跳过。")
                    continue
                
                print(f"2.1 : 读取计算用DAS数据，时间范围: {t_calc_start} to {t_calc_end}")
                data_calc = Read_DAS_data(t_calc_start, t_calc_end, fs,f1,f2, dx,d1,d2)
                
                if data_calc is None or data_calc.size == 0:
                    print(f"  --> 警告: 计算时间窗口内无数据，跳过此事件。")
                    continue

                print(f"2.2 : 计算长时程CCs矩阵")
                CCs_f_calc, _ = CCs_comp(data_calc, temp)
                print(f"CCs_f_calc 的形状: {CCs_f_calc.shape},预计时间长度{t_calc_end- t_calc_start-timedelta(seconds=(temp.shape[0]-1)/fs)}")

                # ==================================================================
                # === 计算开始: 调用新的移动计算函数 ===
                # ==================================================================
                # 旧的占位符函数被替换为从 group_similarity.py 导入的函数
                Ztdt_results = compute_moving_similarity_metrics(CCs_f_calc, config)
                # ==================================================================
                # === 计算结束 ===
                # ==================================================================

                png_fname = f"{disp_Template}_{abs(float(row.CCs_s_trig)):.4f}_{float(row.CCs_s_trig_ofMAD):.1f}_{row.t_trig.strftime('%Y%m%d_%H%M%S_%f')}.png"
                png_path = os.path.join(config.OUTPUT_dir_byPara, png_fname)
                
                # 将计算结果和更长的起始时间传入绘图函数
                event_row_IN_fig_OUTput(row, CCs_f, Ztdt_results, config, png_path)
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

def event_row_IN_fig_OUTput(Event_row, CCs_f, Ztdt_results, config, png_path):
    
    # ------------------------------------------------------------------
    # 1. 准备所有绘图所需的数据
    # ------------------------------------------------------------------
    # ==================================================================
    # === 改动开始: 解包四个指标并创建对应的时间轴 ===
    # ==================================================================
    dist_series, lonely_series, quantile_series, avg_series = Ztdt_results
    
    # 计算移动窗口指标对应的时间轴
    win_before_pts = int(config.Z_win[0] * config.fs)
    win_after_pts = int(config.Z_win[1] * config.fs)
    # 时间点对应于每个窗口的中心
    
    # ==================================================================
    # === 改动结束 ===
    # ==================================================================

    IN_CCs_stack_dir = config.IN_CCs_stack_dir
    mad_value = config.MAD_Avg_dict.get(Event_row.Template)
    Threshold = config.Threshold_level * np.mean(mad_value) if mad_value is not None else 0.1
        
    t_trig = Event_row.t_trig.to_pydatetime()
    t_start_zoom = Event_row.t_trig - timedelta(seconds=config.Search_win[0])
    t_end_zoom = Event_row.t_trig + timedelta(seconds=config.Search_win[1])
    Rt_trig = Event_row.Rt_trig.to_pydatetime()
    CCs_s_trig = Event_row.CCs_s_trig
    Template = Event_row.Template

    pad_pre_sec, pad_post_sec = config.pad_pre_sec, config.pad_post_sec
    t_start_long = t_trig - timedelta(seconds=pad_pre_sec)
    t_end_long = t_trig + timedelta(seconds=pad_post_sec)
    
    fs, f1, f2, dx, d1, d2 = config.fs, config.freq_band[0], config.freq_band[1], config.dx, config.das_dist_start, config.das_dist_end

    try:
        data = Read_DAS_data(t_start_long, t_end_long, fs,f1,f2, dx,d1,d2)
        print(f'--绘图函数: 读取DAS数据从 {t_start_long} 到 {t_end_long}')
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
    # ==================================================================
    # === 改动开始: 调整布局以容纳所有图像 ===
    # ==================================================================
    left_ratios = [2, 2, 1, 1,    1, 1, 1, 1]  # 8行
    right_ratios = [2, 1,         1, 1, 1, 1]      # 6行
    
    fig_height = max(sum(left_ratios), sum(right_ratios)) * 1.8
    fig = plt.figure(figsize=(28, fig_height))
    fig.suptitle(f'Template: {Template}; Filter Band: {f1}-{f2} Hz; Corr: {float(CCs_s_trig):.4f}', fontsize=20)
    
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    # --- 左栏布局 (恢复原始方案) ---
    gs_left = gs_main[0, 0].subgridspec(len(left_ratios), 2, height_ratios=left_ratios, width_ratios=[7.3, 1],hspace=0.55)
    axes_left = [fig.add_subplot(gs_left[0, :]),       # span 2 cols
                 fig.add_subplot(gs_left[1, :]),       # span 2 cols
                 fig.add_subplot(gs_left[2, 0]),
                 fig.add_subplot(gs_left[3, 0]),
                 fig.add_subplot(gs_left[4, 0]),
                 fig.add_subplot(gs_left[5, 0]),
                 fig.add_subplot(gs_left[6, 0]),
                 fig.add_subplot(gs_left[7, 0]),
                ]

    # --- 右栏布局 (恢复原始方案, 并修正使其可运行) ---
    gs_right = gs_main[0, 1].subgridspec(len(right_ratios), 2, height_ratios=right_ratios, width_ratios=[7.3, 1],hspace=0.7)
    axes_right = [fig.add_subplot(gs_right[0, :]),       # span 2 cols
                  fig.add_subplot(gs_right[1, 0]),
                  fig.add_subplot(gs_right[2, 0]),
                  fig.add_subplot(gs_right[3, 0]),
                  fig.add_subplot(gs_right[4, 0]),
                  fig.add_subplot(gs_right[5, 0]),
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
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_long, t_end_long])

    # --- 左栏图 2: 原始数据波形 ---
    ax = axes_left[1]
    waterfall_date(ax, data, fs, t_start_long, dx, d1/1e3, unit, vlim=0)
    ax.set_title(f"Left [1]: Filtered DAS Data ({f1}-{f2} Hz)")
    ax.axvline(x=Rt_trig, color='r', linestyle='--', label='Expected EQ time'); ax.axvline(x=t_trig, color='k', linestyle='--')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_long, t_end_long])

    # --- 左栏图 3: 叠加CCs与包络线 (长时程) ---
    ax = axes_left[2]
    ax.set_title('Left [2]: Stacked CCs Trace (Long window)')
    ax.set_ylabel('CCs')
    ax.scatter(t_trig, CCs_s_trig, c='r', s=80, alpha=0.9, marker='*', zorder=10, label='Trigger point')
    ax.plot(t_s, CCs_s, color='k', alpha=0.7, linewidth=0.7)
    ax.axhline(y=Threshold, color='b', linestyle='--', alpha=0.5, label=f'Thresh={float(Threshold):.3f}')
    ax.axhline(y=-Threshold, color='b', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    rect = patches.Rectangle((Event_row.t_left, -abs(CCs_s_trig)), Event_row.t_right - Event_row.t_left, 2*abs(CCs_s_trig), color='r', alpha=0.1, label='Triggered area')
    ax.add_patch(rect)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_long, t_end_long])

    # --- 左栏图 4: STA/LTA 曲线 ---
    ax = axes_left[3]
    ax.set_title('Left [3]: Centered STA/LTA (unnormed)')
    ax.set_ylabel('STA/LTA')
    sl_threshold = 14
    ax.plot(t_s, sl_unnorm, color='k', alpha=0.7, linewidth=0.7)
    max_sl_unnorm = max(sl_unnorm) if len(sl_unnorm) > 0 else 0
    if max_sl_unnorm > 0:
      ax.text(t_trig, max_sl_unnorm, f"{Event_row['max_stalta']:.2f}", color='b', fontsize=12, ha='left', va='bottom')
    ax.axhline(y=sl_threshold, color='b', linestyle='--', alpha=0.5, label=f'Kurtosis Thresh={sl_threshold}')
    ax.legend(loc='upper right')
    rect = patches.Rectangle((t_trig-timedelta(seconds=0.5), 0), timedelta(seconds=0.5+0.2), max(sl_unnorm), color='b', alpha=0.1)
    ax.add_patch(rect)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_long, t_end_long])


    # --- 右栏图 1: CCs矩阵波形 ---
    ax = axes_right[0]
    ax.set_title('Right [0]: Time-domain CCs Matrix (CCs_f)')
    # CCs_f 的时间轴从 t_start_zoom 开始
    t_start_CCs_f = t_start_zoom
    waterfall_date(ax, CCs_f, fs, t_start_CCs_f, dx, d1/1e3, 'CCs', vlim=5)
    ax.axvline(x=t_trig, color='k', linestyle='--',alpha=0.3, label='Trigger Time')
    ax.legend(loc='upper right')
    ax.set_xticklabels([])  # 清除右栏图1的X轴标签
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_zoom, t_end_zoom])
    
    
    # --- 右栏图 2: 叠加CCs (短时程/放大) ---
    ax = axes_right[1]
    ax.set_title('Right [1]: Stacked CCs Trace (Zoomed window)')
    ax.set_ylabel('CCs')
    ax.plot(t_s, CCs_s, color='k', alpha=0.7, linewidth=0.7)
    ax.scatter(t_trig, CCs_s_trig, c='r', s=80, alpha=0.9, marker='*', zorder=10, label='Trigger point')
    ax.axhline(y=Threshold, color='b', linestyle='--', alpha=0.5, label=f'Thresh={float(Threshold):.3f}')
    ax.axhline(y=-Threshold, color='b', linestyle='--', alpha=0.5)
    ax.grid(linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlim([t_start_zoom, t_end_zoom])  # 使用短时程范围

    # ==================================================================
    # === 改动开始: 循环绘制四个指标的序列图 ===
    # ==================================================================
    # 定义绘图所需的数据和元数据
    t_metrics = t_s
    metrics_data = [dist_series, lonely_series, quantile_series, avg_series]
    titles = [
        f"Top {config.Z_p_ratio*100:.0f}% Outlier Sim",
        f"Loneliest {config.Z_p_ratio*100:.0f}% Sim",
        f"Pairwise Sim {config.Z_q_ratio*100:.0f}-percentile",
        "Average Pairwise Sim (Z)"
    ]
    ylabels = ["Similarity", "Similarity", "Similarity", "Similarity"]
    left_axes_to_plot = axes_left[4:]
    right_axes_to_plot = axes_right[2:]
    
    for i in range(4):
        #* --- 绘制左栏长时程图 ---
        ax = left_axes_to_plot[i]
        ax.set_title(f'Left [{4+i}]: {titles[i]}')
        ax.plot(t_metrics, metrics_data[i], color='k', alpha=0.8, linewidth=0.8)
        ax.set_ylabel(ylabels[i])
        ax.set_xlim([t_start_long, t_end_long])
        ax.grid(linestyle='--', alpha=0.6)
        
        ax.axvline(x=t_trig, color='k', linestyle=':')
        
        #* 标注此时的值
        nearest_idx = np.argmin(np.abs(t_metrics - t_trig))
        nearest_val = metrics_data[i][nearest_idx]
        ax.annotate(f'{nearest_val:.2f}', 
                    xy=(t_trig, nearest_val), 
                    color='b', fontsize=14, ha='left', va='bottom')
        
        # 为平均相似度添加阈值线
        # if i == 3:
        #     ax.axhline(y=config.zlcc_threshold, color='r', linestyle='--', alpha=0.5, label=f'Thresh={config.zlcc_threshold}')
        #     ax.legend(loc='upper right')

        #* --- 绘制右栏短时程图 ---
        ax = right_axes_to_plot[i]
        ax.set_title(f'Right [{2+i}]: {titles[i]} (Zoomed)')
        ax.plot(t_metrics, metrics_data[i], color='k', alpha=0.8, linewidth=0.8)
        ax.set_ylabel(ylabels[i])
        ax.set_xlim([t_start_zoom, t_end_zoom]) # 使用短时程范围
        ax.grid(linestyle='--', alpha=0.6)
        
        ax.axvline(x=t_trig, color='k', linestyle=':')
        
        #* 标注此时的值
        nearest_idx = np.argmin(np.abs(t_metrics - t_trig))
        nearest_val = metrics_data[i][nearest_idx]
        ax.annotate(f'{nearest_val:.2f}', 
                    xy=(t_trig, nearest_val), 
                    color='b', fontsize=14, ha='left', va='bottom')
        # if i == 3:
        #     ax.axhline(y=config.zlcc_threshold, color='r', linestyle='--', alpha=0.5, label=f'Thresh={config.zlcc_threshold}')
        #     ax.legend(loc='upper right')

    # 为所有长、短时程图统一设置X轴格式
    all_axes = left_axes_to_plot + right_axes_to_plot
    for ax in all_axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        ax.tick_params(axis='x', rotation=15)

    # 只在最底部的图显示X轴标签
    axes_left[-1].set_xlabel('Time')
    axes_right[-1].set_xlabel('Time')
    # ==================================================================
    # === 改动结束 ===
    # ==================================================================

    # ------------------------------------------------------------------
    # 4. 最终调整
    # ------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(png_path)
    plt.close(fig)
    print(f'-----> 已保存图像: {png_path}')
    return


if __name__ == "__main__":
    main()
    print('Done!')