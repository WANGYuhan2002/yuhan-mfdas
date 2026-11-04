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
# jupytext:
#   formats: ipynb,py:light
#   text_representation:
#     extension: .py
#     format_name: light
#     format_version: '1.5'
#     jupytext_version: 1.16.4
#   kernelspec:
#     display_name: salvus
#     language: python
#     name: python3
# ---

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
from matplotlib.ticker import FuncFormatter, FixedLocator



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
        self.max_stalta_threshold = 10
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        # print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?++++++++++++++++++ 筛选 max_stalta 和 CCs_s_trig 进入 CCs 的fk分析的范围+++++++++++++++++++++++++++++++++++++++++++++++++
        self.kurtosis_range = [1, 10]
        self.C_trig_range = [0.03, 0.1]
        
        self.kurtosis_range = [20, 1000]
        self.C_trig_range = [0.03, 1]
        
        
        self.Search_win = [1.5, 1.5] # seconds, [left, right] around t_trig
        
        # === 新增参数: 定义STA/LTA峰值搜索的可视化窗口 ===
        self.kurtosis_win = [0.5, 0.2] # seconds, [left, right] around t_trig
        
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
        self.MAD_Avg_dict = load_pkl(self.MAD_Avg_path) # load the MAD_array_dict
        self.subdir = f"level_{self.Threshold_level}/"
        self.IN_CCs_trig_dir = f"{self.HOME_dir}Output_2_Triggered_SCCs_points/{self.subdir}"

        #* 输出路径
        self.OUTPUT_dir = f"{self.HOME_dir}Output_5_CCs_fk/3_CCs_kurtosis_example_for_paper/"
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
        
        
        Temp_lib_df = pd.read_csv(self.temp_lib_csv,sep=',')
        check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(self.Temp_name_list)]
        self.check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)

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

    check_lib =config.check_lib
    
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
                    print(f" --> 警告: 对于此时间窗口没有返回数据，跳过此事件。")
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
                print(f" --> 错误: 在处理位于 {row.t_trig} 的事件时捕获到 UnboundLocalError。")
                print(f" --> 这很可能是因为缺少所请求时间窗口的DAS数据文件。")
                print(f" --> 正在跳过此事件。错误详情: {str(e)}")
                continue
            # except Exception as e:
            #     print(f" --> 在处理位于 {row.t_trig} 的事件时发生意外错误: {str(e)}")
            #     print(f" --> 正在跳过此事件。")
            #     continue

def event_row_IN_fig_OUTput(Event_row, CCs_f, fk_results, config, png_path):
    """
    一个经过优化的绘图函数，用于生成和保存事件的详细分析图。
    V2 更新:
    1. 恢复了在长时程CCs图上标记触发区间的红色矩形。
    2. 调整了STA/LTA文本标签的位置，使其更清晰。
    3. 新增在STA/LTA图上高亮特定分析窗口（绿色矩形）并标记该窗口内的峰值（绿色星号）。
    """
    # ------------------------------------------------------------------
    # 0. 绘图样式配置 (方便统一修改)
    # ------------------------------------------------------------------
    ftz = 20  # 字体大小
    plt.rcParams.update({
        'font.size': ftz,
        'axes.labelsize': ftz,
        'xtick.labelsize': ftz,
        'ytick.labelsize': ftz,
        'legend.fontsize': ftz-2,
        'figure.titlesize': ftz + 2,
        'axes.titlesize': ftz + 1,    
    })
    p_style = {
        # Font sizes
        "suptitle_fs": 26, "title_fs": 23, "label_fs": 20, "tick_fs": 17, "legend_fs": 15, "text_fs": 15,
        # Colors
        "data_color": "black", "trigger_color": "red", "threshold_color": "blue",
        "stalta_color": "darkgreen", "text_color": "blue",
        "stalta_max_color": "green", "stalta_rect_color": "green",
        # Lines and Markers
        "line_width": 1.0,  # <-- MODIFIED: 数据线宽
        "annot_line_width": 2.0, # <-- NEW: 标注线宽
        "line_alpha": 0.8,
        "trigger_marker": '*', "trigger_marker_size": 150, "trigger_marker_alpha": 0.9,
        "stalta_max_marker_size": 180,
        "threshold_style": "--", "trigger_line_style": ":", "expected_time_style": "--",
        # Patches
        "rect_alpha": 0.1,
    }

    # ------------------------------------------------------------------
    # 1. 准备所有绘图所需的数据 (优化数据读取)
    # ------------------------------------------------------------------
    # --- 基本参数 ---
    fs, f1, f2 = config.fs, config.freq_band[0], config.freq_band[1]
    dx, d1, d2 = config.dx, config.das_dist_start, config.das_dist_end
    Template = Event_row.Template
    disp_Template = config.check_lib[config.check_lib['infile_name'] == Template]['plot_name'].values[0]
    t_trig = Event_row.t_trig.to_pydatetime()
    Rt_trig = Event_row.Rt_trig.to_pydatetime()
    CCs_s_trig = Event_row.CCs_s_trig

    # --- 计算绘图所需的最大时间窗口，用于STA/LTA计算 ---
    sta_len_sec, lta_len_sec = 0.5, 1.5 # STA/LTA 时间长度
    pad_pre_sec, pad_post_sec = 7, 15  # 长时程图的padding
    
    t_start_long = t_trig - timedelta(seconds=pad_pre_sec)
    t_end_long = t_trig + timedelta(seconds=pad_post_sec)
    
    t_start_padded = t_start_long - timedelta(seconds=lta_len_sec)
    t_end_padded = t_end_long + timedelta(seconds=lta_len_sec)

    # --- 一次性读取所需数据 ---
    try:
        print(f'-- Reading padded CCs from {t_start_padded} to {t_end_padded} for STA/LTA')
        CCs_s_padded, _ = Read_CCs_stack_h5(config.IN_CCs_stack_dir, Template, t_start_padded, t_end_padded)
        print(f'-- Reading DAS data from {t_start_long} to {t_end_long}')
        data = Read_DAS_data(t_start_long, t_end_long, fs, f1, f2, dx, d1, d2)
    except Exception as e:
        print(f" --> 错误: 绘图函数读取数据失败: {e}。跳过此事件的绘图。")
        return

    temp = Read_template(Template, fs,f1,f2, dx,d1,d2,f'{config.HOME_dir}Output_0_Template_library/Template_npy_files/')

    # --- 从已读数据中计算和切片 ---
    sta_win, lta_win = int(sta_len_sec * fs), int(lta_len_sec * fs)
    sl_unnorm_padded = sta_lta_centred_unnorm(CCs_s_padded, sta_win, lta_win)
    t_padded = np.array([t_start_padded + timedelta(seconds=i/fs) for i in range(len(CCs_s_padded))])
    
    long_mask = (t_padded >= t_start_long) & (t_padded < t_end_long)
    t_s = t_padded[long_mask]
    CCs_s = CCs_s_padded[long_mask]
    sl_unnorm = sl_unnorm_padded[long_mask]

    # --- 其他绘图参数 ---
    mad_value = config.MAD_Avg_dict.get(Template, 0.005)
    if isinstance(mad_value, np.ndarray): mad_value = np.mean(mad_value)
    Threshold = config.Threshold_level * mad_value
    
    t_start_zoom = t_trig - timedelta(seconds=config.Search_win[0])
    t_end_zoom = t_trig + timedelta(seconds=config.Search_win[1])
    unit = 'Strain rate (m/m)/s'
    
    # --- 新增: 计算kurtosis_win内的峰值 ---
    kurt_win_start = t_trig - timedelta(seconds=config.kurtosis_win[0])
    kurt_win_end = t_trig + timedelta(seconds=config.kurtosis_win[1])
    kurt_mask = (t_s >= kurt_win_start) & (t_s <= kurt_win_end)
    if np.any(kurt_mask):
        local_max_sl_val = np.max(sl_unnorm[kurt_mask])
        local_max_sl_time = t_s[kurt_mask][np.argmax(sl_unnorm[kurt_mask])]
    else: # 如果窗口内没有数据点，则使用全局的触发点
        local_max_sl_val = Event_row['max_stalta']
        local_max_sl_time = t_trig

    # ------------------------------------------------------------------
    # 2. 定义双栏布局 (与原代码保持一致)
    # ------------------------------------------------------------------
    left_ratios = [1.7, 1.7, 1.5, 1]; right_ratios = [3, 1.5, 1]
    fig_height = max(sum(left_ratios), sum(right_ratios)) * 2.2
    fig = plt.figure(figsize=(24, fig_height))
    fig.suptitle(f'Template: {Template}; Filter Band: {f1}-{f2} Hz; Corr: {float(CCs_s_trig):.4f}', fontsize=p_style["suptitle_fs"])
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)
    gs_left = gs_main[0, 0].subgridspec(4, 2, height_ratios=left_ratios, width_ratios=[7, 1], hspace=0.5)
    axes_left = [fig.add_subplot(gs_left[0, :]), fig.add_subplot(gs_left[1, :]), fig.add_subplot(gs_left[2, 0]), fig.add_subplot(gs_left[3, 0])]
    gs_right = gs_main[0, 1].subgridspec(3, 2, height_ratios=right_ratios, width_ratios=[7, 1], hspace=0.5)
    axes_right = [fig.add_subplot(gs_right[0, :]), fig.add_subplot(gs_right[1, 0]), fig.add_subplot(gs_right[2, 0])]

    # ------------------------------------------------------------------
    # 3. 模块化绘图 (使用 p_style 配置样式)
    # ------------------------------------------------------------------

    # --- 左栏图 1: 模板波形 ---
    ax = axes_left[0]
    pad_pre_pts = int(pad_pre_sec * fs); pad_post_pts = len(data) - temp.shape[0] - pad_pre_pts
    temp_padded = np.pad(temp, ((pad_pre_pts, max(0, pad_post_pts)), (0, 0)), 'constant')[:len(data), :]
    waterfall_date(ax, temp_padded, fs, t_start_long, dx, d1/1e3, unit, vlim=0)
    ax.set_title(f'Left [1]: Template waveform ({disp_Template})', fontsize=p_style["title_fs"])
    ax.axvline(x=t_trig, color=p_style["data_color"], linestyle=p_style["trigger_line_style"], linewidth=p_style["annot_line_width"])
    ax.axvline(x=Rt_trig, color=p_style["trigger_color"], linestyle=p_style["trigger_line_style"], linewidth=p_style["annot_line_width"])

    # --- 左栏图 2: 原始数据波形 ---
    ax = axes_left[1]
    waterfall_date(ax, data, fs, t_start_long, dx, d1/1e3, unit, vlim=0)
    ax.set_title(f"Left [2]: Filtered DAS Data ({f1}-{f2} Hz)", fontsize=p_style["title_fs"])
    ax.axvline(x=Rt_trig, color=p_style["trigger_color"], linestyle=p_style["expected_time_style"], label='Expected EQ time', linewidth=p_style["annot_line_width"])
    ax.axvline(x=t_trig, color=p_style["data_color"], linestyle=p_style["trigger_line_style"], linewidth=p_style["annot_line_width"])
    ax.legend(loc='upper left', fontsize=p_style["legend_fs"], frameon=True)

    # --- 左栏图 3: 叠加CCs与包络线 (长时程) ---
    ax = axes_left[2]
    ax.set_title('Left [3]: SCC Time Series (Long window)', fontsize=p_style["title_fs"])
    ax.plot(t_s, CCs_s, color=p_style["data_color"], alpha=p_style["line_alpha"], linewidth=p_style["line_width"])
    ax.scatter(t_trig, CCs_s_trig, c=p_style["trigger_color"], s=p_style["trigger_marker_size"], marker=p_style["trigger_marker"], zorder=10, label='Trigger point', alpha=p_style["trigger_marker_alpha"])
    ax.axhline(y=Threshold, color=p_style["threshold_color"], linestyle=p_style["threshold_style"], label=f'Threshold={Threshold:.3f}', linewidth=p_style["annot_line_width"])
    ax.axhline(y=-Threshold, color=p_style["threshold_color"], linestyle=p_style["threshold_style"], linewidth=p_style["annot_line_width"])
    ax.set_ylabel('SCC', fontsize=p_style["label_fs"]); ax.set_xlim([t_start_long, t_end_long])
    rect_height = abs(CCs_s_trig)
    t0,t1 = Event_row['t_left'], Event_row['t_right']
    rect = patches.Rectangle((t0, -rect_height), (t1 - t0), 2*rect_height, color=p_style["trigger_color"], alpha=p_style["rect_alpha"], label='Triggered window', zorder=1)
    ax.add_patch(rect)
    ax.legend(loc='upper right', fontsize=p_style["legend_fs"], frameon=True)


    # --- 左栏图 4: STA/LTA 曲线 ---
    ax = axes_left[3]
    ax.set_title(f"Left [4]: Kurtosis (STA={sta_len_sec}s, LTA={lta_len_sec}s)", fontsize=p_style["title_fs"])
    ax.plot(t_s, sl_unnorm, color=p_style["stalta_color"], alpha=p_style["line_alpha"], linewidth=p_style["line_width"])
    text_time_offset = timedelta(seconds=0.2)
    # ax.text(t_trig + text_time_offset, Event_row['max_stalta'], f"Peak kurtosis: {Event_row['max_stalta']:.2f}", color=p_style["text_color"], fontsize=p_style["text_fs"], ha='left', va='bottom')
    ax.set_ylabel('Kurtosis', fontsize=p_style["label_fs"]); ax.set_xlabel('Time', fontsize=p_style["label_fs"]); ax.set_xlim([t_start_long, t_end_long])
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle((kurt_win_start, ymin), (kurt_win_end - kurt_win_start), ymax - ymin, color=p_style["stalta_rect_color"], alpha=p_style["rect_alpha"], zorder=1)
    ax.add_patch(rect)
    ax.scatter(local_max_sl_time, local_max_sl_val, c=p_style["stalta_max_color"], s=p_style["stalta_max_marker_size"], marker='*', zorder=11, label=f'Kurtosis of the trigger: {local_max_sl_val:.2f}')
    ax.axhline(y=config.max_stalta_threshold, color='C2', linestyle=p_style["threshold_style"], label=f'Threshold={config.max_stalta_threshold:.1f}', linewidth=p_style["annot_line_width"])
    ax.legend(loc='upper right', fontsize=p_style["legend_fs"], frameon=True)

    # --- 右栏图 1: CCs矩阵波形 ---
    ax = axes_right[0]
    ax.set_title('Right [1]: CC Matrix (each column is a CC of one channel)', fontsize=p_style["title_fs"])
    waterfall_date(ax, CCs_f, fs, t_start_zoom, dx, d1/1e3, 'Correlation Coefficients', vlim=5)
    ax.axvline(x=t_trig, color='white', linestyle=p_style["trigger_line_style"], alpha=0.9, label='Trigger Time', linewidth=p_style["annot_line_width"])
    ax.legend(loc='upper right', fontsize=p_style["legend_fs"], frameon=True)

    # --- 右栏图 2: 叠加CCs (短时程/放大) ---
    ax = axes_right[1]
    ax.set_title('Right [2]: SCC Time Series (Zoomed window)', fontsize=p_style["title_fs"])
    ax.plot(t_s, CCs_s, color=p_style["data_color"], alpha=p_style["line_alpha"], linewidth=p_style["line_width"])
    ax.scatter(t_trig, CCs_s_trig, c=p_style["trigger_color"], s=p_style["trigger_marker_size"], marker=p_style["trigger_marker"], zorder=10, label='Trigger point', alpha=p_style["trigger_marker_alpha"])
    ax.axhline(y=Threshold, color=p_style["threshold_color"], linestyle=p_style["threshold_style"], label=f'Threshold={Threshold:.3f}', linewidth=p_style["annot_line_width"])
    ax.axhline(y=-Threshold, color=p_style["threshold_color"], linestyle=p_style["threshold_style"], linewidth=p_style["annot_line_width"])
    ax.set_ylabel('SCC', fontsize=p_style["label_fs"]); ax.set_xlim(t_start_zoom, t_end_zoom)
    t0,t1 = Event_row['t_left'], Event_row['t_right']
    rect = patches.Rectangle((t0, -rect_height), (t1 - t0), 2*rect_height, color=p_style["trigger_color"], alpha=p_style["rect_alpha"], label='Triggered window', zorder=1)
    ax.add_patch(rect)
    ax.legend(loc='upper right', fontsize=p_style["legend_fs"], frameon=True)

    # --- 右栏图 3: STA/LTA (短时程/放大) ---
    ax = axes_right[2]
    ax.set_title('Right [3]: Kurtosis (Zoomed window)', fontsize=p_style["title_fs"])
    ax.plot(t_s, sl_unnorm, color=p_style["stalta_color"], alpha=p_style["line_alpha"], linewidth=p_style["line_width"])
    # ax.text(t_trig + text_time_offset, Event_row['max_stalta'], f"{Event_row['max_stalta']:.2f}", color=p_style["text_color"], fontsize=p_style["text_fs"], ha='left', va='center')
    ax.set_ylabel('Kurtosis', fontsize=p_style["label_fs"]); ax.set_xlabel('Time', fontsize=p_style["label_fs"]); ax.set_xlim(t_start_zoom, t_end_zoom)
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle((kurt_win_start, ymin), (kurt_win_end - kurt_win_start), ymax - ymin, color=p_style["stalta_rect_color"], alpha=p_style["rect_alpha"], zorder=1)
    ax.add_patch(rect)
    ax.scatter(local_max_sl_time, local_max_sl_val, c=p_style["stalta_max_color"], s=p_style["stalta_max_marker_size"], marker='*', zorder=11, label=f'Kurtosis of the trigger: {local_max_sl_val:.2f}')
    ax.axhline(y=config.max_stalta_threshold, color='C2', linestyle=p_style["threshold_style"], label=f'Threshold={config.max_stalta_threshold:.1f}', linewidth=p_style["annot_line_width"])
    ax.legend(loc='upper right', fontsize=p_style["legend_fs"], frameon=True)
    
    # ------------------------------------------------------------------
    # 4. 统一调整所有坐标轴样式 (MODIFIED SECTION)
    # ------------------------------------------------------------------
    def relative_time_formatter(x, pos):
        dt = mdates.num2date(x)
        dt = dt.replace(tzinfo=None)
        relative_seconds = (dt - t_trig).total_seconds()
        if abs(relative_seconds) < 1e-9: return '0.0'
        return f'{relative_seconds:.1f}'
    
    relative_formatter = FuncFormatter(relative_time_formatter)

    # 为左侧图表设置样式 (使用绝对时间)
    for ax in axes_left:
        ax.grid(linestyle='--', alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        ax.tick_params(axis='x', rotation=0, labelsize=p_style["tick_fs"])
        ax.tick_params(axis='y', labelsize=p_style["tick_fs"])
        if ax.get_ylabel(): ax.set_ylabel(ax.get_ylabel(), fontsize=p_style["label_fs"])
        if ax.get_xlabel(): ax.set_xlabel(ax.get_xlabel(), fontsize=p_style["label_fs"])
        
    # 为右侧图表设置样式 (使用相对时间+对称刻度)
    tick_interval_sec = 0.5 
    win_left_sec = -config.Search_win[0]
    win_right_sec = config.Search_win[1]
    
    # 生成刻度值
    ticks_positive = np.arange(0, win_right_sec + tick_interval_sec, tick_interval_sec)
    ticks_negative = np.arange(-tick_interval_sec, win_left_sec - tick_interval_sec, -tick_interval_sec)
    relative_ticks_sec = np.union1d(ticks_positive, ticks_negative)
    relative_ticks_sec = np.round(relative_ticks_sec, 2) # 避免浮点数精度问题
    
    # 转换为datetime对象用于定位器
    tick_datetimes = [t_trig + timedelta(seconds=s) for s in relative_ticks_sec]
    
    for ax in axes_right:
        ax.grid(linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(tick_datetimes)))
        ax.xaxis.set_major_formatter(relative_formatter)
        ax.tick_params(axis='x', rotation=0, labelsize=p_style["tick_fs"])
        ax.tick_params(axis='y', labelsize=p_style["tick_fs"])
        if ax.get_ylabel(): ax.set_ylabel(ax.get_ylabel(), fontsize=p_style["label_fs"])
        ax.set_xlabel("Time relative to trigger (s)", fontsize=p_style["label_fs"])
    
    # ------------------------------------------------------------------
    # 5. 最终调整并保存
    # ------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'-----> 已保存图像: {png_path}')
    return

if __name__ == "__main__":
    main()
    print('Done!')