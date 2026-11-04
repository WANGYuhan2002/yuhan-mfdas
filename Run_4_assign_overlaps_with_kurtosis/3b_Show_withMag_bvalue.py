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
from scipy.stats import linregress

import geopandas
from matplotlib.lines import Line2D
import geodatasets
import contextily as cx

HOME_dir = f'{PROJECT_ROOT}/'
#!__________________________functions_____________________________
sys.path.append(f"{HOME_dir}functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, generate_gs_grid
from trigger import Trigger_with_MADarray
from readers import Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

#!______________specific func for this script____________________
def calculate_mc_maxc(magnitudes, bin_width=0.2):
    """
    使用最大值曲率法（MAXC）计算完整性震级Mc。
    参数:
        magnitudes (list/array): 震级列表
        bin_width (float): 直方图的分箱宽度，默认0.1
    返回:
        mc (float): 估计的完整性震级
    """
    if len(magnitudes) == 0:
        return np.nan
    min_m = np.floor(np.min(magnitudes) / bin_width * bin_width)
    max_m = np.ceil(np.max(magnitudes) / bin_width * bin_width)
    bins = np.arange(min_m, max_m + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    max_count_idx = np.argmax(counts)
    mc = edges[max_count_idx]
    return mc


def calculate_b_value_MLE(magnitudes, mc=None, minimum_events=20):
    """
    使用最大似然估计计算b值。
    参数:
        magnitudes (list/array): 震级列表
        mc (float): 完整性震级，若为None则自动计算
    返回:
        b_value (float): 估计的b值
    """
    if len(magnitudes) < minimum_events:
        # print('insufficient data points for b-value calculation')
        return np.nan  # 至少需要2个事件
    
    # 自动计算Mc（如果未提供）
    if mc is None:
        mc = calculate_mc_maxc(magnitudes)
        if np.isnan(mc):
            return np.nan
    
    # 筛选≥Mc的震级
    mags_above_mc = [m for m in magnitudes if m >= mc]
    if len(mags_above_mc) < minimum_events:
        # print('insufficient data points again for b-value calculation')
        return np.nan
    
    mean_m = np.mean(mags_above_mc)
    b_value = np.log10(np.e) / (mean_m - mc)
    return b_value

def calculate_b_value_LS(magnitudes, mc=None, bin_width=0.2, minimum_events=10):
    """
    使用最小二乘法估计b值
    
    参数:
        magnitudes (array): 地震震级列表
        mc (float): 完整性震级，若为None则使用数据最小值
        bin_width (float): 震级分箱宽度
    
    返回:
        result (dict): 包含键值对：
            'b_value' - 估计的b值
            'a_value' - a值（log10(N) = a - b*M）
            'r_squared' - 拟合优度R²
            'mc_used' - 实际使用的Mc
    """
    # 数据预处理
    mags = np.array(magnitudes)
    if len(mags) < minimum_events:
        # print('insufficient data points for b-value calculation')
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])
    
    # 确定实际使用的Mc
    mc_used = mc if mc is not None else np.min(mags)
    mc_used = max(mc_used, np.min(mags))  # 确保不低与实际最小值
    
    # 筛选有效数据
    valid_mags = mags[mags >= mc_used]
    if len(valid_mags) < minimum_events:
        # print('insufficient data points again for b-value calculation')
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])
    
    # 创建分箱
    bins = np.arange(np.floor(mc_used/bin_width)*bin_width,
                    np.ceil(np.max(valid_mags)/bin_width)*bin_width + bin_width,
                    bin_width)
    
    # 计算累积频度
    counts, _ = np.histogram(valid_mags, bins=bins)
    cum_counts = np.cumsum(counts[::-1])[::-1]  # 从大到小累积
    
    # 准备回归数据
    magnitudes_for_fit = bins[:-1]
    log_counts = np.log10(cum_counts)
    
    # 过滤无效数据点（count=0的情况）
    valid_idx = (cum_counts > 0) & (~np.isinf(log_counts))
    if np.sum(valid_idx) < 2:
        print('insufficient magnitude points and range for b-value calculation')
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])
    
    x = magnitudes_for_fit[valid_idx]
    y = log_counts[valid_idx]
    
    # 执行线性回归
    slope, intercept, r_value, _, _ = linregress(x, y)
    
    print(abs(slope), float(intercept),  float(r_value**2), float(mc_used), len(x.astype(float)), len(y.astype(float)))
    return  abs(slope), float(intercept),  float(r_value**2), float(mc_used), x.astype(float), y.astype(float)


def compute_b_value_time_series(b_method, bin_width, minimum_events, times, magnitudes, window_hours, step_hours, mc):
    """
    计算随时间变化的b值序列（小时单位版本）
    
    参数:
        will not use bin_width if use MLE method
        
        times (list): 地震时间列表（datetime对象）
        magnitudes (list): 对应的震级列表
        window_hours (int): 时间窗口长度（小时），默认720小时（30天）
        step_hours (int): 滑动步长（小时），默认24小时（1天）
        mc (float): 固定Mc，若为None则每个窗口自动计算
    
    返回:
        window_centers (list): 窗口中心时间列表
        b_values (list): 对应的b值列表
    """
    # 转换为DataFrame并按时间排序
    df = pd.DataFrame({'time': times, 'magnitude': magnitudes})
    df.sort_values('time', inplace=True)
    times_sorted = df['time'].to_numpy()
    magnitudes_sorted = df['magnitude'].to_numpy()
    
    window_centers = []
    window_tleft_array = []
    window_tright_array = []
    b_values = []
    current_time = df['time'].iloc[0]  # 起始时间
    end_time = df['time'].iloc[-1]    # 结束时间
    
    # 定义时间间隔对象
    window = timedelta(hours=window_hours)
    step = timedelta(hours=step_hours)
    
    # 滑动窗口遍历
    while current_time + window <= end_time:
        window_start = current_time
        window_end = current_time + window
        
        # 筛选窗口内的事件（闭区间[)）
        mask = (times_sorted >= window_start) & (times_sorted < window_end)
        mags_window = magnitudes_sorted[mask]
        
        # 计算b值
        if b_method == 'MLE':  # 'MLE' or 'LS'
            b = calculate_b_value_MLE(mags_window, mc, minimum_events)
        elif b_method == 'LS':
            b,_,_,_,_,_ = calculate_b_value_LS(mags_window, mc, bin_width, minimum_events)
        
        # 保存结果（窗口中心时间）
        window_centers.append(window_start + window/2)
        window_tleft_array.append(window_start)
        window_tright_array.append(window_end)
        b_values.append(b)
        
        # 滑动到下一个窗口
        current_time += step
    
    return window_tright_array, b_values

def _average_MAD_array_dict(MAD_array_dict, Temp_name_list):
    """
    Args:
        MAD_array_dict: dict, keys are template names, values are MAD_array
        Temp_name_list: list of template names
    Returns:
        MAD_array_dict: dict, keys are template names, values are average MAD_array
    """
    MAD_array_dict_new = {}
    for Template in Temp_name_list:
        MAD_array_dict_new[Template] = np.array([np.mean(MAD_array_dict[Template])])
        MAD_array_dict_new['t_array'] = np.array([MAD_array_dict['t_array'][0], MAD_array_dict['t_array'][-1]])
    return MAD_array_dict_new
#!_______________________DATA IN________________________________
fs = 80

# Temp_name_list = [
#             'temp_0',   'temp_1',   
#             'temp_2',   'temp_3',   
#             'temp_4',   'temp_5',
#             'temp_6',   'temp_7',   
#             'temp_8',   'temp_9',   
#             'temp_10',  'temp_11',
#             'temp_12'
#         ]

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


level = 20
subdir = f'level_{level}/'

IN_CCs_Classification_dir = f'{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/' + subdir

IN_Classified_withMag = f"{IN_CCs_Classification_dir}MagRegression/"
IN_MagEventList_path = IN_Classified_withMag + 'Events_classified_Real_Mag.csv'

IN_MAD_Avg_path = f'{HOME_dir}Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/MAD_Average_dict.pkl'
MAD_Avg_dict = load_pkl(IN_MAD_Avg_path)

#!_______________________DATA OUT________________________________
# OUTPUT_dir = './Output_5_1_Est_Mag_Check_GR_law/' + 'Est_Mag_byCCs/' + subdir
OUTPUT_dir = f"{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/Show_Classification_byCCs/" + subdir
os.makedirs(OUTPUT_dir, exist_ok=True)


# Est magnitude and stored the estimated 'Events_classified' in OUTPUT_dir

#!__________________________PARAS__________________________________
t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')


#!_______________Read Event classified with magnitude estimated_______________________________
Events_cla_mag = pd.read_csv(IN_MagEventList_path, sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])
#? time control
Events_cla_mag = Events_cla_mag[ (Events_cla_mag['Rt_trig'] >= t_start) & (Events_cla_mag['Rt_trig'] <= t_end) ]


for cut_level in [level]:
    #? threshold control
    # mask = Events_cla_mag['CCs_s_trig_ofMAD'].abs() >= cut_level
    # Events_cla_mag = Events_cla_mag[mask]

    print(f'====> Events_classified cut from {t_start} to {t_end}:\n {Events_cla_mag}')


    #!__________________________plot 3______over b value time variation with time-magnitude_________________________


    #*_______1st Plot_______ 

    # adapt the axes according to the number of templates
    # fig = plt.figure(figsize=(30, 40))
    # gs = plt.GridSpec(7,2,figure=fig,height_ratios=[1,1,1,1,1,1,1],width_ratios=[7.3,1]) # adapt the gs according to the number of templates
    # axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:]), fig.add_subplot(gs[6,:])] 

    fig = plt.figure(figsize=(40, 8*(len(Temp_name_list))))
    no = len(Temp_name_list)
    gs = plt.GridSpec(no, 1, figure=fig, height_ratios=[1]*no, width_ratios=[7.3]) # adapt the gs according to the number of templates
    axes = [fig.add_subplot(gs[i, :]) for i in range(no)]


    plt.rcParams['font.size'] = 32

    dt = timedelta(hours=1, minutes=00, seconds=0)
    t_edges = np.arange(t_start, t_end+dt, dt)
    EQ_sum = 0
    
    for tt, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']      
        print(f'\033[35m ************************ 开始画图 3.{tt}: {disp_Template} ************************ \033[0m')
        
        #?_______1_________ plot histogram of classified events
        Real_mag = Events_cla_mag[Events_cla_mag['Template'] == Template]['Real_Mag']
        Rt_trig = pd.to_datetime(
                                        Events_cla_mag.loc[Events_cla_mag['Template'] == Template, 'Rt_trig']
                                        ).dt.to_pydatetime().tolist()
        Real_mag = Real_mag.to_numpy(dtype=float)
        CCs_s_trig = Events_cla_mag[Events_cla_mag['Template'] == Template]['CCs_s_trig']
        CCs_s_trig = CCs_s_trig.to_numpy(dtype=float)
        CCs_s_trig = abs(CCs_s_trig)

        #* scatter plot
        
        #*  legend plot start-----------------------------------------------------------
        # 使用 min-max 归一化将点的大小映射到合适的范围 [10, 300]，确保所有点的大小适中
        min_size = 20  # 最小点的大小
        max_size = 400  # 最大点的大小
        
        min_mag = -0.5
        max_mag = 4
        
        # size_arr = np.interp(Real_mag**2, (min_mag**2, max_mag**2), (min_size, max_size))
        def size_func(mag):
            """将震级映射到点的大小"""
            return np.interp(mag**2, (min_mag**2, max_mag**2), (min_size, max_size))

        mag_legend = [1, 2, 3, 4]
        size_at_this_mag = size_func(np.array(mag_legend))
        labels = [f'ML {s:.2f}' for s in mag_legend]

        legend_handles_mag = [
            Line2D([], [], linestyle='None', marker='o', markersize=np.sqrt(s), 
                color='k', label=labels[ii]) for ii, s in enumerate(size_at_this_mag)
        ]

        # ========== 在第三张图（axes[2]）手动添加“震级图例”，并不影响原 legend ==========
        axes[2].add_artist(
            axes[2].legend(handles=legend_handles_mag, title='Local Magnitude', 
                        loc='upper left', framealpha=0, fontsize=28, title_fontsize=28)
        )
        #*  legend plot end----------------------------------------------------------
        
        
        # #* 2. option 1 -- scatter with (Rt_trig, CCs_s_trig)
        # size_arr = size_func(Real_mag)
        # im = axes[tt].scatter(Rt_trig, CCs_s_trig, c=Real_mag, marker='o', s=size_arr, label=f'{Template}: #{len(Rt_trig)}', alpha=0.6, cmap='viridis')
        # colorbar = plt.colorbar(im, ax=axes[tt])
        # colorbar.set_label('Magnitude')
        
        
        #* 2. option 2 -- scatter with (Rt_trig, CCs_s_trig)
        size_arr = size_func(Real_mag)
        im = axes[tt].scatter(Rt_trig, Real_mag, c=Real_mag, marker='o', edgecolors='k', s=size_arr, label=f'{disp_Template}: #{len(Rt_trig)}', alpha=0.3, cmap='rainbow')
        colorbar = plt.colorbar(im, ax=axes[tt])
        colorbar.set_label('Magnitude')

        print(f"====> Read {Template}: #{len(Rt_trig)} events, display as {disp_Template}")
        EQ_sum += len(Rt_trig)
        
        
        
        #?_______2_________ plot b value variation
        # 创建次坐标系（右侧y轴）
        ax = axes[tt].twinx()

        b_method = 'MLE'  # 'MLE' or 'LS'
        bin_width = 0.1  # 震级分箱宽度, will not use bin_width if use MLE method
        minimum_events = 80  # 最小事件数
        window_hours = 12
        step_hours = 0.2 # 滑动步长
        mc_method = 'minmag'  # 'minmag' or 'maxc'
        if mc_method == 'minmag':
            mc = min(Real_mag)  # 使用最小震级作为Mc
        elif mc_method == 'maxc':
            mc = calculate_mc_maxc(Real_mag, bin_width=0.2)

        t0_array, b_values = compute_b_value_time_series(b_method,bin_width,minimum_events, Rt_trig, Real_mag, window_hours, step_hours, mc)
        
        # 绘制b值随时间变化的图
        ax.plot(t0_array, b_values, 'ro-', lw=3, markersize=3, alpha=0.5)
        ax.set_ylabel('b-value', color='k')
        # ax.legend(loc='upper right')
        ax.set_xlim([t_start, t_end])
        Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0)
        
        #?_______3_________ mark the earthquakes with magnitude > 2.2
        show_mag = 3
        #?+++++++++++++++
        index = np.where(Real_mag >= show_mag)[0]
        show_Real_mag = Real_mag[index]
        for ii in index:
            ax.axvline(x=Rt_trig[ii], color='orange', linestyle='--', alpha=0.5, lw=3)
            # ax.text(Rt_trig[ii], 0.1, f'{show_mag}', color='orange', fontsize=20, ha='center', va='bottom')
        
        
        # 合并图例
        lines1, labels1 = axes[tt].get_legend_handles_labels()
        lines2, labels2 = ax.get_legend_handles_labels()
        axes[tt].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 设置网格
        axes[tt].grid(color=axes[tt].yaxis.label.get_color())
        ax.grid(color=ax.yaxis.label.get_color())
        axes[tt].grid(zorder=0,linestyle=':')  # 底层
        ax.grid(zorder=1,linestyle='--')  # 上层

        # 一起设置一个标题
        title=f'{disp_Template}: Mc={mc:.2f}, minimum events no.={minimum_events}'
        axes[tt].set_title(title)
        

        
    # axes[0].set_title(f'Temporal Distribution of Classified Events.')# Total = {EQ_sum} events')
    print('====> Total number of classified events:', EQ_sum)

    for ax in axes:
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=22)
        ax.set_ylabel(f'Local Magnitude', fontsize=22)
        ax.set_xlim([t_start, t_end])
        Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=22)
        # Ticks_xaxis_minute_or_second(ax, Minute_tick=00, Second_tick=10, rotation=45)
        ax.legend(loc='upper right')
        # ax.grid('--')

    fig.suptitle(f'{b_method} method: b-value Variation ({window_hours}-hour window, {step_hours}-hour step), value at the right point;\n(orange line mark Mag>{show_mag} )')
    # fig.tight_layout()
    fname = f"showMag_bValue_min{minimum_events}_step{step_hours}h_win{window_hours}_{b_method}_mc_{mc_method}.png"
    fig.savefig(OUTPUT_dir + fname)
    print(f'Saved PLOT 3 in {OUTPUT_dir}\n\n')
    
    





    #!__________ZOOM IN________________plot 3______over b value time variation with time - magnitude_________________________
    #? 提取其中的某个template单独zoom in观察某段时间
    choice = 1
    if choice == 1:
        Template = 'template_0423_80Hz'
        disp_Template = 'template_K2'
        t_left = pd.Timestamp('2024-05-02 12:00:00')
        t_right = pd.Timestamp('2024-05-05 12:00:00')
    elif choice == 2:
        Template = 'template_0423_80Hz'
        t_left = pd.Timestamp('2024-05-10 00:00:00')
        t_right = pd.Timestamp('2024-05-12 00:00:00')
        disp_Template = 'template_K2'
    elif choice == 3:
        Template = 'template_10_80Hz'
        t_left = pd.Timestamp('2024-06-01 00:00:00')
        t_right = pd.Timestamp('2024-06-03 00:00:00')
        disp_Template = 'template_K1'
    elif choice == 4:
        Template = 'template_0423_80Hz'
        t_left, t_right = t_start, t_end
        disp_Template = 'template_K2'
    
    t_str_zoom = f"{t_left.strftime('%Y%m%d_%H%M%S')}_{t_right.strftime('%Y%m%d_%H%M%S')}"
    
    
    print(f'\033[35m ************************ 开始画图 4.1: {Template} disp.as {disp_Template} 在时间 {t_left} 到 {t_right} \n的地震的 时间-震级 的图 (NOA, STALTA, MF)************************ \033[0m')
    

    #*_______1st Plot_______ 

    # adapt the axes according to the number of templates
    fig = plt.figure(figsize=(24, 32))
    # fig = plt.figure(figsize=(90, 32))
    
    gs = plt.GridSpec(2,2,figure=fig,height_ratios=[8,1.5],width_ratios=[5.5,1]) # adapt the gs according to the number of templates
    axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0])] 

    plt.rcParams['font.size'] = 40
    
    ax = axes[0]
    
    #?_______1_________ plot histogram of classified events
    Real_mag = Events_cla_mag[Events_cla_mag['Template'] == Template]['Real_Mag']
    Rt_trig = pd.to_datetime(
                                    Events_cla_mag.loc[Events_cla_mag['Template'] == Template, 'Rt_trig']
                                    ).dt.to_pydatetime().tolist()
    Real_mag = Real_mag.to_numpy(dtype=float)
    CCs_s_trig = Events_cla_mag[Events_cla_mag['Template'] == Template]['CCs_s_trig']
    CCs_s_trig = CCs_s_trig.to_numpy(dtype=float)
    CCs_s_trig = abs(CCs_s_trig)

    #* scatter plot
    
    #*  legend plot start-----------------------------------------------------------
    # 使用 min-max 归一化将点的大小映射到合适的范围 [10, 300]，确保所有点的大小适中
    min_size = 70  # 最小点的大小
    max_size = 4000  # 最大点的大小
    
    min_mag = -0.5
    max_mag = 4
    
    # size_arr = np.interp(Real_mag**2, (min_mag**2, max_mag**2), (min_size, max_size))
    def size_func(mag):
        """将震级映射到点的大小"""
        return np.interp(mag**2, (min_mag**2, max_mag**2), (min_size, max_size))

    mag_legend = [1, 2, 3, 4]
    size_at_this_mag = size_func(np.array(mag_legend))
    labels = [f'ML {s:.2f}' for s in mag_legend]

    legend_handles_mag = [
        Line2D([], [], linestyle='None', marker='o', markersize=np.sqrt(s), 
            color='k', label=labels[ii]) for ii, s in enumerate(size_at_this_mag)
    ]

    # ========== 在第三张图（axes[2]）手动添加“震级图例”，并不影响原 legend ==========
    ax.add_artist(
        ax.legend(handles=legend_handles_mag, title='Local Magnitude', 
                    loc='upper left', framealpha=0, fontsize=40, title_fontsize=48)
    )
    #*  legend plot end----------------------------------------------------------
        
        
    # #* 2. option 1 -- scatter with (Rt_trig, CCs_s_trig)
    # size_arr = size_func(Real_mag)
    # im = axes[tt].scatter(Rt_trig, CCs_s_trig, c=Real_mag, marker='o', s=size_arr, label=f'{Template}: #{len(Rt_trig)}', alpha=0.6, cmap='viridis')
    # colorbar = plt.colorbar(im, ax=axes[tt])
    # colorbar.set_label('Magnitude')
    
    
    #* 2. option 2 -- scatter with (Rt_trig, CCs_s_trig)
    size_arr = size_func(Real_mag)
    im = ax.scatter(Rt_trig, Real_mag, c=Real_mag, marker='o', s=size_arr, label=f'{disp_Template}: #{len(Rt_trig)}', alpha=0.6, cmap='rainbow')
    colorbar = plt.colorbar(im, ax=ax)#, fraction=0.05, pad=0.04)
    colorbar.set_label('Magnitude')
    
    print(f"====> Read {Template}: #{len(Rt_trig)} events, display as {disp_Template}")
        
        
        
    #?_______2_________ plot b value variation
    # 创建次坐标系（右侧y轴）
    ax1 = ax.twinx()

    b_method = 'MLE'  # 'MLE' or 'LS'
    bin_width = 0.1  # 震级分箱宽度, will not use bin_width if use MLE method
    minimum_events = 80  # 最小事件数
    window_hours = 12
    step_hours = 1 # 滑动步长
    mc_method = 'minmag'  # 'minmag' or 'maxc'
    if mc_method == 'minmag':
        mc = min(Real_mag)  # 使用最小震级作为Mc
    elif mc_method == 'maxc':
        mc = calculate_mc_maxc(Real_mag, bin_width=0.2)

    t0_array, b_values = compute_b_value_time_series(b_method,bin_width,minimum_events, Rt_trig, Real_mag, window_hours, step_hours, mc)
    
    # 绘制b值随时间变化的图
    ax1.plot(t0_array, b_values, 'ro-', lw=10, markersize=10, alpha=0.5)
    ax1.set_ylabel('b-value', color='k')
    # ax.legend(loc='upper right')
    ax1.set_xlim([t_start, t_end])
    Ticks_xaxis_day_or_hour(ax1, Day_tick=5, Hour_tick=0, font=30)
    
    #?_______3_________ mark the earthquakes with magnitude > 2.2
    show_mag = 3
    index = np.where(Real_mag >= show_mag)[0]
    show_Real_mag = Real_mag[index]
    for ii in index:
        ax1.axvline(x=Rt_trig[ii], color='orange', linestyle='--', alpha=0.4, lw=7)
        # ax.text(Rt_trig[ii], 0.1, f'{show_mag}', color='orange', fontsize=20, ha='center', va='bottom')
    
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 设置网格
    ax1.grid(color=ax1.yaxis.label.get_color())
    ax.grid(color=ax.yaxis.label.get_color())
    ax1.grid(zorder=0,linestyle=':')  # 底层
    ax.grid(zorder=1,linestyle='--')  # 上层

    # 一起设置一个标题
    title=f'{disp_Template}: Mc={mc:.2f}, minimum events no.={minimum_events}'
    ax1.set_title(title)
    # axes[0].set_title(f'Temporal Distribution of Classified Events.')# Total = {EQ_sum} events')

    
    #?_____________________________________2____________________________
    #*----再画出 temporal distribution的hist图
    ax = axes[1]
    # dt = timedelta(hours=0, minutes=10, seconds=0)
    dt = timedelta(hours=0, minutes=10, seconds=0)
    t_edges = np.arange(t_start, t_end+dt, dt)

    # plot histogram of classified events
    Rt_trig_classified = Events_cla_mag[(Events_cla_mag['Template'] == Template) & (Events_cla_mag['Belong_or_not'] == 1)]['Rt_trig']    
    ax.hist(Rt_trig_classified,bins=t_edges, color='r', alpha=0.7, label=f'{disp_Template}: #{len(Rt_trig_classified)}')

    ax.set_title(f'Temporal Distribution of Classified Events.')# Total = {EQ_sum} events')
    ax.set_ylabel(f'Counts per {dt}', fontsize=46)


    for ax in axes:
        ax.tick_params(axis='y', labelsize=42)
        ax.tick_params(axis='x', labelsize=42)
        ax.set_ylabel(f'Counts per {dt}', fontsize=42)
        ax.set_xlim([t_left, t_right])
        # Ticks_xaxis_day_or_hour(ax, Day_tick=0, Hour_tick=12, font=22)
        Ticks_xaxis_day_or_hour(ax, Day_tick=0, Hour_tick=12, font=42)
        # Ticks_xaxis_minute_or_second(ax, Minute_tick=00, Second_tick=10, rotation=45)
        ax.legend(loc='upper right')
        # ax.grid('--')

    fig.suptitle(f'{b_method} method: b-value Variation ({window_hours}-hour window, {step_hours}-hour step), value at the right point;\n(orange line mark Mag>{show_mag} )')
    # fig.tight_layout()
    fname = f"showMag_bValue_min{minimum_events}_step{step_hours}h_win{window_hours}_{b_method}_mc_{mc_method}_{Template}_{t_str_zoom}.png"
    fig.savefig(OUTPUT_dir + fname)
    print(f'Saved PLOT 3 Zoom in in {OUTPUT_dir}\n\n')