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
        
        self.max_stalta_threshold_chosen = 10
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        #     print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #* 测试的max_stalta_threshold序列
        self.max_sl_thr_arr = np.arange(0, 40, 0.2) # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        

        self.t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
        self.t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_str = '_' + self.t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + self.t_end.strftime('%Y-%m-%d %H:%M:%S')


        #* 输入路径
        self.HOME_dir = f'{PROJECT_ROOT}/'
        self.RUN_dir = f"{self.HOME_dir}Run_4_assign_overlaps_with_kurtosis/"

        self.temp_lib_csv = f"{self.HOME_dir}Output_0_Template_library/Template_library.csv"
        self.temp_npy_dir = f"{self.HOME_dir}Output_0_Template_library/Template_npy_files/"

        self.IN_CCs_stack_dir = f"{self.HOME_dir}Output_1_SCCs_and_MAD/SCCs_h5/"
        # Path to the MAD average dictionary
        self.MAD_Avg_path = f"{self.HOME_dir}Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/MAD_Average_dict.pkl"
        self.MAD_Avg_dict = load_pkl(self.MAD_Avg_path)  # load the MAD_array_dict
        
        self.subdir = f"level_{self.Threshold_level}/"
        self.IN_CCs_trig_dir = f"{self.HOME_dir}Output_2_Triggered_SCCs_points/{self.subdir}"

        #* 输出路径
        self.OUTPUT_dir = f"{self.RUN_dir}Output_1b_decide_kurtosis/{self.subdir}"

        # self.OUTPUT_dir_byTime = self.OUTPUT_dir + 'classified' + self.t_str + '/'

        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 1e3
        self.das_dist_end = 15e3
        self.dx = 16*1.02095198631286
        
        # 选择继续进入classification的模板列表
        # self.Temp_name_list = [
        #     'temp_0',   'temp_1',   
        #     'temp_2',   'temp_3',   
        #     'temp_4',   'temp_5',
        #     'temp_6',   
        #     'temp_7',   'temp_8',   
        #     'temp_9',   
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
# template_name P_offset    S_offset
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
    
    # 初始化一个列表来存储每个阈值的结果
    results_data = []

    # 计算初始触发事件总数 (在循环外计算，因为它是常数)
    total_raw_count = len(Collapse_group_raw)
    template_raw_counts = {row['infile_name']: len(Collapse_group_raw[Collapse_group_raw['Template'] == row['infile_name']]) for _, row in check_lib.iterrows()}
    
    print(f"\n开始循环处理不同的 max_stalta_threshold 值...")
    for ss, max_stalta_threshold in enumerate(config.max_sl_thr_arr):
        
        # 字典用于存储当前循环的统计数据
        current_run_data = {'max_stalta_threshold': max_stalta_threshold}

        #? 1. 根据 max_stalta_threshold 筛选事件
        mask = Collapse_group_raw['max_stalta'] > max_stalta_threshold
        Collapse_group_filtered = Collapse_group_raw[mask].copy()
        Collapse_group_filtered.reset_index(drop=True, inplace=True)
        
        # 记录筛选后的总事件数 (即合并前的事件数)
        total_filtered_count = len(Collapse_group_filtered)
        current_run_data['total_filtered'] = total_filtered_count

        #* 2. 使用您的函数合并重叠事件
        Events_classified = Collapse_with_MAX_CCs(Collapse_group_filtered, config.IN_one_event_interval)
        
        # 记录合并后的总事件数
        total_collapsed_count = Events_classified['Belong_or_not'].sum()
        current_run_data['total_collapsed'] = total_collapsed_count
        
        # 3. 为每个模板记录事件数
        for _, row in check_lib.iterrows():
            template_name = row['infile_name']
            
            # 记录筛选后(合并前)的事件数
            filtered_count_template = len(Collapse_group_filtered[Collapse_group_filtered['Template'] == template_name])
            current_run_data[f'{template_name}_filtered'] = filtered_count_template
            
            # 记录合并后的事件数
            collapsed_count_template = Events_classified[Events_classified['Template'] == template_name]['Belong_or_not'].sum()
            current_run_data[f'{template_name}_collapsed'] = collapsed_count_template

        # 将当前阈值的所有统计数据添加到结果列表
        results_data.append(current_run_data)

        # 打印当前循环的进度和结果
        # if ss % 20 == 0: # 每20次循环打印一次，避免刷屏
        print(f"  阈值: {max_stalta_threshold:.2f} | 筛选后事件数: {total_filtered_count} | 最终事件数: {total_collapsed_count}")


    # 将结果列表转换为DataFrame
    df_decide_kurtosis = pd.DataFrame(results_data)
    df_decide_kurtosis.set_index('max_stalta_threshold', inplace=True)

    # 将原始事件数添加到DataFrame中，方便绘图
    df_decide_kurtosis['total_raw'] = total_raw_count
    for template_name, raw_count in template_raw_counts.items():
        df_decide_kurtosis[f'{template_name}_raw'] = raw_count
        
    print("\n数据处理完成，生成的DataFrame如下:")
    print(df_decide_kurtosis.head())

    # 将DataFrame保存到CSV文件
    output_csv_path = os.path.join(config.OUTPUT_dir, 'stalta_threshold_analysis.csv')
    df_decide_kurtosis.to_csv(output_csv_path)
    print(f"\n分析结果已保存到: {output_csv_path}")

    #*____________________数据可视化________________________________________________
    
    # --- 定义字体大小配置，方便全局修改 ---
    ftz = 16
    font_config = {
        'title': ftz + 4,    # 图表标题字体大小
        'label': ftz + 2,    # 坐标轴标签字体大小
        'tick': ftz,     # 坐标轴刻度字体大小
        'legend': ftz,   # 图例字体大小
        'anno': ftz      # 注释文字大小
    }
    
    # --- 图1: 绘制总事件数随阈值的变化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    ax2 = ax1.twinx()

    # 在主轴(ax1)上绘图
    p1, = ax1.plot(df_decide_kurtosis.index, df_decide_kurtosis['total_raw'], label='Initial triggered event no. by SCC thresholding', color='grey', linestyle=':', alpha=0.7, zorder=1)
    p2, = ax1.plot(df_decide_kurtosis.index, df_decide_kurtosis['total_filtered'], label='Event no. (after cleaned by kurtosis control)', color='dodgerblue', marker='o', markersize=3, linestyle='-', zorder=3)
    p3, = ax1.plot(df_decide_kurtosis.index, df_decide_kurtosis['total_collapsed'], label='Final event no. (after assigning overlaps)', color='crimson', marker='x', markersize=4, linestyle='-', zorder=3)
    
    difference = df_decide_kurtosis['total_filtered'] - df_decide_kurtosis['total_collapsed']
    p4, = ax2.plot(df_decide_kurtosis.index, difference, color='darkorange', linestyle='--', label='Overlapping event no.', zorder=2)
    
    # --- 修改代码: 回归到简单的axvline灰色虚线标注 ---
    x_chosen = config.max_stalta_threshold_chosen
    v_line = ax1.axvline(x=x_chosen, color='grey', linestyle='--', linewidth=1.5, 
                         label=f'Chosen kurtosis threshold = {x_chosen}', zorder=4)

    # 设置主轴(ax1)属性，使用字体配置
    ax1.set_title('Overlapping Event Number Changing with Kurtosis Threshold', fontsize=font_config['title'])
    ax1.set_xlabel('Kurtosis Threshold', fontsize=font_config['label'])
    ax1.set_ylabel('Number of Events (Log Scale)', fontsize=font_config['label'])
    ax1.tick_params(axis='both', which='major', labelsize=font_config['tick'])
    ax1.set_yscale('log')

    # 设置次轴(ax2)属性，使用字体配置
    ax2.set_ylabel('Overlapping Event Number after kurtosis control', fontsize=font_config['label'], color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=font_config['tick'])
    ax2.set_ylim(bottom=0)

    # 合并图例
    plots = [p1, p2, p3, p4, v_line]
    ax1.legend(handles=plots, loc='upper right', fontsize=font_config['legend'])
    
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax2.grid(False)

    plt.tight_layout(pad=1.5)
    plot_path1 = os.path.join(config.OUTPUT_dir, 'total_events_vs_stalta_threshold.png')
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    print(f"总体事件数变化图已保存到: {plot_path1}")
    plt.show()


    # --- 图2: 绘制各模板事件数随阈值的变化 (保持上一版的详细设定) ---
    num_templates = len(check_lib)
    if num_templates <= 4: nrows, ncols = 2, 2
    elif num_templates <= 6: nrows, ncols = 3, 2
    elif num_templates <= 9: nrows, ncols = 3, 3
    else: nrows, ncols = int(np.ceil(num_templates / 3)), 3
        
    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for i, row in check_lib.iterrows():
        template_name = row['infile_name']
        disp_name = row['plot_name']
        ax = axes[i]
        
        ax_twin = ax.twinx()
        
        # 在主轴上绘图
        p1_sub, = ax.plot(df_decide_kurtosis.index, df_decide_kurtosis[f'{template_name}_raw'], label='Raw triggers', color='grey', linestyle=':', alpha=0.8)
        p2_sub, = ax.plot(df_decide_kurtosis.index, df_decide_kurtosis[f'{template_name}_filtered'], label='After cleaned by kurtosis', color='dodgerblue', marker='o', markersize=2, linestyle='-')
        p3_sub, = ax.plot(df_decide_kurtosis.index, df_decide_kurtosis[f'{template_name}_collapsed'], label='After assigning overlaps', color='crimson', marker='x', markersize=2, linestyle='-')
        
        # 在次坐标轴上绘制差值
        difference_sub = df_decide_kurtosis[f'{template_name}_filtered'] - df_decide_kurtosis[f'{template_name}_collapsed']
        p4_sub, = ax_twin.plot(df_decide_kurtosis.index, difference_sub, color='darkorange', linestyle='--', label='Overlapping event no.')
        
        # 设置坐标轴和标签
        ax.set_title(f'Template: {disp_name}', fontsize=font_config['label'])
        ax.set_ylabel('Event Count (Log Scale)', fontsize=font_config['tick'])
        ax.tick_params(axis='both', which='major', labelsize=font_config['tick']-2)
        ax.set_yscale('log')
        if ax.get_ylim()[0] < 0.1:
            ax.set_ylim(bottom=0.1)

        ax_twin.set_ylabel('Overlapping event no.', fontsize=font_config['tick']-2, color='darkorange')
        ax_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=font_config['tick']-2)
        ax_twin.set_ylim(bottom=0)

        # 为每个子图都添加X轴标签
        ax.set_xlabel('Kurtosis Threshold', fontsize=font_config['tick'])
        
        # 合并图例
        plots_sub = [p1_sub, p2_sub, p3_sub, p4_sub]
        ax.legend(handles=plots_sub, loc='upper right', fontsize=font_config['legend']-4)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig2.suptitle('Individual Template Event Count vs. Kurtosis Threshold', fontsize=font_config['title']+2, y=1.03)
          
    plot_path2 = os.path.join(config.OUTPUT_dir, 'template_events_vs_stalta_threshold_detailed.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"各模板事件数详细变化图已保存到: {plot_path2}")
    plt.show()


if __name__ == "__main__":
    main()
    # clear_output(wait=True)
    print('Done!')