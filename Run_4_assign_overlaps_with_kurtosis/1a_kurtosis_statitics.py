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
        
        self.max_stalta_threshold = 14
        #?+++++++++++++++++ clean overlaps 关键参数 ++++++++++++++++++++++++++++++++++++++++++++++++++
        self.IN_one_event_interval = 2.5
        # if self.IN_one_event_interval > self.Minimal_event_interval:
        #     print(f'INREALISTIC & UNSTABLE: The IN_one_event_interval={self.IN_one_event_interval} is larger than Minimal_event_interval={self.Minimal_event_interval}')

        #?++++++++++++++++++ 进行 max_stalta 和 CCs_s_trig 统计的关键参数+++++++++++++++++++++++++++++++++++++++++++++++++
        self.kurtosis_arr = np.arange(1, 30, 0.1)  # kurtosis_arr = np.arange(1, 11, 1) # 1 to 10
        self.C_trig_arr = np.arange(0.04, 1, 0.005)  # C_trig = np.arange(0.1, 0.6, 0.1) # 0.1 to 0.5
        

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
        self.OUTPUT_dir = f"{self.HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/{self.subdir}kurtosis_statistics/"

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
    Collapse_group = Read_Collapse_group_df(config.IN_CCs_trig_dir, config.Temp_name_list, DF_Temp_Offset, config.MAD_Avg_dict, config.t_start, config.t_end)
    print('====>Collapse_group by time',Collapse_group)
    #!+++++这里的 Read_Collapse_group_df 是修改过的，包含了新的 max_RAWCCs, t_arg_max 信息

    #!+++++这里的 Read_Collapse_group_df 是修改过的，包含了新的 max_stalta 信息
    
    
    
    mask = Collapse_group['max_stalta'] > config.max_stalta_threshold
    #? 删除 不满足 max_stalta条件的触发事件
    good_Collapse_group = Collapse_group[mask]
    good_Collapse_group.reset_index(drop=True, inplace=True)
    print('====>Collapse_group after max_stalta threshold', len(good_Collapse_group))
    
    #*____________________PROCESS the Collapse_group________________________________________________
    # use pd functions to collapse the overlapped multiple-template events, use parameter: IN_one_event_interval
    # Collapse_group['Rt_trig'] in order
    # mark the row with highest abs(CCs_s_trig) in the group, --> mark the 'Belong_or_not' = 1
    # 初始化标记列

    #* assign overlaps by maximum CCs_s_trig
    Events_classified = Collapse_with_MAX_CCs(good_Collapse_group, config.IN_one_event_interval)
    # Events_classified.to_csv(config.OUTPUT_dir + 'Events_classified_AllTemp.csv', sep='\t', index=False)
    
    #! 处理办法2----如果 max_RAWCCs 已经被记录到了 读取每个事件的最大 RAWCCs_s
    # 读取触发的CCs_s数据
    # 使用新的函数读取最开始的collapsed触发事件数据
    # Events_classified.to_csv(config.OUTPUT_dir + 'Events_classified_AllTemp_addRAWCCs.csv', sep='\t', index=False)
    
    # 统计'Belong_or_not' = 1 的行数， 这也等于group的最大值
    print(f'*********Event number before collapsed:', len(Events_classified))
    print(f'*********Event number after collapsed:', Events_classified['Belong_or_not'].sum())
    print(f'*********Event number after collapsed:', Events_classified['group'].max())

    # this is done!!!!
    Complete_trig_dict = {}
    # initially triggered events of each template
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        print(f'\n********** {Template} disp as {disp_Template}**********')
        df = Events_classified[Events_classified['Template'] == Template]
        print(f'-----Event number before collapsed:', len(df))
        print(f'-----Event number after collapsed:', df['Belong_or_not'].sum())
        
        Complete_trig_dict[Template] = df['Belong_or_not'].sum()



    
    #?++++++++++++++++++ 1-- 分析kurtosis 和 C 的相关性 --- 画两者散点图 (max_stalta, CCs_s_trig) ---分模版画
    print("正在生成图 1: max_stalta vs CCs_s_trig 散点图...")

    n_temp = len(config.Temp_name_list)
    # 创建画布和子图，figsize可以根据需要调整
    fig, axes = plt.subplots(n_temp, 1, figsize=(6, 4 * n_temp))#, sharex=True, sharey=True)
    # 如果只有一个模板，axes不是一个数组，需要转换一下以方便索引
    if n_temp == 1:
        axes = [axes]
    # 循环绘制每个模板的子图
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        
        ax = axes[ii]
        mask = Collapse_group['Template'] == Template
        temp_group = Collapse_group[mask]
        
        # 设置子图标题 (英文)
        ax.set_title(f'(max_stalta v.s. CCs_s_trig) {disp_Template}')
        
        # 绘制散点图
        ax.scatter( temp_group['CCs_s_trig'],temp_group['max_stalta'],
                   alpha=0.1, s=3, c='b', label='Detected Events',
                    edgecolor='k') 
        
        # 设置坐标轴标签 (英文)
        ax.set_xlabel('CCs_s_trig')
        
        # 绘制阈值线
        threshold_label = f'max_stalta threshold = {config.max_stalta_threshold}'
        ax.axhline(y=config.max_stalta_threshold, color='r', linestyle='--', label=threshold_label)

        # 显示图例
        # ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_ylabel('max_stalta')

    # 自动调整布局，防止标签重叠
    fig.tight_layout()

    # 在循环外部保存整个图像
    fig.savefig(config.OUTPUT_dir + 'max_stalta_vs_CCs_s_trig.png', dpi=300)
    plt.close(fig) # 关闭画布，释放内存
    print("图 1 保存完成。")
    
    
    #?++++++++++++++++++ 2-- 分析kurtosis 和 C 的不同限制下 (kurtosis_arr,C_trig_arr) 的事件数量 矩阵imshow
    import matplotlib.colors as mcolors # 导入colors模块以使用LogNorm
    print("正在生成图 2: 事件数量矩阵图...")

    n_temp = len(config.Temp_name_list)
    fig, axes = plt.subplots(n_temp, 1, figsize=(10, 7 * n_temp))

    if n_temp == 1:
        axes = [axes]

    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        
        ax = axes[ii]
        
        mask = Collapse_group['Template'] == Template
        temp_group = Collapse_group[mask]

        count_matrix = np.zeros((len(config.kurtosis_arr), len(config.C_trig_arr)))
        
        # 计算计数矩阵
        for kk, kurtosis in enumerate(config.kurtosis_arr):
            for cc, C_trig in enumerate(config.C_trig_arr):
                mask = (temp_group['max_stalta'] > kurtosis) & (temp_group['CCs_s_trig'].abs() > C_trig)
                count_matrix[kk, cc] = mask.sum()
                
        # 设置子图标题
        ax.set_title(f'Event Count Matrix for {disp_Template}')
        
        # 绘制矩阵热力图
        # 注意: LogNorm不能处理0值。如果count_matrix中有0，imshow会显示为空白。
        # 我们给矩阵加1来避免这个问题，颜色条会自动反映这一点。
        im = ax.imshow(count_matrix + 1, aspect='auto', origin='lower', cmap='viridis', 
                    norm=mcolors.LogNorm(vmin=1, vmax=(count_matrix.max() + 1)))
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Event Count (Log Scale)')
        def get_optimal_ticks(data_array, max_ticks=10):
            """
            计算最佳的刻度位置和标签。

            参数:
            data_array (array-like): 原始数据数组 (例如 config.C_trig_arr)。
            max_ticks (int): 最多显示的刻度数量。

            返回:
            tuple: (刻度位置列表, 刻度标签列表)
            """
            n = len(data_array)
            # 确定最终显示的刻度数量
            num_ticks = min(n, max_ticks)
            
            # 如果没有数据，返回空
            if num_ticks == 0:
                return [], []
            
            # 使用 linspace 生成均匀分布的整数索引
            # 这样可以确保第一个和最后一个点都被包含
            indices = np.linspace(0, n - 1, num_ticks, dtype=int)
            
            # 刻度位置就是这些索引
            tick_positions = indices
            # 刻度标签是原始数据在这些索引上的值
            tick_labels = [f'{data_array[i]:.2f}' for i in indices]
            
            return tick_positions, tick_labels

        # --- 应用到X轴 ---
        x_ticks, x_ticklabels = get_optimal_ticks(config.C_trig_arr, max_ticks=10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=45, ha='right')

        # --- 应用到Y轴 ---
        y_ticks, y_ticklabels = get_optimal_ticks(config.kurtosis_arr, max_ticks=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
        # 设置Y轴标签
        ax.set_ylabel('kurtosis by max_stalta')

    # 设置共享的X轴标签
    axes[-1].set_xlabel('C_trig')

    fig.tight_layout()
    fig.savefig(config.OUTPUT_dir + 'kurtosis_vs_C_trig_CountMatrix.png')
    plt.close(fig)
    print("图 2 保存完成。")
    
    
    
    #?++++++++++++++++++ 3-- 画recall-precision曲线
    print("正在生成图 3: Precision-Recall 曲线...")

    #? 需要指定一个 Complete_trig 的数量， prior，预计的， 可以通过G-R law 预测一下
    # Complete_trig_dict = {
    #     'template_0423_80Hz': 12000,
    #     'template_0425_80Hz': 1200,
    #     'template_10_80Hz': 2000,
    #     'template_8_80Hz': 150,
    #     'template_20_80Hz': 200,
    # }
        
    n_temp = len(config.Temp_name_list)
    fig, axes = plt.subplots(n_temp, 1, figsize=(8, 7 * n_temp))

    if n_temp == 1:
        axes = [axes]
        
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        
        ax = axes[ii]
        
        mask = Collapse_group['Template'] == Template
        temp_group = Collapse_group[mask]

        # 预估的真实正例总数，用于计算召回率
        Complete_trig = Complete_trig_dict[Template]
        
        recall = np.zeros(len(config.C_trig_arr))
        precision = np.zeros(len(config.C_trig_arr))
                            
        for cc, C_trig in enumerate(config.C_trig_arr):
            # 判定为正例的样本 (Predicted Positive)
            PredictedPos_mask = temp_group['CCs_s_trig'].abs() >= C_trig
            
            # 真正例 (True Positive)
            TruePos_mask = (temp_group['max_stalta'] > config.max_stalta_threshold) & PredictedPos_mask
            
            # 假正例 (False Positive)
            FalsePos_mask = (temp_group['max_stalta'] <= config.max_stalta_threshold) & PredictedPos_mask
            
            TP_count = TruePos_mask.sum()
            FP_count = FalsePos_mask.sum()
            
            # 假负例 (False Negative) = 真实正例总数 - 找到的真正例
            FN_count = Complete_trig - TP_count
            
            # 计算 Precision 和 Recall
            # 分母为0时，结果为0，避免除零错误
            precision[cc] = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
            recall[cc] = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0

        # 设置子图标题
        ax.set_title(f'Precision-Recall Curve for {disp_Template}.')
        
        # --- 修正后的绘图代码 ---
        # 1. 绘制背景连接线 (横轴:Recall, 纵轴:Precision)
        ax.plot(recall, precision, color='lightgray', linestyle='-')
        
        # 2. 绘制散点图，并用 C_trig_arr 的值来着色
        scatter = ax.scatter(recall, precision, c=config.C_trig_arr, cmap='viridis', 
                            marker='o', zorder=3)
        for ii in range(len(recall)):
            # 在每个点上添加文本标签
            # if config.C_trig_arr[ii] < 0.2:
            if precision[ii]<0.98:
                ax.text(recall[ii], precision[ii], f'{config.C_trig_arr[ii]:.2f}', fontsize=8, ha='right', va='bottom', alpha=0.7)
        
        # 3. 创建与scatter关联的颜色条
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('CCs_s_trig threshold')
        
        # 设置坐标轴标签 (符合P-R曲线标准)
        ax.set_xlabel(f'Recall (as fraction of {Complete_trig} total events)')
        ax.set_ylabel('Precision')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 设置坐标轴范围
        # ax.set_xlim([0, 1.0])
        # ax.set_ylim([0, 1.05])


    fig.tight_layout()
    fig.savefig(config.OUTPUT_dir + 'PrecRecall_threshold_changing.png')
    plt.close(fig)
    print("图 3 保存完成。")
    
    
    
    
if __name__ == "__main__":
    main()
    # clear_output(wait=True)
    print('Done!')