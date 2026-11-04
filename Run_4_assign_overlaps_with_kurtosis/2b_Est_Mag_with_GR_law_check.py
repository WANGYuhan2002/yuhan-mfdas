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
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from extract_station_info import extract_station_info

def least_squares_fit(x, y, ax, title):
    """
    对输入数据进行最小二乘法线性拟合，并保存图像。
    
    参数：
    x: array-like，自变量数据
    y: array-like，因变量数据
    save_path: str，保存图像的文件路径
    
    返回：
    slope: float，斜率
    intercept: float，截距
    """
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    df = df.astype(float)  # 转换数据类型，避免错误

    # 转换为 NumPy 数组
    x, y = df['x'].to_numpy(), df['y'].to_numpy()
    
    # 进行线性拟合
    slope, intercept = np.polyfit(x, y, 1)  # 1 表示一阶线性拟合

    # 计算拟合直线的 y 值
    y_fit = slope * np.array(x) + intercept

    # 计算 R², evaluation
    ss_total = np.sum((y - np.mean(y)) ** 2)  # 总方差
    ss_residual = np.sum((y - y_fit) ** 2)  # 残差平方和
    R2 = 1 - (ss_residual / ss_total)
    
    # 绘图
    # plt.figure(figsize=(8, 6))
    ax.scatter(x, y, color='blue', label='Original Data')  # 原始数据点
    ax.plot(x, y_fit, color='red', linestyle='--', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')  # 拟合直线

    # 标注斜率和截距
    ax.text(x[0], y_fit[-1], f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nR2: {R2:.3f}', fontsize=12, color='red',
             bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Log10(Amplitude scaling from Cross-Correlation)')
    ax.set_ylabel('Real Magnitude from IMS')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


    return slope, intercept


def from_catalog_to_GR_law(magnitudes,M_c,ax,label,title):
    from scipy.stats import linregress
    # 示例数据：地震震级
    # magnitudes = [2.3, 3.1, 2.8, 3.5, 4.0, 2.7, 3.3, 4.5, 3.0, 3.8, 5.0]
    # 1. 确定完全性震级 M_c
    # M_c = 3.0  # 假设完全性震级为 3.0
    # 2. 筛选 M >= M_c 的震级数据
    filtered_magnitudes = [m for m in magnitudes if m >= M_c]
    # 3. 计算累计频次
    sorted_magnitudes = sorted(filtered_magnitudes)
    cumulative_counts = [len(sorted_magnitudes) - i for i in range(len(sorted_magnitudes))]
    # 转换为 numpy 数组
    sorted_magnitudes = np.array(sorted_magnitudes)
    cumulative_counts = np.array(cumulative_counts)
    # 4. 对频次取对数
    log_counts = np.log10(cumulative_counts)
    # 5. 线性回归拟合
    slope, intercept, r_value, p_value, std_err = linregress(sorted_magnitudes, log_counts)
    b_value = -slope  # b-value 是斜率的绝对值

    # 输出结果
    print(f"b-value: {b_value:.3f}")
    print(f"Intercept (a): {intercept:.3f}")
    print(f"R-squared: {r_value**2:.3f}")

    # 6. 绘制拟合结果
    # fig,ax = plt.subplots(figsize=(10,6))
    ax.scatter(sorted_magnitudes, log_counts, color="blue", marker="o", s=5, alpha=0.6,label=label)
    ax.plot(sorted_magnitudes, intercept + slope * sorted_magnitudes, 'r',label=f'slope = {slope:.3f},\nintercept = {intercept:.3f},\nM_c = {M_c:.3f}')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("log10(N)")
    ax.set_title(title)
    ax.grid()
    ax.legend()
    # fig.savefig(OUT_fname, dpi=300)

    return b_value, intercept


def _find_Template_EQs_in_IMS(df_IMS,Template,Events_classified,Temp_lib_df,t_ERR):
    Evs_with_NOA_Mag = pd.DataFrame()
    
    #*---------- 3.1 Get MF time and search FSK time in surrounding area
    print(f'find===>{Template}')# in df_temp: {df_temp}')
    # df_temp_template = df_temp[df_temp['template_name'] == Template]
    # lat = df_temp_template['Latitude'].values[0]
    # lon = df_temp_template['Longitude'].values[0]
    # print('lat', lat, 'lon', lon)
    df_byTemp = Temp_lib_df[Temp_lib_df['infile_name'] == Template]
    lat = df_byTemp['EQ_latitude'].values[0]
    lon = df_byTemp['EQ_longitude'].values[0]
    
    #*---------- 3.2 Search FSK time in surrounding area
    # convert str type to float type
    df_IMS['EQ_latitude'] = pd.to_numeric(df_IMS['EQ_latitude'], errors='coerce')
    df_IMS['EQ_longitude'] = pd.to_numeric(df_IMS['EQ_longitude'], errors='coerce')
    
    #?----limit the searching area to +-0.05 degree
    search_mask = (abs(df_IMS['EQ_latitude'] - lat) < 0.05) & (abs(df_IMS['EQ_longitude'] - lon) < 0.05)
    df_IMS = df_IMS[search_mask].reset_index(drop=True)
    # find template events in Events_classified,
    # with time difference with df_IMS['STA_arrival_time'] less than t_ERR
    for ii, row in df_IMS.iterrows():
        # print('row', row)
        t_IMS = row['STA_arrival_time']
        t_diff = abs((t_IMS - Events_classified['Rt_trig']).dt.total_seconds())
        # print('t_diff', t_diff)
        if t_diff.min() < t_ERR:
            index_min = t_diff.idxmin()

            # 选取 df_IMS 中的一行，并重置索引
            df_IMS_row = row.to_frame().T.reset_index(drop=True)  

            # 选取 Events_classified 中对应的一行，并重置索引
            Events_classified_row = Events_classified.iloc[[index_min]].reset_index(drop=True)  

            #* 拼接两行数据，确保它们合并到同一行
            new_row = pd.concat([df_IMS_row, Events_classified_row], axis=1)

            # 将新行追加到 Evs_with_NOA_Mag
            Evs_with_NOA_Mag = pd.concat([Evs_with_NOA_Mag, new_row], ignore_index=True)
    return Evs_with_NOA_Mag

#!_______________________DATA IN________________________________
HOME_dir = f'{PROJECT_ROOT}/'

fs = 80

# 会决定画图的数量多少
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

IN_Temp_lib_csv = f"{HOME_dir}/Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv,sep=',')
# trim to the templates we want to check
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Temp_name_list)]
# rank in plot order
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)



subdir = 'level_20/'
IN_CCs_Classification_dir = f"{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/{subdir}"

#!_______________________DATA OUT________________________________
OUTPUT_dir =  f"{IN_CCs_Classification_dir}MagRegression/"
os.makedirs(OUTPUT_dir, exist_ok=True)
# Est magnitude and stored the estimated 'Events_classified' in OUTPUT_dir
os.makedirs(OUTPUT_dir + 'Events_with_NOA_Mag/', exist_ok=True)

#?++++++++++++++++++++++++++++++++++++++
method = 'max_RAWCCs'  # 'Amp' or 'CCs' or 'meanCCs
#?++++++++++++++++++++++++++++++++++++++
if method == 'max_RAWCCs':
    IN_Classified_without_amp_path = IN_CCs_Classification_dir + 'Events_classified_AllTemp.csv'

    IN_trig_MAX_amp_df_path = f"{HOME_dir}Output_3_Trig_PeakAmp/{subdir}Trig_MAX_amp_df.csv"

    IN_Classified_with_max_RAWCCs_path = IN_CCs_Classification_dir + 'Events_classified_AllTemp_addRAWCCs.csv'



    Events_classified = pd.read_csv(IN_Classified_with_max_RAWCCs_path, sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])

    trig_MAX_amp_df = pd.read_csv(IN_trig_MAX_amp_df_path, sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])
    #?+++++++++++++++ find Events_classified element in trig_MAX_amp_df, and get corresponding values of amp_log_mean, amp_max_mean +++++++++++++++++++++++
    Events_classified = pd.merge(Events_classified, trig_MAX_amp_df[['Rt_trig', 'Template', 'amp_log_mean', 'amp_max_mean']], on=['Rt_trig', 'Template'], how='left')
    #?++++++++++++++++++++++++++++++++++++++


#!__________________________PARAS__________________________________
t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
t_str = '_' + t_start.strftime('%Y-%m-%d %H:%M:%S') + '_' + t_end.strftime('%Y-%m-%d %H:%M:%S')

# OUTPUT_dir_byTime = OUTPUT_dir + 'mag'+t_str + '/'
# if not os.path.exists(OUTPUT_dir_byTime):
#     os.makedirs(OUTPUT_dir_byTime)
# if not os.path.exists(OUTPUT_dir_by + 'Events_with_Real_Mag/'):
#     os.makedirs(OUTPUT_dir_byTime + 'Events_with_Real_Mag/')
# OUTPUT_dir_byTime: stores GR law plot and other plots with mag info, specified by t_start and t_end

#!_______________Est magnitude and stored the estimated 'Events_classified' in OUTPUT_dir________________________________

#!)))) 将记录到的 trig 的 Trig_MAX_amp_df.csv 数据中抽取出最后被分类到真正地震事件的 amp_log_mean, amp_max_mean,
#! 最后的带有 QuasiMag 的仍然命名为 Events_classified
# Events_classified = pd.read_csv(IN_amp_record_path, sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])


Events_classified = Events_classified[ (Events_classified['Rt_trig'] >= t_start) & (Events_classified['Rt_trig'] <= t_end) ]
print(f'====> Events_classified cut from {t_start} to {t_end}:\n {Events_classified}')

#? Est. quasi magnitude from unnormalized CCs (Estimate magnitude is proportional to the real magnitude)
# Mag ~= log10(Amplitude) + Const. ======>  Mag = Quasi_Mag + Const.  b value unchanged, a value change
# G-R law: log10(N) = a - b * M, (N is the number of events larger than  magnitude M)

if method == 'max_RAWCCs':
    Events_classified['Quasi_Mag_Amp'] = Events_classified['amp_log_mean']  # use the max mean amplitude as the quasi magnitude
    
    # CC_amp_max_mean = Events_classified['max_RAWCCs'] / check_lib['amp_max_mean']
    #?+++++ perform scaling factor for max_RAWCCs to peak amp ++++++++++++++++++++++++++++++
    for ii, row in check_lib.iterrows():
        Template = row['infile_name']
        index = Events_classified['Template'] == Template
        #选择一种平均的幅度（先对数后平均 或者 先平均后对数）
        # fac_CC2amp = 10**(row['amp_log_mean']) / row['self_CC'] # 先对数后平均
        fac_CC2amp = row['amp_max_mean'] / row['self_CC'] # 先平均后对数

        Events_classified.loc[index, 'fac_CC2amp'] = fac_CC2amp
        Events_classified.loc[index, 'Quasi_Mag'] = np.log10(abs(Events_classified.loc[index, 'max_RAWCCs'] * fac_CC2amp))
        print(f'{Template} fac_CC2amp: {fac_CC2amp:3e}')
    print(f" ------- finished scaling factor for max_RAWCCs to peak amp -------")


#? Get in real magnitude for refering the Quasi_Mag to Real_Mag;
#*---------- 1. Read IMS data and get the real magnitude
IN_IMS_path = f"{HOME_dir}IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt"
STA_name = "FSK"  # 需要查询的台站
Record_dict_ls = extract_station_info(IN_IMS_path, STA_name)
df_IMS = pd.DataFrame(Record_dict_ls)

#*---------- 2. Read template events position and lock the surrounding area
# temp_lat_lon_fname = f'{PROJECT_ROOT}/Input_template_set/template_lat_lon.csv'
# df_temp = pd.read_csv(temp_lat_lon_fname, sep='\t', parse_dates=['EpiTime'])
# #template_name	Latitude	Longitude	Magnitude	Depth	Distance	EpiTime
# print(df_temp)
#?----> now instead we use Temp_lib_df, which contains this part of info



# first value the Real_Mag column with None
Events_classified['Real_Mag'] = None  # 初始化 Real_Mag 列为 None
#*---------- 3. For each template, get MF time and search FSK time in surrounding area. Match them, concat their info
save_path = OUTPUT_dir + 'Events_with_NOA_Mag/' + 'Quasi_Mag_to_NOA_Mag.png'
fig, axes = plt.subplots(len(Temp_name_list), 1, figsize=(10, 3 * len(Temp_name_list)), sharex=False)
fig.suptitle('Quasi-Mag to Real-Mag Estimation', fontsize=16)

for ii, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    ax = axes[ii]
    
    print('\n')
    #?++++++++++++++++++++++++++++
    t_ERR = 5
    #?++++++++++++++++++++++++++++
    df_NOAmag = _find_Template_EQs_in_IMS(df_IMS,Template,Events_classified,Temp_lib_df,t_ERR)
    # print('===>df_NOAmag:', df_NOAmag.shape, df_NOAmag.columns, df_NOAmag)
    
    #*---------- 4. Save the portion of Events_classified with real magnitude matched, for scaling
    df_NOAmag.to_csv(OUTPUT_dir + 'Events_with_NOA_Mag/' + Template + '_Events_with_NOA_Mag.csv', sep='\t', index=False)

    #? Check and Est. the relation from Quasi_Mag to NOA_Mag for each template
    if len(df_NOAmag) == 0:
        Quasi_to_Real_OK = 0
        print(f'No know events can be used for {disp_Template}')
    elif len(df_NOAmag) < 3: 
        Quasi_to_Real_OK = 0
        print(f'{len(df_NOAmag)} events can be used for {disp_Template}, Less than 5, error too large')
    else:
        Quasi_to_Real_OK = 1
        print(f'{len(df_NOAmag)} events can be used for {disp_Template}, more than 5, OKOK HAPPY')
        
        #!-------plot-------
        title = f'Linear Regression from Quasi-Mag. to Real-Mag. of <{disp_Template}>'
        slope, intercept = least_squares_fit(df_NOAmag['Quasi_Mag'], df_NOAmag['EQ_magnitude'], ax, title)

        minQuasiMag = min(Events_classified['Quasi_Mag'])
        maxQuasiMag = max(Events_classified['Quasi_Mag'])
        # ax.set_xlim([-8.8,-4.5])
        ax.set_xlim([minQuasiMag-0.5, maxQuasiMag+0.5])
        

    #? Map the Quasi_Mag to Real_Mag use the relation from Quasi_Mag to NOA_Mag, if the relation holds
    if Quasi_to_Real_OK == 1:
        print(f'\033[33m {disp_Template} to est. real mag or not: {Quasi_to_Real_OK} \033[0m')
        print(f'\033[34m regression:  NOA_Mag = {slope} x Quasi_Mag + {intercept} \033[0m')
        index = Events_classified['Template'] == Template
        Events_classified.loc[index, 'Real_Mag'] = Events_classified.loc[index, 'Quasi_Mag'] * slope + intercept
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应标题
fig.savefig(save_path, dpi=300)
#? save the Events_classified after est. real mag
Event_classified_fname = OUTPUT_dir + 'Events_classified_Real_Mag.csv'
Events_classified.to_csv(Event_classified_fname, sep='\t', index=False)


#!________________OUTPUT_dir_byTime: stores GR law plot and other plots with mag info, specified by t_start and t_end____________________
OUTPUT_dir_GR = OUTPUT_dir + 'GR_law_check/'
os.makedirs(OUTPUT_dir_GR, exist_ok=True)

GR_t0 = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')#('2024-04-24 10:40:00', '%Y-%m-%d %H:%M:%S') data gap in 04-24 10:04:00 to 10:15:00
GR_t1 = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
GR_t_str = '_' + GR_t0.strftime('%Y-%m-%d %H:%M:%S') + '_' + GR_t1.strftime('%Y-%m-%d %H:%M:%S')

# OUTPUT_dir_GR_byTime = OUTPUT_dir_GR + 'sub' + GR_t_str + '/'
OUTPUT_dir_GR_byTime = OUTPUT_dir_GR

# if not os.path.exists(OUTPUT_dir_GR_byTime + 'Real_Mag/'):
#     os.makedirs(OUTPUT_dir_GR_byTime + 'Real_Mag/')
# if not os.path.exists(OUTPUT_dir_GR_byTime + 'Quasi_Mag/'):
#     os.makedirs(OUTPUT_dir_GR_byTime + 'Quasi_Mag/')
    
#? Read the Events_classified_Real_Mag.csv and plot the GR law
Event_classified_fname = OUTPUT_dir + 'Events_classified_Real_Mag.csv'
Events_classified = pd.read_csv(Event_classified_fname, sep='\t', parse_dates=['Rt_trig','Rt_left','Rt_right','t_trig', 'width_trig', 't_left', 't_right'])
Events_classified = Events_classified[ (Events_classified['Rt_trig'] >= GR_t0) & (Events_classified['Rt_trig'] <= GR_t1) ]
# print(f'====> Events_classified cut from {GR_t0} to {GR_t1}:\n {Events_classified}')


fig1, axes1 = plt.subplots(len(Temp_name_list), 1, figsize=(10, 3 * len(Temp_name_list)), sharex=False)
OUT_fname_1 = OUTPUT_dir_GR_byTime + 'Real_Mag_GR_law.png'

fig2, axes2 = plt.subplots(len(Temp_name_list), 1, figsize=(10, 3 * len(Temp_name_list)), sharex=False)
OUT_fname_2 = OUTPUT_dir_GR_byTime + 'Quasi_Mag_GR_law.png'

for ii, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    
    print(f'\033[33m ===>Template: {disp_Template} \033[0m')
    index = Events_classified['Template'] == Template
    
    if not Events_classified.loc[index, 'Real_Mag'].isnull().all():
        print(f'\033[33m Has Real_Mag for {disp_Template}, playing Real_Mag and Quasi_Mag\033[0m')

        Mag = Events_classified.loc[index, 'Real_Mag'].values
        M_c = min(Mag)  # 取震级的最小值作为 M_c, 即采用所有数据点
        
        label = f'Est. Real Magnitude'
        title = f'Gutenberg-Richter Law of {disp_Template}'
        
        #!____this is continuous G-R law version, without bin spliting, to fully check the data points
        ax = axes1[ii]
        b_value, intercept = from_catalog_to_GR_law(Mag, M_c, ax,label,title)
        ax.set_xlim([-0.7,4])
        
    else:
        print(f'No Real_Mag for {disp_Template}, playing Quasi_Mag only')

    QuasiMag = Events_classified.loc[index, 'Quasi_Mag'].values
    if Mag.size == 0:
        print(f'No Quasi_Mag for {disp_Template}, skip')
        continue
    
    M_c = min(QuasiMag)  # 取震级的最小值作为 M_c, 即采用所有数据点
    label = f'Quasi_Magitude from log10(CCs)'
    title = f'(Quasi-) Gutenberg-Richter Law of {disp_Template}'
    
    print(f"compute for quasi magnitude = log10(PeakAmp)")
    ax = axes2[ii]
    b_value, intercept = from_catalog_to_GR_law(QuasiMag, M_c, ax,label,title)
    
    minQuasiMag = min(Events_classified['Quasi_Mag'])
    maxQuasiMag = max(Events_classified['Quasi_Mag'])
    # ax.set_xlim([-8.8,-4.5])
    ax.set_xlim([minQuasiMag-0.5, maxQuasiMag+0.5])
    
fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(OUT_fname_1, dpi=300)
fig2.savefig(OUT_fname_2, dpi=300)



#* EXTRA check the correaltion between Quasi_Mag and Quasi_Mag_Amp
os.makedirs(OUTPUT_dir + 'max_RAWCCs_vs_peakAmp/', exist_ok=True)
save_path = OUTPUT_dir + 'max_RAWCCs_vs_peakAmp/' + 'max_RAWCCs_vs_peakAmp.png'
fig, axes = plt.subplots(len(Temp_name_list), 1, figsize=(8, 6 * len(Temp_name_list)), sharex=False)
fig.suptitle('max_RAWCCs to peak_Amp Relations', fontsize=16)

for ii, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    index = Events_classified['Template'] == Template
    
    ax = axes[ii]
    print(f'\n processing {Template} as {disp_Template}')    
    
    ax.plot(Events_classified.loc[index,'Quasi_Mag'], Events_classified.loc[index, 'Quasi_Mag_Amp'], 'o', markersize=2, alpha=0.3)
    ax.set_xlabel('Log10(Amplitude scaling from Cross-Correlation)')
    ax.set_ylabel(' Log10(PeakAmp)')
    ax.set_title(f' of {disp_Template}')
    ax.grid(True)
    
    ax.axhline(np.log10(20e-9), color='red', linestyle='--', linewidth=1, label='peak = 20 nanostrain')
    
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应标题
fig.savefig(save_path)