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


import os
import gc
from IPython.display import clear_output
import pickle
import re
import glob

import h5py, json
from tqdm import tqdm
import time
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count

import sys
sys.path.append(f"{PROJECT_ROOT}/functions/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from readers import Read_CCs_stack_h5,Read_CCs_trig_pkl, Read_CCs_trig_pkl_toPANDAS
from readers import write_pkl, load_pkl, write_h5, load_h5
from readers import Read_DAS_data, Read_template
# -


def Remove_Temp_Offest_forSeries(SRt_trig, Template, DF_Temp_Offset, offset_Type = 'S'):
    """
    Args:
        SRt_trig (pd.series): triggered time in the frame of CCs_s time
        Template (str): template name
        DF_Temp_Offset (pd.DataFrame): template offset dataframe
        offset_Type (str): 'S' or 'P', S for S_offset, P for P_offset
    Returns:
        Rt_trig: (pd.series): Real triggered time in the actual world and waterfall plot
    """
    if offset_Type == 'S':
        Offset_sec = DF_Temp_Offset['S_offset'][ DF_Temp_Offset['template_name'] == Template ]
    elif offset_Type == 'P':
        Offset_sec = DF_Temp_Offset['P_offset'][ DF_Temp_Offset['template_name'] == Template ]
    else:
        raise ValueError(f"Please check offset_Type. Expected 'S' or 'P'. And {Template} in DF_Temp_Offset")
    Offset_sec = Offset_sec.iloc[0]
    
    if SRt_trig.empty:
        return pd.Series(dtype="float64")  # 避免 FutureWarning
    else:
        # 将 Offset_sec 转换为 Timedelta 类型，并加到 SRt_trig 上
        Rt_trig = SRt_trig + pd.to_timedelta(Offset_sec, unit='s') 
        return Rt_trig


def Collapse_with_MAX_CCs(Collapse_group, IN_one_event_interval):
    """
    group by Rt_trig
    pick by CCs_s_trig_ofMAD maximum
    """
    Collapse_group['Belong_or_not'] = 0 # initial set to 0

    # 按时间排序并不保留原始索引
    Collapse_group = Collapse_group.sort_values(by=['Rt_trig'], ascending=True, ignore_index=True)

    # initial set to False, 标记新组开始（第一个事件默认新组）
    Collapse_group['is_new_group'] = False 
    Collapse_group.loc[0, 'is_new_group'] = True

    # 计算时间差, type 1, Rt_trig difference
    delta = Collapse_group['Rt_trig'].diff()
    
    # 计算时间差, type 2, the (Rt_left[i] - Rt_right[i-1])
    # delta = Collapse_group['Rt_left'].shift(1) - Collapse_group['Rt_right']
    
    print(len(delta), len(Collapse_group['Rt_trig']))
    # 标记新组开始
    Collapse_group.loc[1:, 'is_new_group'] = delta[1:] > pd.to_timedelta(IN_one_event_interval, unit='s') 

    # 生成组编号
    Collapse_group['group'] = Collapse_group['is_new_group'].cumsum()

    # 找到每组中绝对值的最大行索引（在排序后的DataFrame中）
    max_idx = Collapse_group.groupby('group')['CCs_s_trig_ofMAD'].apply(lambda x: x.abs().idxmax())

    # -----不使用原始索引
    # 提取原始索引并标记到原始DataFrame
    # Collapse_group.loc[Collapse_group.loc[max_idx, 'index'], 'Belong_or_not'] = 1
    #- **直接标记最大CCs_s_trig所在行**
    Collapse_group.loc[max_idx, 'Belong_or_not'] = 1

    return Collapse_group


import pandas as pd

def Read_Collapse_group_df(IN_CCs_trig_dir, Temp_name_list, DF_Temp_Offset, MAD_array_dict, t_start, t_end):
    
    # 1. 初始化一个空“列表”来存储 DataFrames，而不是一个空的 DataFrame。
    #    这样做效率高得多。
    processed_dfs = []

    for tt, Template in enumerate(Temp_name_list):
        #? 读取 CCs_s_trig 数据
        CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right, max_RAWCCs, t_arg_max, max_stalta \
            = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
        
        DF_trig = pd.DataFrame({
            'CCs_s_trig': CCs_s_trig,
            'RAWCCs_s_trig': RAWCCs_s_trig,
            't_trig': t_trig,
            'width_trig': width_trig,
            't_left': t_left,
            't_right': t_right,
            'max_RAWCCs': max_RAWCCs,
            't_arg_max': t_arg_max,
            'max_stalta': max_stalta
        })
        
        # 2. 检查 DataFrame 是否为空。如果是，就跳过当前模板。
        #    这是修复 ValueError 的关键步骤。
        if DF_trig.empty:
            print(f"--> 信息: 在给定的时间窗口内，模板 '{Template}' 没有找到任何检测事件。已跳过。")
            continue

        # 3. 使用 .assign() 以一种更清晰的链式方式创建所有新列。
        DF_trig = DF_trig.assign(
            Rt_trig = Remove_Temp_Offest_forSeries(DF_trig['t_trig'], Template, DF_Temp_Offset, offset_Type='S'),
            Rt_left = Remove_Temp_Offest_forSeries(DF_trig['t_left'], Template, DF_Temp_Offset, offset_Type='S'),
            Rt_right = Remove_Temp_Offest_forSeries(DF_trig['t_right'], Template, DF_Temp_Offset, offset_Type='S'),
            CCs_s_trig_ofMAD = DF_trig['CCs_s_trig'].astype('float64') / MAD_array_dict[Template][0],
            Template = Template
        )
        
        # 4. 将完整处理后的 DataFrame 添加到我们的列表中。
        processed_dfs.append(DF_trig)
    
    # 5. 循环结束后，如果列表为空，说明没有任何模板找到了事件。
    if not processed_dfs:
        print("--> 警告: 所有模板都没有找到任何检测事件。将返回一个空的 DataFrame。")
        return pd.DataFrame()

    # 6. 一次性将列表中的所有 DataFrames 合并起来。
    Collapse_group = pd.concat(processed_dfs, ignore_index=True)
    
    print(f'====> 从 {len(processed_dfs)} 个模板中找到了检测事件。总事件数: {len(Collapse_group)}')
    
    # 7. 对最终的 DataFrame 进行排序。'ignore_index=True' 会重新生成从 0 到 N-1 的新索引。
    Collapse_group = Collapse_group.sort_values(by=['Rt_trig'], ascending=True, ignore_index=True)
    
    return Collapse_group


# #!________________UNUSED, OLD BELOW_________________-
        
# def Est_Mag_Bias(OUT_dict):
#     """
#     """
#     if OUT_dict == {}:
#         print('(4) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict['t_trig_s']) == 0:
#         print('(4) No event is found')
#         return {}
    
#     #*TASK 1 -- Estimate magnitude through peak amplitude（1) and cross-correlation value (2 and 3) (unnormalized)
#     #! Magnitude value is invalid, not calibrated, just log value
#     # 1) Take the mean value of the peak amplitude of all channels
#     OUT_dict['quasi_Mag_Amp'] = np.log10(np.mean(abs(OUT_dict['datamax_f']), axis=1))
#     # 2) Take the mean value of the peak value of [all channels cross-correlation stacked series]
#     OUT_dict['quasi_Mag_CCs_s'] = np.log10(abs(OUT_dict['RAWCCs_trig_s']))
#     # 3) Take the mean value of the mean of [peak value of each channel cross-correlation series]
#     OUT_dict['quasi_Mag_CCs_f'] = np.log10(np.mean(abs(OUT_dict['RAWCCsmax_f']), axis=1))
    
#     #*TASK 2 -- Quantify bias level with each channel travel time shift
#     # OUT_dict['Bias_level_arrivals']
#     Bias_level_arrivals = np.zeros(len(OUT_dict['t_trig_s']))
#     for i in range(len(OUT_dict['t_trig_s'])):
#         t_tmp = OUT_dict['t_trig_s'][i]
#         t_f_tmp = OUT_dict['t_CCsmax_f'][:,i]
#         t_diff = np.array([(t_f_tmp[j] - t_tmp).total_seconds() for j in range(len(t_f_tmp))])
#         Bias_level_arrivals[i] = np.mean(abs(t_diff))
#     OUT_dict['Bias_level_arrivals'] = Bias_level_arrivals
    
#     #*TASK 3 -- delete the elements in OUT_dict which will not be used, and take much memory
#     del OUT_dict['t_CCsmax_f'], OUT_dict['CCsmax_f'], OUT_dict['RAWCCsmax_f']
#     del OUT_dict['t_datamax_f'], OUT_dict['datamax_f']
#     return OUT_dict

# def Classification_by_shift(OUT_dict_fromAllTemplates, In_one_event_interval):
#     """
#     """
#     if OUT_dict_fromAllTemplates == {}:
#         print('(6) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict_fromAllTemplates['t_trig_s']) == 0:
#         print('(6) No event is found')
#         return {}
    
#     t_trig_s = OUT_dict_fromAllTemplates['t_trig_s']
#     Bias_level_arrivals = OUT_dict_fromAllTemplates['Bias_level_arrivals']
#     # rank the t_trig_s in datetime order
#     idx_sort = np.argsort(t_trig_s) # from small to large
#     t_rank = t_trig_s[idx_sort]
#     Bias_rank = Bias_level_arrivals[idx_sort]
#     for i in range(len(t_rank)-1):
#         if (t_rank[i+1] - t_rank[i]).total_seconds() < In_one_event_interval:
#             if Bias_rank[i+1] < Bias_rank[i]:
#                 Bias_rank[i] = -1   # mark the one with larger Bias level as collapsed
#             elif Bias_rank[i+1] >= Bias_rank[i]:
#                 Bias_rank[i+1] = -1 # mark the one with larger Bias level as collapsed
#         else:
#             continue
#     idx_valid = np.where(Bias_rank != -1) # the collapsed are marked as -1
#     if type(idx_valid) == tuple:
#         idx_valid = idx_valid[0]
#     elif type(idx_valid) == np.ndarray:
#         pass
    
#     t_valid = t_rank[idx_valid]
#     Bias_valid = Bias_rank[idx_valid]
    
#     OUT_dict_Classified = {}
#     OUT_dict_Classified['t_trig_s'] = t_valid
#     OUT_dict_Classified['Bias_level_arrivals'] = Bias_valid
#     OUT_dict_Classified['quasi_Mag_Amp'] = OUT_dict_fromAllTemplates['quasi_Mag_Amp'][idx_sort][idx_valid]
#     OUT_dict_Classified['quasi_Mag_CCs_s'] = OUT_dict_fromAllTemplates['quasi_Mag_CCs_s'][idx_sort][idx_valid]
#     OUT_dict_Classified['quasi_Mag_CCs_f'] = OUT_dict_fromAllTemplates['quasi_Mag_CCs_f'][idx_sort][idx_valid]
#     OUT_dict_Classified['CCs_trig_s'] = OUT_dict_fromAllTemplates['CCs_trig_s'][idx_sort][idx_valid]
#     OUT_dict_Classified['RAWCCs_trig_s'] = OUT_dict_fromAllTemplates['RAWCCs_trig_s'][idx_sort][idx_valid]
#     OUT_dict_Classified['Template'] = OUT_dict_fromAllTemplates['Template'][idx_sort][idx_valid]
#     return OUT_dict_Classified


# def Classification_by_CCs(OUT_dict_fromAllTemplates, In_one_event_interval):
#     """
#     """
#     if OUT_dict_fromAllTemplates == {}:
#         print('(10) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict_fromAllTemplates['t_trig_s']) == 0:
#         print('(10) No event is found')
#         return {}
#     MAD_dict = get_template_CCs_MAD(1.0)
    
#     t_trig_s = OUT_dict_fromAllTemplates['t_trig_s']
#     CCs_trig_s = OUT_dict_fromAllTemplates['CCs_trig_s']
    
#     idx_sort = np.argsort(t_trig_s) # from small to large
#     t_rank = t_trig_s[idx_sort]
#     CCs_rank = abs(CCs_trig_s[idx_sort])
#     for i in range(len(t_rank)-1):
#         if (t_rank[i+1] - t_rank[i]).total_seconds() < In_one_event_interval:
#             MADi = MAD_dict[OUT_dict_fromAllTemplates['Template'][idx_sort[i]]]
#             MADi1 = MAD_dict[OUT_dict_fromAllTemplates['Template'][idx_sort[i+1]]]
#             if CCs_rank[i+1]/MADi1 < CCs_rank[i]/MADi:
#                 CCs_rank[i+1] = -1   # mark the one with larger Bias level as collapsed
#             elif CCs_rank[i+1]/MADi1 >= CCs_rank[i]/MADi:
#                 CCs_rank[i] = -1 # mark the one with larger Bias level as collapsed
#         else:
#             continue
#     idx_valid = np.where(CCs_rank != -1) # the collapsed are marked as -1
#     if type(idx_valid) == tuple:
#         idx_valid = idx_valid[0]
#     elif type(idx_valid) == np.ndarray:
#         pass
    
#     t_valid = t_rank[idx_valid]
#     CCs_valid = CCs_rank[idx_valid]
    
#     OUT_dict_Classified = {}
#     OUT_dict_Classified['t_trig_s'] = t_valid
#     OUT_dict_Classified['Bias_level_arrivals'] = OUT_dict_fromAllTemplates['Bias_level_arrivals'][idx_sort][idx_valid]
#     OUT_dict_Classified['quasi_Mag_Amp'] = OUT_dict_fromAllTemplates['quasi_Mag_Amp'][idx_sort][idx_valid]
#     OUT_dict_Classified['quasi_Mag_CCs_s'] = OUT_dict_fromAllTemplates['quasi_Mag_CCs_s'][idx_sort][idx_valid]
#     OUT_dict_Classified['quasi_Mag_CCs_f'] = OUT_dict_fromAllTemplates['quasi_Mag_CCs_f'][idx_sort][idx_valid]
#     OUT_dict_Classified['CCs_trig_s'] = CCs_valid
#     OUT_dict_Classified['RAWCCs_trig_s'] = OUT_dict_fromAllTemplates['RAWCCs_trig_s'][idx_sort][idx_valid]
#     OUT_dict_Classified['Template'] = OUT_dict_fromAllTemplates['Template'][idx_sort][idx_valid]
#     return OUT_dict_Classified
    

# def Exlude_Outliers(OUT_dict_OneTemp):
#     """
#     """
#     if OUT_dict_OneTemp == {}:
#         print('(8) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict_OneTemp['Template']) == 0:
#         print('(8) No event is found')
#         return {}
    
#     # check if all events from one template
#     Temp_name = OUT_dict_OneTemp['Template'][0]
#     result = all(item == Temp_name for item in OUT_dict_OneTemp['Template'])
#     if result == False:
#         print('WARNING: The events are not from the same template')
#         return {}
    
#     # check if Bias_level_arrivals is non-negative
#     if any(OUT_dict_OneTemp['Bias_level_arrivals'] < 0):
#         print('WARNING: Bias_level_arrivals should be non-negative')
#         return {}
    
#     # make statistics on the Bias_level_arrivals
#     Bias = OUT_dict_OneTemp['Bias_level_arrivals']
#     # compute 25% and 75% quantile
#     Q1 = np.percentile(Bias, 25)
#     Q3 = np.percentile(Bias, 75)
#     IQR_ = Q3 - Q1
#     upper_bound = Q3 + 1.5 * IQR_
#     lower_bound = 0 # because Bias_level_arrivals should be positive
    
#     idx_valid = np.where(Bias <= upper_bound)
#     if type(idx_valid) == tuple:
#         idx_valid = idx_valid[0]
#     elif type(idx_valid) == np.ndarray:
#         pass
    
#     OUT_dict_OneTemp_cleaned = {}
#     OUT_dict_OneTemp_cleaned['t_trig_s'] = OUT_dict_OneTemp['t_trig_s'][idx_valid]
#     OUT_dict_OneTemp_cleaned['Bias_level_arrivals'] = OUT_dict_OneTemp['Bias_level_arrivals'][idx_valid]
#     OUT_dict_OneTemp_cleaned['quasi_Mag_Amp'] = OUT_dict_OneTemp['quasi_Mag_Amp'][idx_valid]
#     OUT_dict_OneTemp_cleaned['quasi_Mag_CCs_s'] = OUT_dict_OneTemp['quasi_Mag_CCs_s'][idx_valid]
#     OUT_dict_OneTemp_cleaned['quasi_Mag_CCs_f'] = OUT_dict_OneTemp['quasi_Mag_CCs_f'][idx_valid]
#     OUT_dict_OneTemp_cleaned['CCs_trig_s'] = OUT_dict_OneTemp['CCs_trig_s'][idx_valid]
#     OUT_dict_OneTemp_cleaned['RAWCCs_trig_s'] = OUT_dict_OneTemp['RAWCCs_trig_s'][idx_valid]
#     OUT_dict_OneTemp_cleaned['Template'] = OUT_dict_OneTemp['Template'][idx_valid]
    
#     return OUT_dict_OneTemp_cleaned

# def Exclude_Outliers_forAllTemp(OUT_dict_Classified, Temp_name_list):
#     """
#     """
#     if OUT_dict_Classified == {}:
#         print('(9) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict_Classified['Template']) == 0:
#         print('(9) No event is found')
#         return {}
    
#     OUT_dict_Classified_cleaned = {}
#     for Temp_name in Temp_name_list:
#         OUT_dict_OneTemp = Read_OneTemp_from_OUT_dict_Classified(OUT_dict_Classified, Temp_name)
#         OUT_dict_OneTemp_cleaned = Exlude_Outliers(OUT_dict_OneTemp)
#         OUT_dict_Classified_cleaned = Joint_dict_elem(OUT_dict_Classified_cleaned, OUT_dict_OneTemp_cleaned, 'full_chan_info_removed')
#         del OUT_dict_OneTemp, OUT_dict_OneTemp_cleaned
#     return OUT_dict_Classified_cleaned

# def Read_IniTrig_and_Retrig_and_Est_Mag_Bias_and_Classification(t_start, t_end, Temp_name_list, Temp_offset_dict, Retrig_threshold, In_one_event_interval, IN_initial_trig_dir, IS_save_traveltime_bychan=False):
#     """
#     Read the CCs_trig files and classify the events by the CCsmax and datamax.
#     """    
    
#     OUT_dict_fromAllTemplates = dict()
#     for Template in Temp_name_list:
#         OUT_dict = Read_IniTrig(t_start, t_end, Template, IN_initial_trig_dir)
#         # get the key Template since Read_IniTrig() 
#         OUT_dict = Retrig(OUT_dict, Retrig_threshold, Template)
        
#         if OUT_dict == {} or len(OUT_dict['t_trig_s']) == 0:
#             continue
#         OUT_dict = Est_Mag_Bias(OUT_dict)
#         OUT_dict = Remove_template_time_offset(OUT_dict, Template, Temp_offset_dict, offset_Type = 'S')

#         OUT_dict_fromAllTemplates = Joint_dict_elem(OUT_dict_fromAllTemplates, OUT_dict, 'full_chan_info_removed')
#         del OUT_dict
    
#     if OUT_dict_fromAllTemplates == {}:
#         print('No event is found')
#         return {}

#     OUT_dict_Classified = Classification_by_shift(OUT_dict_fromAllTemplates, In_one_event_interval)
#     # OUT_dict_Classified = Classification_by_CCs(OUT_dict_fromAllTemplates, In_one_event_interval)
#     # OUT_dict_Classified_cleaned = Exclude_Outliers_forAllTemp(OUT_dict_Classified, Temp_name_list)
#     OUT_dict_Classified_cleaned = OUT_dict_Classified
    
#     count = len(OUT_dict_Classified_cleaned['t_trig_s'])
#     print(f'>> # {count} found for {t_start}~{t_end}')
#     return OUT_dict_Classified_cleaned


# def Retrig(OUT_dict,Retrig_threshold, Template):
    
#     """
#     Args:
#         OUT_dict (dict):
#             the output dict from the matched filter, only of one template
#         Retrig_threshold (dict):
#         {
#             't_array': np.array(datetime),
#             'Retrig_threshold_level': float,
#             'template_0423_80Hz': np.array(float), MAD_array of the template matched with t_array
#             ...
#             'template_18_80Hz': np.array(float)
#         }
#     """
#     if Retrig_threshold['IS_retrig'] == False:
#         return OUT_dict
    
#     if OUT_dict == {}:
#         print('(1) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict['CCs_trig_s']) == 0:
#         print('(1) No event is found')
        
#     # print('--- Retriggering ---')
#     t_array = Retrig_threshold['t_array']
#     threshold = Retrig_threshold[Template] * Retrig_threshold['Retrig_threshold_level']
        
#     for i,t_trig in enumerate(OUT_dict['t_trig_s']):
#         # find t_trig between which two element of t_array
#         idx_ = np.where(t_array<=t_trig)[0]
#         idx = idx_[-1]
#         # print(idx,idx_,threshold[idx])
#         # print(OUT_dict['CCs_trig_s'][i])
#         if abs(OUT_dict['CCs_trig_s'][i]) < threshold[idx]:
#             OUT_dict['CCs_trig_s'][i] = -1

#     ev_idx = np.where(OUT_dict['CCs_trig_s'] != -1)        
    
#     if type(ev_idx) == tuple:
#         ev_idx = ev_idx[0]
#     elif type(ev_idx) == np.ndarray:
#         pass
    
#     if len(ev_idx) == 0:
#         # print('Retriggered get nothing left')
#         return {}
#     OUT_dict_retrig = {}
#     OUT_dict_retrig['t_trig_s'] = OUT_dict['t_trig_s'][ev_idx]
#     OUT_dict_retrig['CCs_trig_s'] = OUT_dict['CCs_trig_s'][ev_idx]
#     OUT_dict_retrig['RAWCCs_trig_s'] = OUT_dict['RAWCCs_trig_s'][ev_idx]
#     OUT_dict_retrig['t_CCsmax_f'] = OUT_dict['t_CCsmax_f'][ev_idx]
#     OUT_dict_retrig['CCsmax_f'] = OUT_dict['CCsmax_f'][ev_idx]
#     OUT_dict_retrig['RAWCCsmax_f'] = OUT_dict['RAWCCsmax_f'][ev_idx]
#     OUT_dict_retrig['t_datamax_f'] = OUT_dict['t_datamax_f'][ev_idx]
#     OUT_dict_retrig['datamax_f'] = OUT_dict['datamax_f'][ev_idx]
#     # repeat the template name
#     OUT_dict_retrig['Template'] = np.array(list(np.repeat(Template, len(ev_idx))), dtype='str')
    
        
#     return OUT_dict_retrig


# def Read_IniTrig(t_start, t_end, Template, IN_initial_trig_dir):
#     """
#     """
#     file_gap = timedelta(minutes=10)
#     # 提取文件名中的时间，格式为 UTC_YYYYMMDD_HHMMSS.ssssss
#     time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")

#     subdir = Template
#     path = os.path.join(IN_initial_trig_dir, subdir)
#     pkl_files = glob.glob(os.path.join(path, 'CCs_trig_*.pkl'))
#     OUT_dict = {}
#     # print('>> #', len(pkl_files), '>>', Template)
#     for file in pkl_files:
#         # 从文件名中解析时间
#         match = time_pattern.search(file)
#         if match:
#             date_str, time_str, mmsec = match.groups()
#             file_t0 = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
#             file_t1 = file_t0 + file_gap
#             if not (t_start >= file_t1 or t_end <= file_t0):
#                 OUT_dict_tmp = load_pkl(file)
#                 OUT_dict_tmp = Convert_list_to_ndarray_in_OUT_dict(OUT_dict_tmp)
#                 OUT_dict_tmp.pop('Mad_CCs_10min', None)
#                 #! MAD value is safely removed, we can quickly access MAD by time and by template via outer scripts
                
#                 OUT_dict = Joint_dict_elem(OUT_dict, OUT_dict_tmp, 'full_chan_info')
#                 del OUT_dict_tmp

#     if OUT_dict == {}:
#         print(f'\t>> # {0}\t >>{Template}')
#         return {}
#     else:
#         print(f'\t>> # {len(OUT_dict["t_trig_s"])}\t >>{Template}')
#     ##* print the number of events in each template
#     # if OUT_dict == {} or len(OUT_dict['t_trig_s']) == 0:
#     #     print(f'>> # {0}\t >>{Template}')
#     #     return {}
#     # else:
#     #     print(f'>> # {len(OUT_dict["t_trig_s"])}\t >>{Template}')
        
    
#     OUT_dict = Cut_time_window_OUT_dict(OUT_dict, t_start, t_end, 'full_chan_info')
    
#     OUT_dict = Add_temp_name_to_OUT_dict(OUT_dict, Template)
    
#     return OUT_dict
        