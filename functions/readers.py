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
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd



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

#-- To import a function on a relative path:
import sys
sys.path.append(f"{PROJECT_ROOT}/functions/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from compute_correlation import preprocess

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def write_pkl(OUT_dict_stacked,OUTPUT_file):
    with open(OUTPUT_file, 'wb') as f:  # ‘wb’ means write binary mode.
        pickle.dump(OUT_dict_stacked, f)


def write_h5(file_path, data_dict):
    """
    Saves the data to an HDF5 file.
    Only support for dict() with numpy_array as values. And the array.dtype should be float or int.

    Parameters:
    - file_path: str, the path of the HDF5 file to be saved
    - data_dict: dict, the data to be saved, in {key: numpy_array} format.
    """
    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.dtype == "O":
                # dtype=object 
                if all(isinstance(i, (int, float, np.number)) for i in value.flat):
                    value = value.astype(np.float32)  # transform dtype=int, float, np.number to float32
                elif all(isinstance(i, str) for i in value.flat):
                    value = np.array(value, dtype="S")  # transform dtype=str to bytes
                else:
                    print(f"! Unable to save {key}: mixed data types in object array")
                    continue
                
            # Create datasets with support for compression and chunking
            f.create_dataset(key, data=value)
    # print(f “Data successfully saved to {file_path}”)

def load_h5(file_path):
    """
    Read an HDF5 file and restore it to a Python dictionary.

    - Parse data of type np.ndarray
    - Recover dtype=‘S’ (byte strings) as normal strings
    - Compatible with the original float32 conversion
    
    Parameters.
    - file_path: str, HDF5 file path
    
    Returns.
    - data_dict: dict, {key: numpy_array}
    """
    data_dict = {}
    
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            dataset = f[key][()]  # 读取数据
            if dataset.dtype == np.dtype('S'):  
                # 如果是字节字符串，则转换回普通字符串
                dataset = np.array([s.decode('utf-8') for s in dataset], dtype=object)
            data_dict[key] = dataset

    return data_dict

    
    

def Read_template(Template, fs,f1,f2, dx,d1,d2,npy_dir):
    # DIR_templates = f'{PROJECT_ROOT}/Input_template_set/'
    DIR_templates = npy_dir
    temp = np.load(DIR_templates + Template + '.npy')
    # trim
    temp = temp[:,int(d1/dx):int(d2/dx)]
    temp = preprocess(temp,fs,f1,f2)
    return temp

def Read_DAS_data(t_start, t_end_extended, fs,f1,f2, dx,d1,d2):
    das_h5_mnt_raw = f'{PROJECT_ROOT}/IN_DAS_data/DATA_RAW_400Hz_8m/'
    das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
    DIR_das = das_h5_mnt_80Hz
    data, headers, axis = load_das_h5_PE2024.load_das_custom(t_start, t_end_extended, input_dir=DIR_das)
    data = data[:,int(d1/dx):int(d2/dx)]
    data = preprocess(data,fs,f1,f2)
    return data


def Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end):
    """
    ATTENTION: this function will be problematic when (t_end - t_start) mod file_gap != 0
    because in that case, the last file is a <10min file, but we still think we read a complete 10min file.
    
    SOLUTIONS? 
    """
    file_gap = timedelta(minutes=10)
    fs_CCs = 80 # frequency of CCs_s
    # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
    time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")
    
    subdir = Template
    path = os.path.join(IN_CCs_stack_dir, subdir)
    
    # 获取所有 .h5 文件
    h5_files = glob.glob(os.path.join(path, '*.h5'))

    # 定义时间戳匹配正则
    time_pattern = re.compile(r'(\d{6})_(\d{6})\.(\d{3})')

    # 用列表存储数据，提高性能
    data = []
    
    for file in h5_files:
        match = time_pattern.search(os.path.basename(file))  # 只解析文件名
        if match:
            date_str, time_str, mmsec = match.groups()
            file_t0 = pd.to_datetime(date_str + time_str, format="%y%m%d%H%M%S")  # 更健壮
            data.append((file_t0, file))

    # 将数据转换为 DataFrame，并排序
    df_files = pd.DataFrame(data, columns=['file_t0', 'file']).sort_values(by='file_t0')
    df_files['file_t1'] = df_files['file_t0'] + file_gap  # 添加结束时间列
    # find files between t_start and t_end
    df_files = df_files[(df_files['file_t0'] < t_end) & (df_files['file_t1'] > t_start)].reset_index(drop=True)
        
    CCs_s = np.array([])
    RAWCCs_s = np.array([])
    # 读取并拼接数据
    for ii, row in enumerate(df_files.itertuples(index=False)):
        file = row.file
        file_t0 = row.file_t0
        file_t1 = row.file_t1
        
        # 读取 HDF5 文件
        tmp_dict = load_h5(file)

        t0 = max(t_start, file_t0)
        t1 = min(t_end, file_t1)
        idx0 = int((t0 - file_t0).total_seconds() * fs_CCs)
        # idx1 = int((t1 - file_t0).total_seconds() * fs_CCs)
        # index = np.arange(idx0, idx1)

        print(f"---read CCs_stack_file: {ii+1}/{len(df_files)}: from {file_t0} to {file_t1}")
        if ii == len(df_files) - 1:
            idx1 = int((file_t1 - t1).total_seconds() * fs_CCs)
        else:
            idx1 = int((file_t1 - t1).total_seconds() * fs_CCs)  + 1
            
        index = np.arange(idx0, len(tmp_dict['CCs_s']) - idx1)
        
        CCs_s = np.concatenate((CCs_s, tmp_dict['CCs_s'][index])) # 1D array, do not specify axis
        RAWCCs_s = np.concatenate((RAWCCs_s, tmp_dict['RAWCCs_s'][index]))
    
    return CCs_s, RAWCCs_s  
    


#!!错，未对文件进行排序，而文件列表时而是乱的
def MESSUP_Read_CCs_stack_h5(IN_CCs_stack_dir, Template, t_start, t_end):
    """
    ATTENTION: this function will be problematic when (t_end - t_start) mod file_gap != 0
    because in that case, the last file is a <10min file, but we still think we read a complete 10min file.
    
    SOLUTIONS? 
    """
    file_gap = timedelta(minutes=10)
    fs_CCs = 80 # frequency of CCs_s
    # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
    time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")
    
    subdir = Template
    path = os.path.join(IN_CCs_stack_dir, subdir)
    h5_files = glob.glob(os.path.join(path, '*.h5'))
    
    CCs_s = np.array([])
    RAWCCs_s = np.array([])
    for file in h5_files:
        match = time_pattern.search(file)
        if match:
            date_str, time_str, mmsec = match.groups()
            file_t0 = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            file_t1 = file_t0 + file_gap
            if not (t_start > file_t1 or t_end < file_t0): # give up one end, the right end of each time series, the final dim=48001
                tmp_dict = load_h5(file)  # load the h5 file
                # print(f'read {file}')
                # cut the time window
                t0 = max(t_start, file_t0)
                t1 = min(t_end, file_t1)
                idx0 = int((t0 - file_t0).total_seconds() * fs_CCs)
                # idx1 = int((t1 - file_t0).total_seconds() * fs_CCs)
                # index = np.arange(idx0, idx1)
                idx1 = int((file_t1 - t1).total_seconds() * fs_CCs) + 1
                index = np.arange(idx0, len(tmp_dict['CCs_s']) - idx1)
                CCs_s = np.concatenate((CCs_s, tmp_dict['CCs_s'][index])) # 1D array, do not specify axis
                RAWCCs_s = np.concatenate((RAWCCs_s, tmp_dict['RAWCCs_s'][index]))
                
    return CCs_s, RAWCCs_s



#! 类似的使用pd进行读取版本，原版本不会乱序这个没有必要
# def NEW_Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end):
#     """
#     Read the CCs_trig_*.pkl file
#     """
#     file_gap = timedelta(minutes=10)
#     fs_CCs = 80 # frequency of CCs_s
#     # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
#     time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")
    
#     subdir = Template
#     path = os.path.join(IN_CCs_trig_dir, subdir)
    
#     # 获取所有 .pkl 文件
#     pkl_files = glob.glob(os.path.join(path, '*.pkl'))

#     # 定义时间戳匹配正则
#     time_pattern = re.compile(r'(\d{6})_(\d{6})\.(\d{3})')

#     # 用列表存储数据，提高性能
#     data = []
    
#     for file in pkl_files:
#         match = time_pattern.search(os.path.basename(file))  # 只解析文件名
#         if match:
#             date_str, time_str, mmsec = match.groups()
#             file_t0 = pd.to_datetime(date_str + time_str, format="%y%m%d%H%M%S")  # 更健壮
#             data.append((file_t0, file))

#     # 将数据转换为 DataFrame，并排序
#     df_files = pd.DataFrame(data, columns=['file_t0', 'file']).sort_values(by='file_t0')
#     df_files['file_t1'] = df_files['file_t0'] + file_gap  # 添加结束时间列
#     # find files between t_start and t_end
#     df_files = df_files[(df_files['file_t0'] < t_end) & (df_files['file_t1'] > t_start)].reset_index(drop=True)
        
        
#     t_trig = np.array([])
#     CCs_s_trig = np.array([])
#     RAWCCs_s_trig = np.array([])
#     width_trig = np.array([])
#     # 读取并拼接数据
#     for ii, row in df_files.iterrows():
#         file = row['file']
#         file_t0 = row['file_t0']
#         file_t1 = row['file_t1']

#         # 读取 HDF5 文件
#         tmp_dict = load_pkl(file)

#         t0 = max(t_start, file_t0)
#         t1 = min(t_end, file_t1)
#         idx0 = int((t0 - file_t0).total_seconds() * fs_CCs)
#         # idx1 = int((t1 - file_t0).total_seconds() * fs_CCs)
#         # index = np.arange(idx0, idx1)
#         # print(f"{ii}, {len(df_files)}, {file_t0}, {file_t1}, {t0}, {t1}")
#         if ii == len(df_files) - 1:
#             idx1 = int((file_t1 - t1).total_seconds() * fs_CCs)
#         else:
#             idx1 = int((file_t1 - t1).total_seconds() * fs_CCs)  + 1
#         index = np.arange(idx0, len(tmp_dict['CCs_s']) - idx1)
        
#         t_trig = np.concatenate((t_trig, tmp_dict['t_trig'])) # 1D array, do not specify axis
#         CCs_s_trig = np.concatenate((CCs_s_trig, tmp_dict['CCs_s_trig']))
#         RAWCCs_s_trig = np.concatenate((RAWCCs_s_trig, tmp_dict['RAWCCs_trig']))
#         width_trig = np.concatenate((width_trig, tmp_dict['width_trig']))
        
#         # cut the time window
#     index = np.where((t_trig >= t_start) & (t_trig <= t_end)) # index is a tuple, containing an array, can use index or index[0] as the indexes of arrays
#     return CCs_s_trig[index], RAWCCs_s_trig[index], t_trig[index], width_trig[index]
    

#!!也可以使用，即使未对文件进行排序，t_trig已经带有时间信息不会乱套
def Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end):
    """
    Read the CCs_trig_*.pkl file
    """
    file_gap = timedelta(minutes=10)
    fs_CCs = 80 # frequency of CCs_s
    # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
    time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")
    
    subdir = Template
    path = os.path.join(IN_CCs_trig_dir, subdir)
    h5_files = glob.glob(os.path.join(path, '*.pkl'))
    

    t_trig = np.array([])
    CCs_s_trig = np.array([])
    RAWCCs_s_trig = np.array([])
    t_left = np.array([])
    t_right = np.array([])
    width_trig = np.array([])
    
    max_RAWCCs = np.array([])
    t_arg_max = np.array([])
    
    max_stalta = np.array([])
    for file in h5_files:
        match = time_pattern.search(file)
        if match:
            date_str, time_str, mmsec = match.groups()
            file_t0 = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            file_t1 = file_t0 + file_gap
            if not (t_start >= file_t1 or t_end <= file_t0):
                # print(file)
                # tmpOUT_dict = load_pkl(file)  # load the pkl file
                
                with open(file, 'rb') as f:
                    try:
                        tmpOUT_dict = pickle.load(f)
                    except EOFError:
                        print(f"文件可能损坏，无法完整加载: {file}")
                
                # print(f'read {file}')
                # print(tmpOUT_dict.keys())
                t_trig = np.concatenate((t_trig, tmpOUT_dict['t_trig'])) # 1D array, do not specify axis
                CCs_s_trig = np.concatenate((CCs_s_trig, tmpOUT_dict['CCs_s_trig']))
                RAWCCs_s_trig = np.concatenate((RAWCCs_s_trig, tmpOUT_dict['RAWCCs_s_trig']))
                width_trig = np.concatenate((width_trig, tmpOUT_dict['width_trig']))
                t_left = np.concatenate((t_left, tmpOUT_dict['t_left']))
                t_right = np.concatenate((t_right, tmpOUT_dict['t_right'])) 
                
                max_RAWCCs = np.concatenate((max_RAWCCs, tmpOUT_dict['max_RAWCCs']))
                t_arg_max = np.concatenate((t_arg_max, tmpOUT_dict['t_arg_max']))
                
                max_stalta = np.concatenate((max_stalta, tmpOUT_dict['max_stalta']))
                
    
    # cut the time window
    index = np.where((t_trig >= t_start) & (t_trig <= t_end)) # index is a tuple, containing an array, can use index or index[0] as the indexes of arrays
    
    return CCs_s_trig[index], RAWCCs_s_trig[index], t_trig[index], width_trig[index], t_left[index], t_right[index], max_RAWCCs[index], t_arg_max[index], max_stalta[index]
    
def Read_CCs_trig_pkl_toPANDAS(IN_CCs_trig_dir, Template, t_start, t_end):
    """
    Read the CCs_trig_*.pkl file
    """
    file_gap = timedelta(minutes=10)
    fs_CCs = 80 # frequency of CCs_s
    # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
    time_pattern = re.compile(r"UTC_(\d{8})_(\d{6})(\.\d+)?")
    
    subdir = Template
    path = os.path.join(IN_CCs_trig_dir, subdir)
    h5_files = glob.glob(os.path.join(path, '*.pkl'))
    

    # t_trig = np.array([])
    # CCs_s_trig = np.array([])
    # RAWCCs_s_trig = np.array([])
    # t_left = np.array([])
    # t_right = np.array([])
    # width_trig = np.array([])
    DF_trig = pd.DataFrame(columns=['t_trig', 'CCs_s_trig', 'RAWCCs_s_trig', 'width_trig', 't_left', 't_right'])
    for file in h5_files:
        match = time_pattern.search(file)
        if match:
            date_str, time_str, mmsec = match.groups()
            file_t0 = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            file_t1 = file_t0 + file_gap
            if not (t_start >= file_t1 or t_end <= file_t0):
                # print(file)
                # tmpOUT_dict = load_pkl(file)  # load the pkl file
                
                with open(file, 'rb') as f:
                    try:
                        tmpOUT_dict = pickle.load(f)
                    except EOFError:
                        print(f"文件可能损坏，无法完整加载: {file}")
                
                # print(f'read {file}')
                # print(tmpOUT_dict.keys())
                if tmpOUT_dict:  # 只有 tmpOUT_dict 非空时才拼接
                    # DF_trig = pd.concat([DF_trig, pd.DataFrame(tmpOUT_dict)], ignore_index=True)
                    DF_trig = pd.concat([DF_trig, pd.DataFrame(tmpOUT_dict)], ignore_index=True).dropna(axis=1, how='all')
                # t_trig = np.concatenate((t_trig, tmpOUT_dict['t_trig'])) # 1D array, do not specify axis
                # CCs_s_trig = np.concatenate((CCs_s_trig, tmpOUT_dict['CCs_s_trig']))
                # RAWCCs_s_trig = np.concatenate((RAWCCs_s_trig, tmpOUT_dict['RAWCCs_s_trig']))
                # width_trig = np.concatenate((width_trig, tmpOUT_dict['width_trig']))
                # t_left = np.concatenate((t_left, tmpOUT_dict['t_left']))
                # t_right = np.concatenate((t_right, tmpOUT_dict['t_right'])) 
    
    # cut the time window
    # index = np.where((t_trig >= t_start) & (t_trig <= t_end)) # index is a tuple, containing an array, can use index or index[0] as the indexes of arrays
    # return CCs_s_trig[index], RAWCCs_s_trig[index], t_trig[index], width_trig[index], t_left[index], t_right[index]
    DF_trig['t_trig'] = pd.to_datetime(DF_trig['t_trig'])
    DF_trig['t_left'] = pd.to_datetime(DF_trig['t_left'])
    DF_trig['t_right'] = pd.to_datetime(DF_trig['t_right'])
    DF_trig['width_trig'] = pd.to_timedelta(DF_trig['width_trig'])
    return DF_trig[(DF_trig['t_trig'] >= t_start) & (DF_trig['t_trig'] <= t_end)]



#!___这个函数好像没必要
# def Read_Collapse_group_df(IN_CCs_trig_dir, Temp_name_list, t_start, t_end)
#     Collapse_group = pd.DataFrame()
#     for tt,Template in enumerate(Temp_name_list):
#         #? Read CCs_s_trig
#         CCs_s_trig, RAWCCs_s_trig, t_trig, width_trig, t_left, t_right \
#             = Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)
#         DF_trig = pd.DataFrame({
#             'CCs_s_trig': CCs_s_trig,
#             'RAWCCs_s_trig': RAWCCs_s_trig,
#             't_trig': t_trig,
#             'width_trig': width_trig,
#             't_left': t_left,
#             't_right': t_right
#         })
#         # DF_trig = Read_CCs_trig_pkl_toPANDAS(IN_CCs_trig_dir, Template, t_start, t_end)
        
#         #? ADD Template time offset to the DF_trig
#         DF_trig.loc[:, 'Rt_trig'] = Remove_Temp_Offest_forSeries( DF_trig['t_trig'], Template, DF_Temp_Offset, offset_Type = 'S')
#         DF_trig.loc[:, 'Rt_left'] = Remove_Temp_Offest_forSeries( DF_trig['t_left'], Template, DF_Temp_Offset, offset_Type = 'S' )
#         DF_trig.loc[:, 'Rt_right'] = Remove_Temp_Offest_forSeries( DF_trig['t_right'], Template, DF_Temp_Offset, offset_Type = 'S' )
#         #? ADD MAD value normalization
#         DF_trig.loc[:, 'CCs_s_trig_ofMAD'] = DF_trig['CCs_s_trig'].astype('float64') / MAD_array_dict[Template][0]
#         #? ADD template name
#         DF_trig.loc[:, 'Template'] = Template
#         # print(Template,'==',DF_trig)
        
#         Collapse_group = pd.concat([Collapse_group, DF_trig], axis=0)
        
#     Collapse_group.reset_index(drop=True, inplace=True)
#     print('====>Collapse_group:', Collapse_group)
#     Collapse_group = Collapse_group.sort_values(by=['Rt_trig'], ascending=True, ignore_index=True)




# def Read_IniTrig(t_start, t_end, Template, IN_initial_trig_dir):
#     """
#     """
#     file_gap = timedelta(minutes=10)
#     # Extract the time in the filename in the format UTC_YYYYYMMDD_HHMMSS.ssssss
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
        
        

# def Read_OneTemp_from_OUT_dict_Classified(OUT_dict_Classified, Temp_name):
#     """
#     """
#     if OUT_dict_Classified == {}:
#         print('(7) WARNING: The input dict is empty')
#         return {}
#     if len(OUT_dict_Classified['Template']) == 0:
#         print('(7) No event is found')
#         return {}
    
#     idx = np.where(OUT_dict_Classified['Template'] == Temp_name)
#     if type(idx) == tuple:
#         idx = idx[0]
#     elif type(idx) == np.ndarray:
#         pass
    
#     OUT_dict_OneTemp = {}
#     OUT_dict_OneTemp['t_trig_s'] = OUT_dict_Classified['t_trig_s'][idx]
#     OUT_dict_OneTemp['Bias_level_arrivals'] = OUT_dict_Classified['Bias_level_arrivals'][idx]
#     OUT_dict_OneTemp['quasi_Mag_Amp'] = OUT_dict_Classified['quasi_Mag_Amp'][idx]
#     OUT_dict_OneTemp['quasi_Mag_CCs_s'] = OUT_dict_Classified['quasi_Mag_CCs_s'][idx]
#     OUT_dict_OneTemp['quasi_Mag_CCs_f'] = OUT_dict_Classified['quasi_Mag_CCs_f'][idx]
#     OUT_dict_OneTemp['CCs_trig_s'] = OUT_dict_Classified['CCs_trig_s'][idx]
#     OUT_dict_OneTemp['RAWCCs_trig_s'] = OUT_dict_Classified['RAWCCs_trig_s'][idx]
#     OUT_dict_OneTemp['Template'] = OUT_dict_Classified['Template'][idx]
#     return OUT_dict_OneTemp

