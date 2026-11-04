#!/usr/bin/env python
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
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
import sys

import os
import gc
from IPython.display import clear_output # 主要用于Jupyter环境，在纯脚本中可能不需要
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import h5py, json
import time


# 假设这些自定义模块路径和函数是正确的
sys.path.append(f"{PROJECT_ROOT}/functions") # 请确保此路径对您的环境有效
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024 # 示例导入
from readers import  Read_DAS_data, Read_template # 示例导入
from readers import write_pkl, load_pkl, write_h5, load_h5 # 示例导入
from compute_correlation import CCs_comp, moving_average_2d_axis0, preprocess # 示例导入


class Config:
    def __init__(self):
        #* 输入路径
        self.temp_npy_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"
        
        #* 输出路径
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/"

        self.total_cores = os.cpu_count() or 1
        # CPU工作进程数，可根据机器性能和任务特性调整，例如 min(32, int(self.total_cores * 0.75))
        self.cpu_workers = min(10, int(self.total_cores * 0.2)) 
        if self.cpu_workers == 0: self.cpu_workers = 1


        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286 # 示例值
        
        self.Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ] # 示例模板名称列表
        
        self.t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        self.t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_delta = timedelta(minutes=10) # 每个基础时间窗口的长度
        self.t_array = np.arange(self.t_all_start, self.t_all_end, self.t_delta)
        


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


def get_max_template_length(config, all_templates_data):
    """计算所有模板中的最大样本长度"""
    max_len = 0
    if not all_templates_data: # 如果字典为空
        raise ValueError("模板数据字典为空，无法计算最大模板长度。")

    for temp_name in config.Temp_name_list:
        if temp_name in all_templates_data and all_templates_data[temp_name] is not None:
            temp_len = all_templates_data[temp_name].shape[0]
            if temp_len > max_len:
                max_len = temp_len
        # else:
            # print(f"Info: Template {temp_name} not found in preloaded data or is None during max_len calculation.")
            
    if max_len == 0:
        # 这种情况可能是所有模板都加载失败或者所有模板长度都为0
        raise ValueError("未能从加载的模板数据中获取有效的最大模板长度 (max_len is 0)。请检查模板文件。")
    return max_len

def preload_all_templates(config):
    """预加载所有模板数据到内存"""
    all_templates_data = {}
    print("Preloading all templates...")
    for temp_name in tqdm(config.Temp_name_list, desc="Loading Templates"):
        try:
            temp_data = Read_template(temp_name, config.fs, config.freq_band[0], config.freq_band[1],
                                      config.dx, config.das_dist_start, config.das_dist_end, config.temp_npy_dir)
            if temp_data is not None and temp_data.size > 0:
                all_templates_data[temp_name] = temp_data
            else:
                print(f"Warning: Template {temp_name} loaded as None or empty. Skipping.")
                all_templates_data[temp_name] = None # 标记为加载失败或为空
        except Exception as e:
            print(f"Warning: Could not load template {temp_name}: {e}")
            all_templates_data[temp_name] = None # 标记为加载失败

    if not any(data is not None for data in all_templates_data.values()): # 检查是否至少有一个模板成功加载
        raise SystemExit("错误：所有模板均加载失败或为空，程序无法继续。请检查模板路径和文件。")
    return all_templates_data


def process_single_time_window_for_all_templates(args):
    """
    并行处理单个时间窗口，针对所有模板。
    数据只为这个时间窗口读取一次。
    """
    t_start, config, all_templates_data, max_template_len_samples = args
    
    fs = config.fs
    f1 = config.freq_band[0]
    f2 = config.freq_band[1]
    dx = config.dx
    d1 = config.das_dist_start
    d2 = config.das_dist_end
    OUTPUT_dir = config.OUTPUT_dir

    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        # 计算此时间窗口需要读取的数据的实际结束时间
        t_end_basic_window = min(t_start + config.t_delta, config.t_all_end) # 修正A2：直接使用 config.t_all_end
        
        duration_max_template_sec = (max_template_len_samples - 1) / fs if max_template_len_samples > 0 else 0
        t_end_for_read = t_end_basic_window + timedelta(seconds=duration_max_template_sec)
        
        # print(f"Debug: Window {t_start}, basic_end {t_end_basic_window}, read_end {t_end_for_read}")
        continuous_data = Read_DAS_data(t_start, t_end_for_read, fs, f1, f2, dx, d1, d2)
        
        if continuous_data is None or continuous_data.size == 0:
            # print(f"Warning: No data returned for time window starting at {t_start}. Skipping this window for all templates.")
            # 对于此窗口，所有模板都无法处理
            return {'skipped': 0, 'processed': 0, 'errors': len(config.Temp_name_list), 'window_skipped': True}

        # 计算当前基础窗口的预期样本数（可能因数据末尾而不足10分钟）
        actual_duration_seconds = (t_end_basic_window - t_start).total_seconds()
        if actual_duration_seconds < 0: actual_duration_seconds = 0 # 防止 t_start > t_end_basic_window 的极端情况
        expected_samples = int(actual_duration_seconds * fs) + 1

        if expected_samples <= 0 :
            # print(f"Warning: Expected samples for window {t_start} is {expected_samples}. Skipping processing for this window.")
            return {'skipped': 0, 'processed': 0, 'errors': len(config.Temp_name_list), 'window_skipped': True}


        for template_name, temp_data in all_templates_data.items():
            if temp_data is None: # 模板加载失败或为空
                # print(f"Info: Skipping template {template_name} for window {t_start} as its data was not loaded.")
                skipped_count +=1 # 或者算作 error_count，取决于定义
                continue

            OUTPUT_dir_byTemp = os.path.join(OUTPUT_dir, template_name) # 移除末尾斜杠
            if not os.path.exists(OUTPUT_dir_byTemp):
                try:
                    os.makedirs(OUTPUT_dir_byTemp, exist_ok=True) # exist_ok=True更简洁
                except Exception as e_mkdir: # 以防万一，例如权限问题
                    print(f"Error creating directory {OUTPUT_dir_byTemp}: {e_mkdir}. Skipping template {template_name} for this window.")
                    error_count += 1
                    continue
            
            OUT_file = os.path.join(OUTPUT_dir_byTemp, 
                                  f'CCs_stack_{template_name}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.h5')
            
            if os.path.exists(OUT_file):
                skipped_count += 1
                continue
            
            try:
                results = CCs_comp(continuous_data, temp_data) 
                if results is None or not isinstance(results, (tuple, list)) or len(results) < 2:
                    print(f"Warning: CCs_comp for {template_name} at {t_start} returned invalid results. Skipping.")
                    error_count += 1
                    continue
                
                CCs_f_full, RAWCCs_f_full = results

                # --- 修正A1: 裁切 CCs_comp 的输出 ---
                if CCs_f_full is None or CCs_f_full.shape[0] == 0:
                    # print(f"Warning: CCs_comp for {template_name} at {t_start} returned empty CCs_f_full. Skipping.")
                    error_count += 1 # 或者 skipped_count，取决于定义
                    continue
                
                if CCs_f_full.shape[0] > expected_samples:
                    CCs_f = CCs_f_full[:expected_samples, :]
                    RAWCCs_f = RAWCCs_f_full[:expected_samples, :] if RAWCCs_f_full is not None else None
                elif CCs_f_full.shape[0] < expected_samples:
                    # print(f"Warning: Corr result for {template_name} at {t_start} is shorter ({CCs_f_full.shape[0]}) than expected ({expected_samples}). Using as is.")
                    CCs_f = CCs_f_full
                    RAWCCs_f = RAWCCs_f_full
                else: # CCs_f_full.shape[0] == expected_samples
                    CCs_f = CCs_f_full
                    RAWCCs_f = RAWCCs_f_full
                
                # 再次检查裁切后是否有效
                if CCs_f is None or CCs_f.shape[0] == 0 or (RAWCCs_f is not None and CCs_f.shape[0] != RAWCCs_f.shape[0]):
                    # print(f"Warning: Post-slicing/check, CCs_f for {template_name} at {t_start} is invalid. Skipping.")
                    error_count += 1
                    continue
                # --- 裁切结束 ---

                CCs_s = np.mean(CCs_f, axis=1)
                RAWCCs_s = np.mean(RAWCCs_f, axis=1) if RAWCCs_f is not None else np.full_like(CCs_s, np.nan) # 如果RAWCCs_f为None，则创建NaN数组
                
                OUT_dict = {'CCs_s': CCs_s, 'RAWCCs_s': RAWCCs_s}
                write_h5(OUT_file, OUT_dict)
                processed_count += 1
            
            except Exception as e_template:
                print(f"Error processing template {template_name} for window {t_start}: {str(e_template)}")
                error_count += 1
        
        del continuous_data
        gc.collect()
        return {'skipped': skipped_count, 'processed': processed_count, 'errors': error_count, 'window_skipped': False}

    except Exception as e_window:
        # print(f"Major error processing time window starting {t_start}: {str(e_window)}")
        return {'skipped': 0, 'processed': 0, 'errors': len(config.Temp_name_list), 'window_skipped': True}


# ============== 主程序部分 ==============
if __name__ == '__main__':
    config = Config()
    
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    
    # 在传递给工作函数前，确保config内部的时间是Python datetime对象
    if isinstance(config.t_all_start, np.datetime64):
        config.t_all_start = pd.to_datetime(config.t_all_start).to_pydatetime()
    if isinstance(config.t_all_end, np.datetime64):
        config.t_all_end = pd.to_datetime(config.t_all_end).to_pydatetime()

    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print(f"\n配置参数已保存到: {os.path.join(config.OUTPUT_dir, 'config.json')}")

    start_time = time.time() 
    start_cpu_time = time.process_time() 

    try:
        all_templates_data = preload_all_templates(config)
        max_template_len_samples = get_max_template_length(config, all_templates_data)
        print(f"Max template length: {max_template_len_samples} samples ({max_template_len_samples/config.fs:.2f} seconds)")
    except (SystemExit, ValueError) as e:
        print(e)
        sys.exit(1) 

    task_args_list = []
    for t_start_np in config.t_array:
        # np.datetime64 to python datetime for consistency in calculations
        t_start_dt = pd.to_datetime(t_start_np).to_pydatetime() 
        task_args_list.append((t_start_dt, config, all_templates_data, max_template_len_samples))
    
    if not task_args_list:
        print("没有生成任何时间窗口任务，请检查 t_all_start, t_all_end, 和 t_delta 配置。")
        sys.exit(0)

    workers = config.cpu_workers 
    print(f"\n{'='*30}")
    print(f"Available CPU cores: {config.total_cores}")
    print(f"Using {workers} worker processes.")
    print(f"{'='*30}")
    
    stats = {'total_template_tasks':0, 'skipped_template_outputs':0, 'processed_template_outputs':0, 'errors_template_processing':0, 'windows_attempted': 0, 'windows_skipped_data_issue':0}
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_time_window_for_all_templates, arg) for arg in task_args_list]
        
        progress = tqdm(as_completed(futures), total=len(futures), desc="Processing Time Windows", unit="window")
        for future in progress:
            stats['windows_attempted'] += 1
            try:
                result = future.result() 
                if result['window_skipped']:
                    stats['windows_skipped_data_issue'] +=1
                stats['skipped_template_outputs'] += result['skipped']
                stats['processed_template_outputs'] += result['processed']
                stats['errors_template_processing'] += result['errors']
                # 每个窗口的模板任务总数 = 该窗口内（跳过+处理+错误）的模板数
                num_templates_in_window = result['skipped'] + result['processed'] + result['errors']
                stats['total_template_tasks'] += num_templates_in_window
                
                progress.set_postfix({
                    'WinSkip':stats['windows_skipped_data_issue'],
                    'TmpSkip':stats['skipped_template_outputs'],
                    'TmpProc':stats['processed_template_outputs'],
                    'TmpErr':stats['errors_template_processing']
                })
            except Exception as e_future:
                print(f"A time window processing task generated an unhandled exception: {e_future}")
                # 假设此窗口所有模板都失败
                stats['errors_template_processing'] += len(config.Temp_name_list) 
                stats['total_template_tasks'] += len(config.Temp_name_list)
                stats['windows_skipped_data_issue'] += 1 # 标记此窗口存在严重问题

    print(f"\n{'='*40}")
    print(f"All Time Windows Processed:")
    print(f"  Total time windows submitted: {len(task_args_list)}")
    print(f"  Time windows fully attempted (data read ok): {stats['windows_attempted'] - stats['windows_skipped_data_issue']}")
    print(f"  Time windows skipped due to data read/window issues: {stats['windows_skipped_data_issue']}")
    print(f"  --- Breakdown by Template Output Files ---")
    print(f"  Total template processing instances: {stats['total_template_tasks']}")
    print(f"  Skipped (output file existed or template not loaded): {stats['skipped_template_outputs']}")
    print(f"  Processed (new output file created): {stats['processed_template_outputs']}")
    print(f"  Errors (during individual template processing/saving): {stats['errors_template_processing']}")
    print(f"{'='*40}")

    del all_templates_data # 清理内存
    gc.collect()
    
    end_cpu_time = time.process_time()
    end_time = time.time()
    cpu_time_consumed = end_cpu_time - start_cpu_time
    wall_time_consumed = end_time - start_time
    print(f"程序消耗的CPU时间: {cpu_time_consumed:.4f} 秒")
    print(f"程序消耗的墙上时间 (Wall Clock Time): {wall_time_consumed:.4f} 秒")