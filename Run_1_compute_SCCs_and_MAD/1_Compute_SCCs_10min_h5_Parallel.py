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
from IPython.display import clear_output
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np
import h5py, json
import time


sys.path.append(f"{PROJECT_ROOT}/functions")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024
from readers import  Read_DAS_data, Read_template
from readers import write_pkl, load_pkl, write_h5, load_h5
from compute_correlation import CCs_comp, moving_average_2d_axis0, preprocess



class Config:
    def __init__(self):
        #* 输入路径
        self.temp_npy_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"
        
        #* 输出路径
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5_6h/"

        self.total_cores = os.cpu_count() or 1  # 获取逻辑核心数，默认为1
        self.cpu_workers = min(10, int(self.total_cores * 0.1))

        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [5, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 8e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286

        
        self.Temp_name_list = [
            'temp_0',   'temp_1',   
            'temp_2',   'temp_3',   
            'temp_4',   'temp_5',
            'temp_6',   'temp_7',   
            'temp_8',   'temp_9',   
            'temp_10',  'temp_11',
            'temp_12'
        ]
        
        self.t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        self.t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_delta = timedelta(minutes=10)
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


# ============== 新增并行处理函数 ==============
def process_time_window(args):
    """
    并行处理单个时间窗口的函数
    """
    Template, t_start, config, temp_data = args  # 增加模板数据参数
    
    try:
        # 解包参数
        fs = config.fs
        f1 = config.freq_band[0]
        f2 = config.freq_band[1]
        dx = config.dx
        d1 = config.das_dist_start
        d2 = config.das_dist_end
        OUTPUT_dir = config.OUTPUT_dir
        
        # 创建模板目录
        OUTPUT_dir_byTemp = os.path.join(OUTPUT_dir, Template + '/')
        if not os.path.exists(OUTPUT_dir_byTemp):
            os.makedirs(OUTPUT_dir_byTemp)
            
        # 构建输出文件名
        OUT_file = os.path.join(OUTPUT_dir_byTemp, 
                              f'CCs_stack_{Template}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.h5')
        
        # 跳过已处理文件
        if os.path.exists(OUT_file):
            return 0  # 0表示跳过
        
        # 处理时间窗口
        t_end = min(t_start + config.t_delta, config.t_all_end)
        t_end_extended = t_end + timedelta(seconds=(temp_data.shape[0]-1)/fs)
        
        # 读取数据
        data = Read_DAS_data(t_start, t_end_extended, fs, f1, f2, dx, d1, d2)
        
        # 计算相关和触发
        results = CCs_comp(data, temp_data)
        # if results[0] == 0:
        #     return 1  # 1表示无事件

        CCs_f,RAWCCs_f = results
        
        CCs_s = np.mean(CCs_f, axis=1)
        RAWCCs_s = np.mean(RAWCCs_f, axis=1)

        
        # save the results for every 10min
        OUT_dict = {
                    'CCs_s':CCs_s, 'RAWCCs_s':RAWCCs_s
                    }
    
        write_h5(OUT_file, OUT_dict)
            
        # 清理内存
        del data, results, OUT_dict
        gc.collect()
        return 2  # 2表示成功处理
    
    except Exception as e:
        print(f"Error processing {t_start} with {Template}: {str(e)}")
        return -1  # -1表示错误

# ============== 修改主程序部分 ==============
if __name__ == '__main__':
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir, 'config.json'))

    start_cpu_time = time.process_time()
    
    # 并行处理每个模板
    for Template in config.Temp_name_list:
        print(f"\n{'='*40}")
        print(f"Processing template: {Template}")
        print(f"{'='*40}")
        
        # 预加载模板数据（每个模板只加载一次）
        temp_data = Read_template(Template, config.fs, config.freq_band[0], config.freq_band[1],
                                  config.dx, config.das_dist_start, config.das_dist_end, config.temp_npy_dir)
        
        # 生成任务参数列表
        task_args = [(Template, t_start.astype(datetime), config, temp_data) 
                    for t_start in config.t_array]
        
        total_cores = config.total_cores
        workers = config.cpu_workers
        max(1, int(total_cores * 0.1))  # 至少保留1个进程
        print(f"\n{'='*30}")
        print(f"Available CPU cores: {total_cores}")
        print(f"Using 50% cores: {workers}")
        print(f"{'='*30}")
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=workers) as executor:  # 根据CPU核心数调整
            futures = [executor.submit(process_time_window, arg) for arg in task_args]
            
            # 进度条和结果统计
            stats = {'total':0, 'skipped':0, 'processed':0, 'errors':0}
            progress = tqdm(as_completed(futures), total=len(futures), 
                           desc=f"Processing {Template}", unit="window")
            for future in progress:
                result = future.result()
                stats['total'] += 1
                if result == 0:
                    stats['skipped'] += 1
                elif result == 1:
                    stats['processed'] += 1
                elif result == 2:
                    stats['processed'] += 1
                elif result == -1:
                    stats['errors'] += 1
                    
                # 更新进度条描述
                progress.set_postfix({
                    'S':stats['skipped'],
                    'P':stats['processed'],
                    'E':stats['errors']
                })
        
        # 打印统计信息
        print(f"\nTemplate {Template} completed:")
        print(f"  Total windows: {stats['total']}")
        print(f"  Skipped (existing): {stats['skipped']}")
        print(f"  Processed (new): {stats['processed']}")
        print(f"  Errors: {stats['errors']}")
        
        # 清理模板数据
        del temp_data
        gc.collect()
    
    
    # 记录结束后的CPU时间
    end_cpu_time = time.process_time()
    # 计算消耗的CPU时间
    cpu_time_consumed = end_cpu_time - start_cpu_time
    print(f"程序消耗的CPU时间: {cpu_time_consumed:.4f} 秒")