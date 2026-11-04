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
#  ---
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

# 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd

#-- To import a function on a relative path:
import sys
import os
import gc
from matplotlib import ticker
import matplotlib.dates as mdates



#!__________________________functions_______________________
sys.path.append(f"{PROJECT_ROOT}/functions")
from readers import Read_DAS_data
from readers import write_pkl, load_pkl
from plot import Ticks_xaxis_day_or_hour
import load_das_h5_PE2024 as load_das_h5_PE2024

from stalta import sta_lta_centred_multichan, trig_allchan_and_collapsed_index
from stalta import plot_if_really_a_event_STALTA, sta_lta_centred, trig_onechan_and_outindex

from cluster import read_DAS_span, simple_corr_coef_1D, simple_corr_coef_2D, norm_the_flattened_DATA

#!_______________________DATA IN________________________________import numpy as np
import os
import gc
import sys
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import psutil
import time
from scipy.signal import correlate
from scipy.ndimage import uniform_filter1d
import multiprocessing

import json
import yaml
import inspect
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

# ======================== 共享内存初始化函数 ========================
# 修改：移除mu_stats相关的共享内存初始化
def init_worker(shared_data_array, data_shape, data_dtype,
                # shared_mu_array, mu_shape, mu_dtype, # REMOVED
                shared_sigma_array, sigma_shape, sigma_dtype):
    # 转换形状元组为Python整数元组
    data_shape = tuple(map(int, data_shape))
    # mu_shape = tuple(map(int, mu_shape)) # REMOVED
    sigma_shape = tuple(map(int, sigma_shape))
    
    global shared_data, shared_sigma # shared_mu REMOVED
    shared_data = np.frombuffer(shared_data_array, dtype=data_dtype).reshape(data_shape)
    # shared_mu = np.frombuffer(shared_mu_array, dtype=mu_dtype).reshape(mu_shape) # REMOVED
    shared_sigma = np.frombuffer(shared_sigma_array, dtype=sigma_dtype).reshape(sigma_shape)

# %% 配置参数 =======================================================
class Config:
    def __init__(self):
        # 时间范围筛选
        self.time_range = [
            datetime(2024, 4, 24, 10, 15),
            datetime(2024, 6, 22, 16, 15)
            # datetime(2024, 4, 25, 16, 15)
        ]
        self.t_start = self.time_range[0]
        self.t_end = self.time_range[1]
        self.t_str = f"_{self.t_start.strftime('%Y-%m-%d %H:%M:%S')}_{self.t_end.strftime('%Y-%m-%d %H:%M:%S')}"
        
        
        self.input_dir = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
               
        #* 输入路径 1, STA/LTA 目录
        subdir = 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        self.IN_cata_dir = f'{PROJECT_ROOT}/Run_0_preparing_templates_from_catalog/'
        # self.IN_cata_dir += 'extract_threshold_300_trigNO_300_from_threshold_100_trigNO_100/'
        # self.IN_cata_dir += 'extract_threshold_100_trigNO_250_from_threshold_100_trigNO_100/'
        # self.IN_cata_dir += 'extract_threshold_300_trigNO_100_from_threshold_100_trigNO_100/'
        # self.IN_cata_dir += 'threshold_120_trigNO_240_sta0.3s_lta3s/'
        # self.IN_cata_dir += 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        # self.IN_cata_dir += 'threshold_50_trigNO_50_sta0.3s_lta3s/'
        self.IN_cata_dir += subdir
        self.catalog_path = f"{self.IN_cata_dir}/sta_lta_catalog.csv"

        self.IN_SNRed_cata_dir = f"{self.IN_cata_dir}Output_1_Find_NOAeq_Mag2.0_10sec_SNR30_100km/"
        self.SNRed_catalog_path = f"{self.IN_SNRed_cata_dir}matched_df.csv"
        
        # 输出路径
        self.output_dir = f"{self.IN_SNRed_cata_dir}Output_2_Corr_movmax/"

        # 信号处理参数
        self.fs = 80
        self.freq_band = [8, 20]
        self.t_left = timedelta(seconds=10)
        self.t_right = timedelta(seconds=20)
        self.t_pad = timedelta(seconds=10)
        
        # 空间通道范围
        self.dist_start = 8.3e3
        self.dist_end = 14e3
        
        # 并行计算参数
        self.chunk_size = 500
        self.n_workers = 20
        self.dtype = 'float32'

    def _save_json(self, path):
        """增强的 JSON 序列化方法"""
        def serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, timedelta):
                return obj.total_seconds()
            if isinstance(obj, (np.dtype, np.generic)):
                return str(obj)
            if isinstance(obj, MappingProxyType):
                return dict(obj)
            if hasattr(obj, '__dict__'):
                return {k: serializer(v) for k, v in obj.__dict__.items()}
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=serializer, ensure_ascii=False)

# ========================== 数据加载 ==========================
def load_SNRed_df(config):
    """加载匹配的 DataFrame, 经过SNR和magnitude筛选"""
    df = pd.read_csv(config.SNRed_catalog_path, sep='\t', parse_dates=["peak_time","STA_arrival_time",'EQ_time'])
    mask = (df["peak_time"] >= config.time_range[0]) & \
           (df["peak_time"] <= config.time_range[1])
    return df[mask].reset_index(drop=True)


def load_catalog(config):
    """加载STALTA事件目录"""
    df = pd.read_csv(config.catalog_path, sep=',',parse_dates=["peak_time"])
    mask = (df["peak_time"] >= config.time_range[0]) & \
           (df["peak_time"] <= config.time_range[1])
    return df[mask]["peak_time"].tolist()

def read_single_event(event_time, config):
    """读取单个事件的原始数据"""    
    data, _, _ = read_DAS_span(
        event_time,
        t_left=config.t_left + config.t_pad,
        t_right=config.t_right + config.t_pad,
        freq_band=config.freq_band,
        input_dir=config.input_dir,
        dist_start=config.dist_start,
        dist_end=config.dist_end
    )
    return data.astype(config.dtype)

def load_all_events(event_times, config):
    """加载所有事件数据到共享内存"""
    sample = read_single_event(event_times[0], config)
    n_time, n_chan = sample.shape
    n_events = len(event_times)
    
    shared_data_array = multiprocessing.RawArray('f', n_time * n_chan * n_events)
    DATA_tc = np.frombuffer(shared_data_array, dtype=config.dtype).reshape((n_time, n_chan, n_events))
    
    for i, t in enumerate(tqdm(event_times, desc="加载数据到共享内存")):
        DATA_tc[:, :, i] = read_single_event(t, config)
    
    return DATA_tc, shared_data_array, n_time, n_chan

# ========================== 预处理 ==========================
# 修改：compute_sliding_stats 只返回 sigma_j
def compute_sliding_stats(data_j, win_len):
    """快速计算滑动统计量 (仅返回 sigma)"""
    T, C = data_j.shape
    valid_length = T - win_len + 1
    
    sigma_j_output = np.zeros((valid_length, C), dtype=np.float32)
    
    for c in range(C):
        # 滑动均值 (内部计算需要，但不返回)
        mu = np.convolve(data_j[:, c], np.ones(win_len)/win_len, mode='valid')
        
        # 滑动方差
        data_sq = data_j[:, c].astype(np.float64)**2
        mu_sq = np.convolve(data_sq, np.ones(win_len)/win_len, mode='valid')
        
        var = mu_sq - mu.astype(np.float64)**2
        sigma_val = np.sqrt(var, where=var>0, out=np.zeros_like(var)) + 1e-12 # Epsilon for stability
        
        sigma_j_output[:, c] = sigma_val.astype(np.float32)
    
    return sigma_j_output

# 修改：移除 mu_stats 和 shared_mu_array 的处理
def preprocess_sliding_stats(DATA_tc, config):
    """预处理滑动标准差到共享内存"""
    win_len = int((config.t_left + config.t_right).total_seconds() * config.fs)
    n_events = DATA_tc.shape[2]
    valid_length = DATA_tc.shape[0] - win_len + 1
    n_chan = DATA_tc.shape[1]
    
    # 仅为 sigma_stats 创建共享内存
    sigma_shape = (int(n_events), int(valid_length), int(n_chan))
    total_elements_sigma = int(np.prod(sigma_shape))
    
    shared_sigma_array = multiprocessing.RawArray('f', total_elements_sigma)
    sigma_stats = np.frombuffer(shared_sigma_array, dtype=config.dtype).reshape(sigma_shape)
    
    for j in tqdm(range(n_events), desc="预处理滑动标准差到共享内存"):
        data_j = DATA_tc[:, :, j]
        sigma_j_computed = compute_sliding_stats(data_j, win_len) # compute_sliding_stats now only returns sigma
        sigma_stats[j] = sigma_j_computed
    
    return sigma_stats, shared_sigma_array # 只返回 sigma_stats 和其共享数组

# ========================== 核心计算 ==========================
# 修改后的 process_event_pair (已移除 shared_mu[j] 的引用)
def process_event_pair(args, config):
    """处理单个事件对的计算"""
    i, j = args
    try:
        win_len = int((config.t_left + config.t_right).total_seconds() * config.fs)
        pad_samples = int(config.t_pad.total_seconds() * config.fs)
        n_chan = shared_data.shape[1] # Number of channels
        # n_time_shared = shared_data.shape[0] # Total time samples in shared_data for one event trace

        # 从共享内存读取i事件数据并预处理 F_i
        data_i = shared_data[pad_samples : pad_samples + win_len, :, i]
        F_i_mean = data_i.mean(axis=0, keepdims=True)
        F_i_std = data_i.std(axis=0, keepdims=True) + 1e-12
        
        win_len_sqrt_val = np.sqrt(win_len).astype(config.dtype)
        F_i = (data_i - F_i_mean) / F_i_std / win_len_sqrt_val
        
        # 从共享内存读取j事件的滑动标准差
        # mu_j = shared_mu[j] # REMOVED - shared_mu is no longer initialized or passed
        sigma_j = shared_sigma[j] # sigma_j has shape (valid_length, n_chan)
        
        S_j_all_channels = shared_data[:, :, j] # Shape (n_time_shared, n_chan)


# --- CORRECTED CORRELATION CALCULATION ---
        # F_i has shape (win_len, n_chan). Reverse along time axis (axis 0) for convolution.
        # The ::-1 slice reverses the first dimension.
        F_i_reversed_time = F_i[::-1, :] 
        
        # Use signal.fftconvolve for multi-channel correlation along a specified axis
        all_corrs_matrix = signal.fftconvolve(
            S_j_all_channels,
            F_i_reversed_time,
            mode='valid',
            axes=0  # Convolve along the time axis (axis 0) for each channel
        )
# --- END OF CORRECTION ---
        

        denominator = sigma_j * win_len_sqrt_val # Shape (valid_length, n_chan)
        normalized_corrs = all_corrs_matrix / denominator # Shape (valid_length, n_chan)
        
        corr_total = np.sum(normalized_corrs, axis=1) # Shape (valid_length,)
        
        index_max = np.argmax(np.abs(corr_total))
        max_corr = (corr_total[index_max]) / n_chan
        
        center_corr = len(corr_total) // 2
        offset = index_max - center_corr
        
        return (i, j), max_corr, offset
    
    except Exception as e:
        print(f"Error processing ({i},{j}): {str(e)}")
        return (i, j), 0.0, 0

def process_event_pair_loop(args, config):
    """处理单个事件对的计算"""
    i, j = args
    try:
        win_len = int((config.t_left + config.t_right).total_seconds() * config.fs)
        pad_samples = int(config.t_pad.total_seconds() * config.fs)
        n_chan = shared_data.shape[1]
        
        # 从共享内存读取i事件数据
        data_i = shared_data[pad_samples : pad_samples + win_len, :, i]
        F_i_mean = data_i.mean(axis=0, keepdims=True)
        F_i_std = data_i.std(axis=0, keepdims=True) + 1e-12
        F_i = (data_i - F_i_mean) / F_i_std / np.sqrt(win_len)
        
        # 从共享内存读取j统计量
        sigma_j = shared_sigma[j]
        
        corr_total = np.zeros(sigma_j.shape[0], dtype=config.dtype)
        
        for c in range(n_chan):
            data_j = shared_data[:, c, j]
            S_j = data_j[:shared_data.shape[0] - win_len + 1 + win_len - 1]
            corr = correlate(S_j, F_i[:, c], mode='valid', method='fft')
            corr_total += corr / (sigma_j[:, c] * np.sqrt(win_len))
        
        index_max = np.argmax(np.abs(corr_total))
        max_corr = (corr_total[index_max]) / n_chan
        center_corr = len(corr_total) // 2
        offset = index_max - center_corr
        
        return (i, j), max_corr, offset
    
    except Exception as e:
        print(f"Error processing ({i},{j}): {str(e)}")
        return (i, j), 0.0, 0
    
# ========================== 主程序 ==========================
def main():
    start_time = time.time()
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    config._save_json(os.path.join(config.output_dir, 'config.json'))
    print("配置参数已保存到:", os.path.join(config.output_dir, 'config.json'))
    
    print("【1/4】加载事件目录的 event_times...")
    # event_times = load_catalog(config)
    SNRed_df = load_SNRed_df(config)
    event_times = SNRed_df["peak_time"].tolist()
    print(f"加载到 {len(event_times)} 个事件")
    
    
    print("【2/4】加载DAS数据到共享内存...")
    DATA_tc, shared_data_array, n_time, n_chan = load_all_events(event_times, config)
    
    print("【3/4】预处理滑动标准差到共享内存...") # Modified description
    # 修改：preprocess_sliding_stats 现在只返回 sigma 相关的数组
    sigma_stats, shared_sigma_array = preprocess_sliding_stats(DATA_tc, config)
    
    print("【4/4】计算相关矩阵...")
    n_events = len(event_times)
    corr_matrix = np.eye(n_events, dtype=config.dtype)
    offset_matrix = np.zeros((n_events, n_events), dtype=np.int32)
    
    tasks = [(i, j) for i in range(n_events) for j in range(i+1, n_events)]
    
    # 修改：更新 initargs 以移除 mu 相关的共享内存参数
    with ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=init_worker,
        initargs=(
            shared_data_array, DATA_tc.shape, config.dtype,
            # shared_mu_array, mu_stats.shape, config.dtype, # REMOVED
            shared_sigma_array, sigma_stats.shape, config.dtype
        )
    ) as executor:
        futures = []
        for k in range(0, len(tasks), config.chunk_size):
            chunk = tasks[k:k+config.chunk_size]
            futures += [executor.submit(process_event_pair, task, config) for task in chunk]
            
            mem = psutil.virtual_memory()
            print(f"已提交任务: {len(futures)} | 内存使用: {mem.percent}%")
            if mem.percent > 90:
                print("\n内存不足! 停止提交新任务")
                break
        
        with tqdm(total=len(tasks), desc="处理事件对") as pbar:
            for future in as_completed(futures):
                try:
                    (i, j), val, offset = future.result()
                    corr_matrix[i, j] = val
                    corr_matrix[j, i] = val
                    offset_matrix[i, j] = offset
                    offset_matrix[j, i] = -offset
                except Exception as e:
                    print(f"任务失败: {str(e)}")
                finally:
                    pbar.update(1)
                    del future
                    gc.collect()

    # 保存结果
    np.save(os.path.join(config.output_dir, "correlation_matrix.npy"), corr_matrix)
    np.save(os.path.join(config.output_dir, "offset_matrix.npy"), offset_matrix)
    print(f"总处理时间: {time.time()-start_time:.2f}秒")
    
    #!___________________plot only_________________
    # This part remains unchanged as it depends on the final matrices
    corr_matrix_loaded = np.load(os.path.join(config.output_dir, "correlation_matrix.npy")) # Renamed to avoid conflict if run in notebook
    offset_matrix_loaded = np.load(os.path.join(config.output_dir, "offset_matrix.npy")) # Renamed
    
    plt.figure(figsize=(15, 15))
    plt.imshow(abs(corr_matrix_loaded), cmap='Greys', norm=mcolors.LogNorm())
    plt.colorbar(label="Correlation Coefficient")
    plt.title(f"Event Correlation Matrix ({len(event_times)} events)")
    plt.savefig(os.path.join(config.output_dir, "correlation_matrix_abs.png"), dpi=300)
    plt.close() # Close figure to free memory
    
    plt.figure(figsize=(15, 15))
    offset_sec = offset_matrix_loaded / config.fs
    norm = mcolors.SymLogNorm(linthresh=0.1, linscale=1, vmin=-5, vmax=5, base=10)
    plt.imshow(offset_sec, norm=norm, cmap='twilight')  
    plt.colorbar(label="Time Offset (seconds)")
    plt.title(f"Time Offset (in Seconds) Matrix ({len(event_times)} events)")
    plt.savefig(os.path.join(config.output_dir, "offset_matrix.png"), dpi=300)
    plt.close() # Close figure


    plt.figure(figsize=(15, 15))
    # offset_sec already calculated
    plt.imshow(abs(offset_sec), cmap='Greys_r', norm=mcolors.LogNorm())
    plt.colorbar(label="Time Offset (seconds)")
    plt.title(f"abs(Time Offset) (in Seconds) Matrix ({len(event_times)} events)")
    plt.savefig(os.path.join(config.output_dir, "offset_matrix_abs.png"), dpi=300)
    plt.close() # Close figure

if __name__ == "__main__":
    main()