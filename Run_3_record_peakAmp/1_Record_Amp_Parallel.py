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

# +
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# -- To import a function on a relative path:
# -- (Assuming these functions are available in the specified path)
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import Read_DAS_data
from classify import Read_Collapse_group_df
from readers import load_pkl
# -

class Config:
    """
    Configuration class for the event processing script.
    Centralizes all parameters for easy management and modification.
    """
    def __init__(self):
        #?+++++++++++++++++ Key Input Files and Subfolder Parameters +++++++++++++++++++++
        self.Threshold_level = 40

        # --- Time Window Configuration ---
        self.t_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        self.t_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')

        # --- Path Configuration ---
        user_base_path = f"{PROJECT_ROOT}/"
        if os.path.exists(user_base_path):
            self.base_dir_to_use = user_base_path
        else:
            print(f"Warning: User-specified base path '{user_base_path}' does not exist.")
            self.base_dir_to_use = "./temp_kefalonia_testing_data/"
            print(f"Using fallback directory for testing: '{os.path.abspath(self.base_dir_to_use)}'")
            os.makedirs(self.base_dir_to_use, exist_ok=True)

        # --- Input/Output Directories ---
        self.subdir = f"level_{self.Threshold_level}/"
        self.IN_CCs_trig_dir = os.path.join(self.base_dir_to_use, "Output_2_Triggered_SCCs_points/", self.subdir)
        self.IN_Temp_lib_csv = os.path.join(self.base_dir_to_use, "Output_0_Template_library/Template_library.csv")
        self.MAD_Avg_path = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/MAD_Average_dict.pkl'
        self.OUTPUT_dir = os.path.join(self.base_dir_to_use, "Output_3_Trig_PeakAmp/", self.subdir)

        # --- DAS and Filtering Parameters ---
        self.fs = 80
        self.freq_band = [8, 20]
        self.das_dist_start = 8300
        self.das_dist_end = 14000
        self.dx = 16 * 1.02095199

        # --- Template Configuration ---
        # <--- 改进：模板列表不再硬编码，将从 main 函数中动态加载。
        
        self.Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]

        # --- Core Processing Parameters ---
        self.Search_win_right = 2.5  # seconds
        self.Search_win_left = 1.5   # seconds
        
        # --- Parallelization Configuration ---
        self.num_cpus = min(10, cpu_count()) 

        # --- Checkpoint and Output Filenames ---
        os.makedirs(self.OUTPUT_dir, exist_ok=True)
        self.checkpoint_interval_events = 100 
        self.checkpoint_filename = os.path.join(self.OUTPUT_dir, 'Trig_MAX_amp_df_checkpoint.pkl')
        self.final_output_filename = os.path.join(self.OUTPUT_dir, 'Trig_MAX_amp_df.csv')

    def save_config(self, path):
        """Saves the configuration to a JSON file, excluding non-serializable items."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clean_dict = {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, list, dict))}
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(clean_dict, f, indent=4)

def process_event(event_tuple, config):
    """
    Processes a single event: reads DAS data, calculates peak amplitudes, and saves results.
    Designed to be called by a multiprocessing pool. Enhanced with error handling.
    """
    ii, row = event_tuple
    
    # <--- 改进：添加 try...except 块来捕获所有潜在异常，增强稳健性。
    try:
        t_left = row.Rt_left - timedelta(seconds=config.Search_win_left)
        t_right = row.Rt_right + timedelta(seconds=config.Search_win_right)

        data = Read_DAS_data(t_left, t_right, config.fs, config.freq_band[0], config.freq_band[1],
                             config.dx, config.das_dist_start, config.das_dist_end)

        if data is None or data.size == 0:
            return {'index': ii, 'status': 'failed_no_data'}

        amp_max = np.max(np.abs(data), axis=0)
        index_max = np.argmax(np.abs(data), axis=0)
        
        # <--- 改进：使用向量化操作计算 t_max，以提高性能。
        time_deltas = pd.to_timedelta(index_max / config.fs, unit='s')
        t_max = t_left + time_deltas

        amp_max_mean = np.mean(amp_max)
        amp_log_mean = np.mean(np.log10(np.abs(amp_max) + 1e-12)) 

        result_dict = {
            'Rt_trig': row.Rt_trig, 'Rt_left': row.Rt_left, 'Rt_right': row.Rt_right,
            't_trig': row.t_trig, 't_left': t_left, 't_right': t_right,
            'width_trig': row.width_trig, 'Template': row.Template,
            'CCs_s_trig': row.CCs_s_trig, 'RAWCCs_s_trig': row.RAWCCs_s_trig,
            'CCs_s_trig_ofMAD': row.CCs_s_trig_ofMAD, 'amp_max_mean': amp_max_mean,
            'amp_max': amp_max, 't_max': t_max, 'amp_log_mean': amp_log_mean,
        }
        
        OUTPUT_dir_byTemp = os.path.join(config.OUTPUT_dir, row.Template)
        os.makedirs(OUTPUT_dir_byTemp, exist_ok=True)
        
        output_fpath = os.path.join(OUTPUT_dir_byTemp, f'amp_max_at_Rt_trig_{row.Rt_trig.strftime("%Y-%m-%d %H:%M:%S")}.pkl')
        with open(output_fpath, 'wb') as f:
            pickle.dump(result_dict, f)

        return {
            'index': ii,
            'amp_max_mean': amp_max_mean,
            'amp_log_mean': amp_log_mean,
            'status': 'success'
        }
    except Exception as e:
        # 捕获任何可能的异常，打印错误信息并返回失败状态，避免整个程序崩溃。
        print(f"!! Error processing event index {ii} (Time: {row.Rt_trig}): {e}")
        return {'index': ii, 'status': f'failed_exception: {e}'}

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    # 1. --- Configuration and Initialization ---
    config = Config()
    
    print("Starting data processing...")
    print(f"- Using {config.num_cpus} CPU cores for parallel processing.")
    print(f"- Checkpoints will be saved to: {config.checkpoint_filename}")

    # 2. --- Load Initial Data ---
    print("\nLoading initial data...")
    temp_lib_df = pd.read_csv(config.IN_Temp_lib_csv, sep=',')
    
    # <--- 改进：自动从模板库文件生成模板名称列表。
    config.Temp_name_list = temp_lib_df['infile_name'].unique().tolist()
    print(f"- Automatically loaded {len(config.Temp_name_list)} templates from library.")

    DF_Temp_Offset = pd.DataFrame({
        'template_name': temp_lib_df['infile_name'],
        'P_offset': temp_lib_df['t_P_sec'],
        'S_offset': temp_lib_df['t_S_sec']
    })
    
    MAD_Avg_dict = load_pkl(config.MAD_Avg_path)

    # 创建输出目录
    for template_name in config.Temp_name_list:
        os.makedirs(os.path.join(config.OUTPUT_dir, template_name), exist_ok=True)

    # 3. --- Prepare the Main DataFrame of Events ---
    print("Reading and collapsing event groups... (This may take a moment)")
    events_df = Read_Collapse_group_df(
        config.IN_CCs_trig_dir, config.Temp_name_list, DF_Temp_Offset, 
        MAD_Avg_dict, config.t_start, config.t_end
    )
    events_df = events_df.sort_values(by='Rt_trig').reset_index(drop=True)
    print(f"Found a total of {len(events_df)} events to process.")

    # 4. --- Checkpointing: Load existing progress ---
    if os.path.exists(config.checkpoint_filename):
        print(f"Resuming from checkpoint: {config.checkpoint_filename}")
        completed_df = pd.read_pickle(config.checkpoint_filename)
        
        # <--- 改进：修复断点续传的 KeyError。
        # 在 update 之前，必须先初始化结果列，确保 events_df 具有正确的结构。
        events_df['amp_max_mean'] = np.nan
        events_df['amp_log_mean'] = np.nan
        
        # 现在可以安全地用已完成的数据更新主 DataFrame
        events_df.update(completed_df)
        
        tasks_df = events_df[events_df['amp_max_mean'].isna()].copy()
        num_completed = len(events_df) - len(tasks_df)
        print(f"Skipping {num_completed} already processed events.")
    else:
        print("No checkpoint found. Starting from the beginning.")
        events_df['amp_max_mean'] = np.nan
        events_df['amp_log_mean'] = np.nan
        tasks_df = events_df.copy()
        num_completed = 0
    
    config.save_config(config.OUTPUT_dir) # 保存配置，包括动态生成的模板列表

    if tasks_df.empty:
        print("\nAll events have already been processed.")
    else:
        # 5. --- Parallel Processing ---
        print(f"\nProcessing {len(tasks_df)} remaining events...")
        task_processor = partial(process_event, config=config)
        
        with Pool(processes=config.num_cpus) as pool:
            with tqdm(total=len(events_df), initial=num_completed, desc="Processing Events", unit="event") as pbar:
                for result in pool.imap_unordered(task_processor, tasks_df.iterrows()):
                    if result and result['status'] == 'success':
                        idx = result['index']
                        events_df.loc[idx, 'amp_max_mean'] = result['amp_max_mean']
                        events_df.loc[idx, 'amp_log_mean'] = result['amp_log_mean']
                    
                    pbar.update(1)
                    
                    if pbar.n % config.checkpoint_interval_events == 0 and pbar.n > num_completed:
                        events_df.to_pickle(config.checkpoint_filename)

    # 6. --- Finalization ---
    print("\nProcessing complete. Saving final results...")
    events_df.to_pickle(config.checkpoint_filename)
    events_df.to_csv(config.final_output_filename, sep='\t', index=False)
    
    print(f"Final results saved to: {config.final_output_filename}")
    print("Done.")

if __name__ == "__main__":
    main()
# -