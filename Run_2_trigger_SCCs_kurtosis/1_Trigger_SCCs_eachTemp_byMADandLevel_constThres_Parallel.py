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
# +
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
#-- To import a function on a relative path:
import sys
import os
import gc # Retained import, though explicit gc.collect usage is reduced
from IPython.display import clear_output # Retained import, though not used in the parallelized main logic
import pickle
import h5py, json
import numpy as np

# --- Multiprocessing and progress bar ---
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

#!__________________________functions_____________________________
# Ensure this path is correct and accessible by all potential processes
# If functions are in the same directory or installed in the environment, sys.path.append might not be strictly necessary
# For robustness in some multiprocessing start methods, ensure these modules are discoverable.
sys.path.append(f"{PROJECT_ROOT}/functions")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour
from trigger import Trigger_with_MADarray
from stalta import sta_lta_centred, sta_lta_centred_unnorm
# from pydas_readers.readers import load_das_h5, write_das_h5 # Not used in the provided main script
# from pydas_readers.util import block_filters, block_cleaning # Not used
# import load_das_h5_PE2024 as load_das_h5_PE2024 # Not used

# def _average_MAD_array_dict(MAD_array_dict, Temp_name_list): # This function was defined but not used in main
#     """
#     Args:
#         MAD_array_dict: dict, keys are template names, values are MAD_array
#         Temp_name_list: list of template names
#     Returns:
#         MAD_array_dict: dict, keys are template names, values are average MAD_array
#     """
#     MAD_array_dict_new = {}
#     for Template in Temp_name_list:
#         MAD_array_dict_new[Template] = np.array([np.mean(MAD_array_dict[Template])])
#         MAD_array_dict_new['t_array'] = np.array([MAD_array_dict['t_array'][0], MAD_array_dict['t_array'][-1]])
#     return MAD_array_dict_new

#!__________________________config_____________________________

class Config:
    def __init__(self):
        
        #?++++++++++++++++trigger 关键 parameters+++++++++++++++++++++++++++++
        self.Threshold_level =50 # 触发阈值，单位为该模板的MAD值
        self.Minimal_event_interval = 3 # in seconds
        #?+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #* Parallel processing parameters
        self.num_cpus = 10  # Use all available CPUs by default. You can set this to a specific number.
                                     # e.g., self.num_cpus = 4

        #* 输入路径
        self.temp_npy_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"

        self.IN_CCs_stack_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/'
        # Path to the MAD average dictionary
        self.MAD_Avg_path = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/12:00:00/' + 'MAD_Average_dict.pkl'
        self.MAD_Avg_dict = load_pkl(self.MAD_Avg_path)  # load the MAD_array_dict

        #* 输出路径
        self.subdir = f"level_{self.Threshold_level}/"
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/" + self.subdir
        
        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286
        
        self.Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]
        
        
        self.t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        self.t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        self.t_delta = timedelta(minutes=10)
        self.t_array = np.arange(self.t_all_start, self.t_all_end, self.t_delta)
        self.t_array = np.append(self.t_array, self.t_all_end) # Ensure the last segment up to t_all_end is included
        
        
        #!+==++++++ new parameters about STA/LTA on CCs_s

        #!======================================================================
        self.sta_len = 0.5 # seconds
        self.lta_len = 1.5 # seconds

        self.search_win = [0.6,0.2] # # seconds, the window to search for the maximum of STA/LTA
        
        
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


#!________________Worker function for parallel processing_________________
def process_segment(args):
    """
    Worker function to process a single time segment for a given template.
    Args:
        args (tuple): Contains (template_name, t_start, t_end, config, output_dir_by_temp)
    Returns:
        int: Number of triggers found in this segment.
    """
    template_name, t_start, t_end, config, output_dir_by_temp = args

    filename = f'CCs_trig_{template_name}_UTC_{t_start.strftime("%Y%m%d_%H%M%S.%f")}.pkl'
    output_filepath = os.path.join(output_dir_by_temp, filename)

    if os.path.exists(output_filepath):
        try:
            existing_data = load_pkl(output_filepath)
            return len(existing_data.get('t_trig', []))
        except Exception:
            return 0 # If reading existing file fails, assume 0 triggers from it.

    # 1. 定义和计算延伸
    pre_buffer_sec = config.lta_len
    post_buffer_sec = config.sta_len
    t_start_padded = t_start - timedelta(seconds=pre_buffer_sec)
    t_end_padded = t_end + timedelta(seconds=post_buffer_sec)

    # 一次性读取包含前后延伸的“宽”数据
    CCs_s_wide, RAWCCs_s_wide = Read_CCs_stack_h5(config.IN_CCs_stack_dir, template_name, t_start_padded, t_end_padded)
    
    # 如果读取不到数据则提前返回
    if CCs_s_wide is None or CCs_s_wide.size == 0:
        return 0

    # 2. 计算裁剪索引并执行裁剪
    pre_buffer_samples = int(pre_buffer_sec * config.fs)
    post_buffer_samples = int(post_buffer_sec * config.fs)
    
    # 使用 .shape[0] 来获取数组的实际长度，这比用时间计算更精确
    end_index = CCs_s_wide.shape[0] - post_buffer_samples
    start_index = pre_buffer_samples

    # 增加对 start_index < end_index 的检查，防止延伸部分比读取的数据还长
    if start_index >= end_index:
        return 0 # 如果裁剪后没有数据，则返回

    CCs_s = CCs_s_wide[start_index:end_index]
    RAWCCs_s = RAWCCs_s_wide[start_index:end_index]

    if CCs_s.size == 0 or RAWCCs_s.size == 0:
        return 0

    # 3. 计算并裁剪 STA/LTA
    # 对延伸后的完整数据计算，以利用向量化优势
    sl_pad_unnorm = sta_lta_centred_unnorm(CCs_s_wide, int(config.sta_len * config.fs), int(config.lta_len * config.fs))
    # 使用相同的索引裁剪STA/LTA，确保与CCs_s完美对齐
    sl_unnorm = sl_pad_unnorm[start_index:end_index] 

    # 4. 触发检测
    CCs_s_trig, t_trig, t_left, t_right, width_trig = Trigger_with_MADarray(
        t_start, t_end, config.fs, CCs_s,
        config.MAD_Avg_dict['t_array'], config.MAD_Avg_dict[template_name],
        config.Threshold_level, config.Minimal_event_interval
    )
    
    num_triggers_found = len(t_trig)

    if num_triggers_found == 0:
        return 0
    
    # 5. 计算触发点在数组中的索引
    idx_t_trig_float = np.array([(t - t_start).total_seconds() * config.fs for t in t_trig])
    idx_t_trig = np.maximum(0, np.round(idx_t_trig_float)).astype(int)

    # 6. 验证索引并过滤掉越界的触发点
    valid_indices_mask = (idx_t_trig >= 0) & (idx_t_trig < RAWCCs_s.shape[0]) & (idx_t_trig < CCs_s.shape[0])
    
    if not np.all(valid_indices_mask):
        idx_t_trig = idx_t_trig[valid_indices_mask]
        t_trig = [t_trig[i] for i, valid in enumerate(valid_indices_mask) if valid]
        CCs_s_trig = CCs_s_trig[valid_indices_mask]
        
        if width_trig is not None and len(width_trig) == len(valid_indices_mask):
            width_trig = width_trig[valid_indices_mask]
        if t_left is not None and len(t_left) == len(valid_indices_mask):
             t_left = [t_left[i] for i, valid in enumerate(valid_indices_mask) if valid]
        if t_right is not None and len(t_right) == len(valid_indices_mask):
             t_right = [t_right[i] for i, valid in enumerate(valid_indices_mask) if valid]
        
        num_triggers_found = len(t_trig)
        if num_triggers_found == 0:
            return 0

    RAWCCs_s_trig = RAWCCs_s[idx_t_trig]
    
    # 7. 计算 max_RAWCCs
    t_s = np.array([t_start + timedelta(seconds=ii/config.fs) for ii in range(len(CCs_s))])
    max_RAWCCs = []
    t_arg_max = []
    for ii in range(num_triggers_found):
        t_left_tmp = t_left[ii]
        t_right_tmp = t_right[ii]
        win_indices = np.where((t_s >= t_left_tmp) & (t_s <= t_right_tmp))[0]
        if win_indices.size > 0:
            win_RAWCCs = RAWCCs_s[win_indices]
            i_arg_max = np.argmax(np.abs(win_RAWCCs))
            max_RAWCCs.append(win_RAWCCs[i_arg_max])
            t_arg_max.append(t_s[win_indices][i_arg_max])
        else:
            max_RAWCCs.append(np.nan)
            t_arg_max.append(np.nan)

    # 8. 计算 max_stalta (优化后的高效版本)
    max_stalta = []
    # 将搜索窗口从秒转换为采样点数
    search_samples_left = int(config.search_win[0] * config.fs)
    search_samples_right = int(config.search_win[1] * config.fs)
    
    for trig_idx in idx_t_trig:
        # 直接使用索引计算窗口的起止位置
        win_start = trig_idx - search_samples_left
        win_end = trig_idx + search_samples_right

        # **重要：** 确保窗口索引不越界
        win_start_clipped = max(0, win_start)
        win_end_clipped = min(len(sl_unnorm), win_end)

        # 直接在索引上切片，效率极高
        win_stalta = sl_unnorm[win_start_clipped:win_end_clipped]

        # **重要：** 检查切片出的窗口是否为空
        if win_stalta.size > 0:
            max_stalta.append(np.max(win_stalta))
        else:
            # 如果窗口为空，添加一个占位符 NaN (Not a Number)，保持数据对齐
            max_stalta.append(np.nan)

    if len(max_stalta) != num_triggers_found:
        print(f"Warning: Mismatch in max_stalta length for {template_name} in {t_start}-{t_end}.")
        return 0

    # 9. 准备并保存输出字典
    mad_value_for_template = config.MAD_Avg_dict[template_name]
    if isinstance(mad_value_for_template, np.ndarray) and mad_value_for_template.size == 1:
        mad_value_for_template = mad_value_for_template.item()
    
    sccs_of_mad = CCs_s_trig / mad_value_for_template

    OUT_dict = {
        't_trig': t_trig,
        'CCs_s_trig': CCs_s_trig,
        'RAWCCs_s_trig': RAWCCs_s_trig,
        'width_trig': width_trig,
        't_left': t_left,
        't_right': t_right,
        'SCCs_of_MAD': sccs_of_mad,
        'max_RAWCCs': max_RAWCCs,
        't_arg_max': t_arg_max,
        'max_stalta': max_stalta,
    }
    write_pkl(OUT_dict, output_filepath)
            
    return num_triggers_found


def main():
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\nConfiguration parameters saved to:", os.path.join(config.OUTPUT_dir, 'config.json'))

    print(f"\nUsing {config.num_cpus} CPU(s) for parallel processing.")
    print(f'****** Triggering from {config.t_all_start} to {config.t_all_end} with time interval={config.t_delta}\n')

    print(f"Template names: {config.Temp_name_list}\n")
    print(f"Threshold level: {config.Threshold_level}\n")
    print(f"Minimal event interval: {config.Minimal_event_interval} seconds\n")

    overall_triggers_all_templates = 0

    for tt, template_name in enumerate(config.Temp_name_list):  
        OUTPUT_dir_byTemp = os.path.join(config.OUTPUT_dir, template_name)
        os.makedirs(OUTPUT_dir_byTemp, exist_ok=True)
        
        print(f'\n====> Processing Template: {template_name} ({tt + 1} of {len(config.Temp_name_list)})')      

        tasks = []
        for ii in range(len(config.t_array) - 1):
            t_start = config.t_array[ii]  
            t_end = config.t_array[ii+1]
            tasks.append((template_name, t_start, t_end, config, OUTPUT_dir_byTemp))

        NOtrig_Temp_total = 0
        
        # 使用 Pool 进行并行处理
        with Pool(processes=config.num_cpus) as pool:
            # tqdm 提供进度条
            results = list(tqdm(pool.imap_unordered(process_segment, tasks), total=len(tasks), desc=f"Segments for {template_name}"))
        
        NOtrig_Temp_total = sum(results)
        overall_triggers_all_templates += NOtrig_Temp_total
            
        print(f'\033[32mFinished processing for {template_name}. Found # {NOtrig_Temp_total} events.\033[0m')

    print(f"\n\n\033[32mAll templates processed! Total triggers found across all templates: {overall_triggers_all_templates}\033[0m")
    print("Please check the output directory:", config.OUTPUT_dir)


if __name__ == "__main__":
    # 在 __main__ 保护块中运行 main() 是多进程编程的最佳实践
    main()