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
# Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF/1_label_STALTA_with_template_plus_NOAeq.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from tqdm import tqdm

# =================================================================================
# 脚本功能:
# 该脚本旨在丰富原始的STA/LTA地震目录。
# 它通过与两个外部目录进行匹配，为每个STA/LTA事件添加更多维度的信息：
# 1. 与NOA官方目录匹配，以获得权威的震级和位置信息。
# 2. 与您自己生成的Matched Filter (MF)微震目录匹配，以获得MF检测的震级和模板归属。
# 最终输出一个“增强版”的STA/LTA目录，为后续的对比分析提供数据基础。
# =================================================================================


class Config:
    """
    配置类，用于集中管理所有文件路径和分析参数。
    """
    def __init__(self):
        # --- 基础路径 ---
        self.HOME_dir = f'{PROJECT_ROOT}/'
        self.level = 20

        # --- 输入数据路径 ---
        # 1. STA/LTA 原始目录
        # 本发布版自带的 STA/LTA 目录副本（外部 STA/LTA 工程的产物，详见 IN_STALTA_cata/）
        self.stalta_catalog_path = f"{PROJECT_ROOT}/IN_STALTA_cata/sta_lta_catalog.csv"

        # 2. Matched Filter (MF) 完整目录
        self.mf_catalog_path = os.path.join(self.HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{self.level}/MagRegression/Events_classified_Real_Mag.csv')

        # 3. NOA 官方目录
        self.noa_catalog_path = os.path.join(self.HOME_dir, 'IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt')

        # --- 输出路径 ---
        self.output_dir = os.path.join(self.HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{self.level}/eq_list_to_plot/')
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = 'stalta_catalog_enriched.csv'

        # --- 分析参数 ---
        # 时间窗口
        self.t_start = datetime(2024, 4, 24, 10, 20)
        self.t_end = datetime(2024, 6, 22, 17, 10)

        # 匹配容差 (秒)
        self.time_tolerance_noa = timedelta(seconds=3)
        self.time_tolerance_mf = timedelta(seconds=2)
        
        # NOA目录筛选参数
        self.lat_range = [36.5, 40.3800]
        self.lon_range = [18.844, 22.676]
        self.magnitude_cut = 0.0

def get_stalta_catalog(path, t_start, t_end):
    """
    加载并筛选STA/LTA目录。
    """
    print(f"--- 正在加载 STA/LTA 目录: {path}")
    df = pd.read_csv(path)
    df['peak_time'] = pd.to_datetime(df['peak_time'])
    mask = (df['peak_time'] >= t_start) & (df['peak_time'] <= t_end)
    df_filtered = df[mask].sort_values(by='peak_time').reset_index(drop=True)
    print(f"--- STA/LTA 目录加载完成。在指定时间范围内共有 {len(df_filtered)} 个事件。")
    return df_filtered

def read_cut_ims_df(config):
    """
    加载、解析并筛选NOA官方目录。
    此函数直接复用您提供的代码，并整合到配置类中。
    """
    print(f"--- 正在加载 NOA 目录: {config.noa_catalog_path}")
    # 假设 extract_station_info 函数在当前路径或PYTHONPATH中
    try:
        from extract_station_info import extract_station_info
    except ImportError:
        print("\n[错误] 无法导入 `extract_station_info` 函数。请确保该函数所在的脚本在PYTHONPATH中。\n")
        return pd.DataFrame()

    fsk_cata = extract_station_info(config.noa_catalog_path, 'FSK')
    df = pd.DataFrame(fsk_cata)
    df['EQ_time'] = pd.to_datetime(df['EQ_time'])
    df['STA_arrival_time'] = pd.to_datetime(df['STA_arrival_time'])
    for col in ['STA_distance', 'STA_azimuth', 'EQ_latitude', 'EQ_longitude', 'EQ_depth', 'EQ_magnitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    mask = (
        (df["EQ_time"] >= config.t_start) &
        (df["EQ_time"] <= config.t_end) &
        (df["EQ_latitude"] >= config.lat_range[0]) &
        (df["EQ_latitude"] <= config.lat_range[1]) &
        (df["EQ_longitude"] >= config.lon_range[0]) &
        (df["EQ_longitude"] <= config.lon_range[1]) &
        (df["EQ_magnitude"] >= config.magnitude_cut)
    )
    df_filtered = df[mask].copy()
    df_filtered.drop_duplicates(subset=['STA_arrival_time'], keep='first', inplace=True)
    df_filtered.sort_values(by='STA_arrival_time', inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    print(f"--- NOA 目录加载完成。筛选后得到 {len(df_filtered)} 个独立事件。")
    return df_filtered

def load_mf_catalog(path, t_start, t_end):
    """
    加载并筛选Matched Filter目录。
    """
    print(f"--- 正在加载 MF 目录: {path}")
    df = pd.read_csv(path, sep='\t', parse_dates=['Rt_trig'])
    mask = (df['Rt_trig'] >= t_start) & (df['Rt_trig'] <= t_end)
    df_filtered = df[mask].sort_values(by='Rt_trig').reset_index(drop=True)
    print(f"--- MF 目录加载完成。在指定时间范围内共有 {len(df_filtered)} 个事件。")
    return df_filtered

def enrich_stalta_catalog(df_stalta, df_noa, df_mf, config):
    """
    通过与NOA和MF目录进行时间匹配，丰富STA/LTA目录。
    """
    print("\n--- 开始执行目录匹配流程，为STA/LTA事件添加信息...")
    
    # 初始化用于存储匹配信息的新列
    new_cols = {
        'noa_match_flag': False, 'noa_time_diff_s': np.nan, 'noa_magnitude': np.nan,
        'noa_latitude': np.nan, 'noa_longitude': np.nan, 'noa_distance_km': np.nan,
        'mf_match_flag': False, 'mf_time_diff_s': np.nan, 'mf_template_name': 'unmatched',
        'mf_real_mag': np.nan, 'mf_sccs': np.nan
    }
    for col, default_val in new_cols.items():
        df_stalta[col] = default_val

    # 为了加速匹配，将时间列转换为Numpy数组
    noa_times = df_noa['STA_arrival_time'].to_numpy()
    mf_times = df_mf['Rt_trig'].to_numpy()

    # 使用tqdm显示进度条
    for i, stalta_event in tqdm(df_stalta.iterrows(), total=len(df_stalta), desc="[进度] 正在匹配STA/LTA事件"):
        stalta_time = stalta_event['peak_time']

        # --- 第一轮: 匹配NOA目录 ---
        time_diffs_noa = np.abs(noa_times - np.datetime64(stalta_time))
        if len(time_diffs_noa) > 0:
            min_diff_idx_noa = np.argmin(time_diffs_noa)
            # 显式转换为pandas.Timedelta以进行安全比较
            min_diff_noa = pd.to_timedelta(time_diffs_noa[min_diff_idx_noa])

            if min_diff_noa <= config.time_tolerance_noa:
                matched_noa_event = df_noa.iloc[min_diff_idx_noa]
                df_stalta.loc[i, 'noa_match_flag'] = True
                df_stalta.loc[i, 'noa_time_diff_s'] = min_diff_noa.total_seconds()
                df_stalta.loc[i, 'noa_magnitude'] = matched_noa_event['EQ_magnitude']
                df_stalta.loc[i, 'noa_latitude'] = matched_noa_event['EQ_latitude']
                df_stalta.loc[i, 'noa_longitude'] = matched_noa_event['EQ_longitude']
                df_stalta.loc[i, 'noa_distance_km'] = matched_noa_event['STA_distance'] * 111 # 转换为公里

        # --- 第二轮: 匹配MF目录 ---
        time_diffs_mf = np.abs(mf_times - np.datetime64(stalta_time))
        if len(time_diffs_mf) > 0:
            min_diff_idx_mf = np.argmin(time_diffs_mf)
            # 显式转换为pandas.Timedelta以进行安全比较
            min_diff_mf = pd.to_timedelta(time_diffs_mf[min_diff_idx_mf])

            if min_diff_mf <= config.time_tolerance_mf:
                matched_mf_event = df_mf.iloc[min_diff_idx_mf]
                df_stalta.loc[i, 'mf_match_flag'] = True
                df_stalta.loc[i, 'mf_time_diff_s'] = min_diff_mf.total_seconds()
                df_stalta.loc[i, 'mf_template_name'] = matched_mf_event['Template']
                df_stalta.loc[i, 'mf_real_mag'] = matched_mf_event['Real_Mag']
                df_stalta.loc[i, 'mf_sccs'] = matched_mf_event['CCs_s_trig']
    
    print("--- 目录匹配完成。")
    return df_stalta

def main():
    """
    主执行函数
    """
    print("=============================================")
    print("  脚本 1: STA/LTA 目录信息增强")
    print("=============================================\n")
    
    config = Config()

    # 1. 加载所有数据源
    df_stalta = get_stalta_catalog(config.stalta_catalog_path, config.t_start, config.t_end)
    df_noa = read_cut_ims_df(config)
    df_mf = load_mf_catalog(config.mf_catalog_path, config.t_start, config.t_end)

    if df_stalta.empty or df_noa.empty or df_mf.empty:
        print("\n[错误] 至少一个输入数据为空，无法继续执行。请检查文件路径和内容。")
        return

    # 2. 执行匹配和信息增强
    df_enriched = enrich_stalta_catalog(df_stalta, df_noa, df_mf, config)

    # 3. 结果统计和输出
    noa_matches = df_enriched['noa_match_flag'].sum()
    mf_matches = df_enriched['mf_match_flag'].sum()
    
    print("\n--- 最终统计结果 ---")
    print(f"总计 {len(df_enriched)} 个STA/LTA事件中:")
    print(f"  - {noa_matches} 个事件成功匹配到NOA目录 (时间容差: {config.time_tolerance_noa.total_seconds()}s)")
    print(f"  - {mf_matches} 个事件成功匹配到MF目录 (时间容差: {config.time_tolerance_mf.total_seconds()}s)")
    
    # 4. 保存最终的DataFrame
    output_path = os.path.join(config.output_dir, config.output_filename)
    df_enriched.to_csv(output_path, index=False, sep='\t')
    print(f"\n[成功] 增强后的STA/LTA目录已保存至:\n{output_path}")
    print("\n=================== 脚本 1 执行完毕 ===================")


if __name__ == '__main__':
    # 确保在需要时可以导入 extract_station_info
    # 将包含函数的目录添加到系统路径
    # 这是一种常见的做法，当功能函数不在标准库或已安装包中时
    sys.path.append(f'{PROJECT_ROOT}/functions/')
    main()
