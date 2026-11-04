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
# Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF/2_label_NOA_with_template_and_detectByMForNot.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from tqdm import tqdm
from geopy.distance import geodesic

# =================================================================================
# 脚本功能:
# 该脚本从NOA官方目录的视角出发，评估您的Matched Filter (MF)方法的性能。
# 它执行两个核心任务：
# 1. 空间标注：根据地理位置，判断每个NOA地震属于哪个MF模板（微震簇）的区域。
# 2. 检测匹配：根据事件时间，判断每个NOA地震是否被您的MF方法成功检测到。
# 最终输出一个带有丰富标签的NOA目录，用于后续的探测能力分析。
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
        # 1. NOA 官方目录
        self.noa_catalog_path = os.path.join(self.HOME_dir, 'IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt')

        # 2. Matched Filter (MF) 完整目录
        self.mf_catalog_path = os.path.join(self.HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{self.level}/MagRegression/Events_classified_Real_Mag.csv')

        # 3. 模板库文件 (必须包含经纬度信息)
        self.template_library_path = os.path.join(self.HOME_dir, "Output_0_Template_library/Template_library.csv")

        # --- 输出路径 ---
        self.output_dir = os.path.join(self.HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{self.level}/eq_list_to_plot/')
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = 'noa_catalog_labeled.csv'

        # --- 分析参数 ---
        # 时间窗口
        self.t_start = datetime(2024, 4, 24, 10, 20)
        self.t_end = datetime(2024, 6, 22, 17, 10)

        # 匹配容差
        self.time_tolerance_mf = timedelta(seconds=3)
        self.time_search_radius_km = 7.0  # 定义时间搜索模板区域的半径 (公里)
        
        self.spatial_radius_km = 8.0 # 定义模板区域的半径 (公里)

        # NOA目录筛选参数
        self.lat_range = [36.5, 40.3800]
        self.lon_range = [18.844, 22.676]
        self.magnitude_cut = 0.0

def read_cut_ims_df(config):
    """
    加载、解析并筛选NOA官方目录。
    """
    print(f"--- 正在加载 NOA 目录: {config.noa_catalog_path}")
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
    df_filtered.drop_duplicates(subset=['EQ_time'], keep='first', inplace=True) # 使用EQ_time去重
    df_filtered.sort_values(by='EQ_time', inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    print(f"--- NOA 目录加载完成。筛选后得到 {len(df_filtered)} 个独立地震事件。")
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

def load_template_library(path):
    """
    加载模板库，并检查是否包含位置信息。
    """
    print(f"--- 正在加载模板库: {path}")
    df = pd.read_csv(path, sep=',')
    print(f"--- 模板库加载完成:\n {df.describe()}")
    # 检查关键列是否存在
    required_cols = ['infile_name', 'EQ_latitude', 'EQ_longitude']
    if not all(col in df.columns for col in required_cols):
        print(f"\n[错误] 模板库文件 '{path}' 缺少必要的列。必须包含: {required_cols}")
        return None
    print(f"--- 模板库加载完成，共 {len(df)} 个模板。")
    return df

def label_noa_catalog(df_noa, df_mf, df_templates, config):
    """
    对NOA目录进行空间和时间上的双重标注。
    """
    print("\n--- 开始执行NOA目录标注流程...")

    # 初始化新列
    new_cols = {
        'template_region': 'outside_all_templates', 'dist_to_template_center_km': np.nan,
        'is_detected_by_mf': False, 'mf_detection_time_diff_s': np.nan,
        'mf_template_name': 'unmatched'
    }
    for col, default_val in new_cols.items():
        df_noa[col] = default_val

    # 准备用于匹配的数据
    mf_times = df_mf['Rt_trig'].to_numpy()
    template_coords = list(zip(df_templates['EQ_latitude'], df_templates['EQ_longitude']))
    template_names = df_templates['infile_name'].tolist()

    # 使用tqdm显示进度条
    for i, noa_event in tqdm(df_noa.iterrows(), total=len(df_noa), desc="[进度] 正在标注NOA事件"):
        noa_time = noa_event['STA_arrival_time']
        noa_coord = (noa_event['EQ_latitude'], noa_event['EQ_longitude'])

        # --- 第一轮: 空间区域标注 ---
        distances = [geodesic(noa_coord, t_coord).kilometers for t_coord in template_coords]
        if distances:
            min_dist_idx = np.argmin(distances)
            min_dist_km = distances[min_dist_idx]

            if min_dist_km <= config.spatial_radius_km:
                df_noa.loc[i, 'template_region'] = template_names[min_dist_idx]
                df_noa.loc[i, 'dist_to_template_center_km'] = min_dist_km

        # --- 第二轮: MF检测匹配 --- (在扩大一点的区域内)
        time_diffs_mf = np.abs(mf_times - np.datetime64(noa_time))
        if len(time_diffs_mf) > 0:
            min_diff_idx_mf = np.argmin(time_diffs_mf)
            # 显式转换为pandas.Timedelta以进行安全比较
            min_diff_mf = pd.to_timedelta(time_diffs_mf[min_diff_idx_mf])

            if min_diff_mf <= config.time_tolerance_mf and min_dist_km <= config.time_search_radius_km:
                matched_mf_event = df_mf.iloc[min_diff_idx_mf]
                df_noa.loc[i, 'is_detected_by_mf'] = True
                df_noa.loc[i, 'mf_detection_time_diff_s'] = min_diff_mf.total_seconds()
                df_noa.loc[i, 'mf_template_name'] = matched_mf_event['Template']

    print("--- NOA目录标注完成。")
    return df_noa

def main():
    """
    主执行函数
    """
    print("=============================================")
    print("  脚本 2: NOA 目录标注与MF性能评估")
    print("=============================================\n")

    config = Config()

    # 1. 加载所有数据源
    df_noa = read_cut_ims_df(config)
    df_mf = load_mf_catalog(config.mf_catalog_path, config.t_start, config.t_end)
    df_templates = load_template_library(config.template_library_path)

    if df_noa.empty or df_mf.empty or df_templates is None:
        print("\n[错误] 至少一个输入数据为空或格式不正确，无法继续执行。")
        return

    # 2. 执行标注
    df_labeled = label_noa_catalog(df_noa, df_mf, df_templates, config)

    # 3. 结果统计和输出
    mf_detections = df_labeled['is_detected_by_mf'].sum()
    in_template_region = (df_labeled['template_region'] != 'outside_all_templates').sum()
    
    print("\n--- 最终统计结果 ---")
    print(f"总计 {len(df_labeled)} 个NOA事件中:")
    print(f"  - {in_template_region} 个事件位于某个模板的 {config.spatial_radius_km}km 半径范围内。")
    print(f"  - {mf_detections} 个事件被MF方法成功检测到 (时间容差: {config.time_tolerance_mf.total_seconds()}s)。")
    
    # 4. 保存最终的DataFrame
    output_path = os.path.join(config.output_dir, config.output_filename)
    df_labeled.to_csv(output_path, index=False, sep='\t')
    print(f"\n[成功] 标注后的NOA目录已保存至:\n{output_path}")
    print("\n=================== 脚本 2 执行完毕 ===================")


if __name__ == '__main__':
    sys.path.append(f'{PROJECT_ROOT}/functions/')
    main()
