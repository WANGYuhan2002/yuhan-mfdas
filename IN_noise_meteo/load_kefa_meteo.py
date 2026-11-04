# -----------------------------------------------------------------------------
# 脚本名称: 新数据探索与提取工具 (已修正)
# 脚本功能:
# 1. 加载并探索新的业务化（Operational）NetCDF数据。
# 2. 合并数据，并定义一个函数，用于在指定时空点提取连续的时间序列数据。
# 3. 提供一个完整的使用示例，将提取的数据输出为 pandas DataFrame。
#
# --- 改动说明 ---
# 1. [修正] extract_point_data 函数中，将时间坐标 'time' 改为 'valid_time'，以匹配源文件。
# 2. [修正] 示例代码中，将请求的变量 't2m' (不存在) 改为 'sst' (海面温度)，以成功演示。
#
# --- 抽包去硬编码说明 (check_minmag_pkg) ---
# 原脚本 (check_minmag_validity/Run_4_amp_spec/wind_speed/load_kefa_meteo.py:45)
# 把两个气象 CSV 的目录硬编码为绝对路径
#   "<作者原工程目录>/check_minmag_validity/Run_4_amp_spec/wind_speed/kefa_wind/"
# 这里改为：默认从本脚本同目录读取 (两个 CSV 已随包放在 meteo/ 下)，
# 并新增可选入参 data_dir 以便外部覆盖。函数名与位置参数 (t_start, t_end)
# 保持不变，确保 MF 发布版 `from load_kefa_meteo import load_kefa_meteo_data`
# 的调用契约 load_kefa_meteo_data(t_start, t_end) -> (atmos_df, wave_df) 不变。
# -----------------------------------------------------------------------------

# --- 导入所需库 ---
# 注意: 本模块 (load_kefa_meteo_data) 只读取已生成的 CSV, 不需要 xarray。
# 原文件顶部的 `import xarray as xr` 等仅服务于已删除的 NetCDF 探索代码,
# 抽包时移除以避免无谓的重依赖 (xarray 只有重算 CSV 时才需要, 见 acquire_era5/)。
import os
import pandas as pd
from datetime import datetime

# =============================================================================
# 第二部分: 数据预处理与函数定义
# =============================================================================


def load_kefa_meteo_data(t_start, t_end, data_dir=None):
    """
    从CSV文件加载并筛选Kefalonia的气象和海浪数据。

    参数:
    - t_start (datetime): 开始时间，一个datetime对象。
    - t_end (datetime): 结束时间，一个datetime对象。
    - data_dir (str, 可选): 两个 CSV 所在目录。默认 None -> 使用本脚本同目录
      (kefa_atmos_data.csv / kefa_wave_data.csv 已随包放在 meteo/ 下)。

    返回:
    - atmos_df (pd.DataFrame): 包含大气数据的DataFrame。
    - wave_df (pd.DataFrame): 包含海浪数据的DataFrame。
    - 如果文件不存在或出错，则返回 (None, None)。
    """
    print("\n===== 正在从CSV文件加载天气数据... =====")

    # --- [去硬编码] CSV 目录: 默认本脚本同目录 (与两个 CSV 同放 meteo/) ---
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    atmos_csv_path = os.path.join(data_dir, 'kefa_atmos_data.csv')
    wave_csv_path = os.path.join(data_dir, 'kefa_wave_data.csv')

    atmos_df = None
    wave_df = None

    try:
        # --- 1. 加载大气数据CSV ---
        print(f"--- 正在读取大气数据: {atmos_csv_path} ---")
        # index_col=0: 将第一列作为索引
        # parse_dates=True: 尝试将索引列解析为日期时间对象
        temp_atmos_df = pd.read_csv(atmos_csv_path, index_col=0, parse_dates=True)

        # --- 2. 根据时间范围筛选大气数据 ---
        # 直接使用.loc进行时间切片
        atmos_df = temp_atmos_df.loc[t_start:t_end].copy()
        print(f"--- 成功筛选出 {len(atmos_df)} 条大气数据记录 ---")

        # --- 3. 加载海浪数据CSV ---
        print(f"--- 正在读取海浪数据: {wave_csv_path} ---")
        temp_wave_df = pd.read_csv(wave_csv_path, index_col=0, parse_dates=True)

        # --- 4. 根据时间范围筛选海浪数据 ---
        wave_df = temp_wave_df.loc[t_start:t_end].copy()
        print(f"--- 成功筛选出 {len(wave_df)} 条海浪数据记录 ---")

    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print("请检查CSV文件名和路径是否正确。")
        return None, None
    except Exception as e:
        print(f"\n[错误] 读取或处理CSV文件时发生意外: {e}")
        return None, None

    return atmos_df, wave_df
