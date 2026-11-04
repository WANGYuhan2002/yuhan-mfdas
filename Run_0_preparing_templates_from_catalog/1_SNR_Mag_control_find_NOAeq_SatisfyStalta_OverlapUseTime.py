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
from readers import Read_DAS_data, Read_CCs_stack_h5
from readers import write_pkl, load_pkl
from plot import Ticks_xaxis_day_or_hour
from extract_station_info import extract_station_info

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

import geopandas
from matplotlib.lines import Line2D
import geodatasets
import contextily as cx
import cartopy.crs as ccrs # cartopy 坐标参考系统模块
import cartopy.feature as cfeature # cartopy 地理特征模块




class Config:
    def __init__(self):
        # 时间范围筛选
        self.time_range = [
            datetime(2024, 4, 24, 10, 15),
            datetime(2024, 6, 22, 16, 15)
            # datetime(2024, 4, 25, 16, 15)
        ]
        self.t_str = f"_{self.time_range[0].strftime('%Y-%m-%d %H:%M:%S')}_{self.time_range[1].strftime('%Y-%m-%d %H:%M:%S')}"
        

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
        
        
        #* 输入路径, 对比的NOA事件目录, IMS 原始文件读取
        self.lat = [36.5, 40.3800]
        self.lon = [18.844, 22.676]
        
        
        self.as_one_ev_interval = 10 # the maximum gap time between two events to be considered as one event (noa and stalta)
        # note this act as + and -
        
        self.magnitude_cut = 2.0

        self.SNR_cut_again = 30 # SNR cut for the matched filter, to be used in the next step
        self.peak_amp_cut_again = self.SNR_cut_again ** 2
        # SNR in. RMS, = sqrt(peak_amp) for STALTA
        
        
        # 输入路径
        # self.IMS_file_path = f'{PROJECT_ROOT}/IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt'
        self.IMS_file_path = f'{PROJECT_ROOT}/IN_NOA_cata/100km_distance/IMS_20240425_20240623_319.txt'
        
        #* 输出路径
        self.output_dir = self.IN_cata_dir + f'Output_1_Find_NOAeq_Mag{self.magnitude_cut}_{self.as_one_ev_interval}sec_SNR{self.SNR_cut_again}_100km/'
        
        # 信号处理参数
        self.fs = 80
        self.freq_band = [8, 20]

        # 空间通道范围
        self.dist_start = 8.3e3
        self.dist_end = 14e3


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
def load_catalog(config):
    """加载事件目录"""
    df = pd.read_csv(config.catalog_path, parse_dates=["peak_time"], delimiter=',') 
    mask = (df["peak_time"] >= config.time_range[0]) & \
           (df["peak_time"] <= config.time_range[1])
    return df[mask]["peak_time"].tolist()

def load_stalta_df(config):
    """加载事件目录"""
    df = pd.read_csv(config.catalog_path, parse_dates=["peak_time"], delimiter=',') 
    mask = (df["peak_time"] >= config.time_range[0]) & \
           (df["peak_time"] <= config.time_range[1]) & \
            (df["peak_amp"] >= config.peak_amp_cut_again)
           
    return df[mask]

def read_cut_IMS_df(config):
    FSK_cata = extract_station_info(config.IMS_file_path, 'FSK')
    df = pd.DataFrame(FSK_cata)
    df['EQ_time'] = pd.to_datetime(df['EQ_time'])
    df['STA_arrival_time'] = pd.to_datetime(df['STA_arrival_time'])
    for col in ['STA_distance', 'STA_azimuth', 'EQ_latitude', 'EQ_longitude', 'EQ_depth', 'EQ_magnitude']:
        df[col] = df[col].astype(float)
    
    mask = 	(df["EQ_time"] >= config.time_range[0]) & \
        (df["EQ_time"] <= config.time_range[1]) & \
        (df["EQ_latitude"] >= config.lat[0]) & \
        (df["EQ_latitude"] <= config.lat[1]) & \
        (df["EQ_longitude"] >= config.lon[0]) & \
        (df["EQ_longitude"] <= config.lon[1]) & \
        (df["EQ_magnitude"] >= config.magnitude_cut)
    
    print("筛选前的数据:", df)
    print("筛选条件:", mask)
    df = df[mask]
    
    df.drop_duplicates(subset=['STA_arrival_time'], keep='first', ignore_index=True)
    df = df.sort_values(by='STA_arrival_time')
    df = df.reset_index(drop=True)

    print("筛选后的数据:", df)
    print("筛选后的数据长度:", len(df),)

    return df


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    config._save_json(os.path.join(config.output_dir, 'config.json'))
    print("配置参数已保存到:", os.path.join(config.output_dir, 'config.json'))
    
    print("1. 加载 STA/LTA 事件目录...")
    df_stalta = load_stalta_df(config)
    df_stalta.reset_index(drop=True, inplace=True)
    
    event_times_1 = df_stalta["peak_time"].tolist()
    print(f"\t目录1 事件数量 (STA/LTA): {len(event_times_1)}")
    event_times_1 = np.sort(pd.to_datetime(event_times_1))
    # print("event_times_1:", event_times_1)
    # to original python datetime form
    event_times_1 = pd.to_datetime(event_times_1).to_pydatetime().tolist()
    
    print("2. 加载 NOA 事件目录...")
    df_noa = read_cut_IMS_df(config)
    print(f"\tNOA 目录限制条件: \n"
          f"\t\t时间范围: {config.time_range[0]} - {config.time_range[1]}\n"
          f"\t\t经度范围: {config.lon[0]} - {config.lon[1]}\n"
          f"\t\t纬度范围: {config.lat[0]} - {config.lat[1]}\n"
          f"\t\t震级范围: {config.magnitude_cut} 以上\n"
          f"\t\t筛选后的NOA事件数量: {len(df_noa)}")
    
    
    
    #!________________________________利用时间差匹配，将NOA事件和STALTA事件进行匹配____________________
    # 得到STALTA事件的对应的NOA事件，记录在stalta_noa_df中
    # stalta_noa_df = pd.DataFrame(columns=['idx', 'peak_time', 'STA_arrival_time', 'STA_distance', 'STA_azimuth', 'EQ_time', 'EQ_latitude', 'EQ_longitude', 'EQ_depth', 'EQ_magnitude'])
    print(f"\n\033[32m========================================================= \033[0m ")
    stalta_noa_list = list()
    for i, row in df_stalta.iterrows():
        t1 = event_times_1[i]  # 当前STALTA事件的时间
        if t1 != row['peak_time']:
            raise ValueError(f"第{i}个STALTA事件时间 ({t1}) 与目录中的时间不一致 ({row['peak_time']})")
        
        # 找到与当前STALTA事件时间最接近的NOA事件
        closest_idx = (df_noa['STA_arrival_time'] - t1).abs().idxmin()
        closest_event = df_noa.loc[closest_idx]

        # 计算时间差
        time_diff_sec = (closest_event['STA_arrival_time'] - t1)
        time_diff_sec = pd.to_timedelta(time_diff_sec).total_seconds()
        time_diff_sec = abs(time_diff_sec)
        
        if time_diff_sec > config.as_one_ev_interval:
            print(f"XXXXXX 第{i}个STALTA事件 ({t1}) 与最近NOA事件 ({closest_event['STA_arrival_time']}) 的时间差{time_diff_sec}秒, 超过 {config.as_one_ev_interval}秒, 跳过")
            # 标记 match_or_not = 0
            match_or_not = 0
        elif time_diff_sec <= config.as_one_ev_interval:
            print(f"\033[32m YYYYYY 第{i}个STALTA事件 ({t1}) 与最近NOA事件 ({closest_event['STA_arrival_time']}) 的时间差: {time_diff_sec}秒, 进行匹配 \033[0m")
            # 标记 match_or_not = 1
            match_or_not = 1
            
        # 将匹配结果添加到DataFrame中
        stalta_noa_list.append({
            'idx': i,
            'peak_time': t1,
            'peak_amp': row['peak_amp'],
            'peak_score': row['peak_score'],
            'max_amp_mean': row['max_amp_mean'],
            
            'match_or_not': match_or_not,
            'time_diff_sec': time_diff_sec,
            
            'STA_arrival_time': closest_event['STA_arrival_time'],
            'STA_distance': closest_event['STA_distance'],
            'STA_azimuth': closest_event['STA_azimuth'],
            'EQ_time': closest_event['EQ_time'],
            
            'EQ_latitude': closest_event['EQ_latitude'],
            'EQ_longitude': closest_event['EQ_longitude'],
            'EQ_depth': closest_event['EQ_depth'],
            'EQ_magnitude': closest_event['EQ_magnitude']
        })
    print(f"\n\033[32m========================================================= \033[0m ")
    
    
    stalta_noa_df = pd.DataFrame(stalta_noa_list)
    stalta_noa_df = stalta_noa_df.sort_values(by='idx')

    mask = (stalta_noa_df['match_or_not'] == 1 )
        
    matched_df = stalta_noa_df[mask]
    matched_df.reset_index(drop=True, inplace=True)

    # 检查一下有没有不同stalta 的时间对应到同一个地震去的。（就是打印出EQ_time 相同的）
    print(f"检查重合对应到同一个地震的stalta检测: ")
    
    count = 0
    for eq_time, group in matched_df.groupby('EQ_time'):
        if len(group) > 1:
            print(f"\nEQ_time = {eq_time} 匹配了 {len(group)} 个 STALTA 事件:")
            print(group[['idx', 'peak_time', 'STA_arrival_time', 'time_diff_sec', 'match_or_not']])
            count += len(group) - 1
            
    print(f"\n\033[31m========================================================= \033[0m ")
    print(f"实际上的匹配得到的地震是 {len(matched_df)} - 重复的{count} = {len(matched_df) - count} 个, 使用的合并间隔为{config.as_one_ev_interval}秒")
    
    df_noa_unique = df_noa.drop_duplicates(subset=['EQ_time'], keep='first', ignore_index=True)
    df_noa_unique.reset_index(drop=True, inplace=True)
    print(f"---这些匹配成功的地震都在 {len(df_noa)} 个 NOA arrival 去重得到的 {len(df_noa_unique)} 个地震中")
    
    print(f"OK, 下面对这 {len(df_noa_unique)} 个 NOA 地震中的 {len(matched_df) - count} 个 stalta 地震进行对比分析")
    print(f"\033[31m========================================================= \033[0m ")
    
    
    # 存储匹配结果
    matched_df.to_csv(config.output_dir + 'matched_df.csv', sep='\t')
    print('\n >>> matched_df : \n',matched_df)
    
    
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(1, 2, figure=fig, height_ratios=[1], width_ratios=[1, 1])
    # 使用 df_noa_unique, matched_df_unique

    matched_df_unique = matched_df.drop_duplicates(subset=['EQ_time'], keep='first', ignore_index=True)
    matched_df_unique.reset_index(drop=True, inplace=True)
    
    
    label_fsk = f'FSK EQs in NOA network = #{len(df_noa_unique)}'
    label_matched = f'Detectable EQs by STA/LTA = #{len(matched_df_unique)}'
    title = f"Among in total #{len(event_times_1)} STA/LTA detections,\n #{len(matched_df_unique)} detections is also recorded in NOA catalog"
    
    #! 画出匹配的结果 ______________________plot 1_______________magnitude - distance plot_________________
    ax = fig.add_subplot(gs[0, 0])
    # print(df_noa_unique[['STA_distance', 'EQ_magnitude']].head() )
    # 画所有 NOA eq 在 200km半径内
    ax.scatter(df_noa_unique['STA_distance']*111, df_noa_unique['EQ_magnitude'], color='grey', label=label_fsk, alpha=0.3, s=40, marker='o')
    
    # 画所有 STALTA eq 匹配到的 overlapped
    ax.scatter(matched_df_unique['STA_distance']*111, matched_df_unique['EQ_magnitude'], color='red', label=label_matched, alpha=0.3, s=40, marker='*')

    ax.legend(loc='upper left')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Magnitude')
    ax.set_xscale('log')
    ax.set_xlim([3, 250])
    ax.set_title('Magnitude - Distance plot')
    
    
    #! 画出匹配的结果 ______________________plot 2_______________lat lon mag plot on map_________________
    map_projection = ccrs.PlateCarree() # 选择一个地图投影
    ax = fig.add_subplot(gs[0, 1], projection=map_projection) # 这就是您要的 cartopy GeoAxes，名为 ax
    
    # --- 2. 从配置中获取经纬度范围 ---
    lat_range = config.lat
    lon_range = config.lon
    # lat_range: 列表或元组, 定义地图显示的纬度范围, e.g., [min_lat, max_lat].
    # lon_range: 列表或元组, 定义地图显示的经度范围, e.g., [min_lon, max_lon].



    # 1. 添加 Cartopy 地理特征作为底图
    # 使用 '50m' 或 '110m' 分辨率的 Natural Earth 数据。'10m' 更精细但文件更大。
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#a0c3d8', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='gray', linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='darkgray', linewidth=0.6, linestyle=':', zorder=1)
    # ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='#a0c3d8', edgecolor='gray', zorder=0) # 可选
    # ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='#a0c3d8', zorder=0) # 可选

    # 添加网格线和标签
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    # from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter # 如果需要自定义格式
    # gl.xformatter = LongitudeFormatter()
    # gl.yformatter = LatitudeFormatter()

    # 设置地图显示的经纬度范围 (假设 lon_range, lat_range 已定义)
    if 'lon_range' in locals() and 'lat_range' in locals():
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    else: # Fallback if not defined, you might want to remove this or set a default
        ax.set_global() # Or some default extent




    # --- 3. 数据准备和 GeoDataFrame 创建 ---
    # 列: 'STA_distance', 'STA_azimuth', 'EQ_latitude', 'EQ_longitude', 'EQ_depth', 'EQ_magnitude' 和 'STA_arrival_time', 'EQ_time'
    # 画图关键列: 'EQ_latitude', 'EQ_longitude', 'EQ_magnitude' (cluster_id 未在此绘图中使用颜色区分)

    #? 创建 GeoDataFrame for all NOA earthquakes
    gdf_noa = geopandas.GeoDataFrame(
        df_noa_unique, # DataFrame 包含地震数据
        geometry=geopandas.points_from_xy(df_noa_unique.EQ_longitude, df_noa_unique.EQ_latitude), # 从经纬度创建点几何对象
        # 注意: 这里使用了 df_noa 的经纬度, 假设 df_noa_unique 是 df_noa 去重后的版本,
        # 或者 df_noa 包含了原始的、可能重复的记录, 而 df_noa_unique 用于绘图的数据点.
        # 关键是确保 geometry 的行数与 df_noa_unique 的行数匹配.
        crs="EPSG:4326"  # 定义坐标参考系统为 WGS84 (标准的经纬度坐标系)
    )
    # gdf_noa: 一个 GeoDataFrame, 包含了所有FSK网络记录的地震事件的地理空间信息.

    #? 创建 GeoDataFrame for matched earthquakes
    gdf_matched = geopandas.GeoDataFrame(
        matched_df_unique, # DataFrame 包含匹配到的地震数据 (STA/LTA可检测)
        geometry=geopandas.points_from_xy(matched_df_unique.EQ_longitude, matched_df_unique.EQ_latitude), # 从经纬度创建点几何对象
        crs="EPSG:4326"  # 定义坐标参考系统为 WGS84
    )
    # gdf_matched: 一个 GeoDataFrame, 包含了通过STA/LTA算法检测到的地震事件的地理空间信息.
    
    #? --- 3.4 地震点绘制 ---

    # 震级到标记大小的映射
    scaling_factor = 20 # 用于调整标记大小的缩放因子, 使震级差异在视觉上更明显.
    # 创建新列 'markersize', 其值基于地震震级的平方乘以缩放因子.
    # 震级平方使得面积与震级能量释放的某个代理量成正比 (虽然不是严格的能量).
    gdf_noa['markersize'] = (gdf_noa['EQ_magnitude'] ** 2) * scaling_factor
    gdf_matched['markersize'] = (gdf_matched['EQ_magnitude'] ** 2) * scaling_factor

    # 绘制所有FSK网络记录的地震 (gdf_noa)
    gdf_noa.plot(
        ax=ax,                    # 在指定的轴上绘图
        marker='o',               # 标记形状为圆形
        color='Grey',             # 标记颜色为灰色
        markersize=gdf_noa['markersize'], # 标记大小根据计算的 'markersize' 列
        alpha=0.3,                # 标记的透明度 (0=完全透明, 1=完全不透明)
        edgecolor='black',        # 标记边框颜色
        linewidth=0.2,            # 标记边框宽度
        label=label_fsk # 图例中显示的标签
    )

    # 绘制STA/LTA可检测到的地震 (gdf_matched)
    gdf_matched.plot(
        ax=ax,                    # 在同一轴上绘图
        marker='*',               # 标记形状为点 (通常比'o'小, 但这里大小由markersize控制)
        color='Red',              # 标记颜色为红色
        markersize=gdf_matched['markersize'], # 标记大小根据计算的 'markersize' 列
        alpha=0.3,                # 标记的透明度, 比背景点稍不透明以突出显示
        edgecolor='black',        # 标记边框颜色
        linewidth=0.2,            # 标记边框宽度
        label=label_matched # 图例中显示的标签
    )

    #* --- 添加 Contextily 底图 ---
    # 注意：Contextily 会自动根据您数据的范围和 CRS 来添加底图。
    # 确保 gdf_earthquakes 有正确的 CRS 设置 (我们的是 "EPSG:4326")
    try:
        # cx.add_basemap(ax, crs=gdf_earthquakes.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
        # 您可以选择不同的地图源 (provider)
        # 例如 Stamen TonerLite (黑白简约风格):
        # cx.add_basemap(ax, crs=gdf_earthquakes.crs.to_string(), source=cx.providers.Stamen.TonerLite)
        # CartoDB Positron (浅色简约):
        cx.add_basemap(ax, crs=gdf_noa.crs.to_string(), source=cx.providers.CartoDB.Positron, attribution_size=5)
        # 更多提供商: https://contextily.readthedocs.io/en/latest/providers_deepdive.html
    except Exception as e:
        print(f"无法加载 Contextily 底图: {e}")
        print("请确保已安装 'contextily' (pip install contextily) 并且网络连接正常。")



    # --- 创建图例 ---
    # 颜色图例 (解释不同颜色的点代表什么)
    # ax.legend() 会自动查找绘图时通过 `label` 参数指定的标签.
    # 它将使用 gdf_noa.plot 和 gdf_matched.plot 中的 'label' 来创建图例项.
    color_legend = ax.legend(
        # title="Data Source", # 可以给这个图例加一个标题
        loc='upper left',         # 图例位置在左上角
        fontsize=10,              # 图例文字大小
        # title_fontsize=12       # 如果有标题, 这是标题的字体大小
        frameon=True              # 给图例添加边框
    )
    ax.add_artist(color_legend) # 将此图例添加到图中. 如果有多个图例, 需要用 add_artist 添加第一个,
                                # 后续的 legend() 调用会替换前一个, 除非也用 add_artist.
                                # 在这种情况下, 我们有两个独立的 legend() 调用, 所以都需要 add_artist.
                                # 然而, 通常的做法是第一个 legend() 不用 add_artist, 后续的用.
                                # 或者, 一次性创建所有 legend handles 并用一个 legend() 调用.
                                # 为了安全和明确, 我们将为两个图例都使用 add_artist.

    # 震级大小图例
    # 这个图例手动创建, 用于解释不同大小的标记代表的震级.
    legend_magnitudes_display = [1.0, 2.0, 3.0] # 用于图例中显示的示例震级值
    # 根据与绘图时相同的缩放逻辑计算图例标记的面积
    legend_marker_sizes_area = [(m ** 2) * scaling_factor for m in legend_magnitudes_display]

    legend_elements_size_corrected = []
    for i, mag_display in enumerate(legend_magnitudes_display):
        # Line2D 的 markersize 参数与标记的直径(或其某个线性度量)成正比.
        # 由于我们之前用 (mag**2)*scale 计算的是与面积成正比的值,
        # 所以这里需要对面积开方, 得到与直径成正比的值, 以便 Line2D 正确显示大小.
        legend_elements_size_corrected.append(
            Line2D([0], [0],                 # 虚拟的坐标点 (不会实际绘制)
                marker='o',               # 标记形状
                color='w',                # 标记颜色 (设为白色, 仅边框和面色可见)
                label=f'ML {mag_display}', # 图例标签
                markerfacecolor='dimgray',# 标记填充色 (深灰色, 以便在图例中清晰可见)
                markersize=np.sqrt(legend_marker_sizes_area[i]), # 标记大小 (直径)
                linestyle='None'          # 不显示连接线
            )
        )

    size_legend = ax.legend(
        handles=legend_elements_size_corrected, # 手动创建的图例元素
        title="Local magnitude",      # 图例标题
        loc='upper right',            # 图例位置在右上角
        fontsize=10,                  # 图例文字大小
        title_fontsize=12,            # 图例标题字体大小
        frameon=True                  # 给图例添加边框
    )
    ax.add_artist(size_legend) # 将震级大小图例添加到图中, 确保它不会覆盖颜色图例

    # --- 设置地图的显示范围 ---
    ax.set_xlim(lon_range[0], lon_range[1]) # 设置X轴 (经度) 的显示范围
    ax.set_ylim(lat_range[0], lat_range[1]) # 设置Y轴 (纬度) 的显示范围

    # --- 图表美化 ---
    ax.set_title(title, fontsize=16, pad=20)
    # set_title: 设置图表标题.
    # fontsize: 标题字体大小.
    # pad: 标题与图表顶部的间距.

    ax.set_xlabel('Longitude', fontsize=12) # 设置X轴标签
    ax.set_ylabel('Latitude', fontsize=12)  # 设置Y轴标签

    ax.grid(True, linestyle='--', alpha=0.5)
    # grid: 添加网格线.
    # True: 显示网格.
    # linestyle='--': 网格线样式为虚线.
    # alpha=0.5: 网格线透明度.

    # ax.set_aspect('equal') # 可选: 设置X轴和Y轴的单位长度相同.
    # 对于地理图, 这可以确保地图比例在视觉上是正确的 (例如, 一个正方形的地理区域看起来是正方形).
    # 但有时会导致图形在页面上占用过多或过少空间, 具体取决于经纬度范围.
    # 如果使用 contextily, 它通常会处理好比例问题.

    plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域, 避免标签重叠或出界.
    plt.show()         # 显示图形.

    # --- 保存图形 ---
    # 构建保存路径
    output_filename = os.path.join(config.output_dir, "Detectable_STALTA_in_FSK_EQs.png")
    fig.savefig(output_filename, dpi=300)
    # savefig: 将图形保存到文件.
    # output_filename: 文件路径和名称.
    # dpi=300: 设置图像分辨率 (dots per inch), 300 DPI 通常足够出版质量.

    print(f"图表已保存到: {output_filename}")
    
    #______________________________map plot end
    
    
    #!  检查 STA_arrival_time 和 STALTA的 peak_time 各自的时间差 _____________________________
    fig, axes = plt.subplots(2,1,figsize=(10, 6))
    bins = np.arange(0, 30, 0.5)

    ax = axes[0]
    
    stalta_tdiff = np.diff(event_times_1)
    stalta_tdiff = np.array([abs(td.total_seconds()) for td in stalta_tdiff])
    
    ax.hist(stalta_tdiff, bins=bins, color='orange', alpha=0.5)
    ax.set_title('Neighboring time difference of STALTA')
    ax.set_xlabel('Time difference (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_xticks(bins[::4])

    ax = axes[1]
    # STA_arrival_time = df_noa['STA_arrival_time'].values
    # STA_arrival_time = np.sort(pd.to_datetime(STA_arrival_time))
    # noa_tdiff = (np.diff(STA_arrival_time))
    # noa_tdiff = noa_tdiff.astype('timedelta64[s]').astype(float)  # 已是以秒为单位的
    # noa_tdiff = abs(noa_tdiff)  # 取绝对值
    
    # check the time difference of arrivals between different earthquakes!
    noa_tdiff = []
    for i in range(len(df_noa)-1):
        t1 = df_noa['STA_arrival_time'].iloc[i]
        t2 = df_noa['STA_arrival_time'].iloc[i+1]
        
        t1 = pd.to_datetime(t1).to_pydatetime()
        t2 = pd.to_datetime(t2).to_pydatetime()
        tdiff = (t2 - t1).total_seconds()
        if df_noa['EQ_time'].iloc[i] == df_noa['EQ_time'].iloc[i+1]:
            if tdiff > 0:
                print(f"arrivals from the same earthquake: {tdiff}秒")
        else:
            noa_tdiff.append(tdiff)
    print("NOA 有效事件时间差:", len(noa_tdiff), "个 of", len(df_noa)-1, "个")
    
    ax.hist(noa_tdiff, bins=bins, color='blue', alpha=0.5)
    ax.set_title('Neighboring time difference of STA_arrival_time from different earthquakes')
    ax.set_xlabel('Time difference (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_xticks(bins[::3])
    
    plt.tight_layout()
    fig.savefig(config.output_dir + 'Check_time_diff_histogram.png', dpi=300)
    
if __name__ == "__main__":
    main()