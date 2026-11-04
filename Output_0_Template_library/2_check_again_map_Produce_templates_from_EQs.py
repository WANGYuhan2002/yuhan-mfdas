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
# ---
# jupyter:
#   jupytext:
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

#!____________INPUT: EQs info in NOA_______________

#!____________OUTPUT: waterfall plot and .npy unfilterer template file______________________



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from datetime import datetime, timedelta
import glob
import pandas as pd
import time

import h5py
import os
from re import split
#-- To import a function on a relative path:
import sys

import scipy.io as scio
import scipy.signal as signal
import h5py
import gc

from matplotlib import ticker
import matplotlib.dates as mdates

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

#!_______________________FUNCTIONS_____________________
sys.path.append(f"{PROJECT_ROOT}/functions/")
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, waterfall_date
from das_mffi_io import read_mffi_any, read_das_full_sum, das_full_sum_filter_all_in_one, mffi_filter_all_in_one, plot_together
import load_das_h5_PE2024 as load_das_h5_PE2024
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
from compute_correlation import preprocess

#!_______________________DATA IN________________________________import numpy as np

class Config:
    def __init__(self):
        
        # #* 输入路径 1, STA/LTA 目录
        # subdir = 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        # self.IN_cata_dir = f'{PROJECT_ROOT}/Run_0_preparing_templates_from_catalog/'
        # # self.IN_cata_dir += 'extract_threshold_300_trigNO_300_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'extract_threshold_100_trigNO_250_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'extract_threshold_300_trigNO_100_from_threshold_100_trigNO_100/'
        # # self.IN_cata_dir += 'threshold_120_trigNO_240_sta0.3s_lta3s/'
        # # self.IN_cata_dir += 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        # # self.IN_cata_dir += 'threshold_50_trigNO_50_sta0.3s_lta3s/'
        # self.IN_cata_dir += subdir

        # self.catalog_path = f"{self.IN_cata_dir}/sta_lta_catalog.csv"
        
        # self.IN_SNRed_cata_dir = f"{self.IN_cata_dir}Output_1_Find_NOAeq_Mag2.0_10sec_SNR40/"
        # self.SNRed_catalog_path = f"{self.IN_SNRed_cata_dir}matched_df.csv"
        
        # self.IN_SNRed_corr_dir = f"{self.IN_SNRed_cata_dir}Output_2_Corr_movmax/"
        # self.SNRed_corr_path = f"{self.IN_SNRed_corr_dir}correlation_matrix.npy"
        
        # self.IN_duplicated_template_dir = f"{self.IN_SNRed_cata_dir}Output_3_Delete_similar_eqs/"
        # self.duplicated_template_path = f"{self.IN_duplicated_template_dir}dup_temp_df.csv"
        
        self.input_dir = das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        
        
        #* 唯一的输入文件
        self.IN_Template_library_path = f'{PROJECT_ROOT}/Output_0_Template_library/Template_library.csv'
        

        #* 输出路径---1---- 画图输出
        self.OUTPUT_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Check_again_corr_map_waterfall/"

        #* 输出路径---2---- 模板npy保存
        self.npy_output_dir = f"{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/"

        # self.peak_time_left_pad = 8
        # self.peak_time_right_pad = 42


        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = 'filtered ' + str(self.freq_band[0]) + ' - ' + str(self.freq_band[1]) + ' Hz'

        # 空间通道范围.
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16*1.02095198631286

        self.save_as_temp_or_not = 1  # 是否保存为临时文件，1 为是，0 为否
        
        self.lat = [37.2, 39.4]
        self.lon = [20, 22.3]
        

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


def main():
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print("\n配置参数已保存到:", os.path.join(config.OUTPUT_dir, 'config.json'))
    
    
    Temp_lib_df = pd.read_csv(config.IN_Template_library_path, sep=',', parse_dates=["peak_time","STA_arrival_time","EQ_time","t_start","t_end"])
    event_times = Temp_lib_df["t_start"].tolist()
    

    temp_lib_in_plot_order = Temp_lib_df.sort_values(by='plot_order', ascending=True).reset_index(drop=True)

    
    corr_mat_temp = np.load(config.OUTPUT_dir + 'correlation_matrix.npy')
    
    
    os.makedirs(config.npy_output_dir, exist_ok=True)
    for index, row in Temp_lib_df.iterrows():

        t_start = row['t_start']
        t_end = row['t_end']

        tP = t_start + pd.Timedelta(seconds=row['t_P_sec'])
        tS = t_start + pd.Timedelta(seconds=row['t_S_sec'])

        #!---读取数据---
        print(f"\n处理地震事件: {index}: 震级 {row['EQ_magnitude']}, 距离{(row['STA_distance']*111):.1f}, 时间范围: {t_start} 到 {t_end}")
        das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        DIR_das = das_h5_mnt_80Hz
        data, headers, axis = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=DIR_das)
        
        #!存取----
        if config.save_as_temp_or_not == 1:
            # 保存原始数据为临时文件
            # temp_file_path = os.path.join(config.npy_output_dir, f"temp_{index}.npy")
            temp_file_path = os.path.join(config.npy_output_dir, f"{row['infile_name']}.npy")
            np.save(temp_file_path, data)
            # print(f"原始数据已保存到临时文件: {temp_file_path}")
            
        #!---画图---
        # data_plot = data[:,int(d1/dx):int(d2/dx)]
        # data_plot = preprocess(data,fs,f1,f2)
        data_filtered = preprocess(data[:, int(config.das_dist_start/config.dx):int(config.das_dist_end/config.dx)],\
                                    config.fs, config.freq_band[0], config.freq_band[1])
         
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        vlim = np.mean(np.abs(data))*5
        # plot grid
        t_left = t_start
        t_right = t_end
        d_bottom = config.das_dist_start / 1000  # Convert to km
        d_top = config.das_dist_end / 1000  # Convert to km
        im = ax.imshow(data_filtered.T, aspect="auto", cmap='seismic', interpolation='none', 
                extent=(t_left,t_right, d_top, d_bottom),
                vmin=-vlim, vmax=vlim)
        
        
        ax.axvline(tP, color='g', linestyle='--', label='P ')
        ax.axvline(tS, color='g', linestyle='--', label='S ')

        plt.colorbar(im,label='Strain rate (1/s)',ax=ax)
        
        #-- y-axis
        ax.set_ylabel("Distance [km]")
        # ax.set_yticks(np.arange(d_bottom, d_top+1, 1))    
        ax.set_yticks(np.arange(np.ceil(d_bottom), np.floor(d_top)+1, 1))    

        #-- Define the x-tick marks we want, format the labels
        Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=10, rotation=0,font=14)
        ax.set_xlabel('Time [GMT] on {0}'.format(t_start.strftime('%Y-%m-%d')))

        ax.set_title(f"template_{index}, ML={row['EQ_magnitude']}, distance to DAS={(row['STA_distance']*111):.1f} km\n SNR(RMS)={(np.sqrt(abs(row['peak_amp']))):.1f}, {config.filter_str}, {config.das_dist_start/1000:.1f} to {config.das_dist_end/1000:.1f} km", fontsize=14)
        ax.invert_yaxis()
        ax.grid(linestyle='--', linewidth=0.5)
        
        png_fpath = os.path.join(config.OUTPUT_dir, f"{row['infile_name']}_ML{row['EQ_magnitude']}_dist{(row['STA_distance']*111):.1f}.png")
        fig.savefig(png_fpath, dpi=300)
        plt.close(fig)
    

    #!==============画一张总图===============
    fig = plt.figure(figsize=(70, 30), dpi=500) # 宽度，高度
    # plt.figtext(0.02, 0.97, f"Total events in clusters={len(labels)}, noises={n_noise}, {para_str}", fontsize=18, ha='left')
    gs = plt.GridSpec(5, 4, figure=fig, height_ratios=[1,1,1,1,1], width_ratios=[1,1,1,1]) # adapt the gs according to the number of templates
    #!__________________________画各个template的waterfall_____________
    plt.rcParams['font.size'] = 24 # 设置字体大小
    axes = []
    for ii in range(5):
        for jj in range(3):
            axes.append(fig.add_subplot(gs[ii, jj]))
    print(f"axes: {len(axes)} > template numbers: {len(Temp_lib_df)}")
    
    for index, row in temp_lib_in_plot_order.iterrows():
        ax = axes[index]
        print(f"在子图画出地震事件: {index}: 震级 {row['EQ_magnitude']}, 距离{(row['STA_distance']*111):.1f}, 时间范围: {row['peak_time']}")
        t_start = row['t_start']
        t_end = row['t_end']
        
        das_h5_mnt_80Hz = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        DIR_das = das_h5_mnt_80Hz
        data, headers, axis = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=DIR_das)
        
        #!---画图---
        # data_plot = data[:,int(d1/dx):int(d2/dx)]
        # data_plot = preprocess(data,fs,f1,f2)
        data_filtered = preprocess(data[:, int(config.das_dist_start/config.dx):int(config.das_dist_end/config.dx)],\
                                    config.fs, config.freq_band[0], config.freq_band[1])
        vlim = np.mean(np.abs(data))*5
        # plot grid
        t_left = t_start
        t_right = t_end
        d_bottom = config.das_dist_start / 1000  # Convert to km
        d_top = config.das_dist_end / 1000  # Convert to km
        im = ax.imshow(data_filtered.T, aspect="auto", cmap='seismic', interpolation='none', 
                extent=(t_left,t_right, d_top, d_bottom),
                vmin=-vlim, vmax=vlim)
        
        plt.colorbar(im,label='Strain rate (1/s)',ax=ax)
        #-- y-axis
        ax.set_ylabel("Distance [km]")
        # ax.set_yticks(np.arange(d_bottom, d_top+1, 1))   
        ax.set_yticks(np.arange(np.ceil(d_bottom), np.floor(d_top)+1, 1))    

        #-- Define the x-tick marks we want, format the labels
        Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=10, rotation=0,font=14)
        ax.set_xlabel('Time [GMT] on {0}'.format(t_start.strftime('%Y-%m-%d')))
        ax.set_title(f"{row.plot_name}, index{index}, ML={row['EQ_magnitude']}, distance to DAS={(row['STA_distance']*111):.1f} km\n SNR(RMS)={(np.sqrt(abs(row['peak_amp']))):.1f}, {config.filter_str}, {config.das_dist_start/1000:.1f} to {config.das_dist_end/1000:.1f} km", fontsize=14)
        ax.invert_yaxis()
        ax.grid(linestyle='--', linewidth=0.5)
        
    #!_______________________________correlation matrix_____________
    ax = fig.add_subplot(gs[-2:, -1]) # 最后一行的所有列
    title = f"Correlation matrix of templates"
    corr_mat_plot = abs(corr_mat_temp.copy())
    im = ax.imshow(corr_mat_plot, cmap='Greys',norm=mcolors.LogNorm())#,clim=(0.001,0.006)) #viridis,Greys,
    #--label with template name
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.colorbar(im,label='correlation coefficient',ax=ax)
    ax.set_xticks(np.arange(corr_mat_plot.shape[0]))
    ax.set_yticks(np.arange(corr_mat_plot.shape[0]))
    ax.set_xlabel('Template index')
    ax.set_ylabel('Template index')
    #--add the value of each square
    no_temp = len(corr_mat_plot)
    for i in range(no_temp):
        for j in range(no_temp):
            text = ax.text(j, i, f'{corr_mat_plot[i, j]:.3f}', ha="center", va="center", color="w", fontdict={'size': 12})
    ax.set_title(title)

    #! 画出匹配的结果 ______________________plot 2______________lat lon mag plot on map_________________
    map_projection = ccrs.PlateCarree() # 选择一个地图投影
    plt.rcParams['font.size'] = 32 # 设置字体为 Arial, 以确保在图例中显示中文字符
    ax = fig.add_subplot(gs[:3, -1], projection=map_projection) # 这就是您要的 cartopy GeoAxes，名为 ax
    print(f"....画模版事件的分布于地图....")
    
    # matched_df = Temp_lib_df
    matched_df = temp_lib_in_plot_order.copy()  # 使用 temp_lib_in_plot_order 作为 matched_df
    
    title = f"Template library ML>2 with good SNR. #{len(matched_df)} earthquakes"
    # --- 2. 从配置中获取经纬度范围 ---
    lat_range = config.lat
    lon_range = config.lon
    # lat_range: 列表或元组, 定义地图显示的纬度范围, e.g., [min_lat, max_lat].
    # lon_range: 列表或元组, 定义地图显示的经度范围, e.g., [min_lon, max_lon].

    # 1. 添加 Cartopy 地理特征作为底图
    # 使用 '50m' 或 '110m' 分辨率的 Natural Earth 数据。'10m' 更精细但文件更大。
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor="#a3c9e0", zorder=0)
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


    #? 创建 GeoDataFrame for matched earthquakes matched_df
    gdf_matched = geopandas.GeoDataFrame(
        matched_df, # DataFrame 包含匹配到的地震数据 (STA/LTA可检测)
        geometry=geopandas.points_from_xy(matched_df.EQ_longitude, matched_df.EQ_latitude), # 从经纬度创建点几何对象
        crs="EPSG:4326"  # 定义坐标参考系统为 WGS84
    )
    # gdf_matched: 一个 GeoDataFrame, 包含了通过STA/LTA算法检测到的地震事件的地理空间信息.
    
    #? --- 3.4 地震点绘制 ---

    # 震级到标记大小的映射
    scaling_factor = 200 # 用于调整标记大小的缩放因子, 使震级差异在视觉上更明显.
    # 创建新列 'markersize', 其值基于地震震级的平方乘以缩放因子.
    # 震级平方使得面积与震级能量释放的某个代理量成正比 (虽然不是严格的能量).
    gdf_matched['markersize'] = (gdf_matched['EQ_magnitude'] ** 2) * scaling_factor

    # 绘制STA/LTA可检测到的地震 (gdf_matched)
    texts=[]
    for i, row in gdf_matched.iterrows():
        print(f"进度: {i}/{len(gdf_matched)}")
        # 使用 'markersize' 列来设置标记大小
        one_temp_df = gdf_matched.iloc[[i]]  # 获取当前行作为 DataFrame
        one_temp_df.plot(
            ax=ax,                    # 在同一轴上绘图
            marker='*',               # 标记形状为圆点
            # color='Grey',             # 标记颜色为蓝色
            markersize=one_temp_df['markersize'], # 标记大小根据计算的 'markersize' 列
            alpha=0.8,                # 标记的透明度
            edgecolor='black',        # 标记边框颜色
            linewidth=0.2,            # 标记边框宽度
            # label=f"Template {i}, ML={row['EQ_magnitude']}" # 图例中显示的标签
        )

        #----在每个点旁边添加文本标签显示序号----
        txt=ax.text(
            row['EQ_longitude'],        # 经度位置
            row['EQ_latitude'],         # 纬度位置
            f"{row['plot_name']} ML{row['EQ_magnitude']}",                     # 显示的文本 (序号)
            fontsize=20,                 # 字体大小
            ha='right',                 # 水平对齐方式
            va='center',                # 垂直对齐方式
            color='r'               # 文本颜色
        )
        texts.append(txt)  # 将文本对象添加到列表中
        
    from adjustText import adjust_text # 导入 adjust_text
    adjust_text(texts, 
            # 你可以添加一些参数来控制调整的行为，例如：
            force_points=0.2, # 点的排斥力
            force_text=0.2,   # 文本之间的排斥力
            arrowprops=dict(arrowstyle="-", color='black', lw=1) # 如果需要，可以添加指向原始点的箭头
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
        cx.add_basemap(ax, crs=gdf_matched.crs.to_string(), source=cx.providers.CartoDB.Positron, attribution_size=5)
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
    legend_magnitudes_display = [2.0, 3.0, 4.0] # 用于图例中显示的示例震级值
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
        fontsize=20,                  # 图例文字大小
        title_fontsize=20,            # 图例标题字体大小
        frameon=True                  # 给图例添加边框
    )
    ax.add_artist(size_legend) # 将震级大小图例添加到图中, 确保它不会覆盖颜色图例

    # --- 设置地图的显示范围 ---
    ax.set_xlim(lon_range[0], lon_range[1]) # 设置X轴 (经度) 的显示范围
    ax.set_ylim(lat_range[0], lat_range[1]) # 设置Y轴 (纬度) 的显示范围

    # --- 图表美化 ---
    ax.set_title(title, fontsize=24, pad=20)
    # set_title: 设置图表标题.
    # fontsize: 标题字体大小.
    # pad: 标题与图表顶部的间距.

    ax.set_xlabel('Longitude', fontsize=12) # 设置X轴标签
    ax.set_ylabel('Latitude', fontsize=12)  # 设置Y轴标签

    ax.grid(True, linestyle='--', alpha=0.5)
    
    
    png_fname = f"{config.OUTPUT_dir}ALL_templates_waterfall_map.png"
    fig.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    fig.savefig(png_fname, dpi=300)
    print(f"\n所有模板的水瀑图和地图已保存到: {png_fname}")
    
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")