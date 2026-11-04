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
# -----------------------------------------------------------------------------
# ⚠️ 外部依赖 / 对外不可直接运行（非检测主线，仅复现论文插图）
#    本脚本依赖外部出图仓库 GRL_DAS_MF_plots/functions/（不在本发布版内）。
#    隔离环境下会 ImportError 而失败，可跳过；不影响检测主线与模板库产出。
# -----------------------------------------------------------------------------
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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import h5py
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# --- 科学计算与地理信息库 ---
import geopandas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as cx
from adjustText import adjust_text

# --- 自定义函数库 ---
# 确保你的自定义函数路径正确
sys.path.append(f"{PROJECT_ROOT}/external/GRL_DAS_MF_plots/functions/")  # [GitHub] external plotting helpers (not published); plot/load_das_h5_PE2024/compute_correlation resolve from functions/ first
from plot import Ticks_xaxis_minute_or_second
import load_das_h5_PE2024 as load_das_h5_PE2024
from compute_correlation import preprocess

# =============================================================================
# 🎨 1. 集中化绘图样式参数 (Centralized Plotting Style Parameters)
#    在此处修改参数，可统一调整整个图件的视觉风格，便于为发表进行微调
# =============================================================================
ftz = 35 # 全局字体大小
plt.rcParams.update(
    {
        'font.size': ftz,
        
    }
)

plot_style_params = {
    # --- 全局字体大小 ---
    "title_fontsize": ftz + 3,           # 子图标题
    "axis_label_fontsize": ftz,      # 坐标轴标签 (e.g., "Distance [km]")
    "tick_label_fontsize": ftz,      # 坐标轴刻度
    "colorbar_label_fontsize": ftz,  # Colorbar 标签
    "map_label_fontsize": ftz + 3,       # 地图上地震事件的标签
    "corr_matrix_text_fontsize": ftz-2,# 相关矩阵中的数值文本
    
    # --- 图例字体大小 ---
    "legend_fontsize": ftz,          # 图例文字
    "legend_title_fontsize": ftz+2,    # 图例标题
    
    # --- 其他参数 ---
    "figure_dpi": 160,              # 输出图像的DPI
    "vlim_percentile": 90,          # 瀑布图颜色范围的百分位
}


class Config:
    def __init__(self):
        
        HOME_dir = f"{PROJECT_ROOT}/Output_0_Template_library/"
        #* 唯一的输入文件
        self.IN_Template_library_path = f'{HOME_dir}Template_library.csv'

        self.input_dir = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        
        #* 输出路径---1---- 画图输出
        self.OUTPUT_dir = f"{HOME_dir}Check_again_corr_map_waterfall_for_paper/"
        self.corr_mat_path = f"{HOME_dir}Check_again_corr_map_waterfall/" + f"correlation_matrix.npy"

        #* 输出路径---2---- 模板npy保存
        self.npy_output_dir = f"{HOME_dir}Template_npy_files/"
        self.save_as_temp_or_not = 0  # 是否保存为临时文件，1 为是，0 为否

        # bandpass parameters,,, for visualization only
        self.fs = 80
        self.freq_band = [8, 20]
        self.filter_str = f'filtered {self.freq_band[0]} - {self.freq_band[1]} Hz'

        # 空间通道范围
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16 * 1.02095198631286

        # 地图经纬度范围
        self.lat = [37.9, 39.0]
        self.lon = [20.12, 21.04]

    def _save_json(self, path):
        """将配置保存为JSON文件以便追溯"""
        def is_json_serializable(value):
            try:
                json.dumps(value)
                return True
            except (TypeError, OverflowError):
                return False

        clean_dict = {k: v for k, v in self.__dict__.items() if is_json_serializable(v)}

        with open(path, 'w') as f:
            json.dump(clean_dict, f, indent=4, ensure_ascii=False)

def plot_waterfalls(axes, df, config, style_params):
    """
    绘制所有模板的瀑布图
    """
    print("...开始绘制瀑布图...")
    for index, row in df.iterrows():
        ax = axes[index]
        print(f"  在子图 {index} 中绘制事件: ML={row['EQ_magnitude']:.1f}, Dist={(row['STA_distance']*111):.1f} km")
        t_start, t_end = row['t_start'], row['t_end']
        
        # --- 读取数据 ---
        data, _, _ = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=config.input_dir)
        
        # --- 数据预处理 ---
        ch_start = int(config.das_dist_start / config.dx)
        ch_end = int(config.das_dist_end / config.dx)
        data_filtered = preprocess(data[:, ch_start:ch_end], config.fs, config.freq_band[0], config.freq_band[1])
        
        # --- 绘图 ---
        vlim = np.percentile(np.abs(data_filtered), style_params["vlim_percentile"])
        d_bottom_km = config.das_dist_start / 1000
        d_top_km = config.das_dist_end / 1000
        
        im = ax.imshow(data_filtered.T, aspect="auto", cmap='seismic', interpolation='none', 
                       extent=(t_start, t_end, d_top_km, d_bottom_km),
                       vmin=-vlim, vmax=vlim)
        
        # --- 设置坐标轴和标题 ---
        cbar = plt.colorbar(im, label='Strain Rate (1/s)', ax=ax, fraction=0.02, pad=0.04)
        cbar.ax.yaxis.label.set_size(style_params["colorbar_label_fontsize"])
        
        ax.set_ylabel("Distance (km)", fontsize=style_params["axis_label_fontsize"])
        ytick = np.arange(np.ceil(d_bottom_km), np.floor(d_top_km) + 1, 1)
        ax.set_yticks(ytick)
        
        Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=5, rotation=0, font=style_params["tick_label_fontsize"])
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', labelsize=style_params["tick_label_fontsize"])
        
        ax.set_xlabel(f'Time (GMT) on {t_start.strftime("%Y-%m-%d")}', fontsize=style_params["axis_label_fontsize"])
        title_text = (f"{row.plot_name}: ML={row['EQ_magnitude']:.1f}, "
                      f"d={(row['STA_distance']*111):.1f} km, "
                      f"SNR={(row['SNR_P']):.0f} (P), {(row['SNR_S']):.0f} (S)")
        ax.set_title(title_text, fontsize=style_params["title_fontsize"], pad=20, loc='center')
        
        ax.invert_yaxis()
        ax.grid(linestyle='--', linewidth=0.5)

def plot_correlation_matrix(ax, corr_matrix, labels, style_params):
    """
    绘制模板间的相关系数矩阵
    """
    print("...开始绘制相关系数矩阵...")
    im = ax.imshow(np.abs(corr_matrix), cmap='Greys', norm=mcolors.LogNorm())
    
    plt.colorbar(im, label='Correlation Coefficient', ax=ax)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=style_params["tick_label_fontsize"], rotation=-30, ha='left')
    ax.set_yticklabels(labels, fontsize=style_params["tick_label_fontsize"])
    
    ax.set_xlabel('Template Name', fontsize=style_params["axis_label_fontsize"])
    ax.set_ylabel('Template Name', fontsize=style_params["axis_label_fontsize"])
    
    # 在每个格子中添加数值
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr_matrix[i, j]:.3f}', ha="center", va="center", 
                    color="red", fontsize=style_params["corr_matrix_text_fontsize"])
            
    ax.set_title(f"Correlation Matrix of Templates", fontsize=style_params["title_fontsize"], loc ='center', pad=10)

def plot_map(ax, df, config, style_params):
    """
    在地图上绘制模板地震的位置
    """
    print("...开始绘制模板事件地理分布图...")
    
    # --- 添加地图底图和地理特征 ---
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor="#a3c9e0", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='gray', linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='darkgray', linewidth=0.6, linestyle=':', zorder=1)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': style_params["tick_label_fontsize"]}
    gl.ylabel_style = {'size': style_params["tick_label_fontsize"]}

    # --- 设置地图范围 ---
    ax.set_extent([config.lon[0], config.lon[1], config.lat[0], config.lat[1]], crs=ccrs.PlateCarree())
    
    # --- 准备数据 ---
    gdf_matched = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.EQ_longitude, df.EQ_latitude), crs="EPSG:4326"
    )
    
    # --- 绘制地震点 ---
    scaling_factor = 400
    gdf_matched['markersize'] = (gdf_matched['EQ_magnitude'] ** 2) * scaling_factor
    
    texts = []
    for _, row in gdf_matched.iterrows():
        gdf_matched.loc[[row.name]].plot(ax=ax, marker='*', markersize=row['markersize'], alpha=0.8, edgecolor='black', linewidth=0.2)
        texts.append(
            ax.text(row['EQ_longitude'], row['EQ_latitude'], f"{row['plot_name']} ML{row['EQ_magnitude']:.1f}",
                    fontsize=style_params["map_label_fontsize"], ha='right', va='center', color='r')
        )
    
    # 自动调整文本标签避免重叠
    adjust_text(texts, ax=ax, force_points=0.5, force_text=2.5,
                arrowprops=dict(arrowstyle="-", color='black', lw=1.5))
                
    # --- 添加在线地图底图 ---
    try:
        cx.add_basemap(ax, crs=gdf_matched.crs.to_string(), source=cx.providers.CartoDB.Positron, attribution_size=5)
    except Exception as e:
        print(f"无法加载 Contextily 底图: {e}")

    # --- 创建震级图例 ---
    legend_magnitudes_display = [2.0, 3.0, 4.0]
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label=f'ML {mag}',
               markerfacecolor='dimgray', markersize=np.sqrt((mag**2) * scaling_factor), linestyle='None')
        for mag in legend_magnitudes_display
    ]
    size_legend = ax.legend(handles=legend_elements, title="Local Magnitude", loc='upper right',
                              fontsize=style_params["legend_fontsize"], title_fontsize=style_params["legend_title_fontsize"], frameon=False)
    ax.add_artist(size_legend)
    
    # --- 美化 ---
    ax.set_title(f"Template Library: {len(df)} Earthquakes", fontsize=style_params["title_fontsize"], pad=20, loc='center')
    ax.set_xlabel('Longitude', fontsize=style_params["axis_label_fontsize"])
    ax.set_ylabel('Latitude', fontsize=style_params["axis_label_fontsize"])

def main():
    config = Config()
    os.makedirs(config.OUTPUT_dir, exist_ok=True)
    config._save_json(os.path.join(config.OUTPUT_dir, 'config.json'))
    print(f"\n配置参数已保存到: {os.path.join(config.OUTPUT_dir, 'config.json')}")
    
    # --- 读取和准备数据 ---
    Temp_lib_df = pd.read_csv(config.IN_Template_library_path, sep=',', parse_dates=["peak_time","STA_arrival_time","EQ_time","t_start","t_end"])
    Temp_lib_df.sort_values(by="plot_order", inplace=True)
    Temp_lib_df.reset_index(drop=True, inplace=True)
    
    corr_mat = np.load(config.corr_mat_path)
    
    # --- 设置全局绘图风格 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 创建主图和子图布局 ---
    fig = plt.figure(figsize=(31, 30), dpi=plot_style_params["figure_dpi"])
    gs = plt.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1.3, 1])
    
    #* --- 创建瀑布图的子图列表 ---
    waterfall_axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    
    # --- 调用模块化绘图函数 ---
    plot_waterfalls(waterfall_axes, Temp_lib_df, config, plot_style_params)
    
    #* --- 绘制相关矩阵 ---
    corr_ax = fig.add_subplot(gs[-2:, -1])
    plot_name_list = Temp_lib_df['plot_name'].tolist()
    plot_correlation_matrix(corr_ax, corr_mat, plot_name_list, plot_style_params)
    
    #* --- 绘制地图 ---
    map_ax = fig.add_subplot(gs[:3, -1], projection=ccrs.PlateCarree())
    plot_map(map_ax, Temp_lib_df, config, plot_style_params)


    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']
    axes = waterfall_axes + [corr_ax, map_ax]
    for ii,ax in enumerate(axes):
        plot_label = labels[ii]
        ax.text(-0.1, 1.05, plot_label,
        transform=ax.transAxes,
        fontsize=plot_style_params["title_fontsize"]+5,
        fontweight='bold',  # <--- 这个参数就是用来加粗的
        va='bottom',
        ha='left')
    
    # --- 保存最终图像 ---
    fig.tight_layout()  # 调整子图间距
    png_fname = os.path.join(config.OUTPUT_dir, "ALL_templates_waterfall_map.png")
    fig.savefig(png_fname)
    print(f"\n✅ 所有模板的水瀑图和地图已保存到: {png_fname}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")