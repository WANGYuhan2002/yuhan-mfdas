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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import json
import time
import pickle
from datetime import datetime, timedelta
from types import MappingProxyType

# --- 科学计算、聚类与地理信息库 ---
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import geopandas
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as cx

# --- 自定义函数库 ---
# 确保你的自定义函数路径正确
sys.path.append(f"{PROJECT_ROOT}/functions")
from readers import Read_DAS_data, Read_CCs_stack_h5, write_pkl, load_pkl
from plot import Ticks_xaxis_day_or_hour
from extract_station_info import extract_station_info

# =============================================================================
# 🎨 1. 集中化绘图样式参数 (Centralized Plotting Style Parameters)
# =============================================================================
ftz = 30
plot_style_params = {
    # --- 全局字体大小 ---
    "figure_title_fontsize": ftz + 6,
    "subplot_title_fontsize": ftz + 6,
    "axis_label_fontsize": ftz,
    "tick_label_fontsize": ftz ,
    "corr_matrix_text_fontsize": ftz, # 相关矩阵中的簇编号

    # --- 图例字体大小 ---
    "legend_fontsize": ftz-6,
    "legend_title_fontsize": ftz -4,

    # --- 其他参数 ---
    "figure_dpi": 200,
    "map_marker_scaling_factor": 50, # 地图上震级标记的缩放因子
}


class Config:
    def __init__(self):
        # 时间范围筛选
        self.time_range = [
            datetime(2024, 4, 24, 10, 15),
            datetime(2024, 6, 22, 16, 15)
        ]
        self.t_str = f"_{self.time_range[0].strftime('%Y-%m-%d %H:%M:%S')}_{self.time_range[1].strftime('%Y-%m-%d %H:%M:%S')}"
        self.input_dir = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        #* 输入路径 1, STA/LTA 目录
        subdir = 'threshold_100_trigNO_100_sta0.3s_lta3s/'
        self.IN_cata_dir = f'{PROJECT_ROOT}/Run_0_preparing_templates_from_catalog/'
        self.IN_cata_dir += subdir
        self.catalog_path = f"{self.IN_cata_dir}/sta_lta_catalog.csv"
        self.IN_SNRed_cata_dir = f"{self.IN_cata_dir}Output_1_Find_NOAeq_Mag2.0_10sec_SNR30_100km/"
        self.SNRed_catalog_path = f"{self.IN_SNRed_cata_dir}matched_df.csv"
        self.IN_SNRed_corr_dir = f"{self.IN_SNRed_cata_dir}Output_2_Corr_movmax/"
        self.SNRed_corr_path = f"{self.IN_SNRed_corr_dir}correlation_matrix.npy"
        # 输出路径
        self.output_dir = f"{self.IN_SNRed_cata_dir}Output_3_Delete_similar_eqs_for_paper/"
        #* DBSCAN 聚类的参数
        self.eps = 0.90
        self.min_samples = 2
        self.lat = [38, 39.65]
        self.lon = [19.5, 21.6]
        self.magnitude_cut = 2.0
        # 信号处理参数
        self.fs = 80
        self.freq_band = [8, 20]
        # 空间通道范围
        self.dist_start = 8e3
        self.dist_end = 14e3

    def _save_json(self, path):
        """增强的 JSON 序列化方法"""
        def serializer(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            if isinstance(obj, timedelta): return obj.total_seconds()
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, MappingProxyType): return dict(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # 创建一个可序列化的字典副本
        serializable_dict = {}
        for k, v in self.__dict__.items():
            try:
                json.dumps(v, default=serializer)
                serializable_dict[k] = v
            except TypeError:
                serializable_dict[k] = f"NOT_SERIALIZABLE: <{type(v).__name__}>"

        with open(path, 'w') as f:
            json.dump(serializable_dict, f, indent=4, default=serializer, ensure_ascii=False)

def corr_DBSCAN_cluster(corr, eps, min_samples, output_dir):
    """使用DBSCAN对相关矩阵进行聚类分析"""
    print("\033[34m 执行 DBSCAN 聚类......=============================\033[0m")
    # 步骤 1: 准备距离矩阵
    R = np.abs(corr)
    np.fill_diagonal(R, 1) # 确保对角线为1
    R[np.isnan(R)] = 0 # 处理可能的NaN值
    D = 1 - R # 距离矩阵

    # 步骤 2: 执行DBSCAN
    print(f"\033[34m \t>>>参数: 领域半径eps={eps}, 最小样本数min_samples={min_samples}\033[0m")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    labels = dbscan.fit_predict(D)
    
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"\033[34m事件总个数: {len(labels)}, 其中 {len(labels)-n_noise} 个事件聚类为 {no_clusters} 簇, 噪声点个数: {n_noise}\033[0m")

    # 步骤 3: 保存聚类结果
    cluster_data = {cluster: np.where(labels == cluster)[0] for cluster in set(labels)}
    with open(os.path.join(output_dir, "cluster_data.pkl"), "wb") as f:
        pickle.dump(cluster_data, f)
    
    print("DBSCAN 聚类分析完成.\n")
    return cluster_data, labels

def run_clustering_and_select_templates(matched_df, correlation_matrix, config):
    """执行聚类并筛选出代表性模板"""
    # 步骤 1: 运行DBSCAN聚类
    cluster_data, labels = corr_DBSCAN_cluster(
        correlation_matrix, config.eps, config.min_samples, config.output_dir
    )

    # 步骤 2: 将聚类标签添加到DataFrame
    print("---为 DataFrame 添加聚类标签---")
    index_to_label = {idx: int(label) for label, indices in cluster_data.items() for idx in indices}
    matched_df['cluster'] = matched_df.index.map(index_to_label)

    # 步骤 3: 从每个簇中选取震级最大的事件作为代表
    print("---选取每个簇中震级最大的事件作为代表模板---")
    non_noise_df = matched_df[matched_df['cluster'] != -1]
    # 使用 .loc 避免 SettingWithCopyWarning
    idx_max_mag = non_noise_df.loc[non_noise_df.groupby('cluster')['EQ_magnitude'].idxmax()]
    
    # 步骤 4: 合并代表模板和所有噪声事件
    noise_df = matched_df[matched_df['cluster'] == -1]
    final_templates_df = pd.concat([idx_max_mag, noise_df], ignore_index=True)
    
    print(f"---->筛选完成，共获得 {len(final_templates_df)} 个模板 (每个簇中震级最大者 + 所有孤立事件)。")
    
    # 步骤 5: 提取这些最终模板对应的相关系数矩阵
    original_indices = final_templates_df['idx'].map(
        pd.Series(matched_df.index, index=matched_df['idx'])
    ).astype(int).values
    
    final_corr_matrix = correlation_matrix[original_indices, :][:, original_indices]
    np.save(os.path.join(config.output_dir, 'correlation_matrix_final_templates.npy'), final_corr_matrix)
    
    return final_templates_df, cluster_data, labels

def plot_rearranged_correlation_matrix(ax, corr, cluster_data, style_params):
    """绘制根据聚类结果重排的相关系数矩阵"""
    print("...绘制重排后的相关系数矩阵...")
    # 步骤 1: 生成重排后的索引
    sorted_indices = np.concatenate([indices for label, indices in sorted(cluster_data.items()) if label != -1] + 
                                    [cluster_data.get(-1, [])]).astype(int)
    
    # 步骤 2: 重排矩阵并绘图
    corr_rearranged = np.abs(corr[sorted_indices, :][:, sorted_indices])
    im = ax.imshow(corr_rearranged, cmap='rainbow', norm=plt.matplotlib.colors.LogNorm(vmin=1e-2, vmax=1))
    cbar = plt.colorbar(im, label='Correlation Coefficient', ax=ax)
    cbar.ax.tick_params(labelsize=style_params["tick_label_fontsize"])
    cbar.ax.set_ylabel('Pairwise Correlation Coefficient', fontsize=style_params["axis_label_fontsize"])

    # 步骤 3: 标记每个簇的区域
    current_pos = -0.5
    for label, indices in sorted(cluster_data.items()):
        if label == -1: continue # 跳过噪声
        size = len(indices)
        # 绘制矩形框
        rect = plt.Rectangle((current_pos, current_pos), size, size, linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        # 添加簇标签
        text_pos = current_pos + size / 2
        ax.text(text_pos, text_pos, str(label), color='k', 
                fontsize=style_params["corr_matrix_text_fontsize"], 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
        current_pos += size
        
    ax.set_xlabel('Event Index (rearranged by cluster)', fontsize=style_params["axis_label_fontsize"])
    ax.set_ylabel('Event Index (rearranged by cluster)', fontsize=style_params["axis_label_fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=style_params["tick_label_fontsize"])
    return ax

def plot_template_map(ax, df, config, style_params):
    """在地图上绘制聚类结果和模板分布"""
    print("...绘制模板地理分布图...")
    # --- 1. 设置地图背景和特征 ---
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor="#a3c9e0", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='gray', linewidth=0.7, zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': style_params["tick_label_fontsize"]}
    gl.ylabel_style = {'size': style_params["tick_label_fontsize"]}
    ax.set_extent([config.lon[0], config.lon[1], config.lat[0], config.lat[1]], crs=ccrs.PlateCarree())

    # --- 2. 准备地理数据 ---
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.EQ_longitude, df.EQ_latitude), crs="EPSG:4326")
    scaling_factor = style_params["map_marker_scaling_factor"]
    gdf['markersize'] = (gdf['EQ_magnitude'] ** 2) * scaling_factor

    # --- 3. 分组绘制不同簇的地震点 ---
    # 使用 tab20 颜色映射，确保有足够多的不同颜色
    cmap = plt.get_cmap('tab20')
    unique_clusters = sorted([c for c in gdf['cluster'].unique() if c != -1])
    
    for i, label in enumerate(unique_clusters):
        cluster_gdf = gdf[gdf['cluster'] == label]
        cluster_gdf.plot(ax=ax, marker='o', color=cmap(i % 20), markersize=cluster_gdf['markersize'],
                         alpha=0.7, edgecolor='black', linewidth=0.5, label=f'Cluster {label} (#{len(cluster_gdf)})')
    
    # 单独绘制噪声点
    noise_gdf = gdf[gdf['cluster'] == -1]
    if not noise_gdf.empty:
        noise_gdf.plot(ax=ax, marker='o', color='grey', markersize=noise_gdf['markersize'],
                       alpha=0.4, edgecolor='black', linewidth=0.5, label=f'Noise or 1-element cluster (#{len(noise_gdf)})')

    # --- 4. 添加在线地图底图 ---
    try:
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron, attribution_size=5)
    except Exception as e:
        print(f"无法加载 Contextily 底图: {e}")

    # --- 5. 创建图例 ---
    color_legend = ax.legend(loc='upper left', fontsize=style_params["legend_fontsize"], frameon=True)
    ax.add_artist(color_legend)

    legend_mags = [2.0, 3.0, 4.0]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'ML {mag}', markerfacecolor='dimgray',
                              markersize=np.sqrt((mag ** 2) * scaling_factor), linestyle='None') for mag in legend_mags]
    size_legend = ax.legend(handles=legend_elements, title="Local Magnitude", loc='upper right',
                              fontsize=style_params["legend_fontsize"], title_fontsize=style_params["legend_title_fontsize"], frameon=True)
    ax.add_artist(size_legend)

    # --- 6. 美化 ---
    ax.set_xlabel('Longitude', fontsize=style_params["axis_label_fontsize"])
    ax.set_ylabel('Latitude', fontsize=style_params["axis_label_fontsize"])
    return ax

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    config._save_json(os.path.join(config.output_dir, 'config.json'))
    print(f"配置参数已保存到: {os.path.join(config.output_dir, 'config.json')}")
    
    # --- 1. 数据加载 ---
    print("\n1. 加载备选模板事件序列 (经SNR和震级筛选)...")
    initial_templates_df = pd.read_csv(config.SNRed_catalog_path, sep='\t', parse_dates=["peak_time","STA_arrival_time",'EQ_time'])
    initial_templates_df.reset_index(drop=True, inplace=True) # 确保索引从0开始
    
    print("2. 加载相关系数矩阵...")
    correlation_matrix = np.load(config.SNRed_corr_path, allow_pickle=True)

    # --- 2. 核心分析：聚类与去重 ---
    print("\n3. 对备选模板序列进行 DBSCAN 聚类与去重...")
    final_templates_df, cluster_data, labels = run_clustering_and_select_templates(initial_templates_df, correlation_matrix, config)

    # --- 3. 创建并保存最终图件 ---
    print("\n4. 开始生成汇总图件...")
    fig = plt.figure(figsize=(32, 12)) 
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
    
    # 左图：重排后的相关矩阵
    ax_corr = fig.add_subplot(gs[0, 0])
    plot_rearranged_correlation_matrix(ax_corr, correlation_matrix, cluster_data, plot_style_params)
    ax_corr.set_title(f"Rearranged PCC Matrix by DBSCAN Clusters",  #\n(eps={config.eps}, min_samples={config.min_samples})
                      fontsize=plot_style_params["subplot_title_fontsize"])

    # 右图：模板分布地图
    ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    plot_template_map(ax_map, initial_templates_df, config, plot_style_params)
    ax_map.set_title(f"{len(initial_templates_df)} earthquakes grouped by waveform similarity", 
                     fontsize=plot_style_params["subplot_title_fontsize"])
    
    labels = ['a)', 'b)']
    axes = [ax_corr, ax_map]
    for ii,ax in enumerate(axes):
        plot_label = labels[ii]
        ax.text(-0.1, 1.05, plot_label,
        transform=ax.transAxes,
        fontsize=plot_style_params["subplot_title_fontsize"]+2,
        fontweight='bold',  # <--- 这个参数就是用来加粗的
        va='bottom',
        ha='left')

    fig.tight_layout(pad=3.0)
    png_fpath = os.path.join(config.output_dir, f"DBSCAN_clustering_and_map_summary.png")
    fig.savefig(png_fpath, dpi=plot_style_params["figure_dpi"], bbox_inches='tight')
    print(f"\n✅ 汇总图像已保存到: {png_fpath}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序总运行时间: {end_time - start_time:.2f} 秒")