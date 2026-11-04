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
        
        self.IN_SNRed_cata_dir = f"{self.IN_cata_dir}Output_1_Find_NOAeq_Mag2.0_10sec_SNR30/"
        self.SNRed_catalog_path = f"{self.IN_SNRed_cata_dir}matched_df.csv"
        
        self.IN_SNRed_corr_dir = f"{self.IN_SNRed_cata_dir}Output_2_Corr_movmax/"
        self.SNRed_corr_path = f"{self.IN_SNRed_corr_dir}correlation_matrix.npy"
        
        # 输出路径
        self.output_dir = f"{self.IN_SNRed_cata_dir}Output_3_Delete_similar_eqs/"

        #* DBSCAN 聚类的参数
        self.eps = 0.90
        self.min_samples = 2
        
        # self.lat = [36.5, 40.3800]
        # self.lon = [18.844, 22.676]
        
        self.lat = [37.2, 39.4]
        self.lon = [20, 22.3]
        
        self.magnitude_cut = 2.0
        # # 输入路径
        # self.IMS_file_path = f'{PROJECT_ROOT}/IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt'

        # 信号处理参数
        self.fs = 80
        self.freq_band = [5, 20]

        # 空间通道范围
        self.dist_start = 8e3
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
def plot_corr_rearranged_by_cluster(ax, corr, cluster_data):
    #? Step 5.1: 根据聚类重新排列corr矩阵， 以及画出各个聚类的时间分布
    #!## cluster_data[ii] is the index of the the iith cluster, cluster_data[-1] is the index of the noise
    if -1 in cluster_data:
        no_clusters = len(cluster_data) - 1
    else:
        no_clusters = len(cluster_data)
    
    print(f"总样本数: {corr.shape[0]}, 聚类的簇数量: {no_clusters}:")
    for k, v in list(cluster_data.items()):  # 只显示前3个簇
        print(f"  Cluster {k}: {len(v)} samples")
    if -1 in cluster_data[0]:
        print(f"  Noise: {len(cluster_data[-1])} samples")
    
    index_rearranged = np.array([])
    for ii in range(no_clusters):
        index_rearranged = np.append(index_rearranged,cluster_data[ii])
    if -1 in cluster_data:
        index_rearranged = np.append(index_rearranged,cluster_data[-1])
    index_rearranged = index_rearranged.astype(int)

    #*-------plot the corr_matrix according to the rearranged index-----------
    corr_matrix_rearranged = corr[index_rearranged,:][:,index_rearranged]
    corr_matrix_rearranged = abs(corr_matrix_rearranged)
    #?_____________________________________________________________________-
    im = ax.imshow(corr_matrix_rearranged, cmap='rainbow',norm=mcolors.LogNorm(),clim=(1e-2,1)) #viridis,Greys,

    # label each index block with the cluster index
    pos = 0.5
    for i in range(no_clusters):
        pos = pos + len(cluster_data[i]) / 2
        ax.text(pos,pos,str(i),color='k',fontsize=30,horizontalalignment='center', verticalalignment='center')
        pos = pos + len(cluster_data[i]) / 2
        
    edge = -0.5
    for i in range(no_clusters):
        # plot a rectangle marking the cluster square
        rect = plt.Rectangle((edge,edge),len(cluster_data[i]),len(cluster_data[i]),linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        edge = edge + len(cluster_data[i])

    plt.colorbar(im,label='correlation value',ax=ax)
    # ax.set_title(title)
    ax.set_xlabel('index of the events')
    ax.set_ylabel('index of the events')
    return ax
    
    
def corr_DBSCAN_cluster(corr, eps, min_samples, OUTPUT_dir):
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    import pickle


    #? Step 1: check corr
    # 加载互相关矩阵
    R = abs(corr)
    # 将对角元素设置为1（确保自相关性为1）
    # check the correlation matrix: if symmetry and if diag=1
    print("\033[33m 检查相关系数矩阵... ==========================\033[0m")
    print('R shape',R.shape)
    print('--1----R diag')#,np.where(np.diag(R)!=1))
    # print(np.diag(R))
    is_symmetric = np.array_equal(R, R.T)
    print("\n--2----Matrix is symmetric:" if is_symmetric else "Matrix is not symmetric.")

    # 检查是否存在负元素
    has_negative = np.any(R < 0)
    print("\n--3----Matrix has negative elements:" if has_negative else "Matrix has no negative elements.")
    num_negatives = np.sum(R < 0)
    print(f"Number of negative elements: {num_negatives}")

    epsilon = 1e-5  # 容差范围
    print('\n--4----must be abs<1+ε, the >1 elements number:', len(np.where(np.abs(R) > 1 + epsilon)[0]) )

    # 检查是否有 NaN 值
    if np.isnan(R).any():
        print("\n--5----Distance matrix contains NaN values.")
    nan_count = np.sum(np.isnan(R))
    print(f"Number of NaN values: {nan_count}")
    # 用一个较大的常量（如最大距离）替换 NaN
    R[np.isnan(R)] = 0#np.nanmean(D)

    np.fill_diagonal(R, 1)

    # Step 1: 计算距离矩阵
    D = 1 - (R)  # 直接用 1 - R 表示距离矩阵
    # print(D)

    # #? Step 2: DBSCAN聚类

    # 文件路径
    labels_file = os.path.join(OUTPUT_dir, "cluster_labels.npy")
    tsne_file = os.path.join(OUTPUT_dir, "tsne_coordinates.npy")
    plot_file = os.path.join(OUTPUT_dir, "dbscan_clustering_plot.png")

    # 检查是否已有保存的结果
    if os.path.exists(labels_file) and os.path.exists(tsne_file):
        print("covering old results...")
        
    #!________________________________________________

    # Step 2: 聚类分析（DBSCAN）
    print("\033[34m 执行 DBSCAN 聚类......=============================\033[0m")
    print(f"\033[34m \t>>>参数: 领域半径eps={eps}, 最小样本数min_samples={min_samples}.................\033[0m")
    # eps=0.28 # 领域半径，值越小越严格，聚类越精细；值越大越宽松，聚类越粗糙广泛
    # min_samples=4 # 样本数阈值，值越小越容易形成簇，值越大越难形成簇
    para_str = f"eps={eps}, min_samples={min_samples}"
    # print(f"DBSCAN 参数: {para_str}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)  # 开启多线程并行
    labels = dbscan.fit_predict(D)
    np.save(labels_file, labels)

    # Step 3: 降维（t-SNE）
    print("执行 t-SNE 降维...")
    data_2d = TSNE(n_components=2, metric="precomputed", init="random", random_state=42, n_jobs=-1).fit_transform(D)
    np.save(tsne_file, data_2d)

    # 查看聚类结果
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"\033[34m事件总个数: {len(labels)}, 其中 {len(labels)-n_noise} 聚类为 {no_clusters} 簇, 剩下的噪声点个数: {n_noise}\033[0m")

    # Step 4: 绘制聚类结果
    print("...plotting 降维图...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab20", s=10, alpha=0.8)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("DBSCAN Clustering Results:" + para_str)
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.savefig(plot_file, dpi=300)
    plt.close()    
    print(f"......plots saved to {plot_file}")
    
    # Step 5: 保存每个簇对应的样本事件
    print("......cluster data saving...")
    cluster_data = {cluster: np.where(labels == cluster)[0] for cluster in set(labels)}
    with open(os.path.join(OUTPUT_dir, "cluster_data.pkl"), "wb") as f:
        pickle.dump(cluster_data, f)

    print("DONE! 聚类分析完成.\n")
    return cluster_data, labels

def load_SNRed_df(config):
    """加载匹配的 DataFrame, 经过SNR和magnitude筛选"""
    df = pd.read_csv(config.SNRed_catalog_path, sep='\t', parse_dates=["peak_time","STA_arrival_time",'EQ_time'])
    mask = (df["peak_time"] >= config.time_range[0]) & \
           (df["peak_time"] <= config.time_range[1])
    return df[mask].reset_index(drop=True)

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
           (df["peak_time"] <= config.time_range[1])
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
    
    print(f"1. 加载备选template序列 (SNR Magnitude control 过的 STALTA 和 NOA 综合事件序列)...")
    matched_df = load_SNRed_df(config)
    
    print(f"2. 加载备选 template 序列的 correlation matrix")
    correlation_matrix = np.load(config.SNRed_corr_path, allow_pickle=True)

    print(f"-------------对备选 template 序列 依照 correlation matrix 进行 DBSCAN 聚类---------------\n")
    cluster_data, labels = corr_DBSCAN_cluster(correlation_matrix, config.eps, config.min_samples, config.output_dir)
    
    
    fig = plt.figure(figsize=(40, 15)) 
    # plt.figtext(0.02, 0.97, f"Total events in clusters={len(labels)}, noises={n_noise}, {para_str}", fontsize=18, ha='left')
    gs = plt.GridSpec(1, 4, figure=fig, height_ratios=[1], width_ratios=[2, 1.2, 1.2, 2]) # adapt the gs according to the number of templates


    print("...plot 重新排列的correlation matrix...")#!_____________________________
    ax =  fig.add_subplot(gs[0, 0])
    title = f"Rearranged correlation matrix by DBSCAN clusters, eps={config.eps}, min_samples={config.min_samples}"
    ax = plot_corr_rearranged_by_cluster(ax, correlation_matrix, cluster_data)
    ax.set_title(title, fontsize=20)


    print(f"---给matched_df添加聚类标签---")
    # 第一步：把 cluster_data 转成“索引 → 标签”的形式
    index_to_label = {}
    for label, indices in cluster_data.items():
        for i in indices:
            index_to_label[i] = int(label)  # 或者写成 label 保持字符串
    # 第二步：加到 matched_df 中
    matched_df['cluster'] = matched_df.index.map(lambda x: index_to_label.get(x, -1))

    print(f"---选取每组中 magnitude 最大的那个作为模版")
    # 先筛选出 cluster 不等于 -1 的数据
    filtered_df = matched_df[matched_df['cluster'] != -1]
    # 找到每个 cluster 中 EQ_magnitude 最大值对应的索引
    idx_max_eq = filtered_df.groupby('cluster')['EQ_magnitude'].idxmax()
    # 用这些索引从 filtered_df 中取出对应的行
    dup_temp_df = filtered_df.loc[idx_max_eq]
    # 加上noise中所有事件
    dup_temp_df = pd.concat([dup_temp_df, matched_df[matched_df['cluster'] == -1]], ignore_index=True)
    
    print(f"---->筛选出每个簇中 EQ_magnitude 最大的事件 + 孤立事件 作为模版，共 {len(dup_temp_df)} 个事件。")
    print(dup_temp_df[['cluster', 'EQ_magnitude', 'EQ_time', 'STA_arrival_time']])
    
    #* 从correlation_matrix 中抽出 dup_temp_df 的idx和 matched_df中的idx相同的对应的列______
    indices = np.zeros(len(dup_temp_df), dtype=int)
    for ii, row in dup_temp_df.iterrows():
        idx = row['idx']
        # 在matched_df中找到对应idx相等的行数
        indices[ii] = matched_df.index[matched_df['idx'] == idx].tolist()[0]
        print(f'{ii}th row, idx={idx}, matched_df index={indices[ii]}')
    corr_matrix_dup_temp = correlation_matrix[indices.astype(int), :][:, indices.astype(int)]
    np.save(os.path.join(config.output_dir, 'corr_matrix_dup_temp.npy'), corr_matrix_dup_temp)
    
    
    ax =  fig.add_subplot(gs[0, 3])
    title = f"Correlation matrix of duplicate templates"
    im = ax.imshow(corr_matrix_dup_temp, cmap='Greys',norm=mcolors.LogNorm())#,clim=(0.001,0.006)) #viridis,Greys,
    #--label with template name
    plt.colorbar(im,label='correlation coefficient',ax=ax)
    #--add the value of each square
    no_temp = corr_matrix_dup_temp.shape[0]
    for i in range(no_temp):
        for j in range(no_temp):
            text = ax.text(j, i, f'{corr_matrix_dup_temp[i, j]:.3f}', ha="center", va="center", color="w", fontsize=7)
    ax.set_title(title)

    
    #*_________________________________
    
    
    dup_temp_df.sort_values(by='cluster', inplace=True)
    dup_temp_df.to_csv(os.path.join(config.output_dir, 'dup_temp_df.csv'), index=False)
    matched_df.sort_values(by='cluster', inplace=True)
    matched_df.to_csv(os.path.join(config.output_dir, 'matched_df_with_clusters.csv'), index=False)

    #! 画出匹配的结果 ______________________plot 1______________lat lon mag plot on map_________________
    map_projection = ccrs.PlateCarree() # 选择一个地图投影
    ax = fig.add_subplot(gs[0, 1], projection=map_projection) # 这就是您要的 cartopy GeoAxes，名为 ax
    
    title = f"Initial template library ML>2 with good SNR = #{len(matched_df)}"
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
    scaling_factor = 20 # 用于调整标记大小的缩放因子, 使震级差异在视觉上更明显.
    # 创建新列 'markersize', 其值基于地震震级的平方乘以缩放因子.
    # 震级平方使得面积与震级能量释放的某个代理量成正比 (虽然不是严格的能量).
    gdf_matched['markersize'] = (gdf_matched['EQ_magnitude'] ** 2) * scaling_factor

    # 绘制STA/LTA可检测到的地震 (gdf_matched)
    for label in gdf_matched['cluster'].unique():
        # 过滤出当前簇的地震数据
        cluster_df = gdf_matched[gdf_matched['cluster'] == label]
        if not cluster_df.empty:
            if label != -1:
                # 绘制每个簇的地震点
                cluster_df.plot(
                    ax=ax,                    # 在同一轴上绘图
                    marker='o',               # 标记形状为圆点
                    # color='blue',             # 标记颜色为蓝色
                    markersize=cluster_df['markersize'], # 标记大小根据计算的 'markersize' 列
                    alpha=0.5,                # 标记的透明度
                    edgecolor='black',        # 标记边框颜色
                    linewidth=0.2,            # 标记边框宽度
                    label=f'Cluster {label} = #{len(cluster_df)} ' # 图例中显示的标签
                )
            elif label == -1:
                cluster_df.plot(
                    ax=ax,                    # 在同一轴上绘图
                    marker='o',               # 标记形状为圆点
                    color='Grey',             # 标记颜色为蓝色
                    markersize=cluster_df['markersize'], # 标记大小根据计算的 'markersize' 列
                    alpha=0.3,                # 标记的透明度
                    edgecolor='black',        # 标记边框颜色
                    linewidth=0.2,            # 标记边框宽度
                    label=f'Noise = #{len(cluster_df)} ' # 图例中显示的标签
                )
    
    # gdf_matched.plot(
    #     ax=ax,                    # 在同一轴上绘图
    #     marker='*',               # 标记形状为点 (通常比'o'小, 但这里大小由markersize控制)
    #     color='Red',              # 标记颜色为红色
    #     markersize=gdf_matched['markersize'], # 标记大小根据计算的 'markersize' 列
    #     alpha=0.3,                # 标记的透明度, 比背景点稍不透明以突出显示
    #     edgecolor='black',        # 标记边框颜色
    #     linewidth=0.2,            # 标记边框宽度
    #     label=f'initial template library ML>2 with good SNR = #{len(matched_df)}' # 图例中显示的标签
    # )


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
    
    
    #! 画出匹配的结果 ______________________plot 2______________lat lon mag plot on map_________________
    map_projection = ccrs.PlateCarree() # 选择一个地图投影
    ax = fig.add_subplot(gs[0, 2], projection=map_projection) # 这就是您要的 cartopy GeoAxes，名为 ax
    matched_df = dup_temp_df # 使用筛选后的 DataFrame
    title = f"Duplicated template library ML>2 with good SNR = #{len(matched_df)}"
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
    scaling_factor = 20 # 用于调整标记大小的缩放因子, 使震级差异在视觉上更明显.
    # 创建新列 'markersize', 其值基于地震震级的平方乘以缩放因子.
    # 震级平方使得面积与震级能量释放的某个代理量成正比 (虽然不是严格的能量).
    gdf_matched['markersize'] = (gdf_matched['EQ_magnitude'] ** 2) * scaling_factor

    # 绘制STA/LTA可检测到的地震 (gdf_matched)
    texts = []  # 用于存储文本标签对象
    for i, row in gdf_matched.iterrows():
        # 使用 'markersize' 列来设置标记大小
        if row['cluster'] != -1:
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
            
        elif row['cluster'] == -1:
            one_temp_df = gdf_matched.iloc[[i]]  # 获取当前行作为 DataFrame
            one_temp_df.plot(
                ax=ax,                    # 在同一轴上绘图
                marker='*',               # 标记形状为圆点
                color='Grey',             # 标记颜色为蓝色
                markersize=one_temp_df['markersize'], # 标记大小根据计算的 'markersize' 列
                alpha=0.8,                # 标记的透明度
                edgecolor='black',        # 标记边框颜色
                linewidth=0.2,            # 标记边框宽度
                # label=f"Template {i}, ML={row['EQ_magnitude']}" # 图例中显示的标签
            )

        #----在每个点旁边添加文本标签显示序号----
        txt = ax.text(
            row['EQ_longitude'],        # 经度位置
            row['EQ_latitude'],         # 纬度位置
            f"Temp-{i}-ML{row['EQ_magnitude']}",                     # 显示的文本 (序号)
            fontsize=10,                 # 字体大小
            ha='right',                 # 水平对齐方式
            va='center',                # 垂直对齐方式
            color='r'               # 文本颜色
        )
        texts.append(txt)  # 将文本对象添加到列表中
        
    from adjustText import adjust_text # 导入 adjust_text
    adjust_text(texts, 
            # 你可以添加一些参数来控制调整的行为，例如：
            # force_points=0.2, # 点的排斥力
            force_text=0.2,   # 文本之间的排斥力
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5) # 如果需要，可以添加指向原始点的箭头
           )
    # gdf_matched.plot(
    #     ax=ax,                    # 在同一轴上绘图
    #     marker='*',               # 标记形状为点 (通常比'o'小, 但这里大小由markersize控制)
    #     color='Red',              # 标记颜色为红色
    #     markersize=gdf_matched['markersize'], # 标记大小根据计算的 'markersize' 列
    #     alpha=0.3,                # 标记的透明度, 比背景点稍不透明以突出显示
    #     edgecolor='black',        # 标记边框颜色
    #     linewidth=0.2,            # 标记边框宽度
    #     label=f'initial template library ML>2 with good SNR = #{len(matched_df)}' # 图例中显示的标签
    # )


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
    #*——————————————————————————————————————————————    
    
    # 保存图像
    png_fpath = os.path.join(config.output_dir, f"corr_initial_duplicate_templates_map.png")
    fig.savefig(png_fpath, dpi=300)
    print(f"图像已保存到: {png_fpath}")
    
if __name__ == "__main__":
    main()