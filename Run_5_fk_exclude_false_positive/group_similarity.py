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
# 文件名: group_similarity.py

import numpy as np
from tqdm import tqdm # 使用tqdm来显示进度条，更友好

def calculate_top_outlier_masscenter_distance(matrix: np.ndarray, p: float = 0.1) -> float:
    """
    计算离群体中心最远的 p% 的向量到中心的平均余弦距离。
    """
    d, n = matrix.shape
    if n < 2: return 0.0
    
    epsilon = 0
    centroid = np.mean(matrix, axis=1, keepdims=True)
    
    norm_matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + epsilon)
    norm_centroid = centroid / (np.linalg.norm(centroid, axis=0, keepdims=True) + epsilon)
    
    similarities = (norm_centroid.T @ norm_matrix)[0]
    distances = 1.0 - abs(similarities)
    
    k = int(np.ceil(n * p))
    if k == 0 and n > 0: k = 1

    # 计算前 k 个最远点的平均距离
    sorted_distances = np.sort(distances)[::-1]
    
    return 1-np.mean(sorted_distances[:k])

def calculate_loneliest_neighborhood_similarity(matrix: np.ndarray, p: float = 0.1) -> float:
    """
    计算最“孤独”的 p% 的向量与其最相似邻居的平均余弦相似度。
    """
    d, n = matrix.shape
    if n < 2: return 1.0

    epsilon = 0
    norm_matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + epsilon)
    sim_matrix = norm_matrix.T @ norm_matrix
    
    sim_matrix = abs(sim_matrix)  # 确保相似度矩阵是非负的
    
    np.fill_diagonal(sim_matrix, -1.0) # 将这一行置于最小值，避免自身相似度干扰
    
    best_neighbor_sims = np.max(sim_matrix, axis=1)
    
    k = int(np.ceil(n * p))
    if k == 0 and n > 0: k = 1

    sorted_sims = np.sort(best_neighbor_sims)
    return np.mean(sorted_sims[:k])

def calculate_pairwise_similarity_quantile(matrix: np.ndarray, q: float = 0.1) -> float:
    """
    计算所有向量对的余弦相似度，并返回指定分位点 q 的值。
    """
    d, n = matrix.shape
    if n < 2: return 1.0

    epsilon = 0
    norm_matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + epsilon)
    sim_matrix = norm_matrix.T @ norm_matrix
    
    sim_matrix = abs(sim_matrix)
    
    unique_pairwise_sims = sim_matrix[np.triu_indices(n, k=1)]
    if unique_pairwise_sims.size == 0: return 1.0
    
    return np.quantile(unique_pairwise_sims, q)

def calculate_average_pairwise_similarity(matrix: np.ndarray) -> float:
    """
    计算数据集中所有唯一向量对的平均余弦相似度。
    """
    d, n = matrix.shape
    if n < 2: return 1.0

    epsilon = 0
    norm_matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + epsilon)
    sim_matrix = norm_matrix.T @ norm_matrix
    
    sim_matrix = abs(sim_matrix)
    
    unique_pairwise_sims = sim_matrix[np.triu_indices(n, k=1)]
    if unique_pairwise_sims.size == 0: return 1.0
    
    return np.mean(unique_pairwise_sims)

def compute_moving_similarity_metrics(CCs_matrix, config):
    """
    在滑动窗口上计算四个群组相似度指标。

    参数:
    CCs_matrix (np.ndarray): 输入的CCs矩阵，形状为 (time_steps, channels)。
    config (Config): 包含参数的配置对象。

    返回:
    tuple: 包含四个指标时间序列的元组。
           (dist_series, lonely_series, quantile_series, avg_series)
    """
    # 从配置中获取参数
    win_before_pts = int(config.Z_win[0] * config.fs)
    win_after_pts = int(config.Z_win[1] * config.fs)
    p_ratio = config.Z_p_ratio
    q_ratio = config.Z_q_ratio
    
    total_window_pts = win_before_pts + win_after_pts + 1
    num_time_steps, num_channels = CCs_matrix.shape
    
    # 输出序列的长度会比输入短
    num_outputs = num_time_steps - total_window_pts + 1
    
    # 初始化四个输出序列
    dist_series = np.zeros(num_outputs)
    lonely_series = np.zeros(num_outputs)
    quantile_series = np.zeros(num_outputs)
    avg_series = np.zeros(num_outputs)

    print(f"\033[32m |---> 开始计算移动窗口相似度，总计 {num_outputs} 个窗口... \033[0m")
    # 使用tqdm创建进度条
    for i in tqdm(range(num_outputs), desc="Processing windows"):
        # 提取当前窗口的数据，注意要转置以满足函数输入（列是向量）
        window_data = CCs_matrix[i : i + total_window_pts, :].T
        
        # 调用四个函数计算指标
        dist_series[i] = calculate_top_outlier_masscenter_distance(window_data, p=p_ratio)
        lonely_series[i] = calculate_loneliest_neighborhood_similarity(window_data, p=p_ratio)
        quantile_series[i] = calculate_pairwise_similarity_quantile(window_data, q=q_ratio)
        avg_series[i] = calculate_average_pairwise_similarity(window_data)

    print(f"\033[32m |---> 移动窗口相似度计算完成。 \033[0m")
    return (dist_series, lonely_series, quantile_series, avg_series)