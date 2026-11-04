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
# ℹ️ 非检测主线（辅助出图：MAD 时序叠加气象/STALTA）。所需气象模块 load_kefa_meteo + CSV
#    已自带于本发布版 IN_noise_meteo/（由外部 check_minmag 工程产出），已无外部依赖、可直接出图。
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import h5py, json, pickle

# --- 自定义函数库 ---
sys.path.append(f"{PROJECT_ROOT}/functions/")
from readers import load_pkl
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second

sys.path.append(f"{PROJECT_ROOT}/IN_noise_meteo")  # 本发布版自带的气象模块+CSV（外部 check_minmag 工程产物，详见 IN_noise_meteo/）
from load_kefa_meteo import load_kefa_meteo_data


# =============================================================================
# 🎨 1. 配置与样式参数 (Configuration and Style Parameters)
# =============================================================================
class Config:
    """集中管理所有配置参数"""
    def __init__(self):
        # --- 数据路径 ---
        HOME_dir = f"{PROJECT_ROOT}" # 统一HOME目录
        self.base_dir = f'{HOME_dir}/Output_1_SCCs_and_MAD/MAD_arrays/'
        self.mf_catalog_path = f'{HOME_dir}/Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_20/MagRegression/Events_classified_Real_Mag.csv'
        # [新增] 模板库文件路径
        self.template_library_path = f"{HOME_dir}/Output_0_Template_library/Template_library.csv"

        # --- 处理参数 ---
        # 这个列表现在作为筛选器，决定从template_library中选取哪些模板
        self.templates_to_check = [
            'template_0423_80Hz', 'template_0425_80Hz', 'template_10_80Hz',
            'template_8_80Hz', 'template_20_80Hz'
        ]
        self.time_delta_minutes = 60  # MAD计算的时间窗口

        # --- 绘图参数 ---
        self.Day_tick = 5
        self.Hour_tick = 0
        
        # 根据处理参数生成输入目录
        self.t_start = datetime(2024, 4, 24, 10, 20)
        self.t_end = datetime(2024, 6, 22, 17, 10)
        t_str = self.t_start.strftime('%Y%m%d_%H%M%S') + '_to_' + self.t_end.strftime('%Y%m%d_%H%M%S')
        t_delta_str = str(timedelta(minutes=self.time_delta_minutes))
        self.input_dir = os.path.join(self.base_dir, t_str, t_delta_str)

ftz = 20  # 字体大小
plt.rcParams.update({
    'font.size': ftz,
    'axes.labelsize': ftz,
    'xtick.labelsize': ftz,
    'ytick.labelsize': ftz,
    'legend.fontsize': ftz,
    'figure.titlesize': ftz + 2,
    'axes.titlesize': ftz + 1,    
})

plot_style_params = {
    # --- 字体大小 ---
    "label_fontsize": ftz,
    "tick_fontsize": ftz,
    "legend_fontsize": ftz,
    "title_fontsize": ftz + 4,

    # --- 颜色与样式 ---
    "mad_alpha": 0.4,
    "mad_marker_size": 5,
    
    "mf_hist_color": 'k',
    "mf_hist_alpha": 0.7,
    "swh_color": 'r', 
    "wind_speed_color": 'C5',
    
    # --- Y轴缩放因子 ---
    "hist_ylim_scale_factor": 4.0, 
}


# =============================================================================
# ⚙️ 2. 数据加载与处理函数 (Data Loading and Processing Functions)
# =============================================================================
def _get_mf_catalog(cata_path, t_start, t_end, temp_name_list):
    """从CSV加载并筛选模板匹配(MF)目录"""
    print(f'===> Reading Matched Filter (MF) catalog...')
    events_classified = pd.read_csv(cata_path, sep='\t', parse_dates=['Rt_trig'])
    mask = (events_classified['Rt_trig'] >= t_start) & (events_classified['Rt_trig'] <= t_end)\
        & (events_classified['Template'].isin(temp_name_list))
    filtered_df = events_classified.loc[mask].reset_index(drop=True)
    print(f'     Found {len(filtered_df)} MF events from {t_start} to {t_end}')
    return filtered_df

def load_data_for_plotting(cfg):
    """加载所有用于绘图的已处理数据"""
    print("\n--- Loading pre-computed data for plotting ---")
    data = {}
    
    # [新增] 加载并处理模板库信息
    print(f"===> Reading Template Library from: {cfg.template_library_path}")
    temp_lib_df = pd.read_csv(cfg.template_library_path, sep=',')
    check_lib = temp_lib_df[temp_lib_df['infile_name'].isin(cfg.templates_to_check)]
    check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)
    data['template_info'] = check_lib
    print(f"     Loaded and ordered {len(check_lib)} templates for plotting.")
    
    # 使用处理后的模板文件名列表进行后续数据加载
    template_infiles_ordered = data['template_info']['infile_name'].tolist()

    # 加载MAD数据
    print(f"Reading MAD data from: {cfg.input_dir}")
    data['mad'] = load_pkl(os.path.join(cfg.input_dir, 'MAD_array_dict.pkl'))
    data['t_array'] = data['mad']['t_array']
    plot_t_start, plot_t_end = data['t_array'][0], data['t_array'][-1]
    
    # 加载MF事件
    data['mf_events'] = _get_mf_catalog(cfg.mf_catalog_path, plot_t_start, plot_t_end, template_infiles_ordered)
    
    # 加载气象数据
    print("===> Reading meteorological data...")
    data['atmos_df'], data['wave_df'] = load_kefa_meteo_data(plot_t_start, plot_t_end)
    
    print("--- Data loading complete ---")
    return data


# =============================================================================
# 📊 3. 模块化绘图函数 (Modular Plotting Functions)
# =============================================================================
def plot_mad(ax, data, styles):
    """绘制MAD时间序列图 (左Y轴), 使用template_info中的信息"""
    ax.set_title(f'Hourly MAD of SCC From 5 Templates', fontsize=styles["title_fontsize"])
    
    # [修改] 循环遍历处理好的 template_info DataFrame
    for i, row in data['template_info'].iterrows():
        infile_name = row['infile_name']   # 用于获取数据的键
        display_name = row['plot_name']    # 用于图例显示
        
        # ax.scatter(data['t_array'][:-1], data['mad'][infile_name], 
        #            label=display_name, s=styles["mad_marker_size"], 
        #            alpha=styles["mad_alpha"], color=f'C{i}')
        ax.plot(data['t_array'][:-1], data['mad'][infile_name],
                label=display_name, linewidth=2, marker='.', markersize=styles["mad_marker_size"],
                alpha=styles["mad_alpha"], color=f'C{i}')
                   
    ax.set_ylabel('MAD of SCC', fontsize=styles["label_fontsize"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=styles["tick_fontsize"])
    ax.legend(loc='upper left', fontsize=styles["legend_fontsize"])

def plot_meteo(ax, data, cfg, styles):
    """绘制气象数据图（SWH和风速）"""
    ax.set_title('Meteorological Data', fontsize=styles["title_fontsize"])
    color = styles["swh_color"]
    ax.set_ylabel('Sig. Wave Height (m)', color=color, fontsize=styles["label_fontsize"])
    ax.plot(data['wave_df'].index, data['wave_df']['swh'], color=color, linestyle='-', linewidth=2, label='SWH')
    ax.tick_params(axis='y', labelcolor=color, labelsize=styles["tick_fontsize"])
    ax.legend(loc='upper left', fontsize=styles["legend_fontsize"])

    ax_twin = ax.twinx()
    color = styles["wind_speed_color"]
    ax_twin.set_ylabel('Wind Speed (m/s)', color=color, fontsize=styles["label_fontsize"])
    ax_twin.plot(data['atmos_df'].index, data['atmos_df']['wind_speed'], color=color, linestyle='--', linewidth=2, label='Wind Speed')
    ax_twin.tick_params(axis='y', labelcolor=color, labelsize=styles["tick_fontsize"])
    ax_twin.legend(loc='upper right', fontsize=styles["legend_fontsize"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ax.set_xlabel('Time', fontsize=styles["label_fontsize"])
    Ticks_xaxis_day_or_hour(ax, Day_tick=cfg.Day_tick, Hour_tick=cfg.Hour_tick, rotation=30, font=styles["tick_fontsize"])
    ax.set_xlim(data['t_array'][0], data['t_array'][-1])

def plot_mf_histogram(ax, data, styles):
    """绘制MF事件的直方图 (在给定的twinx轴上)"""
    t_array = data['t_array']
    mf_events = data['mf_events']
    
    bin_delta = timedelta(hours=1)
    bins = np.arange(t_array[0], t_array[-1] + bin_delta, bin_delta)
    counts, _ = np.histogram(mf_events['Rt_trig'], bins=bins)
    
    ax.bar(bins[:-1], counts, width=bin_delta, 
           color=styles["mf_hist_color"], alpha=styles["mf_hist_alpha"], 
           label=f'Matched filter of all templates (Total: {len(mf_events)})')
    
    ax.set_ylabel('Hourly Detections', fontsize=styles["label_fontsize"], color=styles["mf_hist_color"])
    ax.tick_params(axis='y', labelcolor=styles["mf_hist_color"], labelsize=styles["tick_fontsize"])
    
    max_count = np.max(counts) if len(counts) > 0 else 1
    ax.set_ylim(0, max_count * styles["hist_ylim_scale_factor"])
    ax.legend(loc='upper right', fontsize=styles["legend_fontsize"])

def create_summary_plot(data, cfg, styles):
    """创建并保存最终的汇总图"""
    print("\n--- Creating summary plot ---")
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True, height_ratios=[3, 1])
    
    plot_mad(axes[0], data, styles)
    ax_hist = axes[0].twinx()
    plot_mf_histogram(ax_hist, data, styles)
    
    plot_meteo(axes[1], data, cfg, styles)
    
    fig.tight_layout()
    figpath = os.path.join(cfg.input_dir, f'Summary_Plot_MAD_Meteo_MF_Combined_v2.png')
    fig.savefig(figpath, dpi=300)
    print(f"✅ Summary plot saved to: {figpath}")
    plt.show()
    plt.close(fig)

# =============================================================================
# ✨ 4. 主逻辑 (Main Logic)
# =============================================================================
def main():
    """主执行函数"""
    cfg = Config()
    styles = plot_style_params
    plot_data = load_data_for_plotting(cfg)
    create_summary_plot(plot_data, cfg, styles)
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()