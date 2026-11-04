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
# -*- coding: utf-8 -*-
"""
脚本功能：
该脚本读取通过模板匹配识别的地震事件目录，并针对【单个特定模板】，在多个用户定义的时间窗口内，
进行并排对比分析。分析内容包括：
1. 频率-震级分布（FMD），图中同时包含累积曲线和频次直方图。
2. 时间-震级散点图，其中点的大小和颜色反映震级，并使用固定的气泡样式图例。

此版本修复了因输入数据中包含NaN/inf值而导致的崩溃问题。
"""

# +
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import json
from scipy.stats import linregress
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
# -

HOME_dir = f'{PROJECT_ROOT}/'
#!__________________________functions_____________________________
import sys
sys.path.append(f"{HOME_dir}functions/")
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl, write_h5, load_h5
from plot import Ticks_xaxis_day_or_hour, Ticks_xaxis_minute_or_second, generate_gs_grid
# =============================================================================
# 1. 配置类 (Config Class)
# =============================================================================
class Config:
    """
    一个用于保存所有脚本参数并管理输出路径的类。
    """
    def __init__(self):
        self.home_dir = f'{PROJECT_ROOT}/'
        self.sub_dir = 'level_20/'
        self.input_dir = os.path.join(self.home_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/{self.sub_dir}/MagRegression/')
        self.output_base_dir = os.path.join(self.home_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/Show_Classification_byCCs/{self.sub_dir}/')
        self.output_final_dir = os.path.join(self.output_base_dir, 'FMD_moving_window/')
        self.event_catalog_csv = 'Events_classified_Real_Mag.csv'
        
        self.selected_template = 'template_0423_80Hz'
        
        # self.time_windows = [
        #     {'name': 'window_1', 'start': '2024-05-02 12:00:00', 'end': '2024-05-03 12:00:00'},
        #     {'name': 'window_2', 'start': '2024-05-03 12:00:00', 'end': '2024-05-04 12:00:00'},
        #     {'name': 'window_3', 'start': '2024-05-04 12:00:00', 'end': '2024-05-05 12:00:00'}
        # ]
        self.time_windows = [
            {'name': '60 days', 'start': '2024-04-24 16:20:00', 'end': '2024-06-22 17:10:00'},
        ]
        
        
        self.fmd = {'bin_width': 0.03, 'mc_method': 'maxc', 'min_events_for_fit': 15}
        self.plot = {
            'subplot_base_width': 16, 'subplot_height': 10,
            'title_fontsize': 28, 'label_fontsize': 24,
            'tick_fontsize': 20, 'legend_fontsize': 18,
            'cmap': 'rainbow',
            'mag_min_size': 20, 'mag_max_size': 1000, 'dpi': 150,
            'legend_min_mag': 0.5, 'legend_max_mag': 4.0,
            'legend_magnitudes': [1, 2, 3, 4],
            
            'DayTicks': 5, 'HourTicks': 0,
        }
        os.makedirs(self.output_final_dir, exist_ok=True)

    def save_to_json(self):
        """将配置保存到JSON文件，以供记录。"""
        path = os.path.join(self.output_final_dir, f'config_run_{self.selected_template}.json')
        with open(path, 'w', encoding='utf-8') as f:
             json.dump({k: str(v) for k, v in self.__dict__.items()}, f, indent=4, ensure_ascii=False)
        print(f"配置已保存至: {path}")

# =============================================================================
# 2. 核心功能函数 (已修复)
# =============================================================================

def calculate_mc_maxc(magnitudes, bin_width=0.1):
    """使用最大曲率法（MAXC）计算完整性震级Mc。"""
    mags = np.array(magnitudes)
    # *** 关键修复：移除所有非有限值 (NaN, inf, -inf) ***
    mags = mags[np.isfinite(mags)]
    
    # 检查清洗后是否还有足够的数据
    if len(mags) < 2: return np.nan

    min_m, max_m = np.floor(np.min(mags) / bin_width) * bin_width, np.ceil(np.max(mags) / bin_width) * bin_width
    
    # 再次检查以确保 min_m 和 max_m 是有效的
    if not np.isfinite(min_m) or not np.isfinite(max_m) or min_m >= max_m:
        return np.nan
        
    bins = np.arange(min_m, max_m + bin_width, bin_width)
    counts, _ = np.histogram(mags, bins=bins)
    if np.sum(counts) == 0: return np.nan
    return bins[np.argmax(counts)]

def calculate_b_value_LS(magnitudes, mc, bin_width=0.1, minimum_events=20):
    """使用最小二乘法估计b值。"""
    mags = np.array(magnitudes)
    # *** 良好实践：同样在此处添加数据清洗 ***
    mags = mags[np.isfinite(mags)]

    if len(mags) < minimum_events: return np.nan, np.nan, np.array([]), np.array([])
    
    valid_mags = mags[mags >= mc]
    if len(valid_mags) < minimum_events: return np.nan, np.nan, np.array([]), np.array([])
    
    bins = np.arange(np.floor(mc / bin_width) * bin_width, np.ceil(np.max(valid_mags) / bin_width) * bin_width + bin_width, bin_width)
    if len(bins) < 2: return np.nan, np.nan, np.array([]), np.array([])
    
    counts, _ = np.histogram(valid_mags, bins=bins)
    cum_counts = np.cumsum(counts[::-1])[::-1]
    magnitudes_for_fit = bins[:-1]
    log_counts = np.log10(cum_counts, where=cum_counts > 0)
    valid_idx = cum_counts > 0
    if np.sum(valid_idx) < 2: return np.nan, np.nan, np.array([]), np.array([])
    
    x, y = magnitudes_for_fit[valid_idx], log_counts[valid_idx]
    slope, intercept, _, _, _ = linregress(x, y)
    return abs(slope), intercept, x, y

def plot_fmd_on_axis(ax, magnitudes, plot_params, fmd_params):
    """在给定的坐标轴上绘制FMD图（累积+频次）并拟合b值。"""
    # *** 重要：在调用计算函数之前，先在此处清洗震级数据 ***
    valid_magnitudes = np.array(magnitudes)[np.isfinite(magnitudes)]
    
    min_events_for_fit = fmd_params['min_events_for_fit']
    if len(valid_magnitudes) < min_events_for_fit:
        ax.text(0.5, 0.5, f'Not enough events ({len(valid_magnitudes)})', ha='center', va='center', transform=ax.transAxes, fontsize=plot_params['label_fontsize'])
        return

    mc = calculate_mc_maxc(valid_magnitudes, fmd_params['bin_width']) if fmd_params['mc_method'] == 'maxc' else np.min(valid_magnitudes)
    if np.isnan(mc):
        ax.text(0.5, 0.5, 'Could not compute Mc', ha='center', va='center', transform=ax.transAxes, fontsize=plot_params['label_fontsize'])
        return

    bins = np.arange(np.floor(np.min(valid_magnitudes) / fmd_params['bin_width']) * fmd_params['bin_width'],
                     np.ceil(np.max(valid_magnitudes) / fmd_params['bin_width']) * fmd_params['bin_width'] + fmd_params['bin_width'],
                     fmd_params['bin_width'])
    
    cum_counts = np.cumsum(np.histogram(valid_magnitudes, bins=bins)[0][::-1])[::-1]
    ax.plot(bins[:-1], cum_counts, marker='o', linestyle='-', color='k', label=f"Cumulative (N={len(valid_magnitudes)})")
    
    b_val, a_val, fit_x, fit_y = calculate_b_value_LS(valid_magnitudes, mc, fmd_params['bin_width'], min_events_for_fit)
    if not np.isnan(b_val):
        fit_line = 10**(a_val - b_val * fit_x)
        ax.plot(fit_x, fit_line, color='b', linestyle='--', lw=3, label=f'b={b_val:.2f}, Mc={mc:.1f}')
        ax.axvline(mc, color='b', linestyle=':', lw=3, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude (ML)', fontsize=plot_params['label_fontsize'])
    ax.set_ylabel('Cumulative Number (log scale)', fontsize=plot_params['label_fontsize'], color='k')
    ax.tick_params(axis='y', labelcolor='k', labelsize=plot_params['tick_fontsize'])
    ax.tick_params(axis='x', labelsize=plot_params['tick_fontsize'])
    ax.grid(True, which='both', linestyle=':', color='gray')
    
    ax_hist = ax.twinx()
    ax_hist.hist(valid_magnitudes, bins=bins, color='r', alpha=0.6, label='Binned Counts')
    ax_hist.set_ylabel('Counts per Bin', fontsize=plot_params['label_fontsize'], color='r')
    ax_hist.tick_params(axis='y', labelcolor='r', labelsize=plot_params['tick_fontsize'])
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_hist.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=plot_params['legend_fontsize'])


# =============================================================================
# 3. 主执行逻辑
# =============================================================================
def main():
    """主函数，用于加载数据、运行分析并生成图像。"""
    config = Config()
    config.save_to_json()

    input_path = os.path.join(config.input_dir, config.event_catalog_csv)
    print(f"从以下路径加载事件目录: {input_path}")
    events_df = pd.read_csv(input_path, sep='\t', parse_dates=['Rt_trig'])

    if config.selected_template:
        print(f"正在筛选模板: {config.selected_template}")
        events_df = events_df[events_df['Template'] == config.selected_template].copy()
        if events_df.empty:
            print(f"错误: 未找到模板 '{config.selected_template}' 的任何事件，脚本退出。")
            return
        print(f"筛选后剩余 {len(events_df)} 个事件。")

    num_windows = len(config.time_windows)
    figsize_width = config.plot['subplot_base_width'] * num_windows
    figsize_height = config.plot['subplot_height'] * 2
    
    fig, axes = plt.subplots(nrows=2, ncols=num_windows, figsize=(figsize_width, figsize_height), squeeze=False)
    
    for i, window in enumerate(config.time_windows):
        ax_mag_time, ax_fmd = axes[0, i], axes[1, i]
        print(f"\n--- 正在处理窗口: {window['name']} ---")
        
        t_start = datetime.strptime(window['start'], '%Y-%m-%d %H:%M:%S')
        t_end = datetime.strptime(window['end'], '%Y-%m-%d %H:%M:%S')
        window_df = events_df[(events_df['Rt_trig'] >= t_start) & (events_df['Rt_trig'] <= t_end)]
        
        ax_mag_time.set_title(window['name'], fontsize=config.plot['title_fontsize'])

        if window_df.empty:
            print(f"警告: 在窗口 '{window['name']}' 中没有找到事件。")
            for ax in [ax_mag_time, ax_fmd]: ax.text(0.5, 0.5, 'No Events', ha='center', va='center')
            continue

        magnitudes, times = window_df['Real_Mag'].to_numpy(), window_df['Rt_trig'].to_numpy()
        
        # 在这里也对震级数据进行一次初步清洗
        finite_mask = np.isfinite(magnitudes)
        magnitudes, times = magnitudes[finite_mask], times[finite_mask]
        
        if len(magnitudes) == 0:
            print(f"警告: 在窗口 '{window['name']}' 中没有有效的震级数据。")
            for ax in [ax_mag_time, ax_fmd]: ax.text(0.5, 0.5, 'No valid magnitude data', ha='center', va='center')
            continue

        # 1. 绘制时间-震级图
        def size_func(mag):
            return np.interp(mag**2, (config.plot['legend_min_mag']**2, config.plot['legend_max_mag']**2), 
                             (config.plot['mag_min_size'], config.plot['mag_max_size']))
        
        sc = ax_mag_time.scatter(times, magnitudes, s=size_func(magnitudes), c=magnitudes, 
                                 cmap=config.plot['cmap'], alpha=0.7, ec='k', lw=0.5)
        
        cbar = fig.colorbar(sc, ax=ax_mag_time, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label('Magnitude (ML)', fontsize=config.plot['label_fontsize'])
        
        mag_legend_values = np.array(config.plot['legend_magnitudes'])
        labels = [f'ML {s:.1f}' for s in mag_legend_values]
        
        legend_handles_mag = [
            Line2D([], [], linestyle='None', marker='o', markersize=np.sqrt(size_func(m)), 
                   color='k', label=labels[ii]) for ii, m in enumerate(mag_legend_values)
        ]
        legend1 = ax_mag_time.legend(handles=legend_handles_mag, title='Magnitude Size', loc='upper left', 
                                     framealpha=0.3, fontsize=config.plot['legend_fontsize'], 
                                     title_fontsize=config.plot['legend_fontsize'])
        
        # ax_mag_time.add_artist(legend1) # 不加图例
        
        ax_mag_time.set(xlabel='', ylabel='Magnitude (ML)', xlim=(t_start, t_end))
        # ax_mag_time.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H'))
        Ticks_xaxis_day_or_hour(ax_mag_time, config.plot['DayTicks'], config.plot['HourTicks'])

        ax_mag_time.tick_params(axis='both', labelsize=config.plot['tick_fontsize'])
        ax_mag_time.grid(True, linestyle=':')
        
        # 2. 绘制FMD图
        plot_fmd_on_axis(ax_fmd, magnitudes, config.plot, config.fmd)
        ax_fmd.set_xticks(np.arange(0.5, 4, 0.5))

    template_str = config.selected_template if config.selected_template else "All_Templates"
    fig.suptitle(f'Comparative Analysis for Template: {template_str}', fontsize=config.plot['title_fontsize'] + 6, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    window_names_str = '_'.join([w['name'] for w in config.time_windows])
    output_filename = f"FMD_Analysis_{template_str}_{window_names_str}.png"
    output_path = os.path.join(config.output_final_dir, output_filename)
    
    plt.savefig(output_path, dpi=config.plot['dpi'], bbox_inches='tight')
    print(f"\n分析完成。图表已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()