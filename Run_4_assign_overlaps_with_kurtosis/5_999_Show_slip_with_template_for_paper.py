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
#    依赖：外部出图仓库 GRL_DAS_MF_plots/、连续 DAS 大数据（外部归档或 IN_DAS_data/continuous_scan_1h/ 样本）。
#    隔离 / 无样本环境下会失败，可跳过；不影响检测主线。
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

#!____________INPUT: Template_library.csv 和 Events_classified_Real_Mag.csv_______________
#!____________OUTPUT: 一张 5x2 的组合图，左边是模板瀑布图，右边是对应的滑移时间序列。______________________

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import h5py
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# --- 自定义函数库 ---
# 确保你的自定义函数路径正确
# !! 注意: 运行此脚本的环境必须能够访问这个路径 !!
CUSTOM_FUNC_PATH = f"{PROJECT_ROOT}/external/GRL_DAS_MF_plots/functions/"  # [GitHub] external plotting helpers (not published); functions/ is searched first
if CUSTOM_FUNC_PATH not in sys.path:
    sys.path.append(CUSTOM_FUNC_PATH)

try:
    # 尝试导入你脚本4和脚本5中使用的自定义函数
    from plot import Ticks_xaxis_minute_or_second
    import load_das_h5_PE2024 as load_das_h5_PE2024
    from compute_correlation import preprocess
except ImportError:
    print(f"!!!!!!!!!! 警告 !!!!!!!!!!")
    print(f"无法从 '{CUSTOM_FUNC_PATH}' 导入自定义模块。")
    print(f"请确保 'plot.py', 'load_das_h5_PE2024.py', 'compute_correlation.py' 存在于该路径。")
    print(f"瀑布图 (左侧) 将无法绘制。")
    # 定义占位函数以避免脚本完全崩溃
    def Ticks_xaxis_minute_or_second(ax, **kwargs):
        ax.text(0.5, 0.5, "Ticks_xaxis_minute_or_second 未找到", ha='center', transform=ax.transAxes)
    def preprocess(data, *args):
        return data
    class MockLoader:
        def load_das_custom(self, t_start, t_end, **kwargs):
            print(f"警告: load_das_h5_PE2024 未找到。正在返回模拟数据。")
            time_len = int((t_end - t_start).total_seconds() * 80) # 假设 fs=80
            return np.random.rand(time_len, 500), None, None
    load_das_h5_PE2024 = MockLoader()


# =============================================================================
# 🎨 1. 集中化绘图样式参数 (来自脚本4，用于统一风格)
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
    "axis_label_fontsize": ftz,      # 坐标轴标签
    "tick_label_fontsize": ftz,      # 坐标轴刻度
    "colorbar_label_fontsize": ftz,  # Colorbar 标签
    "panel_label_fontsize": ftz + 5,   # 面板标签 (a), (b)
    
    # --- 图例字体大小 ---
    "legend_fontsize": ftz,          # 图例文字
    
    # --- 其他参数 ---
    "figure_dpi": 160,              # 输出图像的DPI
    "vlim_percentile": 90,          # 瀑布图颜色范围的百分位
}


# =============================================================================
# 🛠 2. 合并的配置类 (Config Class)
# =============================================================================
class Config:
    def __init__(self):
        # --- 1. 基础路径 (来自脚本4和5) ---
        self.MF_HOME_dir = f'{PROJECT_ROOT}/'
        self.DAS_input_dir = f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'
        self.level = 20  # 分类级别

        # --- 2. 输入文件 ---
        self.IN_Template_library_path = os.path.join(self.MF_HOME_dir, 'Output_0_Template_library/Template_library.csv')
        self.IN_MagEventList_path = os.path.join(self.MF_HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{self.level}/MagRegression/Events_classified_Real_Mag.csv')

        # --- 3. 输出路径 ---
        self.OUTPUT_dir =  os.path.join(self.MF_HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/Show_Classification_byCCs/level_{self.level}/5_999_Show_slip_with_template_for_paper/')

        self.OUTPUT_png_path = os.path.join(self.OUTPUT_dir, "Combined_Template_Slip_Figure.png")
        os.makedirs(self.OUTPUT_dir, exist_ok=True)

        # --- 4. 瀑布图参数 (来自脚本4) ---
        self.fs = 80
        self.freq_band = [8, 20]
        self.das_dist_start = 8.3e3
        self.das_dist_end = 14e3
        self.dx = 16 * 1.02095198631286

        # --- 5. 滑移图参数 (来自脚本5) ---
        self.t_start_slip = pd.Timestamp('2024-04-24 10:20:00', tz='UTC')
        self.t_end_slip = pd.Timestamp('2024-06-22 17:10:00', tz='UTC')
        self.STRESS_DROP = 1e6
        self.SHEAR_MODULUS = 30e9
        self.time_bin_minutes = 10
        self.slip_day_tick = 5 # X轴滑移图的主要刻度（天）

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


# =============================================================================
#  হেল 3. 辅助函数 (HELPER FUNCTIONS)
# =============================================================================

# --- 来自脚本 5 ---
def Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=18, is_bottom=True):
    """设置日期时间x轴的主要和次要刻度。"""
    if Day_tick > 0:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=Day_tick))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        if is_bottom:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=font)
        else:
            # 如果不是底部图，隐藏x轴标签
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            plt.setp(ax.get_xticklabels(), visible=False)
            
    elif Hour_tick > 0:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=Hour_tick))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        if is_bottom:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=font)
        else:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            plt.setp(ax.get_xticklabels(), visible=False)

def Slip_from_Mw_Brune(sigma, mu, Mw):
    """使用特定的缩放关系从Mw计算地震滑移。"""
    return (sigma ** (2/3)) / mu * 571.73 * (10 ** (0.5 * Mw))

def Moment_from_Mw(Mw):
    """从Mw计算地震矩。"""
    return 10 ** (1.5 * (Mw + 6.03)) # 矩，单位 N.m

def plot_panel_label(ax, label, style_params):
    """在子图的左上角添加一个粗体标签（例如, 'a)'）。"""
    ax.text(-0.08, 1.05, label,
            transform=ax.transAxes,
            fontsize=style_params["panel_label_fontsize"],
            fontweight='bold',
            va='bottom',
            ha='left')

# =============================================================================
# 📊 4. 绘图函数 (PLOTTING FUNCTIONS)
# =============================================================================

def plot_single_waterfall(ax, row, config, style_params, is_bottom_plot=False):
    """
    绘制单个模板的瀑布图 (改编自脚本4的 plot_waterfalls)
    """
    t_start, t_end = row['t_start'], row['t_end']
    
    # --- 读取数据 ---
    try:
        data, _, _ = load_das_h5_PE2024.load_das_custom(t_start, t_end, input_dir=config.DAS_input_dir)
    except Exception as e:
        print(f"  !! 错误: 无法加载DAS数据: {e}")
        ax.text(0.5, 0.5, f"无法加载DAS数据\n{e}", color='red', ha='center', transform=ax.transAxes)
        return

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
    cbar.ax.tick_params(labelsize=style_params["tick_label_fontsize"])
    
    ax.set_ylabel("Distance (km)", fontsize=style_params["axis_label_fontsize"])
    ytick = np.arange(np.ceil(d_bottom_km), np.floor(d_top_km) + 1, 1)
    ax.set_yticks(ytick)
    
    Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=5, rotation=0, font=style_params["tick_label_fontsize"])
    ax.tick_params(axis='x', pad=10, labelsize=style_params["tick_label_fontsize"])
    ax.tick_params(axis='y', labelsize=style_params["tick_label_fontsize"])
    
    
    ax.set_xlabel(f'Time (GMT) on {t_start.strftime("%Y-%m-%d")}', fontsize=style_params["axis_label_fontsize"])
    # if is_bottom_plot:
    #     ax.set_xlabel(f'Time (GMT) on {t_start.strftime("%Y-%m-%d")}', fontsize=style_params["axis_label_fontsize"])
    # else:
    #     ax.set_xlabel("")
    #     plt.setp(ax.get_xticklabels(), visible=False)

    # --- 标题 --- 原来的标题弃用，放弃标注不清不清楚的SNR
    # title_text = (f"{row.plot_name}: ML={row['EQ_magnitude']:.1f}, "
    #               f"d={(row['STA_distance']*111):.1f} km, "
    #               f"SNR={(row['SNR_P']**0.5):.0f} (P), {(row['SNR_S']**0.5):.0f} (S)")
    title_text = (f"{row.plot_name.replace('Temp_', 'Template ')}: " 
                  f"ML={row['EQ_magnitude']:.1f}, "
                  f"distance={(row['STA_distance']*111):.1f} km, " )
    
    ax.set_title(title_text, fontsize=style_params["title_fontsize"], pad=20, loc='center')
    
    ax.invert_yaxis()
    ax.grid(linestyle='--', linewidth=0.5)


def plot_single_slip(ax, template_events, config, style_params, disp_Template, is_bottom_plot=False):
    """
    绘制单个模板的滑移和累积滑移 (改编自脚本5的 plot_cumulative_bar)
    """
    title = f"{disp_Template.replace('Temp_', 'Template ')} Slip"
    ax.set_title(title, fontsize=style_params["title_fontsize"])
    
    # --- 时间分箱 ---
    t_edges = pd.date_range(start=config.t_start_slip, end=config.t_end_slip, freq=f"{config.time_bin_minutes}min")
    
    ylabel = f'Slip (m)\nper {config.time_bin_minutes} min'
    twin_ylabel = 'Cumulative\nSlip (m)'
    color = 'b' # 累积曲线的颜色

    if not template_events.empty:
        time_bins = pd.cut(template_events['Rt_trig'], bins=t_edges, right=False)
        
        # observed=False 确保所有时间箱都被包括
        data_per_bin = template_events.groupby(time_bins, observed=False)['Slip'].sum().reindex(
            pd.IntervalIndex.from_breaks(t_edges, closed='left'), fill_value=0
        )
        
        # 绘制每个时间箱的净滑移（条形图）
        ax.bar(data_per_bin.index.left, data_per_bin.values, width=t_edges.freq.delta, 
               align='edge', edgecolor='k', alpha=0.7, 
               label=ylabel, color='gray')
        ax.set_ylabel(ylabel, fontsize=style_params["axis_label_fontsize"])

        # --- 绘制累积滑移（阶梯图） ---
        ax_twin = ax.twinx()
        time_for_step = pd.concat([pd.Series([config.t_start_slip]), template_events['Rt_trig']])
        cumulative_for_step = pd.concat([pd.Series([0]), template_events['cum_Slip']])

        ax_twin.step(time_for_step, cumulative_for_step, where='post', color=color, lw=3.5, 
                     label=f'Cumulative Slip')
        ax_twin.set_ylabel(twin_ylabel, color=color, fontsize=style_params["axis_label_fontsize"])
        ax_twin.tick_params(axis='y', labelcolor=color, labelsize=style_params["tick_label_fontsize"])
        
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(style_params["tick_label_fontsize"]-2)
        ax_twin.yaxis.get_offset_text().set_fontsize(style_params["tick_label_fontsize"]-2)
        
        ax_twin.grid(True, linestyle='--', alpha=0.6, color=color)
    else:
        ax.text(0.5, 0.5, 'No Events Detected', transform=ax.transAxes, ha='center', va='center', 
                fontsize=style_params["axis_label_fontsize"])

    ax.set_xlim(config.t_start_slip, config.t_end_slip)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=style_params["tick_label_fontsize"])
    
    # --- 控制X轴标签 ---
    Ticks_xaxis_day_or_hour(ax, Day_tick=config.slip_day_tick, 
                            font=style_params["tick_label_fontsize"], 
                            is_bottom=is_bottom_plot)
    if is_bottom_plot:
        ax.set_xlabel(f'Time (GMT) in 2024', fontsize=style_params["axis_label_fontsize"])
    


# =============================================================================
# 🚀 5. 主函数 (MAIN FUNCTION)
# =============================================================================
def main():
    print("--- 开始运行组合绘图脚本 ---")
    config = Config()
    config._save_json(os.path.join(config.OUTPUT_dir, 'config_combined_plot.json'))
    print(f"配置参数已保存到: {config.OUTPUT_dir}")
    
    # --- 1. 加载数据 ---
    try:
        Temp_lib_df = pd.read_csv(config.IN_Template_library_path, sep=',', 
                                  parse_dates=["peak_time","STA_arrival_time","EQ_time","t_start","t_end"])
        Temp_lib_df.sort_values(by="plot_order", inplace=True)
        Temp_lib_df.reset_index(drop=True, inplace=True)
        # 仅选择前5个模板 (a-e)
        Temp_lib_df = Temp_lib_df.head(5).copy()
        print(f"成功加载 {len(Temp_lib_df)} 个模板信息。")
    except FileNotFoundError:
        print(f"!! 致命错误: 模板库文件未找到: {config.IN_Template_library_path}")
        return
    except Exception as e:
        print(f"!! 致命错误: 加载模板库时出错: {e}")
        return

    try:
        date_columns = ['Rt_trig', 't_trig', 'Rt_left', 'Rt_right', 't_right', 't_left']
        events_df = pd.read_csv(
            config.IN_MagEventList_path, sep='\t', 
            parse_dates=date_columns
        )
        # 确保时区一致
        for col in date_columns:
             if events_df[col].dt.tz is None:
                events_df[col] = events_df[col].dt.tz_localize('UTC')
             else:
                events_df[col] = events_df[col].dt.tz_convert('UTC')
        
        # 过滤到全局时间范围
        events_df = events_df[(events_df['Rt_trig'] >= config.t_start_slip) & (events_df['Rt_trig'] <= config.t_end_slip)].copy()
        print(f"成功加载 {len(events_df)} 个检测事件。")
    except FileNotFoundError:
        print(f"!! 致命错误: 事件列表文件未找到: {config.IN_MagEventList_path}")
        return
    except Exception as e:
        print(f"!! 致命错误: 加载事件列表时出错: {e}")
        return

    # --- 2. 设置全局绘图风格 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # --- 3. 创建主图和子图布局 ---
    # 5行 (每个模板一行), 2列 (左:瀑布图, 右:滑移图)
    fig = plt.figure(figsize=(40, 45), dpi=plot_style_params["figure_dpi"])
    
    # 设定边框和刻度线的磅数 (针对大尺寸图表适当加粗)
    # 建议值: 2.0 - 4.0，根据实际视觉效果微调
    border_linewidth = 4.5  
    
    # 为瀑布图分配稍多的宽度 (1.2)，滑移图为 1
    gs = plt.GridSpec(5, 2, figure=fig, 
                      width_ratios=[1, 1.2], 
                      height_ratios=[1, 1, 1, 1, 1],
                      hspace=0.4,  # 增加垂直间距
                      wspace=0.3) # 增加水平间距
    
    slip_axes_list = [] # 用于共享X轴
    panel_labels_left = ['a1)', 'b1)', 'c1)', 'd1)', 'e1)']
    panel_labels_right = ['a2)', 'b2)', 'c2)', 'd2)', 'e2)']

    print("\n--- 开始循环绘制子图 ---")
    
    # 辅助函数：统一设置边框和刻度粗细
    def set_axes_linewidth(ax, width):
        # 设置四个边框的粗细
        for spine in ax.spines.values():
            spine.set_linewidth(width)
        # 设置刻度线的粗细 (width参数控制刻度线本身的粗细)
        ax.tick_params(width=width)

    # --- 4. 循环绘制每个模板 ---
    for index, row in Temp_lib_df.iterrows():
        
        is_bottom_plot = (index == (len(Temp_lib_df) - 1))
        
        # --- A. 设置子图 ---
        ax_left = fig.add_subplot(gs[index, 0])
        set_axes_linewidth(ax_left, border_linewidth) # <--- 应用边框加粗
        
        if index == 0:
            ax_right = fig.add_subplot(gs[index, 1])
        else:
            # 共享右侧滑移图的X轴
            ax_right = fig.add_subplot(gs[index, 1], sharex=slip_axes_list[0])
        
        set_axes_linewidth(ax_right, border_linewidth) # <--- 应用边框加粗
        slip_axes_list.append(ax_right)
        
        # --- B. 绘制左侧 (瀑布图) ---
        print(f"  正在绘制左侧 ({panel_labels_left[index]}): {row['plot_name']} 瀑布图")
        plot_single_waterfall(ax_left, row, config, plot_style_params, is_bottom_plot=is_bottom_plot)
        
        # --- C. 绘制右侧 (滑移图) ---
        print(f"  正在绘制右侧 ({panel_labels_right[index]}): {row['plot_name']} 滑移图")
        disp_Template = row['plot_name']
        Template = row['infile_name']
        
        # 筛选此模板的事件
        template_events = events_df[events_df['Template'] == Template].sort_values(by='Rt_trig').reset_index(drop=True)
        
        if not template_events.empty:
            # 计算滑移
            template_events['Slip'] = Slip_from_Mw_Brune(config.STRESS_DROP, config.SHEAR_MODULUS, template_events['Real_Mag'])
            # 考虑滑移方向 (来自脚本5)
            template_events['Slip'] *= template_events['CCs_s_trig'].apply(lambda x: -1 if x < 0 else 1)
            template_events['cum_Slip'] = template_events['Slip'].cumsum()
        
        plot_single_slip(ax_right, template_events, config, plot_style_params, disp_Template, is_bottom_plot=is_bottom_plot)

        # --- D. 添加面板标签 ---
        plot_panel_label(ax_left, panel_labels_left[index], plot_style_params)
        plot_panel_label(ax_right, panel_labels_right[index], plot_style_params)

    # --- 5. 添加总标题和保存 ---
    fig_title = (f"Template Waveforms and Corresponding Seismic Slip\n"
                 rf"Slip Calculated with Brune's Model ($\sigma = {(config.STRESS_DROP*1e-6):.0f}\,\mathrm{{MPa}},\, \mu = {(config.SHEAR_MODULUS*1e-9):.0f}\,\mathrm{{GPa}}$)")
    
    # 标题也可以稍微加粗一点点，如果需要的话
    fig.suptitle(fig_title, fontsize=plot_style_params["title_fontsize"] + 10, weight='bold', y=0.96)
    
    # 调整布局以适应总标题
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    fig.savefig(config.OUTPUT_png_path, bbox_inches='tight')
    print(f"\n✅ 组合图已成功保存到: {config.OUTPUT_png_path}")
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序总运行时间: {end_time - start_time:.2f} 秒")
