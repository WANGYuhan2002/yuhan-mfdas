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
# Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF/3_plot_catalog_MF_stalta_NOA.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import os
from scipy.stats import linregress

# =================================================================================
# 脚本功能:
# 该脚本严格复现 `fig3_catalog_plot.py` 的核心功能和视觉风格。
# 它使用由前序脚本生成的规整化数据，创建综合性多面板图表，
# 用于定量对比MF、STA/LTA和NOA三个目录的分布特性。
# 唯一的改动是数据源的读取，所有绘图参数和逻辑均与原始脚本保持一致。
# =================================================================================

# --- 辅助函数 (严格从参考脚本复制) ---
def fit_catalog_to_GR_law(magnitudes,M_c):
    from scipy.stats import linregress
    if M_c is None or np.isnan(M_c) or len(magnitudes) < 10:
        return np.nan, np.nan
    filtered_magnitudes = magnitudes[magnitudes >= M_c]
    if len(filtered_magnitudes) < 10:
        return np.nan, np.nan
    sorted_magnitudes = np.sort(filtered_magnitudes)
    cumulative_counts = np.arange(len(sorted_magnitudes), 0, -1)
    log_counts = np.log10(cumulative_counts)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(sorted_magnitudes, log_counts)
        b_value = -slope
        a_value = intercept
        return b_value, a_value
    except:
        return np.nan, np.nan

def calculate_mc_maxc(magnitudes, bin_width=0.2):
    if len(magnitudes) == 0: return np.nan
    min_m = np.floor(np.min(magnitudes) / bin_width * bin_width)
    max_m = np.ceil(np.max(magnitudes) / bin_width * bin_width)
    bins = np.arange(min_m, max_m + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    max_count_idx = np.argmax(counts)
    mc = edges[max_count_idx]
    return mc

def calculate_b_value_MLE(magnitudes, mc=None, minimum_events=20):
    if len(magnitudes) < minimum_events: return np.nan
    if mc is None:
        mc = calculate_mc_maxc(magnitudes)
        if np.isnan(mc): return np.nan
    mags_above_mc = np.array(magnitudes)[np.array(magnitudes) >= mc]
    if len(mags_above_mc) < minimum_events: return np.nan
    mean_m = np.mean(mags_above_mc)
    b_value = np.log10(np.e) / (mean_m - mc)
    return b_value

def compute_b_value_time_series(times, magnitudes, window_hours, step_hours, mc, minimum_events=100):
    df = pd.DataFrame({'time': pd.to_datetime(times), 'magnitude': magnitudes})
    df.sort_values('time', inplace=True)
    window_centers, b_values = [], []
    if df.empty or len(df) < minimum_events: return [], []
    current_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]
    window = timedelta(hours=window_hours)
    step = timedelta(hours=step_hours)

    while current_time + window <= end_time:
        window_start, window_end = current_time, current_time + window
        mask = (df['time'] >= window_start) & (df['time'] < window_end)
        mags_window = df['magnitude'][mask].to_numpy()
        
        # 确保窗口内有足够事件
        if len(mags_window) >= minimum_events:
            b = calculate_b_value_MLE(mags_window, mc, minimum_events)
        else:
            b = np.nan
            
        window_centers.append(window_start + window / 2)
        b_values.append(b)
        current_time += step
    return window_centers, b_values

def Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=38, rotation=0):
    if Day_tick > 0:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=Day_tick))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="center", fontsize=font)

def size_func(mag):
    min_size, max_size = 20, 800
    min_mag, max_mag = -0.5, 4.0
    mag = np.nan_to_num(mag, nan=min_mag)
    return np.interp(np.square(mag), (min_mag**2, max_mag**2), (min_size, max_size))

def main():
    print("=============================================")
    print("  脚本 3: 目录综合对比绘图 (严格复刻版)")
    print("=============================================\n")
    
    # --- 配置 (严格参考 fig3_catalog_plot.py) ---
    HOME_dir = f'{PROJECT_ROOT}/'
    level = 20
    data_dir = os.path.join(HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{level}/eq_list_to_plot/')
    output_dir = os.path.join(HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{level}/compare_figures/')
    os.makedirs(output_dir, exist_ok=True)
    
    t_start = datetime(2024, 4, 24, 10, 15)
    t_end = datetime(2024, 6, 22, 16, 15)
    
    Temp_name_list = ['template_10_80Hz', 'template_0423_80Hz', 'template_8_80Hz', 'template_20_80Hz', 'template_0425_80Hz']
    disp_Temp_name = {'template_10_80Hz': 'temp_K1', 'template_0423_80Hz': 'temp_K2', 'template_8_80Hz': 'temp_K3', 'template_20_80Hz': 'temp_I1', 'template_0425_80Hz': 'temp_I2'}

    # --- 数据加载 ---
    print("--- 正在加载绘图数据...")
    try:
        df_sl = pd.read_csv(os.path.join(data_dir, 'stalta_catalog_enriched.csv'), sep='\t', parse_dates=['peak_time'])
        df_noa_lb = pd.read_csv(os.path.join(data_dir, 'noa_catalog_labeled.csv'), sep='\t', parse_dates=["STA_arrival_time","EQ_time"])
        Events_cla_mag = pd.read_csv(os.path.join(HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{level}/MagRegression/Events_classified_Real_Mag.csv'), sep='\t', parse_dates=["Rt_trig"])
        Events_cla_mag = Events_cla_mag[(Events_cla_mag['Rt_trig'] >= t_start) & (Events_cla_mag['Rt_trig'] <= t_end)]
        print("--- 所有数据加载成功。")
    except FileNotFoundError as e:
        print(f"\n[错误] 数据文件未找到: {e.filename}")
        return

    # --- 开始画图 (严格复刻) ---
    no_temp = len(Temp_name_list)
    no = (no_temp + 1) * 2
    
    fig = plt.figure(figsize=(55, 11 * (no_temp + 1)), dpi=150)
    height_ratios = [1.5, 1, 0.6] * no_temp + [2.3, 2]
    gs = plt.GridSpec((no_temp) * 3 + 2, 2, figure=fig, height_ratios=height_ratios, width_ratios=[3.5, 1])
    fig.subplots_adjust(hspace=0.02, wspace=0.17)
    
    axes_upper_left = [fig.add_subplot(gs[i*3, 0]) for i in range(no_temp + 1)]
    axes_lower_left = [fig.add_subplot(gs[i*3+1, 0], sharex=axes_upper_left[i]) for i in range(no_temp + 1)]
    axes_upper_right = [fig.add_subplot(gs[i*3, 1]) for i in range(no_temp + 1)]
    axes_lower_right = [fig.add_subplot(gs[i*3+1, 1], sharex=axes_upper_right[i]) for i in range(no_temp + 1)]
    
    plt.rcParams['font.size'] = 36

    # --- 循环绘制每个模板 ---
    for tt, Template in enumerate(Temp_name_list):
        ax_tm, ax_th = axes_upper_left[tt], axes_lower_left[tt]
        ax_mf, ax_gr = axes_upper_right[tt], axes_lower_right[tt]
        disp_Template = disp_Temp_name.get(Template, Template)
        
        mf_data = Events_cla_mag[Events_cla_mag['Template'] == Template]
        stalta_data = df_sl[df_sl['mf_template_name'] == Template]
        noa_data = df_noa_lb[df_noa_lb['template_region'] == Template]

        # Panel 1: 时间-震级图
        # ax_tm.set_title(f'Temporal and Magnitude Distribution of Template: {disp_Template}', fontsize=42)
        ax_tm.scatter(mf_data['Rt_trig'], mf_data['Real_Mag'], color='r', alpha=0.2, s=size_func(mf_data['Real_Mag']), edgecolor='k')
        stalta_mag = stalta_data['mf_real_mag'].fillna(stalta_data['noa_magnitude'])
        ax_tm.scatter(stalta_data['peak_time'], stalta_mag, color='g', alpha=0.2, s=size_func(stalta_mag), marker='P', edgecolor='k')
        ax_tm.scatter(noa_data['STA_arrival_time'], noa_data['EQ_magnitude'], color='b', alpha=0.3, s=size_func(noa_data['EQ_magnitude']), marker='*')
        ax_tm.set_xlim([t_start, t_end])
        
        # #* show mag axvlines ------>
        # show_mag = 2.2
        # large_eqs = mf_data[mf_data['Real_Mag'] >= show_mag]
        # for _, eq_row in large_eqs.iterrows():
        #     ax_tm.axvline(x=eq_row['Rt_trig'], color='orange', linestyle='--', alpha=0.5, lw=3)
        # #* <-----end---show mag axvlines ------
        
        # #* show b-value  ------>
        # ax_b = ax_tm.twinx()
        # t_b, b_vals = compute_b_value_time_series(mf_data['Rt_trig'], mf_data['Real_Mag'], window_hours=12, step_hours=0.2, mc=mf_data['Real_Mag'].min())
        # ax_b.plot(t_b, b_vals, 'ko-', lw=3, markersize=3, alpha=0.5)
        # ax_b.set_ylabel('b-value', color='k'); ax_b.set_xticks([])
        # #* <-----end---show b-value ------

        # Panel 2: 时间直方图
        dt = timedelta(hours=2)
        bins = mdates.date2num(np.arange(t_start, t_end + dt, dt))
        ax_th.hist(mdates.date2num(mf_data['Rt_trig']), bins=bins, color='r', alpha=0.7, label=f'MF: #{len(mf_data)}')
        ax_th.hist(mdates.date2num(stalta_data['peak_time']), bins=bins, color='g', alpha=0.5, label=f'STA/LTA: #{len(stalta_data)}')
        ax_th.hist(mdates.date2num(noa_data['STA_arrival_time']), bins=bins, color='b', alpha=0.5, label=f'NOA: #{len(noa_data)}')
        ax_th.legend(loc='upper right', fontsize=36, framealpha=0)
        ax_th.set_xlim([t_start, t_end])
        
        # Panel 3 & 4: 震级频次与G-R图
        mag_bins = np.arange(-1, 4.1, 0.1)
        mf_mag = mf_data['Real_Mag'].dropna()
        stalta_mag = stalta_data['mf_real_mag'].fillna(stalta_data['noa_magnitude']).dropna()
        noa_mag = noa_data['EQ_magnitude'].dropna()

        ax_mf.hist(mf_mag, bins=mag_bins, color='r', alpha=0.5)
        ax_mf.hist(stalta_mag, bins=mag_bins, color='g', alpha=0.5)
        ax_mf.hist(noa_mag, bins=mag_bins, color='b', alpha=0.5)
        if tt == 0:
            ax_mf.set_title('FMD and G-R Law', fontsize=42)

        for mag_data, color in [(mf_mag, 'r'), (stalta_mag, 'g'), (noa_mag, 'b')]:
            if not mag_data.empty:
                counts, _ = np.histogram(mag_data, bins=mag_bins)
                cum_counts = np.cumsum(counts[::-1])[::-1]
                valid = cum_counts > 0
                ax_gr.plot(mag_bins[:-1][valid], cum_counts[valid], color=color, lw=5.5, marker='o', markersize=6)
        
        b_val, a_val = fit_catalog_to_GR_law(mf_mag, mf_mag.min())
        if not np.isnan(b_val):
            fit_x = np.linspace(mf_mag.min(), 4, 100)
            fit_y = 10**(a_val - b_val * fit_x)
            gr_fit, = ax_gr.plot(fit_x, fit_y, 'k--', label=f'b={b_val:.2f}, a={a_val:.2f}', lw=4.5)
            ax_gr.legend(handles=[gr_fit], loc='upper right', fontsize=36, framealpha=0)

    #! --- 绘制总览图 (补全部分) ---
    tt = no_temp
    ax_tm, ax_th = axes_upper_left[tt], axes_lower_left[tt]
    ax_mf, ax_gr = axes_upper_right[tt], axes_lower_right[tt]

    mf_data_all = Events_cla_mag
    stalta_data_all = df_sl
    noa_data_all = df_noa_lb

    # Panel 1: 时间-震级图 (总览)
    # ax_tm.set_title('Temporal and Magnitude Distribution of All Events', fontsize=42)
    ax_tm.scatter(mf_data_all['Rt_trig'], mf_data_all['Real_Mag'], color='r', alpha=0.2, s=size_func(mf_data_all['Real_Mag']), edgecolor='k')
    stalta_mag_all = stalta_data_all['mf_real_mag'].fillna(stalta_data_all['noa_magnitude'])
    ax_tm.scatter(stalta_data_all['peak_time'], stalta_mag_all, color='g', alpha=0.2, s=size_func(stalta_mag_all), marker='P', edgecolor='k')
    ax_tm.scatter(noa_data_all['STA_arrival_time'], noa_data_all['EQ_magnitude'], color='b', alpha=0.3, s=size_func(noa_data_all['EQ_magnitude']), marker='*')
    ax_tm.set_xlim([t_start, t_end])
    
    # Panel 2: 时间直方图 (总览)
    dt = timedelta(hours=2)
    bins = mdates.date2num(np.arange(t_start, t_end + dt, dt))
    ax_th.hist(mdates.date2num(mf_data_all['Rt_trig']), bins=bins, color='r', alpha=0.7, label=f'MF: #{len(mf_data_all)}')
    ax_th.hist(mdates.date2num(stalta_data_all['peak_time']), bins=bins, color='g', alpha=0.5, label=f'STA/LTA: #{len(stalta_data_all)}')
    ax_th.hist(mdates.date2num(noa_data_all['STA_arrival_time']), bins=bins, color='b', alpha=0.5, label=f'NOA: #{len(noa_data_all)}')
    ax_th.legend(loc='upper right', fontsize=36, framealpha=0)
    ax_th.set_xlim([t_start, t_end])

    # Panel 3 & 4: 震级频次与G-R图 (总览)
    mag_bins = np.arange(-1, 4.1, 0.1)
    mf_mag_all = mf_data_all['Real_Mag'].dropna()
    stalta_mag_all = stalta_data_all['mf_real_mag'].fillna(stalta_data_all['noa_magnitude']).dropna()
    noa_mag_all = noa_data_all['EQ_magnitude'].dropna()

    ax_mf.hist(mf_mag_all, bins=mag_bins, color='r', alpha=0.5)
    ax_mf.hist(stalta_mag_all, bins=mag_bins, color='g', alpha=0.5)
    ax_mf.hist(noa_mag_all, bins=mag_bins, color='b', alpha=0.5)

    for mag_data, color in [(mf_mag_all, 'r'), (stalta_mag_all, 'g'), (noa_mag_all, 'b')]:
        if not mag_data.empty:
            counts, _ = np.histogram(mag_data, bins=mag_bins)
            cum_counts = np.cumsum(counts[::-1])[::-1]
            valid = cum_counts > 0
            ax_gr.plot(mag_bins[:-1][valid], cum_counts[valid], color=color, lw=5.5, marker='o', markersize=6)

    # --- 统一格式化 ---
    for i in range(no_temp + 1):
        axes_upper_left[i].set_ylabel('ML', fontsize=38)
        axes_upper_left[i].grid(True, linestyle='--'); axes_upper_left[i].set_ylim(-1, 4.5); axes_upper_left[i].tick_params(axis='y', labelsize=38)
        plt.setp(axes_upper_left[i].get_xticklabels(), visible=False)
        
        axes_lower_left[i].set_ylabel('Counts per 2h', fontsize=36)
        axes_lower_left[i].grid(True, linestyle='--'); axes_lower_left[i].set_yscale('log'); axes_lower_left[i].tick_params(axis='y', labelsize=36)
        Ticks_xaxis_day_or_hour(axes_lower_left[i], Day_tick=5, font=38)
        
        axes_upper_right[i].set_ylabel('Count', fontsize=36)
        axes_upper_right[i].grid(True, linestyle='--'); axes_upper_right[i].set_yscale('log'); axes_upper_right[i].tick_params(axis='y', labelsize=36)
        plt.setp(axes_upper_right[i].get_xticklabels(), visible=False)
        axes_upper_right[i].set_xlim(-1, 4)
        axes_upper_right[i].set_yticks([10, 100, 1000])
        
        if i == no_temp+1-1:
            axes_lower_right[i].set_xlabel('Magnitude', fontsize=38)
        axes_lower_right[i].set_ylabel('Cum Count', fontsize=36)
        axes_lower_right[i].grid(True, linestyle='--'); axes_lower_right[i].set_yscale('log'); axes_lower_right[i].tick_params(axis='both', labelsize=38)
        axes_lower_right[i].set_xlim(-1, 4); axes_lower_right[i].set_xticks([-1, 0, 1, 2, 3, 4])
        axes_lower_right[i].set_yticks([10, 100, 1000, 10000])

    # 添加震级大小图例
    mag_legend = [1, 2, 3, 4]
    legend_handles_mag = [Line2D([], [], linestyle='None', marker='o', markersize=np.sqrt(size_func(m)), color='k', label=f'ML {m}') for m in mag_legend]
    axes_upper_left[0].add_artist(axes_upper_left[0].legend(handles=legend_handles_mag, title='Local Magnitude', loc='upper left', framealpha=0, fontsize=28, title_fontsize=28))
    
    # 添加suptitle
    # fig.suptitle(f'Matched Filter Detections', fontsize=68, weight='bold', y=0.95)

    # 添加每个模版的标记在图左侧
    row_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for tt, Template in enumerate(Temp_name_list):
            ax_tm, ax_th = axes_upper_left[tt], axes_lower_left[tt]
            ax_mf, ax_gr = axes_upper_right[tt], axes_lower_right[tt]
            disp_Template = disp_Temp_name.get(Template, Template)
            
            ax_tm.text(-0.06, 0.91, f'{row_labels[tt]}1)', transform=ax_tm.transAxes,
                    fontsize=45, fontweight='bold', va='bottom', ha='left')
            
            ax_mf.text(-0.24, 0.91, f'{row_labels[tt]}2)', transform=ax_mf.transAxes,
                    fontsize=45, fontweight='bold', va='bottom', ha='left')

            ax_tm.text(-0.08, 0.0, disp_Template.replace('temp_', 'template '), transform=ax_tm.transAxes,
                    fontsize=55, fontweight='bold', va='center', ha='center', rotation=90)
            
    # label the last sum figure
    tt = no_temp
    ax_tm_sum, ax_mf_sum = axes_upper_left[tt], axes_upper_right[tt]
    
    ax_tm_sum.text(-0.06, 0.91, f'{row_labels[tt]}1)', transform=ax_tm_sum.transAxes,
            fontsize=45, fontweight='bold', va='bottom', ha='left')
    
    ax_mf_sum.text(-0.24, 0.91, f'{row_labels[tt]}2)', transform=ax_mf_sum.transAxes,
            fontsize=45, fontweight='bold', va='bottom', ha='left')

    ax_tm_sum.text(-0.08, 0.0, 'Overall', transform=ax_tm_sum.transAxes,
            fontsize=55, fontweight='bold', va='center', ha='center', rotation=90)
    
    
    # 保存
    png_name = os.path.join(output_dir, "3_catalog_comparison_plot.png")
    fig.savefig(png_name, bbox_inches='tight')
    print(f"\n[成功] 目录对比图已保存至:\n{png_name}")
    plt.close(fig)
    print("\n=================== 脚本 3 执行完毕 ===================")

if __name__ == '__main__':
    main()
