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
# ℹ️ 非检测主线（综合出图：震级-Mc-b值-滑移-地震矩）。所需噪声成品 noise_trends h5
#    在 Zenodo 发布版的 IN_noise_meteo/ 里（由外部 check_minmag 工程产出，详见配套仓库 WANGYuhan2002/yuhan_das_spectra（Zenodo 压缩包内目录名 check_minmag_pkg/）；本仓库不含该 h5，见 README.md 的数据一节），
#    已无外部依赖、可直接出图；检测主线核心在 2a、2b。
# -----------------------------------------------------------------------------
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

# # Combined Analysis Script: Magnitude, b-value, Mc, Noise, Slip, and Moment
# This script generates a comprehensive 4-panel plot for each seismic template.
# 1. Top Panel (a): Magnitude-time distribution and noise level.
# 2. Second Panel (b): b-value and Mc time series.
# 3. Third Panel (c): Binned and cumulative seismic slip.
# 4. Bottom Panel (d): Binned and cumulative seismic moment.

# +
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import h5py
import json

from scipy.stats import linregress
# -

# !__________________________ CONFIGURATION CLASS _____________________________
class Config:
    """
    Configuration class to hold all parameters for the analysis and plotting.
    """
    def __init__(self):
        # --- 1. Base Paths and Data Location ---
        self.HOME_dir = f'{PROJECT_ROOT}/'
        self.level = 20  # Classification level for file paths

        # --- 2. Input Data ---
        self.IN_MagEventList_path = os.path.join(self.HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{self.level}/MagRegression/Events_classified_Real_Mag.csv')
        self.IN_Temp_lib_csv = os.path.join(self.HOME_dir, "Output_0_Template_library/Template_library.csv")
        self.noise_level_h5_path = os.path.join(self.HOME_dir, "IN_noise_meteo/noise_trends_f8-20Hz_p5.h5")  # 本发布版自带的噪声成品（外部 check_minmag 工程产物，详见 IN_noise_meteo/）
        
        # --- 3. Output Parameters ---
        self.OUTPUT_dir = os.path.join(self.HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/Show_Classification_byCCs/level_{self.level}/5_show_with_mag_mc_bvalue_mc_noise_slip_momentum/')
        os.makedirs(self.OUTPUT_dir, exist_ok=True)

        # --- 4. Analysis Time Window (TIMEZONE-AWARE) ---
        self.t_start = pd.Timestamp('2024-04-24 10:20:00', tz='UTC')
        self.t_end = pd.Timestamp('2024-06-22 17:10:00', tz='UTC')

        # --- 5. b-value Calculation Parameters (Panel b) ---
        self.b_method = 'MLE'
        self.mc_method = 'maxc'
        self.minimum_events = 100
        self.window_hours = 12
        self.step_hours = 2
        self.b_bin_width = 0.02
        
        # --- 6. Slip & Moment Calculation Parameters (Panels c, d) ---
        self.STRESS_DROP = 1e6
        self.SHEAR_MODULUS = 30e9
        self.time_bin_minutes = 10

        # --- 7. Plotting Style & Font Sizes ---
        self.fig_size = (30, 25)
        self.height_ratios = [3, 1.2, 1, 1] 

        self.font_size = 37
        plt.rcParams['font.size'] = self.font_size

        self.panel_label_fontsize = self.font_size + 4
        self.mag_plot_title_fontsize = self.font_size
        self.mag_plot_axis_label_fontsize = self.font_size - 2
        self.mag_plot_tick_fontsize = self.font_size - 4
        self.mag_legend_fontsize = self.font_size - 4
        self.mag_legend_title_fontsize = self.font_size - 2

        self.scatter_min_size = 40
        self.scatter_max_size = 1600
        self.scatter_min_mag = -0.5
        self.scatter_max_mag = 4.0
        self.mag_legend_values = [1, 2, 3]
        
        self.slip_moment_title_fontsize = self.font_size
        self.slip_moment_axis_label_fontsize = self.font_size - 2
        self.slip_moment_tick_fontsize = self.font_size - 4
        self.slip_moment_legend_fontsize = self.font_size - 4

    def save_config(self, template_name):
        """Saves the configuration to a JSON file in the output directory."""
        path = os.path.join(self.OUTPUT_dir, f'config_{template_name}.json')
        serializable_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (datetime, timedelta, pd.Timestamp)):
                serializable_dict[key] = str(value)
            else:
                serializable_dict[key] = value
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, indent=4, ensure_ascii=False)
        print(f"Configuration for {template_name} saved to: {path}")

# !__________________________ HELPER FUNCTIONS _____________________________

def Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0, font=18, is_bottom=True):
    """Sets major and minor ticks for a datetime x-axis."""
    if Day_tick > 0:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=Day_tick))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        if is_bottom:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=font)
        else:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
    elif Hour_tick > 0:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=Hour_tick))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        if is_bottom:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=font)
        else:
            ax.xaxis.set_major_formatter(plt.NullFormatter())


def calculate_mc_maxc(magnitudes, bin_width=0.1):
    """Calculates completeness magnitude (Mc) using the Maximum Curvature (MAXC) method."""
    if len(magnitudes) == 0:
        return np.nan
    min_m = np.floor(np.min(magnitudes) / bin_width) * bin_width
    max_m = np.ceil(np.max(magnitudes) / bin_width) * bin_width
    bins = np.arange(min_m, max_m + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    if np.sum(counts) == 0:
        return np.nan
    max_count_idx = np.argmax(counts)
    mc = edges[max_count_idx]
    return mc

def calculate_b_value_MLE(magnitudes, mc, minimum_events=20):
    """Calculates b-value using Maximum Likelihood Estimation."""
    mags_above_mc = magnitudes[magnitudes >= mc]
    if len(mags_above_mc) < minimum_events:
        return np.nan, np.nan
    mean_m = np.mean(mags_above_mc)
    b_value = np.log10(np.e) / (mean_m - mc)
    return b_value, mc

def compute_b_value_time_series(times, magnitudes, window_hours, step_hours, bin_width, minimum_events):
    """Computes a time series of b-value and Mc using a sliding window."""
    df = pd.DataFrame({'time': pd.to_datetime(times), 'magnitude': magnitudes})
    df.sort_values('time', inplace=True)
    
    window_centers, b_values, mc_arr = [], [], []
    if df.empty:
        return [], [], []

    tz = df['time'].dt.tz

    start_time_num = mdates.date2num(df['time'].iloc[0])
    end_time_num = mdates.date2num(df['time'].iloc[-1])
    window_days = window_hours / 24.0
    step_days = step_hours / 24.0

    current_time_num = start_time_num
    while current_time_num + window_days <= end_time_num:
        window_start = mdates.num2date(current_time_num, tz=tz)
        window_end = mdates.num2date(current_time_num + window_days, tz=tz)
        
        mask = (df['time'] >= window_start) & (df['time'] < window_end)
        mags_window = df['magnitude'][mask].to_numpy()
        
        if len(mags_window) >= minimum_events:
            mc = calculate_mc_maxc(mags_window, bin_width)
            if not np.isnan(mc):
                b, mc_out = calculate_b_value_MLE(mags_window, mc, minimum_events)
                if not np.isnan(b):
                    b_values.append(b)
                    mc_arr.append(mc_out)
                    window_centers.append(window_start + (window_end - window_start) / 2)
        
        current_time_num += step_days
        
    return window_centers, b_values, mc_arr


def read_noise_level(h5_filepath):
    """Reads noise trend data from an HDF5 file."""
    print(f"--> Reading noise data from: {h5_filepath}")
    try:
        with h5py.File(h5_filepath, 'r') as f:
            noise_matrix = f['noise_amplitude'][:]
            channels = f['channels_km'][:]
            time_strs = [t.decode('utf-8') for t in f['timestamps_utc_str'][:]]
            time_steps = [pd.Timestamp(ts, tz='UTC') for ts in time_strs]
    except Exception as e:
        print(f"Error reading HDF5 file '{h5_filepath}': {e}")
        return None, None
    
    channels_to_avg = [9, 10, 11, 12, 13, 14]
    indices = [np.where(channels == ch)[0][0] for ch in channels_to_avg if ch in channels]
    if not indices:
        print("Warning: None of the specified channels for noise averaging were found.")
        return None, None
    noise_level = noise_matrix[:, indices].mean(axis=1)
    
    return time_steps, noise_level

def Slip_from_Mw_Brune(sigma, mu, Mw):
    """Calculates seismic slip from Mw using a specific scaling relation."""
    return (sigma ** (2/3)) / mu * 571.73 * (10 ** (0.5 * Mw))

def Moment_from_Mw(Mw):
    """Calculates seismic moment from Mw."""
    return 10 ** (1.5 * (Mw + 6.03)) # Moment in N.m

# !__________________________ PLOTTING FUNCTIONS _____________________________

def plot_panel_label(ax, label, config):
    """Adds a bold label (e.g., 'a)') to the top-left of a subplot."""
    ax.text(-0.05, 1.05, label,
            transform=ax.transAxes,
            fontsize=config.panel_label_fontsize,
            fontweight='bold',
            va='bottom',
            ha='left')

def plot_magnitude_bvalue(ax, events_df, noise_times, noise_mag, config, disp_Template):
    """
    Plots Panel (a): Magnitude-time scatter and noise level.
    """
    ax.set_title(f'Magnitude and Noise Level for {disp_Template}', fontsize=config.mag_plot_title_fontsize)
    
    Real_mag = events_df['Real_Mag'].to_numpy(dtype=float)
    Rt_trig = events_df['Rt_trig']
    
    def size_func(mag):
        """Maps magnitude to marker size."""
        return np.interp(mag**2, (config.scatter_min_mag**2, config.scatter_max_mag**2), (config.scatter_min_size, config.scatter_max_size))

    size_arr = size_func(Real_mag)
    im = ax.scatter(Rt_trig, Real_mag, c=Real_mag, marker='o', edgecolor='k', lw=0.5, s=size_arr, alpha=0.7, cmap='rainbow', zorder=10)
    
    colorbar = plt.colorbar(im, ax=ax, pad=0.01)
    colorbar.set_label('Magnitude', fontsize=config.mag_plot_axis_label_fontsize)
    colorbar.ax.tick_params(labelsize=config.mag_plot_tick_fontsize)
    
    mag_legend_handles = [Line2D([], [], linestyle='None', marker='o', markersize=np.sqrt(s), color='gray', label=f'ML {m}') 
                          for m, s in zip(config.mag_legend_values, size_func(np.array(config.mag_legend_values)))]

    all_handles = mag_legend_handles
    if noise_times is not None and noise_mag is not None:
        noise_line, = ax.plot(noise_times, noise_mag, 'C6.-', linewidth=1, markersize=2.5, alpha=0.5, label='Noise Magnitude', zorder=11)
        all_handles = mag_legend_handles + [noise_line]

    all_labels = [h.get_label() for h in all_handles]
    ax.legend(all_handles, all_labels, loc='upper right', framealpha=0.2,
              fontsize=config.mag_legend_fontsize, title_fontsize=config.mag_legend_title_fontsize)


    ax.set_ylabel('Local Magnitude', fontsize=config.mag_plot_axis_label_fontsize)
    ax.set_xlim(config.t_start, config.t_end)
    ax.tick_params(axis='both', labelsize=config.mag_plot_tick_fontsize)
    ax.grid(color='k', linestyle=':', alpha=0.5, zorder=0)

def plot_bvalue_mc(ax, events_df, config, disp_Template):
    """
    Plots Panel (b): b-value and Mc time series.
    """
    ax.set_title(f'b-value and Magnitude of Completeness', fontsize=config.slip_moment_title_fontsize)
    Real_mag = events_df['Real_Mag'].to_numpy(dtype=float)
    Rt_trig = events_df['Rt_trig']

    if len(Real_mag) < config.minimum_events:
        ax.text(0.5, 0.5, 'Not enough events for b-value/Mc analysis',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=config.slip_moment_axis_label_fontsize)
        ax.set_yticks([])
        return

    t_center, b_vals, mc_vals = compute_b_value_time_series(
        Rt_trig, Real_mag, config.window_hours, config.step_hours,
        config.b_bin_width, config.minimum_events
    )

    if not t_center:
        ax.text(0.5, 0.5, 'No windows with sufficient events for b-value/Mc',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=config.slip_moment_axis_label_fontsize)
        ax.set_yticks([])
        return

    line_mc, = ax.plot(t_center, mc_vals, 'k-', lw=2., markersize=3, alpha=0.6, label='Mc', zorder=21)
    ax.set_ylabel('Mc', color='k', fontsize=config.slip_moment_axis_label_fontsize)
    ax.tick_params(axis='y', labelcolor='k', labelsize=config.slip_moment_tick_fontsize)
    
    ax_b = ax.twinx()
    line_b, = ax_b.plot(t_center, b_vals, 'r-o', lw=2., markersize=3, alpha=0.6, label='b-value', zorder=20)
    ax_b.set_ylabel('b-value', color='r', fontsize=config.slip_moment_axis_label_fontsize)
    ax_b.tick_params(axis='y', labelcolor='r', labelsize=config.slip_moment_tick_fontsize)
    ax_b.grid(True, linestyle='--', alpha=0.6, color='r')

    ax.set_xlim(config.t_start, config.t_end)
    # ax.legend(handles=[line_mc, line_b], loc='upper left', fontsize=config.slip_moment_legend_fontsize)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_cumulative_bar(ax, events_df, data_col, cumulative_col, ylabel, twin_ylabel, title, color, config):
    """
    Plots Panel (c) or (d): A bar plot of a quantity per time bin and its cumulative value.
    *** THIS FUNCTION HAS BEEN FIXED ***
    """
    ax.set_title(title, fontsize=config.slip_moment_title_fontsize)
    # FIX 1: Changed 'T' to 'min' to address FutureWarning
    t_edges = pd.date_range(start=config.t_start, end=config.t_end, freq=f"{config.time_bin_minutes}min")
    
    if not events_df.empty:
        time_bins = pd.cut(events_df['Rt_trig'], bins=t_edges, right=False)
        
        # FIX 2 (CORE FIX): Select the column *before* summing to avoid TypeError.
        # FIX 3: Added observed=False to groupby to address FutureWarning.
        data_per_bin = events_df.groupby(time_bins, observed=False)[data_col].sum().reindex(
            pd.IntervalIndex.from_breaks(t_edges, closed='left'), fill_value=0
        )
        
        ax.bar(data_per_bin.index.left, data_per_bin.values, width=t_edges.freq.delta, align='edge', edgecolor='k', alpha=0.7, label=f'{ylabel} per {config.time_bin_minutes} min')
        ax.set_ylabel(ylabel, fontsize=config.slip_moment_axis_label_fontsize)

        ax_twin = ax.twinx()
        time_for_step = pd.concat([pd.Series([config.t_start]), events_df['Rt_trig']])
        # The cumulative sum for the step plot needs to be calculated correctly.
        # The 'cumulative_col' from the dataframe is already a cumsum, we should use it directly.
        cumulative_for_step = pd.concat([pd.Series([0]), events_df[cumulative_col]])

        ax_twin.step(time_for_step, cumulative_for_step, where='post', color=color, lw=2.5, label=f'Cumulative {ylabel}')
        ax_twin.set_ylabel(twin_ylabel, color=color, fontsize=config.slip_moment_axis_label_fontsize)
        ax_twin.tick_params(axis='y', labelcolor=color, labelsize=config.slip_moment_tick_fontsize)
        
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax_twin.grid(True, linestyle='--', alpha=0.6, color=color)
    else:
        ax.text(0.5, 0.5, 'No Events', transform=ax.transAxes, ha='center', va='center', fontsize=20)


    ax.set_xlim(config.t_start, config.t_end)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=config.slip_moment_tick_fontsize)

# !__________________________ MAIN SCRIPT _____________________________
def main():
    """Main execution function."""
    print("--- Starting Comprehensive Analysis and Plotting Script ---")
    config = Config()

    print(f"--> Loading event data from: {config.IN_MagEventList_path}")
    try:
        date_columns = ['Rt_trig', 't_trig', 'Rt_left', 'Rt_right', 't_right', 't_left']
        events_df = pd.read_csv(
            config.IN_MagEventList_path, sep='\t', 
            parse_dates=date_columns
        )
        
        for col in date_columns:
             if events_df[col].dt.tz is None:
                events_df[col] = events_df[col].dt.tz_localize('UTC')
             else:
                events_df[col] = events_df[col].dt.tz_convert('UTC')

        events_df = events_df[(events_df['Rt_trig'] >= config.t_start) & (events_df['Rt_trig'] <= config.t_end)].copy()
        print(f"    Loaded {len(events_df)} total events in the specified time range.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Event data file not found at {config.IN_MagEventList_path}. Exiting.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    try:
        temp_lib_df = pd.read_csv(config.IN_Temp_lib_csv, sep=',')
        temp_lib_df = temp_lib_df.sort_values(by='plot_order', ascending=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: Template library not found at {config.IN_Temp_lib_csv}. Exiting.")
        return
        
    noise_times, noise_level = read_noise_level(config.noise_level_h5_path)
    
    for _, row in temp_lib_df.iterrows():
        Template = row['infile_name']
        disp_Template = row['plot_name']
        
        disp_Template = disp_Template.replace('Temp_', 'Template ')
        
        print(f"\n\033[1;35m === Processing Template: {disp_Template} ({Template}) === \033[0m")
        
        template_events = events_df[events_df['Template'] == Template].sort_values(by='Rt_trig').reset_index(drop=True)
        print(f"    Found {len(template_events)} events for this template.")

        if template_events.empty:
            print("    No events to plot. Skipping.")
            continue
            
        noise_mag = None
        if len(template_events) > 1 and noise_level is not None:
            valid_fit_data = template_events[['amp_log_mean', 'Real_Mag']].dropna()
            if len(valid_fit_data) > 1:
                a, b = np.polyfit(valid_fit_data['amp_log_mean'], valid_fit_data['Real_Mag'], deg=1)
                print(f"    Linear fit for noise: Real_mag ≈ {a:.4f} * log10(amplitude) + {b:.4f}")
                log_noise = np.log10(noise_level  * 5)
                noise_mag = a * log_noise + b
            else:
                print("    Not enough valid data points to perform linear fit for noise magnitude.")
        
        template_events['Slip'] = Slip_from_Mw_Brune(config.STRESS_DROP, config.SHEAR_MODULUS, template_events['Real_Mag'])
        #?++++++++++ add sign for the slip with CCs_s_trig sign
        template_events['Slip'] *= template_events['CCs_s_trig'].apply(lambda x: -1 if x < 0 else 1)
        
        template_events['Moment'] = Moment_from_Mw(template_events['Real_Mag'])
        template_events['cum_Slip'] = template_events['Slip'].cumsum()
        template_events['cum_Moment'] = template_events['Moment'].cumsum()

        fig = plt.figure(figsize=config.fig_size, constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=config.height_ratios)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        
        axes = [ax1, ax2, ax3, ax4]
        
        plot_magnitude_bvalue(ax1, template_events, noise_times, noise_mag, config, disp_Template)
        plot_bvalue_mc(ax2, template_events, config, disp_Template)
        plot_cumulative_bar(ax3, template_events, 'Slip', 'cum_Slip', 
                            f'Slip (m)\nper {config.time_bin_minutes} min', 'Cumulative\nSlip (m)', 
                            rf"Slip Calculated with Brune's Model ($\sigma = {(config.STRESS_DROP*1e-6):.0f}\,\mathrm{{MPa}},\, \mu = {(config.SHEAR_MODULUS*1e-9):.0f}\,\mathrm{{GPa}}$)", # title
                            'b', config)
        plot_cumulative_bar(ax4, template_events, 'Moment', 'cum_Moment',
                            f'Moment (N·m)\nper {config.time_bin_minutes} min', 'Cumulative\nMoment (N·m)', 
                            f"Seismic Moment Release", #title
                            'C1', config)

        panel_labels = ['a)', 'b)', 'c)', 'd)']
        for i, ax in enumerate(axes):
            plot_panel_label(ax, panel_labels[i], config)
            is_bottom_plot = (ax == axes[-1])
            Ticks_xaxis_day_or_hour(ax, Day_tick=5, font=config.mag_plot_tick_fontsize, is_bottom=is_bottom_plot)
            if not is_bottom_plot:
                plt.setp(ax.get_xticklabels(), visible=False)

        fig.suptitle(f'Matched Filter Detections of {disp_Template}', fontsize=config.panel_label_fontsize+5, weight='bold')
        
        fname = f"Matched_Filter_Results_{disp_Template}.png"
        output_path = os.path.join(config.OUTPUT_dir, fname)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"\033[1;32m    Successfully saved plot to: {output_path}\033[0m")
        config.save_config(disp_Template)

    print("\n--- Script finished. ---")


if __name__ == '__main__':
    main()