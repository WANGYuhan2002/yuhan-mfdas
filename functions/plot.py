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

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import numpy as np
from datetime import datetime, timedelta


    
    
# def plot_style_1(fig, t_start, t_end, IN_retrig_dir, Temp_name_list, Day_tick, Hour_tick):
#     ax = fig.add_subplot(111)
#     ax.set_title(f'Events from {t_start} to {t_end}')

#     # 为每个 Template 分配不同的颜色
#     Color_by_temp = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#     color_map = {Template: Color_by_temp[i % len(Color_by_temp)] for i, Template in enumerate(Temp_name_list)}

#     # 遍历模板，绘制散点图
#     for Template in Temp_name_list:
#         OUT_dict = Read_Retrig_OneTemp_from_pkl(t_start, t_end, Template, IN_retrig_dir)
#         t_trig_s = OUT_dict['t_trig_s']
#         quasi_Mag = OUT_dict['quasi_Mag_CCs_s']
#         # 使用每个模板对应的颜色
#         im = ax.scatter(t_trig_s, quasi_Mag, c=color_map[Template], label=Template, s=10, alpha=0.5)
#         del OUT_dict

#     ax.set_ylabel('Quasi-magnitude from CCs')
#     ax.set_xlim(t_start, t_end)
#     # ax.set_ylim(np.percentile(quasi_Mag, 5), np.percentile(quasi_Mag, 95))

#     Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick)

#     # 添加图例，显示每个模板的颜色
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Templates')


# def plot_style_2(fig, t_start, t_end, IN_retrig_dir, Template, Day_tick, Hour_tick):
#     """
#     """
#     OUT_dict = Read_Retrig_OneTemp_from_pkl(t_start, t_end, Template, IN_retrig_dir)
#     quasi_Mag = OUT_dict['quasi_Mag_CCs_s']
    
#     gs = plt.GridSpec(2,2,figure=fig,height_ratios=[3,1],width_ratios=[7.3,1])
#     axes = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0])]
    
#     axes[0].set_title(f'Events from {t_start} to {t_end}')
#     vmin = 0
#     vmax = np.percentile(OUT_dict['Bias_level_arrivals'], 95)
#     # --1--plot scatter
#     im = axes[0].scatter(OUT_dict['t_trig_s'], quasi_Mag, c=OUT_dict['Bias_level_arrivals'], cmap='YlOrBr', s=10, alpha=0.6,vmin=vmin,vmax=vmax)
#     axes[0].set_ylabel('Quasi-magnitude from CCs')
#     axes[0].set_xlim(t_start, t_end) 
#     axes[0].set_ylim(np.percentile(quasi_Mag, 5), np.percentile(quasi_Mag, 95))
#     Ticks_xaxis_day_or_hour(axes[0], Day_tick, Hour_tick)
#     # --1b--add colorbar
#     colorbar = plt.colorbar(im, ax=axes[0])
#     colorbar.set_label('Bias_level_arrivals')
#     im.set_clim([vmin,vmax])
    
#     # --2--plot histogram
#     bins = ((t_end - t_start).days*24 + (t_end - t_start).seconds//3600)
#     axes[1].hist(OUT_dict['t_trig_s'], bins, color='k', alpha=1, range=(t_start, t_end))
#     axes[1].set_title('Temporal Distribution')
#     axes[1].set_xlabel('time')
#     axes[1].set_ylabel('Counts per hour')
#     axes[1].set_xlim(t_start, t_end)
#     axes[1].set_yscale('log')
#     Ticks_xaxis_day_or_hour(axes[1], Day_tick, Hour_tick)
    
    
def Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick, rotation=0, font=14):
    if Day_tick > 0 and Hour_tick == 0:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=Day_tick))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    elif Day_tick == 0 and Hour_tick > 0:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=Hour_tick))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d.%H'))
    else: 
        print('WARNING: Day_tick or Hour_tick should be non-zero. Do not support both non-zero')
    ax.grid(True)
    ax.xaxis.set_tick_params(rotation=rotation, labelsize=font)  # ✅ 设置字体大小和旋转
    
def Ticks_xaxis_minute_or_second(ax, Minute_tick, Second_tick, rotation=0,font=14):
    if Minute_tick > 0 and Second_tick == 0:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=Minute_tick))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d.%H:%M'))
    elif Minute_tick == 0 and Second_tick > 0:
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=Second_tick))
        ax.xaxis.set_minor_locator(mdates.SecondLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:
        print('WARNING: Minute_tick or Second_tick should be non-zero. Do not support both non-zero')
    ax.grid(True)
    ax.xaxis.set_tick_params(rotation=rotation, labelsize=font)  # ✅ 设置字体大小


def waterfall_date(ax, data, fs, t_start, dx, d_start, unit, vlim=0):
    #-- imshow data
    if(vlim==0):
        vlim = np.mean(np.abs(data))*3
    else:
        vlim = np.mean(np.abs(data))*vlim

    # plot grid
    t_left = t_start
    t_right = t_start + timedelta(seconds=data.shape[0]/fs)
    d_bottom = d_start
    d_top = d_start + data.shape[1]*dx/1000
    im = ax.imshow(data.T, aspect="auto", cmap='seismic', interpolation='antialiased', 
            extent=(t_left,t_right, d_top, d_bottom),
            vmin=-vlim, vmax=vlim)
    
    #-- y-axis
    ax.set_ylabel("Distance [km]")
    yticks = np.arange(np.ceil(d_start), np.floor(d_top), 1)
    if len(yticks) > 10:
        yticks = np.arange(np.ceil(d_start), np.floor(d_top), 2)
    elif len(yticks) > 20:
        yticks = np.arange(np.ceil(d_start), np.floor(d_top), 3)
    elif len(yticks) > 30:
        yticks = np.arange(np.ceil(d_start), np.floor(d_top), 4)
    elif len(yticks) > 50:
        yticks = np.arange(np.ceil(d_start), np.floor(d_top), 5)
    
    if yticks[0]> d_start:
        yticks = np.insert(yticks, 0, d_start)
    if yticks[-1]< d_top:
        yticks = np.append(yticks, d_top)
        
    ax.set_yticks(yticks)

    #-- Define the x-tick marks we want, format the labels
    Ticks_xaxis_minute_or_second(ax, Minute_tick=0, Second_tick=10, rotation=0,font=14)

    plt.colorbar(im,label=unit,ax=ax)
    
    ax.set_title('Data on {0} GMT'.format(t_start.strftime('%Y.%m.%d')))
    ax.invert_yaxis()
    ax.grid(linestyle='--', linewidth=0.5)


def generate_gs_grid(fig,no):
    gs = plt.GridSpec(no, 2, figure=fig, height_ratios=[1]*no, width_ratios=[7.3, 1]) # adapt the gs according to the number of templates
    axes = [fig.add_subplot(gs[i, :]) for i in range(no)]
    return axes