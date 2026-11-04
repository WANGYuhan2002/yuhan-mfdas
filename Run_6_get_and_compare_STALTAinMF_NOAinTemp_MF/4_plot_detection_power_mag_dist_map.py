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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import os
import geopandas
import contextily as cx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
from datetime import datetime
import re
import requests
from io import StringIO
import io
import sys

sys.path.append(f'{PROJECT_ROOT}/functions/')
from noa import extract_station_codes, load_station_inventory

# =================================================================================
# 脚本功能:
# 该脚本严格复现 `fig4_Mag_dist_map.py` 和 `6c_compare_Mag_Dist_stalta_MF.py` 的核心视觉风格。
# 它使用由前序脚本生成的规整化数据，创建双面板图，从空间维度评估和对比探测能力。
# 关键修正：在震级-距离图中，将匹配到模板的STA/LTA事件也绘制在对应模板的固定距离上，
# 以便与MF的检测能力进行直接对比。
# =================================================================================



def main():
    print("=============================================")
    print("  脚本 4: 探测能力与空间分布图 (严格复刻版)")
    print("=============================================\n")

    # --- 配置 (严格参考 fig4_Mag_dist_map.py 和 6c_compare_Mag_Dist_stalta_MF.py) ---
    HOME_dir = f'{PROJECT_ROOT}/'
    level = 20
    data_dir = os.path.join(HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{level}/eq_list_to_plot/')
    output_dir = os.path.join(HOME_dir, f'Output_6_Compare_MF_stalta_NOA/level_{level}/compare_figures/')
    os.makedirs(output_dir, exist_ok=True)
    
    inset_image_path = os.path.join(HOME_dir, 'Output_6_Compare_MF_stalta_NOA/insert_fig.png')
    das_lon, das_lat = 20.59861, 38.43328
    map_lat_range = [36.5, 40.5]
    map_lon_range = [18.9, 22.7]

    Temp_name_list = ['template_10_80Hz', 'template_0423_80Hz', 'template_8_80Hz', 'template_20_80Hz', 'template_0425_80Hz']
    disp_Temp_name = {'template_10_80Hz': 'temp_K1', 'template_0423_80Hz': 'temp_K2', 'template_8_80Hz': 'temp_K3', 'template_20_80Hz': 'temp_I1', 'template_0425_80Hz': 'temp_I2'}

    ims_file = f'{PROJECT_ROOT}/IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt'

    # --- 数据加载 ---
    print("--- 正在加载绘图数据...")
    try:
        df_stalta = pd.read_csv(os.path.join(data_dir, 'stalta_catalog_enriched.csv'), sep='\t', parse_dates=['peak_time'])
        df_noa = pd.read_csv(os.path.join(data_dir, 'noa_catalog_labeled.csv'), sep='\t', parse_dates=['STA_arrival_time'])
        df_mf = pd.read_csv(os.path.join(HOME_dir, f'Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_{level}/MagRegression/Events_classified_Real_Mag.csv'), sep='\t', parse_dates=['Rt_trig'])
        df_templates = pd.read_csv(os.path.join(HOME_dir, "Output_0_Template_library/Template_library.csv"))
        
        # --- [新流程] 使用官方Inventory绘制台站 ---
        print("正在从IMS文件提取台站代码以进行匹配...")
        station_codes_from_ims = extract_station_codes(ims_file)
        print(f"从IMS文件中发现 {len(station_codes_from_ims)} 个唯一台站代码。")
        # [新流程] 加载官方台站Inventory
        station_inventory = load_station_inventory()
        # 筛选出在IMS文件中出现过的台站
        plot_stations = station_inventory[station_inventory['code'].isin(station_codes_from_ims)]
        # 进一步筛选出在地图范围内的台站
        lon_min, lon_max, lat_min, lat_max = map_lon_range[0], map_lon_range[1], map_lat_range[0], map_lat_range[1]
        plot_stations_in_map = plot_stations[
            (plot_stations['lon'] >= lon_min) & (plot_stations['lon'] <= lon_max) &
            (plot_stations['lat'] >= lat_min) & (plot_stations['lat'] <= lat_max)
        ]
        
        print("数据加载完毕。\n")

        print("--- 所有数据加载成功。")

        das_coord = (das_lat, das_lon)
        df_templates['distance_km'] = df_templates.apply(lambda row: geodesic((row['EQ_latitude'], row['EQ_longitude']), das_coord).kilometers, axis=1)
    except FileNotFoundError as e:
        print(f"\n[错误] 数据文件未找到: {e.filename}")
        return
    except Exception as e:
        print(f"\n[错误] 数据加载过程中发生未知错误: {e}")
        # 在调试时，可以取消下面这行注释来查看完整的错误追溯
        # import traceback; traceback.print_exc()
        return

    # --- 开始画图 (严格复刻) ---
    fig = plt.figure(figsize=(20, 21))
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], width_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.rcParams['font.size'] = 20

    # --- Panel 1: 震级-距离图 ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Detection Power Comparison in Magnitude-Distance Space", fontsize=30)
    
    # label_fsk = f'Events recorded by NOA station FSK = #{len(df_noa)}'
    label_fsk = f'Events recorded by NOA station FSK'
    ax1.scatter(df_noa['STA_distance'] * 111, df_noa['EQ_magnitude'], color='grey', label=label_fsk, alpha=0.3, s=200, marker='o')

    df_mf_loc = df_mf.merge(df_templates[['infile_name', 'distance_km']], left_on='Template', right_on='infile_name', how='left')
    df_mf_loc.dropna(subset=['distance_km'], inplace=True)
    # label_MF = f"Events by Matched Filter = #{len(df_mf)}"
    label_MF = f"Events by Matched Filter"

    for i, Template in enumerate(Temp_name_list):
        disp_Template = disp_Temp_name.get(Template, Template)
        
        dist = df_templates[df_templates['infile_name'] == Template]['distance_km'].mean()
        if pd.isna(dist): continue

        # 绘制 STA/LTA + MF 重叠部分
        stalta_template_data = df_stalta[df_stalta['mf_template_name'] == Template]
        mag_stalta = stalta_template_data['mf_real_mag'].dropna()
        label_sl = f'Events by STA/LTA (overlapped part)' if i == 0 else ""
        ax1.scatter((dist + 0.5) * np.ones_like(mag_stalta), mag_stalta, alpha=0.2, s=200, marker='*', color='b', label=label_sl)

        # 绘制 STA/LTA + NOA 重叠部分
        stalta_noa_data = df_stalta[(df_stalta['noa_match_flag']==True)]
        ax1.scatter(stalta_noa_data['noa_distance_km'], stalta_noa_data['noa_magnitude'], color='b', alpha=0.06, s=200, marker='*')


        # 绘制 MF 检测结果
        mf_template_data = df_mf_loc[df_mf_loc['Template'] == Template]
        mag_mf = mf_template_data['Real_Mag']
        label = label_MF if i == 0 else ""
        ax1.scatter(dist * np.ones_like(mag_mf), mag_mf, alpha=0.04, s=200, marker='o', color='r', label=label)

        # 标注模板名称
        if not mag_mf.empty:
            if Template in ['template_0423_80Hz']: #, 'template_10_80Hz'
                ax1.text(dist + 1.2, mag_mf.min() + 2.0, disp_Template, rotation=90, va='bottom', ha='center', fontsize=20, color='r', weight='bold')
            else:
                ax1.text(dist, mag_mf.min() - 0.1, disp_Template, rotation=90, va='top', ha='center', fontsize=20, color='r', weight='bold')
    ax1.plot([70, 200], [0.9, 2.5], color='b', linestyle='--', linewidth=2.5)
    ax1.plot([10, 40], [-0.8, 0.8], color='b', linestyle='--', linewidth=2.5)
    ax1.plot([10, 40], [-1, 0.2], color='r', linestyle='--', linewidth=2.5)
    
    ax1.set_xlabel('Distance (km)', fontsize=26, loc='left')
    ax1.set_ylabel('Magnitude', fontsize=26)
    ax1.set_xscale('log')
    ax1.set_xlim([9, 200])
    ax1.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    leg=ax1.legend(loc='upper right', markerscale=0.8, framealpha=0.1, fontsize=20)
    for lh in leg.legend_handles:
        lh.set_alpha(0.8)  # 设置图例里的符号透明度

    # --- Panel 2: 插入图 ---
    ax2 = fig.add_subplot(gs[1, 0])
    try:
        img = mpimg.imread(inset_image_path)
        ax2.imshow(img, aspect='auto'); ax2.axis('off')
    except FileNotFoundError:
        ax2.text(0.5, 0.5, 'Inset image not found', ha='center', va='center'); ax2.axis('off')

    # --- Panel 3: 地图 ---
    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax3.set_title("Matched filter and STA/LTA detections\nincluded in NOA catalog", fontsize=24, pad=20)
    ax3.set_extent(map_lon_range + map_lat_range, crs=ccrs.PlateCarree())
    ax3.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0f0', zorder=0)
    ax3.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#a0c3d8', zorder=0)
    ax3.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='gray', zorder=1)
    cx.add_basemap(ax3, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, attribution_size=5)
    gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False

    gdf_noa = geopandas.GeoDataFrame(df_noa, geometry=geopandas.points_from_xy(df_noa.EQ_longitude, df_noa.EQ_latitude), crs="EPSG:4326")
    stalta_loc = df_stalta[df_stalta['noa_match_flag']]
    gdf_stalta = geopandas.GeoDataFrame(stalta_loc, geometry=geopandas.points_from_xy(stalta_loc.noa_longitude, stalta_loc.noa_latitude), crs="EPSG:4326")
    scaling_factor = 30
    gdf_noa['markersize'] = (gdf_noa['EQ_magnitude'] ** 2) * scaling_factor
    
    gdf_noa.plot(ax=ax3, marker='o', color='Grey', markersize=gdf_noa['markersize'], alpha=0.3, edgecolor='k', linewidth=0.2, zorder=2)
    
    #* detectable by MF
    mf_loc = df_noa[df_noa['is_detected_by_mf'] == True]
    gdf_mf = geopandas.GeoDataFrame(mf_loc, geometry=geopandas.points_from_xy(mf_loc.EQ_longitude, mf_loc.EQ_latitude), crs="EPSG:4326")
    gdf_mf['markersize'] = (gdf_mf['EQ_magnitude'] ** 2) * scaling_factor
    gdf_mf.plot(ax=ax3, marker='*', color='red', markersize=gdf_mf['markersize'], alpha=0.7, edgecolor='k', linewidth=0.2, zorder=4)

    #* detectable by STA/LTA
    gdf_stalta_nomf = gdf_stalta
    gdf_stalta_nomf['markersize'] = (gdf_stalta_nomf['noa_magnitude'] ** 2) * scaling_factor
    gdf_stalta_nomf.plot(ax=ax3, marker='*', color='white', markersize=gdf_stalta_nomf['markersize'], alpha=0.5, edgecolor='k', linewidth=0.2, zorder=3)
    
    # *绘制 DAS 位置
    ax3.plot(das_lon, das_lat, marker='^', markersize=12, color='blue', markeredgecolor='k', transform=ccrs.PlateCarree(), zorder=10)
    ax3.text(das_lon + 0.05, das_lat, 'DAS', transform=ccrs.PlateCarree(), fontsize=24, ha='left', va='center', zorder=10)

    #* 绘制 IMS 台站
    if not plot_stations_in_map.empty:
        ax3.scatter(plot_stations_in_map['lon'], plot_stations_in_map['lat'],
                    marker='^', s=70, alpha=0.7,
                    facecolors='g', edgecolors='k', linewidths=1,
                    transform=ccrs.Geodetic(), 
                    label='Seismic Stations', zorder=10)
        # print(f"将在地图上绘制 {len(plot_stations_in_map)} 个台站。")
        # for idx, row in plot_stations_in_map.iterrows():
        #     ax3.text(row['lon'] + 0.02, row['lat'], row['code'], transform=ccrs.Geodetic(), fontsize=12, fontweight='bold', color='black', zorder=11)


    # 图例
    #* 有计数版本的图例
    color_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'NOA events = {len(gdf_noa)}', markerfacecolor='grey', markersize=22, alpha=0.9),
        Line2D([0], [0], marker='*', color='w', label=f'Detectable by\nSTA/LTA = {len(gdf_stalta)}', markerfacecolor='white', markersize=22, alpha=0.9, markeredgecolor='k'),
        Line2D([0], [0], marker='*', color='w', label=f'Detectable by\nmatched filter = {len(gdf_mf)}', markerfacecolor='red', markersize=22, alpha=0.9, markeredgecolor='k'),        
    ]
    
    # #* 没有计数版本的图例
    # color_legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label=f'Events in NOA catalog', markerfacecolor='grey', markersize=22, alpha=0.9),
    #     Line2D([0], [0], marker='*', color='w', label=f'Detectable by STA/LTA', markerfacecolor='white', markersize=22, alpha=0.9, markeredgecolor='k'),
    #     Line2D([0], [0], marker='*', color='w', label=f'Detectable by matched filter', markerfacecolor='red', markersize=22, alpha=0.9, markeredgecolor='k'),        
    # ]
        
    color_legend = ax3.legend(handles=color_legend_elements, loc='upper right', fontsize=14, frameon=True)
    color_legend.get_frame().set_alpha(0.1)
    ax3.add_artist(color_legend)
    
    # station legend
    # 图例
    station_legend_elements = [
        Line2D([0], [0], marker='^', color='w', label=f'NOA Stations', markerfacecolor='g', markersize=12, alpha=0.7, markeredgecolor='k'),
        Line2D([0], [0], marker='^', color='w', label=f'DAS', markerfacecolor='blue', markersize=12, alpha=0.7, markeredgecolor='k'),
    ]
    station_legend = ax3.legend(handles=station_legend_elements, loc='upper left', fontsize=14, frameon=True)
    station_legend.get_frame().set_alpha(0.1)
    ax3.add_artist(station_legend)

    
    legend_magnitudes_display = [1.0, 2.0, 3.0]
    legend_marker_sizes_area = [(m ** 2) * scaling_factor for m in legend_magnitudes_display]
    legend_elements_size_corrected = [Line2D([0], [0], marker='o', color='w', label=f'ML {m}', markerfacecolor='dimgray', markersize=np.sqrt(s), linestyle='None') for m, s in zip(legend_magnitudes_display, legend_marker_sizes_area)]
    size_legend = ax3.legend(handles=legend_elements_size_corrected, title="Local magnitude", loc='lower left', fontsize=16, title_fontsize=18, frameon=True)
    size_legend.get_frame().set_alpha(0.1)
    ax3.add_artist(size_legend)


    # --- 设置 a,b,c 标注---
    row_labels = ['a', 'b', 'c']
    for ii, ax in enumerate([ax1, ax2, ax3]):
        if ii ==0:
            ax.text(-0.05, 1.02, f'{row_labels[ii]})', transform=ax.transAxes,
                    fontsize=28, fontweight='bold', va='bottom', ha='left')
            continue
        ax.text(-0.1, 1.02, f'{row_labels[ii]})', transform=ax.transAxes,
                    fontsize=28, fontweight='bold', va='bottom', ha='left')
    # --- 保存 ---
    output_filename = os.path.join(output_dir, "4_detection_power.png")
    fig.savefig(output_filename, dpi=250, bbox_inches='tight')
    print(f"\n[成功] 探测能力图已保存至:\n{output_filename}")
    plt.close(fig)
    print("\n=================== 脚本 4 执行完毕 ===================")

if __name__ == '__main__':
    main()