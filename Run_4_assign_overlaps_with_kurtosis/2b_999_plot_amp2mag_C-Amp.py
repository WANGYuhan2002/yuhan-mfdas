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

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. 绘图参数设置 (针对A4排版优化) ---
# 为了在A4页面中字体接近14pt，这里的字号需要设置得更大一些。
# 您可以微调这些值以获得最佳效果。
COLUMN_TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
ANNOTATION_FONTSIZE = 24 # 用于标注模板名称
LEGEND_FONTSIZE = 12
TEXT_FONTSIZE = 13 # 用于回归参数标注的字体大小
SUBPLOT_LABEL_FONTSIZE = 18 # a1), a2) 的字体

# --- 2. 定义文件和目录路径 ---
# 核心目录结构
HOME_dir = f'{PROJECT_ROOT}/'
subdir = 'level_20/'

# 输入数据目录
IN_CCs_Classification_dir = f"{HOME_dir}Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/{subdir}"
IN_MagRegression_dir =  f"{IN_CCs_Classification_dir}MagRegression/"
IN_Events_with_NOA_Mag_dir = os.path.join(IN_MagRegression_dir, 'Events_with_NOA_Mag')

# 输出目录
OUT_PLOT_dir = f"{IN_CCs_Classification_dir}999_plot_regression_CviaAmp_for_paper/"
os.makedirs(OUT_PLOT_dir, exist_ok=True)


# --- 3. 加载所需数据 ---

# 加载模板库信息
IN_Temp_lib_csv = f"{HOME_dir}/Output_0_Template_library/Template_library.csv"
Temp_lib_df = pd.read_csv(IN_Temp_lib_csv, sep=',')

# 筛选和排序模板
Temp_name_list = [
    'template_0423_80Hz', 'template_0425_80Hz', 'template_10_80Hz',
    'template_8_80Hz', 'template_20_80Hz',
]
check_lib = Temp_lib_df[Temp_lib_df['infile_name'].isin(Temp_name_list)]
check_lib = check_lib.sort_values(by='plot_order', ascending=True).reset_index(drop=True)

# 加载主数据文件
Event_classified_fname = os.path.join(IN_MagRegression_dir, 'Events_classified_Real_Mag.csv')
Events_classified = pd.read_csv(Event_classified_fname, sep='\t')
# 将log10振幅转换为线性振幅，以匹配新的坐标轴标签要求
# Quasi_Mag = log10(Scaled Peak Amp) => Scaled Peak Amp = 10**Quasi_Mag
# Quasi_Mag_Amp = log10(Peak DAS Amp) => Peak DAS Amp = 10**Quasi_Mag_Amp
Events_classified['Scaled_Peak_Amp'] = 10**Events_classified['Quasi_Mag']
Events_classified['Peak_Amp'] = 10**Events_classified['Quasi_Mag_Amp']

print("数据加载完成。")

# --- 4. 预计算统一的坐标轴范围 ---
all_noa_dfs = []
for template_name in check_lib['infile_name']:
    noa_mag_file = os.path.join(IN_Events_with_NOA_Mag_dir, f"{template_name}_Events_with_NOA_Mag.csv")
    if os.path.exists(noa_mag_file):
        df = pd.read_csv(noa_mag_file, sep='\t')
        df['Scaled_Peak_Amp'] = 10**df['Quasi_Mag']
        all_noa_dfs.append(df)

# 左列坐标范围
if all_noa_dfs:
    full_noa_df = pd.concat(all_noa_dfs)
    left_xlim = (full_noa_df['Scaled_Peak_Amp'].min() * 0.8, full_noa_df['Scaled_Peak_Amp'].max() * 1.2)
    left_ylim = (full_noa_df['EQ_magnitude'].min() - 0.2, full_noa_df['EQ_magnitude'].max() + 0.2)
else:
    left_xlim = (1e-9, 1e-4); left_ylim = (0, 3) # Fallback

# 右列坐标范围
right_xlim = (Events_classified['Scaled_Peak_Amp'].min() * 0.8, Events_classified['Scaled_Peak_Amp'].max() * 1.2)
right_ylim = (Events_classified['Peak_Amp'].min() * 0.8, Events_classified['Peak_Amp'].max() * 1.2)

print(f"左列统一坐标范围: xlim={left_xlim}, ylim={left_ylim}")
print(f"右列统一坐标范围: xlim={right_xlim}, ylim={right_ylim}")


# --- 5. 创建组合图 ---

# 创建一个更扁平的 5x2 子图网格 (figsize的宽高比决定了图像形状)
fig, axes = plt.subplots(len(check_lib), 2, figsize=(15, 16), 
                         gridspec_kw={'width_ratios': [1.1, 1], 'height_ratios': [1]*len(check_lib)})
fig.subplots_adjust(hspace=0.6, wspace=3.5) #   

# 定义子图序号
row_labels = ['a', 'b', 'c', 'd', 'e']

# 循环遍历每个模板
for ii, row in check_lib.iterrows():
    Template = row['infile_name']
    disp_Template = row['plot_name']
    
    ax_left = axes[ii, 0]
    ax_right = axes[ii, 1]

    # --- (A) 添加子图序号和模板名称标注 ---
    # 左图序号 (调整了X位置以避免与Y轴标签太近)
    ax_left.text(-0.05, 1.1, f'{row_labels[ii]}1)', transform=ax_left.transAxes,
                 fontsize=SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom', ha='left')
    # 右图序号
    ax_right.text(-0.05, 1.1, f'{row_labels[ii]}2)', transform=ax_right.transAxes,
                  fontsize=SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom', ha='left')
    # 模板名称标注 (在左图的左侧)
    ax_left.text(-0.15, 0.5, disp_Template.replace('Temp_', 'Template '), transform=ax_left.transAxes,
                 fontsize=ANNOTATION_FONTSIZE-2, fontweight='bold', va='center', ha='center', rotation=90)


    # --- (B) 左侧列: 震级标定 ---
    noa_mag_file = os.path.join(IN_Events_with_NOA_Mag_dir, f"{Template}_Events_with_NOA_Mag.csv")
    if os.path.exists(noa_mag_file):
        df_NOAmag = pd.read_csv(noa_mag_file, sep='\t')
        df_NOAmag['Scaled_Peak_Amp'] = 10**df_NOAmag['Quasi_Mag']
        
        if len(df_NOAmag) >= 3:
            # 注意：线性回归在对数空间进行
            x_log, y = df_NOAmag['Quasi_Mag'], df_NOAmag['EQ_magnitude']
            slope, intercept = np.polyfit(x_log, y, 1)
            
            # ******** MODIFICATION START ********
            # 1. 计算 R²
            y_fit_at_points = slope * x_log + intercept
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_fit_at_points)**2)
            R2 = 1 - (ss_residual / ss_total)

            # 绘制数据点
            ax_left.scatter(df_NOAmag['Scaled_Peak_Amp'], y, color='blue', s=45, ec='k', lw=0.5, label='Catalog Events')
            
            # 绘制拟合线
            x_fit_log = np.linspace(np.log10(left_xlim[0]), np.log10(left_xlim[1]), 100)
            y_fit = slope * x_fit_log + intercept
            ax_left.plot(10**x_fit_log, y_fit, color='red', linestyle='--', label=f'Fit Line: y = {slope:.2f}x + {intercept:.2f}')
            ax_left.set_xscale('log') # X轴使用对数尺度
            
            # 2. 在左上角添加回归结果标注
            text_str = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\n$R^2$: {R2:.3f}'
            ax_left.text(0.05, 0.95, text_str, transform=ax_left.transAxes, fontsize=TEXT_FONTSIZE,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

            # 3. 将图例放置在右下角
            ax_left.legend(fontsize=LEGEND_FONTSIZE, loc='upper center', frameon=False)
            # ******** MODIFICATION END ********

    # 设置左图的标签、范围和网格
    ax_left.set_ylabel('ML', fontsize=LABEL_FONTSIZE)
    ax_left.set_xlabel('Scaled Peak Amplitude [(m/m)/s]', fontsize=LABEL_FONTSIZE)
    ax_left.set_xlim(left_xlim)
    ax_left.set_ylim(left_ylim)
    ax_left.set_yticks([1,2,3,4]) # 设置X轴刻度为整数
    ax_left.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.3)
    ax_left.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)


    # --- (C) 右侧列: 振幅一致性 ---
    template_events = Events_classified[Events_classified['Template'] == Template]
    if not template_events.empty:
        # 绘制散点图 (KDE密度着色)
        from scipy.stats import gaussian_kde
        # 使用对数转换后的数据进行KDE计算，因为数据在对数空间更均匀
        x = np.log10(template_events['Scaled_Peak_Amp'].values)
        y = np.log10(template_events['Peak_Amp'].values)

        xy = np.vstack([x, y])
        # 增加 bw_method 因子以获得更平滑的密度估计
        z = gaussian_kde(xy, bw_method=0.2)(xy)

        # 绘图时使用原始线性数据
        ax_right.scatter(template_events['Scaled_Peak_Amp'], template_events['Peak_Amp'], 
                         c=z, s=25, cmap='viridis', edgecolors='k', linewidth=0.2, alpha=0.3)
        
        #* 添加噪声幅度参考线
        # amp_noise = 1e-9
        # ax_right.plot(y=amp_noise, 'k--', alpha=0.7, linewidth=1.5, label='DAS noise amplitude')
        # ax_right.legend(fontsize=LEGEND_FONTSIZE)


    # 设置右图的标签、范围和网格
    ax_right.set_ylabel('Peak Amp. [(m/m)/s]', fontsize=LABEL_FONTSIZE)
    ax_right.set_xlabel('Scaled Peak Amplitude [(m/m)/s]', fontsize=LABEL_FONTSIZE)
    ax_right.set_xscale('log') # X轴和Y轴都使用对数尺度
    ax_right.set_yscale('log')
    ax_right.set_xlim(right_xlim)
    ax_right.set_ylim(right_ylim)
    ax_right.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.3)
    ax_right.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)


# --- (D) 添加列标题 ---
axes[0, 0].set_title('Regression from Scaled Peak Amplitude to ML', fontsize=COLUMN_TITLE_FONTSIZE, pad=45)
axes[0, 1].set_title('Peak Amplitude v.s. Scaled Peak Amplitude', fontsize=COLUMN_TITLE_FONTSIZE, pad=45)

# --- (E) 调整与保存 ---
# 调整整体布局 (调整了rect的left值以适应新的模板名称标注位置)
plt.tight_layout(rect=[0.08, 0.02, 1, 0.95])

# 保存图像
output_filename = os.path.join(OUT_PLOT_dir, "Fig_Magnitude_Regression_for_Paper_v2.png")
plt.savefig(output_filename, dpi=300) # bbox_inches='tight'有时会与tight_layout冲突,可先移除
print(f"\n图像已成功保存至: {output_filename}")

plt.show()

# - 结束