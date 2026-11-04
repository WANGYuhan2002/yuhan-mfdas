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

import os
import numpy as np
import matplotlib.pyplot as plt

# 建议将此文件保存为 spectrum_analyzer.py


def Calculate_fk_amplitude_spectrum(signal_2d, dt, dx):
    """
    计算二维实数信号的单边频率 f-k 振幅谱。

    该函数经过优化，增加了输入检查，确保了代码的健壮性，适合广泛调用。

    Args:
        signal_2d (np.ndarray): 二维实数信号数组 (时间, 空间)。
                                 第一个维度是时间，第二个是空间。
        dt (float): 时间采样间隔 (秒)。必须为正数。
        dx (float): 空间采样间隔 (米)。必须为正数。

    Returns:
        tuple: 一个包含以下元素的元组:
            - freqs (np.ndarray): 一维非负频率数组 (f >= 0) (Hz)。
            - wavenums (np.ndarray): 一维波数数组 (1/m)，经过fftshift处理以使零点居中。
            - ampspec (np.ndarray): 二维物理振幅谱数组。
            
    Raises:
        ValueError: 如果输入信号为复数，或者dt/dx非正数。
    """
    # --- 新增：输入参数健壮性检查 ---
    if np.iscomplexobj(signal_2d):
        raise ValueError("输入信号 `signal_2d` 必须是实数数组。")
    if dt <= 0 or dx <= 0:
        raise ValueError("时间间隔 `dt` 和空间间隔 `dx` 必须是正数。")

    # 获取维度信息
    Nt, Nx = signal_2d.shape

    # 执行二维傅里叶变换
    fk_transform = np.fft.fft2(signal_2d)

    # 为了获得物理振幅，需要进行归一化。
    # 对于实数信号，一个振幅为A的余弦波分量，其能量会分布在正负频率上，
    # FFT峰值的模是 A * N / 2。为了从单边频谱中恢复A，我们需要乘以 2/N。
    # 这里的N是总采样点数 Nt * Nx。
    ampspec_full = np.abs(fk_transform) / (Nt * Nx)

    # 使用 fftshift 将波数轴的零点移动到中心
    ampspec_shifted_k = np.fft.fftshift(ampspec_full, axes=1)

    # --- 创建频率和波数坐标轴 ---
    wavenums = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    freqs_full = np.fft.fftfreq(Nt, dt)
    
    # 我们只需要 f >= 0 的单边频谱。
    # 对于长度为 Nt 的FFT结果，前 Nt//2 + 1 个点对应从0到奈奎斯特频率。
    num_positive_freqs = Nt // 2 + 1
    freqs = freqs_full[:num_positive_freqs]
    
    # 提取单边频谱 (对应 f >= 0)
    # 新增：使用 .copy() 确保我们得到一个副本，而不是视图，这是更安全的做法。
    ampspec_single_sided = ampspec_shifted_k[:num_positive_freqs, :].copy()
    
    # 将除了直流(f=0)和奈奎斯特频率之外的所有频率分量的振幅乘以2
    # 注意：奈奎斯特频率仅在Nt为偶数时存在
    if Nt % 2 == 0:
        # 对 (0, Nyquist) 范围内的频率乘以2
        ampspec_single_sided[1:-1, :] *= 2.0
    else:
        # 对 (0, Nyquist) 范围内的频率乘以2
        ampspec_single_sided[1:, :] *= 2.0

    return freqs, wavenums, ampspec_single_sided


# --- Plotting helper functions ---
def plot_fk_spectrum(freqs, wavenums, ampspec, title_suffix, A_max_for_plot, expected_peaks_plot, OUTPUT_dir, filename_suffix):
    """Helper function to plot f-k spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax_val = A_max_for_plot * 1.1 if A_max_for_plot > 0 else 0.1
    if A_max_for_plot == 0 and np.any(ampspec > 0): # If A_max is 0 but spectrum has energy
        vmax_val = np.max(ampspec) * 1.1
    if vmax_val == 0 : vmax_val = 0.1 # Ensure vmax is not zero if spectrum is all zero

    im = ax.pcolormesh(wavenums, freqs, ampspec, shading='gouraud', cmap='viridis', 
                       vmin=0, vmax=vmax_val)
    
    plt.colorbar(im,label='Physical Amplitude (Same units as input signal)')
    ax.set_xlabel('Wavenumber (1/m)')
    ax.set_ylabel('Frequency (Hz) (f >= 0)')
    ax.set_ylim(0, max(freqs))  # Ensure y-axis starts from 0
    ax.set_title(f'2D Amplitude Spectrum (Single-sided Freq. f-k Domain) - {title_suffix}')

    if expected_peaks_plot: 
        for k_coords, f_coords, label_text in expected_peaks_plot:
            if k_coords and f_coords: 
                 ax.scatter(k_coords, f_coords, color='red', marker='x', s=100, label=label_text)
        if any(p[0] for p in expected_peaks_plot): # Only show legend if there are labels
            ax.legend()

    ax.axhline(0, color='white', linestyle=':', linewidth=0.5) 
    ax.axvline(0, color='white', linestyle=':', linewidth=0.5) 
    fig.savefig(os.path.join(OUTPUT_dir, f'2d_fk_spectrum_{filename_suffix}.png'), dpi=300)
    plt.close(fig)