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
# =============================================================================
# demo.py  ——  一键冒烟测试 / 快速演示：几秒钟看到一个真实的模板匹配检测
# =============================================================================
#
#   运行：   python demo.py
#
# 它做什么：
#   - 完全使用本发布版【已附带】的中间产物（2024-04-25 的叠加互相关 SCCs + MAD 阈值），
#     不需要连续 DAS 大数据、不需要联网、不需要重算。
#   - 取模板 template_0423（Temp_K2）在 2024-04-25 11:00–11:10 的 SCCs 序列：
#     这一窗里正好有该模板自身那次地震（11:02:14 UTC），所以匹配峰非常清晰。
#   - 用「阈值 = level × MAD」（level=20，与 Output_2/level_20 一致）做触发，
#     打印检测结果，并把检测图保存为 demo_detection.png。
#
# 这就是整套「模板匹配检测」最核心的一步的最小可视化：
#   连续数据里凡是波形长得像模板的时刻，叠加互相关 CCs 就会冒出一个尖峰，
#   超过 level×MAD 阈值就记为一次检测。
# =============================================================================

import os
import glob
import pickle
import datetime as dt

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")          # 无显示环境也能出图
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------- 可调参数 ------------------------------------
TEMPLATE = "template_0423_80Hz"     # 用哪个模板（Temp_K2）
DATE     = "20240425"               # 哪天（发布版自带这一天的 SCCs）
WINDOW   = "110000"                 # 哪个 10 分钟窗起点 HHMMSS（含 11:02 的真实地震）
LEVEL    = 20                       # 触发阈值 = LEVEL × MAD（与 level_20 一致）
FS       = 80                       # 采样率 Hz
# -----------------------------------------------------------------------------

print("=" * 70)
print(" yuhan_dasmf 一键演示：一次真实的模板匹配检测")
print("=" * 70)

# 1) 找到 SCCs h5（叠加互相关序列）-------------------------------------------
scc_glob = (f"{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/{TEMPLATE}/"
            f"*{DATE}_{WINDOW}*.h5")
scc_files = glob.glob(scc_glob)
if not scc_files:
    raise SystemExit(f"找不到 SCCs 文件：{scc_glob}\n（请确认发布版自带了 {DATE} 的 SCCs_h5）")
scc_path = scc_files[0]
with h5py.File(scc_path, "r") as h:
    ccs = h["CCs_s"][:]            # (48001,) 叠加互相关序列
print(f"[1] 读入 SCCs 序列: {os.path.basename(scc_path)}  ({ccs.size} 点 = 10 分钟 @ {FS}Hz)")

# 2) 读 MAD 阈值基准 ----------------------------------------------------------
mad_files = glob.glob(f"{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/*/12:00:00/MAD_Average_dict.pkl")
if not mad_files:
    raise SystemExit("找不到 MAD_Average_dict.pkl")
with open(mad_files[0], "rb") as f:
    mad_dict = pickle.load(f)
mad = float(np.asarray(mad_dict[TEMPLATE]).ravel()[0])
thr = LEVEL * mad
print(f"[2] MAD({TEMPLATE}) = {mad:.3e}   →   触发阈值 = {LEVEL} × MAD = {thr:.3e}")

# 3) 触发：|CCs| 超过 level×MAD 的局部峰 = 检测 ------------------------------
ratio = ccs / mad                                  # 检测统计量（多少倍 MAD）
peaks, props = find_peaks(np.abs(ratio), height=LEVEL, distance=FS * 2)  # 至少隔 2s
t0 = dt.datetime.strptime(f"{DATE}_{WINDOW}", "%Y%m%d_%H%M%S")
peak_times = [t0 + dt.timedelta(seconds=float(i) / FS) for i in peaks]

print(f"[3] 在这 10 分钟里，共有 {len(peaks)} 个时刻的匹配度超过 {LEVEL}×MAD 阈值：")
order = np.argsort(props["peak_heights"])[::-1]
for rank, k in enumerate(order[:5], 1):
    i = peaks[k]
    print(f"      #{rank}  {peak_times[k].strftime('%H:%M:%S.%f')[:-3]} UTC   "
          f"CCs/MAD = {ratio[i]:+.1f}   (CCs = {ccs[i]:+.3e})")
if len(peaks) > 5:
    print(f"      ... 其余 {len(peaks) - 5} 个略")

ibest = peaks[order[0]]
print(f"\n  ✅ 最强检测：{peak_times[order[0]].strftime('%Y-%m-%d %H:%M:%S')} UTC，"
      f"匹配度 {ratio[ibest]:+.0f}×MAD")
print(f"     （这正是 {TEMPLATE} 模板自身那次地震 ~11:02:14，所以匹配近乎满分）")

# 4) 出图 ---------------------------------------------------------------------
t_axis = np.arange(ccs.size) / FS                  # 窗内秒数
# 注：图内文字用英文，避免 matplotlib 默认字体缺中文字形（控制台输出仍是中文）。
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_axis, ratio, lw=0.6, color="0.3", label="CCs / MAD  (detection statistic)")
ax.axhline(LEVEL, color="r", ls="--", lw=1, label=f"trigger threshold  +/-{LEVEL} x MAD")
ax.axhline(-LEVEL, color="r", ls="--", lw=1)
ax.plot(t_axis[peaks], ratio[peaks], "v", color="C1", ms=7, label=f"detections ({len(peaks)})")
ax.annotate("template self-match", xy=(t_axis[ibest], ratio[ibest]),
            xytext=(t_axis[ibest] + 40, ratio[ibest]),
            arrowprops=dict(arrowstyle="->"), va="center", fontsize=9)
ax.set_xlabel(f"time in window (s),  start {t0:%Y-%m-%d %H:%M:%S} UTC")
ax.set_ylabel("CCs / MAD")
ax.set_title(f"Matched-filter detection demo  —  {TEMPLATE} @ {DATE} {WINDOW[:2]}:{WINDOW[2:4]}  (level={LEVEL})")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, t_axis[-1])
fig.tight_layout()
out_png = f"{PROJECT_ROOT}/demo_detection.png"
fig.savefig(out_png, dpi=130)
print(f"\n[4] 检测图已保存：{out_png}")
print("=" * 70)
print(" 完成。这一步对应 Run_2（触发）；要看完整链路见 说明文档_总览.md。")
print("=" * 70)
