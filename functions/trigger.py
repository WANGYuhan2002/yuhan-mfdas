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
import numpy as np
from datetime import datetime, timedelta

def Trigger_with_MADarray(t_start, t_end, fs, CCs_s, t_array, MAD_array, Threshold_level, Minimal_event_interval):
    """
    Args:
        t_start, t_end: datetime
        fs: float
        CCs_s: 1D np.array from t_start to t_end
        t_array: 1D np.array, datetime64. len>=2
        MAD_array: 1D np.array, len = len(t_array)-1
        Threshold_level: float
        Minimal_event_interval: float, seconds
    """
    # step 1: find t_start and t_end between which two t_array elements
    # first check [t_start,t_end] is within the t_array
    if t_start < t_array[0] or t_end > t_array[-1]:
        print(f'Error: t_start={t_start} or t_end={t_end} is out of the t_array range')
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    t_start = np.datetime64(t_start)
    t_end = np.datetime64(t_end)
    idx_start = np.where(t_array <= t_start)[0][-1]
    idx_end = np.where(t_array >= t_end)[0][0]
    
    # step 2: trigger the CCs_s by segment
    # <t_array>------[idx_start--------------idx_start+1-----.........----- idx_end-1--------------idx_end]----- 
    # ----------------------------[t_start---------|----------------------------|-----------t_end]----------------
    # ----------------------------|------<seg1>----|----------<seg2>------------|---<seg3>-----|-----------
    
    # satisfying: t_start >= t_array[idx_start] AND t_end <= t_array[idx_end]
    if idx_start+1 > idx_end-1:
        t_grid  = [t_start, t_end]
    else:
        t_grid = [t_start] + list(t_array[idx_start+1:idx_end]) + [t_end] # idx_end not included, actually idx_start+1 to idx_end-1
    t_grid = np.array(t_grid)
    MAD_grid = MAD_array[idx_start:idx_end] # MAD_array[i] cover for t_array[i] to t_array[i+1]
    #idx_end is end point not included, idx_start ~ idx_end-1 is the range of MAD_array, covers t_array[idx_start] to t_array[idx_end] overall
    # print(f't_grid: {t_grid}, MAD_grid: {MAD_grid}')
    
    CCs_s_trig = np.array([])
    t_trig = np.array([])
    t_left = np.array([])
    t_right = np.array([])
    width_trig = np.array([])
    
    for ii in range(len(t_grid)-1):
        t0 = t_grid[ii].astype(datetime)
        t1 = t_grid[ii+1].astype(datetime)
        Threshold = MAD_grid[ii] * Threshold_level
        
        idx0_CCs = int((t0 - t_start.astype(datetime)).total_seconds() * fs)
        idx1_CCs = int((t1 - t_start.astype(datetime)).total_seconds() * fs)
        
        tmp_CCs, tmp_t, tmp_t_left, tmp_t_right, tmp_width = Trigger_with_Constant(t0, fs, CCs_s[idx0_CCs:idx1_CCs], Threshold, Minimal_event_interval)
        # tmp_CCs, tmp_t are np.array (use extend), not elements (use append);
        # tmp_CCs, tmp_t can be empty list []
        CCs_s_trig = np.concatenate((CCs_s_trig, tmp_CCs))
        t_trig = np.concatenate((t_trig, tmp_t))
        t_left = np.concatenate((t_left, tmp_t_left))
        t_right = np.concatenate((t_right, tmp_t_right))
        width_trig = np.concatenate((width_trig, tmp_width))
    return CCs_s_trig, t_trig, t_left, t_right, width_trig
    
def Trigger_with_Constant(t_start, fs, CCs_s, Threshold, Minimal_event_interval):
    """
    Args:
        t_start, t_end: datetime
        fs: float
        CCs_s: 1D array from t_start to t_end
        Threshold: float
        Minimal_event_interval: float, seconds
        
    Returns:
        CCs_s_trig: 1D np.array, triggered CCs_s points can be empty list []
        t_trig: 1D np.array, datetime, can be []
        
    """
    # step 1: trigger the CCs_s by segment
    #* find out the index of the peaks
    CCs_s_trig_01 = np.zeros(CCs_s.shape)
    CCs_s_trig_01[abs(CCs_s) > Threshold] = 1
    # print('CCs_s_trig_01 shape:', CCs_s_trig_01.shape)
    index_1 = np.where(CCs_s_trig_01 == 1)
    
    if index_1[0].size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # print('all index: ',(index_1))
    #*-- cluster the events with minimal_event_interval
    #--- index_2 saves the index of the first peak in each cluster 
    index_2 = [] 
    index_2.append(index_1[0][0])
    for i in range(1, len(index_1[0])):
        if abs(index_1[0][i] - index_1[0][i-1]) > int(Minimal_event_interval*fs):
            index_2.append(index_1[0][i])
    # print('start index',index_2)
    
    #--- index_3 saves the index of the last peak in each cluster
    index_3 = []
    index_3.append(index_1[0][-1])
    for i in range(len(index_1[0])-2, -1, -1):
        if abs(index_1[0][i+1] - index_1[0][i]) > int(Minimal_event_interval*fs):
            index_3.append(index_1[0][i])
    index_3 = index_3[::-1]
    # print('end index' ,index_3)
    
    #--- pick the maximum in each cluster
    index_arg_max = []
    index_2 = np.array(index_2)
    index_3 = np.array(index_3)
    
    print('index_2:', len(index_2), 'index_3:', len(index_3))
    if len(index_2) != len(index_3):
        print('Error: index_2 and index_3 have different length')
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    for i in range(len(index_2)):
        if index_2[i] == index_3[i]:
            index_arg_max.append(index_2[i])
        else:
            index_arg_max.append(index_2[i] + np.argmax(abs(CCs_s[index_2[i]:index_3[i]])))
            # NOTE: np.argmax returns the first occurrence of the maximum value
    # start_time = [t_datetime[i] for i in index_2]
    # end_time = [t_datetime[i] for i in index_3]
    # max_time = [t_datetime[i] for i in index_arg_max]
    # max_value = [corr_stack_trig[i] for i in index_arg_max]
    Nt_trig_s = index_arg_max
    Nt_trig_left = index_2
    Nt_trig_right = index_3
        
    CCs_s_trig = np.array([CCs_s[i] for i in Nt_trig_s])
    t_trig = np.array([t_start + timedelta(seconds=i/fs) for i in Nt_trig_s])
    t_left = np.array([t_start + timedelta(seconds=i/fs) for i in Nt_trig_left])
    t_right = np.array([t_start + timedelta(seconds=i/fs) for i in Nt_trig_right])
    width_trig = np.array([timedelta(seconds=(Nt_trig_right[i] - Nt_trig_left[i])/fs) for i in range(len(Nt_trig_right))])
    return CCs_s_trig, t_trig, t_left, t_right, width_trig
