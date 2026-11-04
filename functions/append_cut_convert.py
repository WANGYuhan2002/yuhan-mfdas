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

#
# +
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
#-- To import a function on a relative path:
import sys
sys.path.append(f"{PROJECT_ROOT}/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters, block_cleaning
import load_das_h5_PE2024 as load_das_h5_PE2024

import os
import gc
from IPython.display import clear_output
import pickle
import re
import glob

import h5py, json
from tqdm import tqdm
import time
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count

# -


def Convert_list_to_ndarray_in_OUT_dict(OUT_dict):
    """
    Check the list element in OUT_dict, and neccessarily convert to numpy array
    """
    for key in OUT_dict.keys():
        if type(OUT_dict[key]) == list:
            OUT_dict[key] = np.array(OUT_dict[key])
    return OUT_dict
    
        
def Joint_dict_elem(dict1, dict2, dictType):
    """
    No MAD value in the dict take part in the joint
    """
    #* dictType = 'full_chan_info' or 'summed_chan_info'
    if dict1 == {}:
        return dict2
    elif dict2 == {}:
        return dict1
    elif dict1.keys() == dict2.keys():
        if dictType == 'full_chan_info':
            dict1_joint = {}
            dict1_joint['t_trig_s'] = np.concatenate((dict1['t_trig_s'], dict2['t_trig_s']))
            dict1_joint['CCs_trig_s'] = np.concatenate((dict1['CCs_trig_s'], dict2['CCs_trig_s']))
            dict1_joint['RAWCCs_trig_s'] = np.concatenate((dict1['RAWCCs_trig_s'], dict2['RAWCCs_trig_s']))
            dict1_joint['t_CCsmax_f'] = np.concatenate((dict1['t_CCsmax_f'], dict2['t_CCsmax_f']), axis=0)
            dict1_joint['CCsmax_f'] = np.concatenate((dict1['CCsmax_f'], dict2['CCsmax_f']), axis=0)
            dict1_joint['RAWCCsmax_f'] = np.concatenate((dict1['RAWCCsmax_f'], dict2['RAWCCsmax_f']), axis=0)
            dict1_joint['t_datamax_f'] = np.concatenate((dict1['t_datamax_f'], dict2['t_datamax_f']), axis=0)
            dict1_joint['datamax_f'] = np.concatenate((dict1['datamax_f'], dict2['datamax_f']), axis=0)
        
        elif dictType == 'full_chan_info_removed':
            dict1_joint = {}
            dict1_joint['t_trig_s'] = np.concatenate((dict1['t_trig_s'], dict2['t_trig_s']))
            dict1_joint['CCs_trig_s'] = np.concatenate((dict1['CCs_trig_s'], dict2['CCs_trig_s']))
            dict1_joint['RAWCCs_trig_s'] = np.concatenate((dict1['RAWCCs_trig_s'], dict2['RAWCCs_trig_s']))
            dict1_joint['quasi_Mag_Amp'] = np.concatenate((dict1['quasi_Mag_Amp'], dict2['quasi_Mag_Amp']))
            dict1_joint['quasi_Mag_CCs_s'] = np.concatenate((dict1['quasi_Mag_CCs_s'], dict2['quasi_Mag_CCs_s']))
            dict1_joint['quasi_Mag_CCs_f'] = np.concatenate((dict1['quasi_Mag_CCs_f'], dict2['quasi_Mag_CCs_f']))
            dict1_joint['Bias_level_arrivals'] = np.concatenate((dict1['Bias_level_arrivals'], dict2['Bias_level_arrivals']))
            dict1_joint['Template'] = np.concatenate((dict1['Template'], dict2['Template']))
            
    else:
        print('WARNING: dict1 and dict2 have different keys')
        print('dict1.keys():', dict1.keys())
        print('dict2.keys():', dict2.keys())
        return {}
    return dict1_joint

def Cut_time_window_OUT_dict(OUT_dict, t_start, t_end, dictType):
    """
    No MAD value in the dict take part in the cut
    """
    if OUT_dict == {}:
        print('(2) WARNING: The input dict is empty')
        return {}
    if len(OUT_dict['t_trig_s']) == 0:
        print('(2) No event is found')
        return {}
    
    idx_win = np.where((OUT_dict['t_trig_s'] >= t_start) & (OUT_dict['t_trig_s'] < t_end))
    if type(idx_win) == tuple:
        idx_win = idx_win[0]
    elif type(idx_win) == np.ndarray:
        pass        
    if dictType == 'full_chan_info':
        OUT_dict_cut = {}
        OUT_dict_cut['t_trig_s'] = OUT_dict['t_trig_s'][idx_win]
        OUT_dict_cut['CCs_trig_s'] = OUT_dict['CCs_trig_s'][idx_win]
        OUT_dict_cut['RAWCCs_trig_s'] = OUT_dict['RAWCCs_trig_s'][idx_win]
        OUT_dict_cut['t_CCsmax_f'] = OUT_dict['t_CCsmax_f'][idx_win]
        OUT_dict_cut['CCsmax_f'] = OUT_dict['CCsmax_f'][idx_win]
        OUT_dict_cut['RAWCCsmax_f'] = OUT_dict['RAWCCsmax_f'][idx_win]
        OUT_dict_cut['t_datamax_f'] = OUT_dict['t_datamax_f'][idx_win]
        OUT_dict_cut['datamax_f'] = OUT_dict['datamax_f'][idx_win]
    elif dictType == 'full_chan_info_removed':
        OUT_dict_cut = {}
        OUT_dict_cut['t_trig_s'] = OUT_dict['t_trig_s'][idx_win]
        OUT_dict_cut['CCs_trig_s'] = OUT_dict['CCs_trig_s'][idx_win]
        OUT_dict_cut['RAWCCs_trig_s'] = OUT_dict['RAWCCs_trig_s'][idx_win]
        OUT_dict_cut['quasi_Mag_Amp'] = OUT_dict['quasi_Mag_Amp'][idx_win]
        OUT_dict_cut['quasi_Mag_CCs_s'] = OUT_dict['quasi_Mag_CCs_s'][idx_win]
        OUT_dict_cut['quasi_Mag_CCs_f'] = OUT_dict['quasi_Mag_CCs_f'][idx_win]  
        OUT_dict_cut['Bias_level_arrivals'] = OUT_dict['Bias_level_arrivals'][idx_win]
        OUT_dict_cut['Template'] = OUT_dict['Template'][idx_win]
    
    return OUT_dict_cut

def Add_temp_name_to_OUT_dict(OUT_dict, Template):
    """
    """
    if OUT_dict == {}:
        print('(3) WARNING: The input dict is empty')
        return {}
    if len(OUT_dict['t_trig_s']) == 0:  
        print('(3) No event is found')
        return {}
    
    OUT_dict['Template'] = np.array(list(np.repeat(Template, len(OUT_dict['t_trig_s']))), dtype='str')
    return OUT_dict

def Remove_template_time_offset(OUT_dict, Template, Temp_offset_dict, offset_Type = 'S'):
    """
    """
    if OUT_dict == {}:
        print('(5) WARNING: The input dict is empty')
        return {}
    if len(OUT_dict['t_trig_s']) == 0:
        print('(5) No event is found')
        return {}
    
    if offset_Type == 'S':
        offset = Temp_offset_dict['S_offset'][Temp_offset_dict['template_name'] == Template]
    elif offset_Type == 'P':
        offset = Temp_offset_dict['P_offset'][Temp_offset_dict['template_name'] == Template]
    else:
        print('WARNING: offset_Type should be S or P')
        return {}
    
    if len(offset) == 0:
        print('WARNING: No offset value for the template')
        return {}
    elif len(offset) > 1:
        print('WARNING: More than one offset value for the template')
        return {}
    elif len(offset) == 1:
        offset = offset[0]
        
    OUT_dict['t_trig_s'] = np.array([t + timedelta(seconds=offset) for t in OUT_dict['t_trig_s']])
    return OUT_dict

    """
    Read the CCs_trig files and classify the events by the CCsmax and datamax.
    """    
    
    OUT_dict_fromAllTemplates = dict()
    for Template in Temp_name_list:
        OUT_dict = Read_IniTrig(t_start, t_end, Template, IN_initial_trig_dir)
        # get the key Template since Read_IniTrig() 
        OUT_dict = Retrig(OUT_dict, Retrig_threshold, Template)
        
        if OUT_dict == {} or len(OUT_dict['t_trig_s']) == 0:
            continue
        OUT_dict = Est_Mag_Bias(OUT_dict)
        OUT_dict = Remove_template_time_offset(OUT_dict, Template, Temp_offset_dict, offset_Type = 'S')

        OUT_dict_fromAllTemplates = Joint_dict_elem(OUT_dict_fromAllTemplates, OUT_dict, 'full_chan_info_removed')
        del OUT_dict
    
    if OUT_dict_fromAllTemplates == {}:
        print('No event is found')
        return {}

    OUT_dict_Classified = Classification_by_shift(OUT_dict_fromAllTemplates, In_one_event_interval)
    # OUT_dict_Classified = Classification_by_CCs(OUT_dict_fromAllTemplates, In_one_event_interval)
    # OUT_dict_Classified_cleaned = Exclude_Outliers_forAllTemp(OUT_dict_Classified, Temp_name_list)
    OUT_dict_Classified_cleaned = OUT_dict_Classified
    
    count = len(OUT_dict_Classified_cleaned['t_trig_s'])
    print(f'>> # {count} found for {t_start}~{t_end}')
    return OUT_dict_Classified_cleaned

    # 移除原有的性能分析代码，直接调用main()
    main()