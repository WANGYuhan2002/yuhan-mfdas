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
# +
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
# import scipy.signal as signal # Not used in the provided parallelizable section
# from scipy.ndimage import uniform_filter1d # Not used
# import pandas as pd # Not used in the parallelizable section, but kept for later plotting
import pandas as pd # Kept for the plotting part later if uncommented by user

import sys
import os
# import gc # Not explicitly used, can be removed if not needed elsewhere
# from IPython.display import clear_output # Not used in script logic
# import pickle # Used by write_pkl/load_pkl

import h5py, json # h5py likely used by Read_CCs_stack_h5

# -- To import a function on a relative path:
sys.path.append(f"{PROJECT_ROOT}/functions/") # Ensure this path is correct
from readers import Read_CCs_stack_h5
from readers import write_pkl, load_pkl
from plot import Ticks_xaxis_day_or_hour
# from pydas_readers.readers import load_das_h5, write_das_h5 # Not used in this script part
# from pydas_readers.util import block_filters, block_cleaning # Not used
# import load_das_h5_PE2024 as load_das_h5_PE2024 # Not used

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

#!_________newly defined functions_________
def mad(data):
    """
    only for 1D array
    """
    if data is None or len(data) == 0:
        return np.nan # Handle empty or None data
    return np.median(np.abs(data - np.median(data)))
#!______________________________________

# Worker function for parallel processing
def calculate_mad_for_chunk(args):
    """
    Calculates MAD for a single template and time chunk.
    Designed to be called by ProcessPoolExecutor.map.
    """
    template_name, t_start_chunk, t_end_chunk, ccs_dir = args
    # It's critical that Read_CCs_stack_h5 is process-safe.
    # Typically, reading different parts of files or different files is fine.
    # If Read_CCs_stack_h5 involves complex state or writes, it might need adjustment.
    
    CCs_s, _ = Read_CCs_stack_h5(ccs_dir, template_name, t_start_chunk, t_end_chunk)
    
    median = np.nanmedian(CCs_s)  # Calculate median ignoring NaNs
    median_abs_dev = np.nanmedian(np.abs(CCs_s - median)) # Calculate absolute deviation from the median
    mean = np.nanmean(CCs_s)  # Calculate mean ignoring NaNs
    mean_abs_dev = np.nanmean(np.abs(CCs_s - mean))  # Calculate absolute deviation from the mean
    peak_CCs = np.nanmax(np.abs(CCs_s)) # Calculate peak CCs ignoring NaNs

    return median, median_abs_dev, mean, mean_abs_dev, peak_CCs

def main_processing_logic(num_cpus=10, nt_delta_arr=[60*4]):
    #!__________________________PARAS_____________________________
    Temp_name_list = [
            'template_0423_80Hz',
            'template_0425_80Hz',
            'template_10_80Hz',
            'template_8_80Hz',
            'template_20_80Hz',
        ]
    # nt delta in minutes, the time window for MAD calculation
    # Reduced loop for quicker testing, original was: [240,300,360,480,600,720]
    for nt_delta in nt_delta_arr:
        t_all_start = datetime.strptime('2024-04-24 10:20:00', '%Y-%m-%d %H:%M:%S')
        t_all_end = datetime.strptime('2024-06-22 17:10:00', '%Y-%m-%d %H:%M:%S')
        t_delta = timedelta(minutes=nt_delta)
        t_array_np = np.arange(t_all_start, t_all_end, t_delta)
        # t_all_end > t_array[-1] must be True
        t_array_np = np.append(t_array_np, t_all_end) # t_array is now a numpy array of datetime objects

        print('t_array segment count:', len(t_array_np)-1, '\n')
        print(f'********* calculate MAD array from {t_all_start} to {t_all_end} with time interval={t_delta} using {num_cpus} CPUs\n')

        IN_CCs_stack_dir = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/'

        t_str = t_all_start.strftime('%Y%m%d_%H%M%S') + '_to_' + t_all_end.strftime('%Y%m%d_%H%M%S') + '/'
        OUTPUT_dir_base = f'{PROJECT_ROOT}/Output_1_SCCs_and_MAD/' + 'MAD_arrays/' + t_str
        # Corrected timedelta string representation for path
        OUTPUT_dir = OUTPUT_dir_base + str(t_delta)+ '/' # e.g. "4_00_00" for 4 hours
        if not os.path.exists(OUTPUT_dir):
            os.makedirs(OUTPUT_dir)
            print(f"Created directory: {OUTPUT_dir}")


    #!________________________Heavy Reading Load here, the real computing part____________________________________
        Median_array_dict = {}
        Median_array_dict['t_array'] = t_array_np # Store the numpy array of datetimes
        
        MAD_array_dict = {}
        MAD_array_dict['t_array'] = t_array_np # Store the numpy array of datetimes

        MEAN_array_dict = {}
        MEAN_array_dict['t_array'] = t_array_np # Store the numpy array of datetimes
        
        MEANAbsDev_array_dict = {}
        MEANAbsDev_array_dict['t_array'] = t_array_np # Store the numpy array of datetimes
        
        PeakCCs_array_dict = {}
        PeakCCs_array_dict['t_array'] = t_array_np # Store the numpy array of datetimes
        
        
        for Template in Temp_name_list:
            print(f'\n=====> Processing Template: {Template}')

            # Prepare arguments for all chunks for the current template
            tasks_args = []
            for ii in range(len(t_array_np)-1):
                t_start_chunk = t_array_np[ii]
                t_end_chunk = t_array_np[ii+1]
                tasks_args.append((Template, t_start_chunk, t_end_chunk, IN_CCs_stack_dir))

            # Initialize tmpMAD_array for this template
            tmpMAD_array = np.full(len(t_array_np)-1, np.nan) # Use NaN for potential errors

            # Parallel execution using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                # Using executor.map to process tasks and collect results in order
                # tqdm provides a progress bar
                results = list(tqdm(executor.map(calculate_mad_for_chunk, tasks_args),
                                    total=len(tasks_args),
                                    desc=f"MAD for {Template} (nt_delta={nt_delta}min)"))

            # Unzip the list of tuples into separate lists
            median, median_abs_dev, mean, mean_abs_dev, PeakCCs = zip(*results)

            tmpMedian_array = np.array(median)  # Convert to numpy array
            tmpMAD_array = np.array(median_abs_dev)  # Convert to numpy array
            tmpMean_array = np.array(mean)  # Convert to numpy array
            tmpMeanAbsDev_array = np.array(mean_abs_dev)  # Convert to numpy array
            tmpPeakCCs_array = np.array(PeakCCs)  # Convert to numpy array

            Median_array_dict[Template] = tmpMedian_array
            MAD_array_dict[Template] = tmpMAD_array
            MEAN_array_dict[Template] = tmpMean_array
            MEANAbsDev_array_dict[Template] = tmpMeanAbsDev_array
            PeakCCs_array_dict[Template] = tmpPeakCCs_array

        #? saving the completed MAD_array_dict
        write_pkl(Median_array_dict, OUTPUT_dir + 'Median_array_dict.pkl')
        write_pkl(MAD_array_dict, OUTPUT_dir + 'MAD_array_dict.pkl')
        write_pkl(MEAN_array_dict, OUTPUT_dir + 'MEAN_array_dict.pkl')
        write_pkl(MEANAbsDev_array_dict, OUTPUT_dir + 'MEANAbsDev_array_dict.pkl')
        write_pkl(PeakCCs_array_dict, OUTPUT_dir + 'PeakCCs_array_dict.pkl')
        print('Saved: ', OUTPUT_dir)


        #? saving the average MAD for each template, t_array take the first and last time
        MAD_Average_dict = {}
        MAD_Average_dict['t_array'] = np.array([t_array_np[0], t_array_np[-1]])
        for Template in Temp_name_list:
            # Calculate mean ignoring NaNs if any chunk failed
            MAD_Average_dict[Template] = np.array([np.nanmean(MAD_array_dict[Template])])
        filename = 'MAD_Average_dict.pkl'
        write_pkl(MAD_Average_dict, OUTPUT_dir + filename)
        print('Saved: ', OUTPUT_dir + filename)

    # #!_________________________________________________________________________________

        # If just want to read the dict to plot
        #* read MAD_array_dict from pkl
        file_path = OUTPUT_dir + 'MAD_array_dict.pkl'
        loaded_MAD_array_dict = load_pkl(file_path) # Use a different variable name to avoid confusion
        loaded_t_array = loaded_MAD_array_dict['t_array']
        print('Loaded: ', file_path)
        loaded_Median_array_dict = load_pkl(OUTPUT_dir + 'Median_array_dict.pkl')
        loaded_MEAN_array_dict = load_pkl(OUTPUT_dir + 'MEAN_array_dict.pkl')
        loaded_MEANAbsDev_array_dict = load_pkl(OUTPUT_dir + 'MEANAbsDev_array_dict.pkl')
        loaded_PEAKCCs_array_dict = load_pkl(OUTPUT_dir + 'PeakCCs_array_dict.pkl')
        

        Day_tick=5
        Hour_tick=0

        #? plot MAD_array for each template in seperate figures
        no = len(Temp_name_list)
        fig, axes = plt.subplots(no, 5, figsize=(10*4, 2.5 * no), sharex=True) # Adjusted height
        if no == 1: # If only one template, axes is not a list
            axes = [axes]

        for ii, Template in enumerate(Temp_name_list):
            ax = axes[ii,0]
            # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_MAD_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')
            avg_MAD = np.nanmean(loaded_MAD_array_dict[Template]) # Use nanmean
            ax.axhline(avg_MAD, color='r', linestyle='--', label=f'Avg MAD: {avg_MAD:.3e}')

            ax.set_title(f'SCCs MAD every {t_delta} for {Template}')
            ax.set_ylabel('MAD of SCCs')
            Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick) # Ensure this function handles datetime objects
            ax.legend()
            
            ax = axes[ii,1]
            # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_Median_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')
            avg = np.nanmean(loaded_Median_array_dict[Template]) # Use nanmean
            ax.axhline(avg, color='r', linestyle='--', label=f'Avg: {avg:.3e}')

            ax.set_title(f'SCCs Median every {t_delta} for {Template}')
            ax.set_ylabel('Median of SCCs')
            Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick) # Ensure this function handles datetime objects
            ax.legend()
            
            
            ax = axes[ii,2]
            # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_MEAN_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')
            avg = np.nanmean(loaded_MEAN_array_dict[Template]) # Use nanmean
            ax.axhline(avg, color='r', linestyle='--', label=f'Avg: {avg:.3e}')

            ax.set_title(f'SCCs Mean every {t_delta} for {Template}')
            ax.set_ylabel('Mean of SCCs')
            Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick) # Ensure this function handles datetime objects
            ax.legend()
            
            ax = axes[ii,3]
            # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_MEANAbsDev_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')
            avg = np.nanmean(loaded_MEANAbsDev_array_dict[Template]) # Use nanmean
            ax.axhline(avg, color='r', linestyle='--', label=f'Avg: {avg:.3e}') 
            ax.set_title(f'SCCs Mean Abs Dev every {t_delta} for {Template}')
            ax.set_ylabel('Mean Abs Dev of SCCs')
            Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick) # Ensure this function handles datetime objects
            ax.legend()
            
            
            ax = axes[ii,4]
            # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_PEAKCCs_array_dict[Template], label=Template, s=10, alpha=0.8, color='k')
            avg = np.nanmean(loaded_PEAKCCs_array_dict[Template]) # Use nanmean
            ax.axhline(avg, color='r', linestyle='--', label=f'Avg: {avg:.3e}') 
            ax.set_title(f'SCCs Peak value every {t_delta} for {Template}')
            ax.set_ylabel('Peak value of SCCs')
            Ticks_xaxis_day_or_hour(ax, Day_tick, Hour_tick) # Ensure this function handles datetime objects
            ax.legend()
            
            
        
        for ax in axes[-1,:]:
            ax.set_xlabel('Time') # Set xlabel only on the last subplot
        plt.tight_layout()
        plt.show()
        fig.savefig(OUTPUT_dir + f'SCCs_MAD_array_AllTemp.png')
        print('Saved: ', OUTPUT_dir + f'SCCs_MAD_array_AllTemp.png')
        plt.close(fig) # Close the figure properly


        #? plot MAD_array of all templates in one figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for ii, Template in enumerate(Temp_name_list):
             # Use loaded_t_array for plotting
            ax.scatter(loaded_t_array[:-1], loaded_MAD_array_dict[Template], label=Template, s=10, alpha =0.5, color=f'C{ii}')
        ax.set_xlabel('Time')
        ax.set_ylabel('MAD')
        ax.set_title(f'MAD every {t_delta} for all templates')
        Ticks_xaxis_day_or_hour(ax, Day_tick=5, Hour_tick=0) # Ensure this function handles datetime objects
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Templates')
        plt.tight_layout()
        plt.show()
        fig.savefig(OUTPUT_dir + 'SCCs_Statistics_array_AllTemp_inOne.png')
        print('Saved: ', OUTPUT_dir + 'SCCs_Statistics_array_AllTemp_inOne.png')
        plt.close(fig) # Close the figure properly

        del loaded_MAD_array_dict # Clean up loaded dictionary

# This guard is important for multiprocessing
if __name__ == "__main__":
    # Define the number of CPUs to use
    # You can set this via an environment variable or command-line argument for more flexibility
    NUM_CPUS = 20
    nt_delta_arr = [1 * 60]  # in minutes, You can modify this list to include other time deltas if needed
    main_processing_logic(num_cpus=NUM_CPUS, nt_delta_arr=nt_delta_arr)

    # The rest of your script (commented out plotting sections) can be placed here
    # if they are meant to be run after the main MAD calculation.
    # Ensure that any variables they rely on (like t_all_start, t_all_end, OUTPUT_dir, etc.)
    # are appropriately defined or passed if you refactor further.

    # Example: If you wanted to run the SSA plotting part, you would need to ensure
    # MAD_array_dict and t_array are available. The current script re-loads them
    # for plotting, which is fine.

    print("Processing complete.")