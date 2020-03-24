import scipy.io as sio
import numpy as np
STEPS_PER_DAY = 96

def  load_total_power_from_mat_file(path, day_start, day_end, included_years):
    mat_contents = sio.loadmat('loadprofiles_1min.mat')
    pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
    pges = np.mean(np.reshape(pges, (-1,15,74)), axis=1) #zetiliche Auflüsung verringern
    ts = np.reshape(pges[day_start*STEPS_PER_DAY: day_end*STEPS_PER_DAY, included_years], (-1,))
    return ts

def get_power_differences(power_vals):
    return np.diff(power_vals, axis=0)

def standardize_power_differences(p_diffs):
    mean = np.mean(p_diffs, axis=0)
    std  = np.std(p_diffs, axis=0)
    return (p_diffs - mean) / std, mean, std
