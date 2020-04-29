import scipy.io as sio
import numpy as np
STEPS_PER_DAY = 24*20

def  load_total_power_from_mat_file(path, day_start=0, day_end=365, included_years=None):
    mat_contents = sio.loadmat(path)
    pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
    pges = np.swapaxes(pges, 0,1)
    pges = np.mean(np.reshape(pges, (74,-1,3)), axis=2) #zetiliche Auflüsung verringern
    if included_years is not None:
        ts = np.reshape(pges[included_years, day_start*STEPS_PER_DAY: day_end*STEPS_PER_DAY], (-1,))
    else:
        ts = np.reshape(pges[:, day_start*STEPS_PER_DAY: day_end*STEPS_PER_DAY], (-1,))
    return ts

def get_power_differences(power_vals):
    return np.diff(power_vals, axis=0)

def standardize_power_differences(p_diffs):
    mean = np.mean(p_diffs, axis=0)
    std  = np.std(p_diffs, axis=0)
    return (p_diffs - mean) / std, mean, std
