import scipy.io as sio
import numpy as np



def get_power_differences(power_vals):
    return np.diff(power_vals, axis=0)

def standardize_power_differences(p_diffs):
    mean = np.mean(p_diffs, axis=0)
    std  = np.std(p_diffs, axis=0)
    return (p_diffs - mean) / std, mean, std
