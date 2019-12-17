import scipy.io as sio
import numpy as np


def  load_total_power_from_mat_file(path):
    mat_contents = sio.loadmat('loadprofiles_1min.mat')
    pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
    pges = np.mean(np.reshape(pges, (-1,15,74)), axis=1)  #zetiliche Auflüsung verringern


def get_power_differences(power_vals):
    return np.diff(power_vals, axis=0)

def standardize_power_differences(p_diffs):
    return (p_diffs - np.mean(p_diffs, axis=0)) / np.std(p_diffs, axis=0)
