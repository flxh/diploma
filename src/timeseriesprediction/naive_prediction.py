import warnings
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)

i_household = 69
steps_ahead = 1

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm = standardize_power_differences(p_diffs)
pv_norm, pv_mean, pv_std = standardize_power_differences(power_vals)

ts = pv_norm[:364*96, i_household][steps_ahead:]
predict = pv_norm[:364*96, i_household][:-steps_ahead]
with open('zero_change_prediction_69.csv', 'a') as file:
    for yt, yp in zip(ts, predict):
        file.write('{};{}\n'.format(yt, yp))
