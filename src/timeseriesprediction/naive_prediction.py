import warnings
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)


power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm = standardize_power_differences(p_diffs)

ts = p_diffs_norm[:364*96, :].reshape((-1,))
with open('zero_change_prediction.csv', 'a') as file:
    for yt in ts:
        file.write('{};{}\n'.format(yt, 0))
