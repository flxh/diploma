import numpy as np
import warnings
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm = standardize_power_differences(p_diffs)
p_diffs_norm = (p_diffs - np.mean(p_diffs, axis=0)) / np.std(p_diffs, axis=0)

season = 4*24

ts = p_diffs_norm[:364*96, :].reshape((-1,))

y_test_season = []
y_predict_season = []
for i in range(2000, 5000, 10):
    ts_train = ts[:i*96]
    ts_test = ts[i*96: (i+10)*96]

    model = np.mean(ts_train.reshape((96,-1)), axis=1)
    y_predict_season.extend(np.tile(model, 10))
    y_test_season.extend(ts_test)

with open('season_prediction.csv', 'a') as file:
    for yt, yp in zip(y_test_season, y_predict_season):
        file.write('{};{}\n'.format(yt, yp))

