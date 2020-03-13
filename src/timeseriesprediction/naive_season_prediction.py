import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm,_ ,_ = standardize_power_differences(p_diffs)

pvals_norm, pvals_mean, pvals_std = standardize_power_differences(power_vals)

i_household = 69
season = 4*24

ts = p_diffs_norm[:364*season, :].reshape((-1,))
ts_no_transform = pvals_norm[:,i_household]
#ts_no_transform = pvals_norm.swapaxes(0,1).reshape(-1)

#trend = np.mean(ts_no_transform.reshape(season, -1), axis = 1)
#y_predict_no_transform = np.tile(trend, 365)

#print(r2_score(ts_no_transform, y_predict_no_transform))
#print(mean_squared_error(ts_no_transform, y_predict_no_transform)**0.5)
#print(mean_absolute_error(ts_no_transform, y_predict_no_transform))


y_test_season = []
y_predict_season = []

for i in range(30,360,5):
    ts_train = ts_no_transform[(i-30)*season:i*season]
    ts_test = ts_no_transform[i*season: (i+5)*season]

    model = np.mean(ts_train.reshape((season,-1)), axis=1)
    y_predict_season.extend(np.tile(model, 5))
    y_test_season.extend(ts_test)

    print(i,'   ',mean_absolute_error(np.array(y_test_season), np.array(y_predict_season)))

plt.plot(range(len(y_predict_season)),y_predict_season)
plt.plot(range(len(y_predict_season)),y_test_season)
plt.show()

with open('season_prediction_{}.csv'.format(i_household), 'a') as file:
    for yt, yp in zip(y_test_season, y_predict_season):
        file.write('{};{}\n'.format(yt, yp))

