from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARMA
import warnings
import numpy as np
import matplotlib.pyplot as plt
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)

n_steps_predict = 5
i_household = 69
output_path = 'arima_prediction_{}.csv'.format(i_household)

power_vals = load_total_power_from_mat_file('../../loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm, pd_mean, pd_std = standardize_power_differences(p_diffs)
pv_norm, pv_mean, pv_std = standardize_power_differences(power_vals)

ts = pv_norm[:, i_household].reshape((-1,))
#y_test_retrans_all = power_vals[1:,i_household]
#y_before = power_vals[:-1,i_household]

print('dataset built')
print(len(ts))

y_predict_arima = []
y_test_arima = []

min_train_size = 10000
train_end = len(ts)

for i in range(min_train_size, train_end, n_steps_predict):
    ts_train, ts_test = ts[i-min_train_size:i], ts[i:i+n_steps_predict]

    model = ARMA(ts_train, order=(5,0))
    model_fit = model.fit(disp=False)
    prediction = model_fit.forecast(len(ts_test))

    y_test_arima.extend(ts_test)
    y_predict_arima.extend(prediction[0])
    print(i)
    #print(r2_score(y_test_arima, y_predict_arima))
    #print(mean_squared_error(y_test_arima, y_predict_arima))

    with open(output_path, 'a') as file:
        for k in range(len(ts_test)):
            file.write('{};{}\n'.format(ts_test[k], prediction[0][k]))

#y_predict_arima = np.array(y_predict_arima)
#y_predict_retrans = (y_predict_arima * pd_std[i_household]) + pd_mean[i_household] + y_before[min_train_size:train_end]
#y_test_retrans = y_test_retrans_all[min_train_size:train_end]
#plt.plot(range(len(y_predict_retrans)), y_predict_retrans)
#plt.plot(range(len(y_test_retrans)), y_test_retrans)
#plt.show()


#print(r2_score(y_test_retrans, y_predict_retrans))
#print(mean_squared_error(y_test_retrans, y_predict_retrans))
#print(mean_absolute_error(y_test_retrans, y_predict_retrans))

print(r2_score(y_test_retrans, y_before[min_train_size:train_end]))
print(mean_squared_error(y_test_retrans, y_before[min_train_size:train_end]))
print(mean_absolute_error(y_test_retrans, y_before[min_train_size:train_end]))


print()
