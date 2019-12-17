from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima_model import ARMA
import warnings
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
warnings.simplefilter('ignore', FutureWarning)

n_steps_predict = 5
output_path = 'arima_prediction.csv'

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm = standardize_power_differences(p_diffs)

ts = p_diffs_norm[:, 10].reshape((-1,))

print('dataset built')
print(len(ts))

y_predict_arima = []
y_test_arima = []

for i in range(10000, len(ts), n_steps_predict):
    ts_train, ts_test = ts[:i], ts[i:i+n_steps_predict]

    model = ARMA(ts_train, order=(5,0))
    model_fit = model.fit(disp=False)
    prediction = model_fit.forecast(len(ts_test))

    y_test_arima.extend(ts_test)
    y_predict_arima.extend(prediction[0])
    print(r2_score(y_test_arima, y_predict_arima))
    print(mean_squared_error(y_test_arima, y_predict_arima))

    with open(output_path, 'a') as file:
        for k in range(len(ts_test)):
            file.write('{};{}\n'.format(ts_test[k], prediction[0][k]))
