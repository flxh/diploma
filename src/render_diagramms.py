import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import pandas as pd

image_folder = '/home/florus/Documents/Uni+/da/latex/images/'

mat_contents = sio.loadmat('loadprofiles_1min.mat')
pges_1min = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
pges = np.mean(np.reshape(pges_1min, (-1,15,74)), axis=1)  #zetiliche Auflüsung verringern
p_diffs = np.diff(pges, axis=0)
p_diffs_1min = np.diff(pges_1min, axis=0)

twoweeks = 4 * 24 * 14
oneweek = 4*24*7

datevecs = mat_contents['time_datevec_MEZ']
date_times_1min = np.array([datetime(x[0], x[1], x[2], x[3], x[4], x[5]) for x in datevecs])
date_times = date_times_1min[range(0, len(date_times_1min), 15)]


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

ax1.plot(date_times[:oneweek], pges[:oneweek, 0])
ax1.set_ylabel('$P_{ges}$')

#ax3.plot(date_times[:twoweeks], p_diffs[:twoweeks, 0])
#ax3.set_ylabel('$ΔP_{ges}$')

plt.gcf().autofmt_xdate()

auto_correll =[1]
auto_correll_first_derivative = [1]

for i in range(1,200):
    auto_correll.append(np.corrcoef(pges[i:,0], pges[:-i,0])[0,1])
    auto_correll_first_derivative.append(np.corrcoef(p_diffs[i:,0], p_diffs[:-i, 0])[0,1])

ax2.plot(range(200), auto_correll)
ax2.set_ylabel('ACF')

#ax4.plot(range(200), auto_correll_first_derivative)
#ax4.set_ylabel('ACF')
#ax4.set_xlabel('Verzögerung n Schritte')

plt.tight_layout()
plt.savefig(image_folder+"correlogramm.pdf")
plt.close(fig1)


fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 5))
ax1.plot(date_times_1min[:7*60*24], pges_1min[:7*60*24,0])
ax1.plot(date_times[:7*4*24], pges[:7*4*24,0])
ax1.legend(['1 min', '15 min'])
ax1.set_ylabel('$P_{ges}$')

#ax2.plot(date_times_1min[:7*60*24], p_diffs_1min[:7*60*24,0])
#ax2.plot(date_times[:7*4*24], p_diffs[:7*4*24,0])
#ax2.legend(['1 min', '15 min'])
#ax2.set_ylabel('$ΔP_{ges}$')

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig(image_folder+"1min15min.pdf")
plt.close(fig1)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(date_times[:twoweeks], pges[:twoweeks,69])
ax1.plot(date_times[:twoweeks], pges[:twoweeks,42])
ax1.set_ylabel('$P_{ges}$')
ax1.legend(['Messstelle 70', 'Messstelle 43'])

ax2.plot(date_times[:twoweeks], (pges[:twoweeks,69]-np.mean(pges[:,69]))/np.std(pges[:,69]))
ax2.plot(date_times[:twoweeks], (pges[:twoweeks,42]-np.mean(pges[:,42]))/np.std(pges[:,42]))
ax2.set_ylabel('$P\'_{ges}$')
ax2.legend(['Messstelle 70', 'Messstelle 43'])

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig(image_folder+"standardization.pdf")
plt.close(fig1)

arima_results_df = pd.read_csv('timeseriesprediction/arima_prediction_10.csv', delimiter=';', names=['true', 'predict'])
y_true = np.array(arima_results_df['true'])
y_predict = np.array(arima_results_df['predict'])

r2_arima_convergence = []
mse_arima_convergence = []
n_datapoints = []
for i in range(100, len(y_true), 10):
    r2_arima_convergence.append(r2_score(y_true[:i], y_predict[:i]))
    mse_arima_convergence.append(mean_squared_error(y_true[:i], y_predict[:i]))
    print(mean_squared_error(y_true[:i], y_predict[:i]))
    n_datapoints.append(i)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(n_datapoints, mse_arima_convergence)
ax1.set_ylabel('$MSE$')

ax2.plot(n_datapoints, r2_arima_convergence)
ax2.set_ylabel('$R2$')
ax2.set_xlabel('$n_{Datenpunkte}$')

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig(image_folder+"arima_convergence.pdf")
plt.close(fig1)

lstm_single_df = pd.read_csv('timeseriesprediction/lstm_single_iteration_10.csv', delimiter=';', names=['r2', 'mse'])
mse_lstm = np.array(lstm_single_df['mse'])

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.plot(range(len(mse_lstm)), mse_lstm)
ax1.set_ylabel('$MSE$')
ax1.set_xlabel('$n_{Epochen}$')

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig(image_folder+"lstm_convergence.pdf")
plt.show()
plt.close(fig1)