import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import pandas as pd
import pickle as pkl


plt.rcParams.update({'font.size': 8})

image_folder = '/home/florus/Documents/Uni+/da/Latex/images/'
page_width = 6.2


mat_contents = sio.loadmat('../loadprofiles_1min.mat')
pges_1min = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
pges = np.mean(np.reshape(pges_1min, (-1,15,74)), axis=1)  #zetiliche Auflüsung verringern
p_diffs = np.diff(pges, axis=0)
p_diffs_1min = np.diff(pges_1min, axis=0)

twoweeks = 4 * 24 * 14
oneweek = 4*24*7
light_blue = '#cbe3f0'



files = [
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_159_no_adaption_year_radiation.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_2_high_year_adaption.csv'
]
names = ['trajectory_low_reg.pdf','trajectory_high_reg.pdf']
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(page_width*1.5, 6),sharex='all')

df = None
for file, name in zip(files, names):
    df = pd.read_csv(file, delimiter=';', names=['soc', 'e_last', 'e_pv', 'e_speicher'], usecols=[0,1,2,3])
    df['p_speicher'] = df['e_speicher'] / (15*60)
    ax1.plot(range(17800, 18550), df['p_speicher'][17800:18550])
    ax2.plot(range(17800, 18550), df['soc'][17800:18550])
    ax2.set_xlabel('Zeitintervall t')
    ax2.set_ylabel('SOC')
    ax1.set_ylabel('P [W]')
    ax2.set_ylim(0,1)
df['p_last'] = df['e_last'] / (15*60)
df['p_pv'] = df['e_pv'] / (15*60)
df['p_diff'] = -df['p_last'] - df['p_pv']
ax1.plot(range(17800, 18550), df['p_diff'][17800:18550])
ax1.legend(['$P_{Speicher}; \\alpha_{Reg}=4.0$','$P_{Speicher}; \\alpha_{Reg}=1.6$', '$P_{Diff}$'])
plt.tight_layout()
plt.savefig(image_folder+name)
plt.close(fig1)


datevecs = mat_contents['time_datevec_MEZ']
date_times_1min = np.array([datetime(x[0], x[1], x[2], x[3], x[4], x[5]) for x in datevecs])
date_times = date_times_1min[range(0, len(date_times_1min), 15)]

clusters = pkl.load(open('clusters.pkl', 'rb'))

df = pd.read_csv('/home/florus/Documents/Uni+/da/files/soc_collapse.csv', delimiter=',', names=['time', 'x', 'y'], skiprows=1)
df['y_ma'] =  df['y'].rolling(window=25).mean()
fig1, ax1 = plt.subplots(1, 1, figsize=(page_width, 3))
ax1.plot(df['x'], df['y'], color=light_blue)
ax1.plot(df['x'], df['y_ma'])
ax1.set_xlabel('$n_{iter}$')
ax1.set_ylabel('$VAR(\mu)$')
plt.tight_layout()
#plt.show()
plt.savefig(image_folder+"soc_collapse.pdf")
plt.close(fig1)


ods = pd.ExcelFile('/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_summary.ods', engine='odf')
names = ['$k_{SEK}$ [€/kWh]', '$k_{EV}$','$k_{SV}$', '$k_{AR}$']
ids = [10,7,6,8]

fig1, axes = plt.subplots(2, 2, figsize=(page_width, 5))
axes = axes.flatten()

for name, idx, ax in zip(names, ids, axes):
    for sheet in [1,2,3]:
        y = pd.read_excel(ods, sheet_name=sheet, usecols=[idx], names=['y'])['y']
        ax.scatter(np.ones(len(y))*sheet, y)

    ax.set_xlim((0.,4.))
    ax.set_xticks([])
    ax.set_ylabel(name)
axes[0].legend(['Reinforcement L.', 'Prioritätsbasiert','Dynamische Einsp.'])
plt.tight_layout()
#plt.show()
plt.savefig(image_folder+"eval_metrics.pdf")
plt.close()



fig1, ax = plt.subplots(1, 1, figsize=(page_width, 3.5))
ax.plot(date_times[:oneweek], pges[:oneweek, 42])
ax.set_ylabel('$P_{ges}  [W]$')
ax.legend(['Messstelle 42'])
plt.gcf().autofmt_xdate()
plt.tight_layout()
#plt.show()
plt.savefig(image_folder+"consumption_data.pdf")
plt.close()


df = pd.read_csv('/home/florus/IdeaProjects/diploma/ihm-daten_20252.csv', names=['datetime', 'obb', 'tharandt'], header=0, delimiter=';')
df.fillna(0, inplace=True)
ts_obb = np.array(df['tharandt'].iloc[120*120:120*120+6*24*7])
dt = pd.to_datetime(df['datetime'].iloc[120*120:120*120+6*24*7], format='%d/%m/%Y %H:%M')

fig1, ax = plt.subplots(1, 1, figsize=(page_width, 3.5))
ax.plot(dt, ts_obb)
ax.set_ylabel('$G  [W/m^2]$')
plt.gcf().autofmt_xdate()
plt.tight_layout()
#plt.show()
plt.savefig(image_folder+"radiation_data.pdf")
plt.close()



fig1, axes = plt.subplots(3, 1, figsize=(page_width, 5),sharex='all')
files = [
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/policy_collapse_pi_mu_mean.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/policy_collapse_pi_mu_var.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/policy_collapse_SOC_mean.csv'
]

names = ['$\\bar{\mu}$', '$VAR(\mu)$', '$\\bar{SOC}$']

for ax, file, name in zip(axes, files, names):
    df = pd.read_csv(file, delimiter=',', names=['time', 'x', 'y'], skiprows=1)
    df['y_ma'] =  df['y'].rolling(window=25).mean()
    ax.plot(df['x'], df['y'], color=light_blue)
    ax.plot(df['x'], df['y_ma'])
    ax.set_ylabel(name)
axes[2].set_xlabel('$n_{iter}$')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"policy_collapse.pdf")
plt.close(fig1)


fig1, axes = plt.subplots(4, 1, figsize=(page_width, 6.5),sharex='all')
files = [
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/training_progress_GRID_BOUGHT_mean.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/training_progress_GRID_SOLD_mean.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/training_progress_GRID_WASTED_mean.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/training_progress_eval_reward_mean.csv'
]
names = ['$E_{Bezogen}$ [kWh]', '$E_{Eingespeist}$ [kWh]', '$E_{Abgeregelt}$ [kWh]', '$\\bar{r}$']
data = [pd.read_csv(file, delimiter=',', names=['y'],skiprows=1, usecols=[2])['y'] for file in files]
data_factor = []
for i in range(3):
    data_factor.append(data[i] * 96 *365 / 3600000)
data_factor.append(data[3])

for ax, d, name in zip(axes, data_factor, names):
    ax.plot(range(len(d)), d)
    ax.set_ylabel(name)
axes[3].set_xlabel('$n_{Evaluierung}$')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"training_progress.pdf")
plt.close(fig1)



fig1, axes = plt.subplots(2, 1, figsize=(page_width, 5),sharex='all')
files = [
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_159_no_adaption_year_radiation.csv',
    '/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_2_high_year_adaption.csv'
]

for ax, file in zip(axes, files):
    df = pd.read_csv(file, delimiter=';', names=['soc'], usecols=[0])
    df['soc_ma'] =  df['soc'].rolling(window=3000).mean()
    ax.plot(range(len(df['soc'])), df['soc'], color=light_blue)
    ax.plot(range(len(df['soc_ma'])), df['soc_ma'])
    ax.set_ylim((0,1))
    ax.set_ylabel('SOC')
axes[1].set_xlabel('Zeitintervall t')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"wo_soc_reg.pdf")
plt.close(fig1)

fig1, ax1 = plt.subplots(1, 1, figsize=(page_width, 3),sharex='all')
df = pd.read_csv('/home/florus/Documents/Uni+/da/Daten_Diagramme/eval_2_high_year_adaption.csv', delimiter=';', names=['soc'], usecols=[0])
ax1.hist(df['soc'], bins=50)
ax1.set_ylabel('Häufigkeit')
ax1.set_xlabel('SOC')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"soc_hist.pdf")
plt.close(fig1)

fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
for c in clusters:
    ax1.scatter(c[:,0], c[:,1])
ax1.set_ylabel('Var(P) (Normalisiert)')
ax1.set_xlabel('Jahresverbrauch (Normalisiert)')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"cluster_plot.pdf")
plt.close(fig1)

fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
x = np.linspace(0.,1., 200)
ax1.plot(x, (2*x-1)**2)
ax1.set_ylabel('Regularisierung / $\\alpha_{SOC}$')
ax1.set_xlabel('SOC')
#plt.show()
plt.tight_layout()
plt.savefig(image_folder+"soc_regularization.pdf")
plt.close(fig1)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(page_width, 2.5))

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


fig1, ((ax1)) = plt.subplots(1, 1, figsize=(page_width, 4))
ax1.plot(date_times_1min[:7*60*24], pges_1min[:7*60*24,0])
ax1.plot(date_times[:7*4*24], pges[:7*4*24,0])
ax1.legend(['1 min', '15 min'])
ax1.set_ylabel('$P_{ges}$')

#ax2.plot(date_times_1min[:7*60*24], p_diffs_1min[:7*60*24,0])
#ax2.plot(date_times[:7*4*24], p_diffs[:7*4*24,0])
#ax2.legend(['1 min', '15 min'])
#ax2.set_ylabel('$ΔP_{ges}$')

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(image_folder+"1min15min.pdf")
plt.close(fig1)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(page_width, 5))
ax1.plot(date_times[:twoweeks], pges[:twoweeks,69])
ax1.plot(date_times[:twoweeks], pges[:twoweeks,42])
ax1.set_ylabel('$P_{ges}$')
ax1.legend(['Messstelle 70', 'Messstelle 43'])

ax2.plot(date_times[:twoweeks], (pges[:twoweeks,69]-np.mean(pges[:,69]))/np.std(pges[:,69]))
ax2.plot(date_times[:twoweeks], (pges[:twoweeks,42]-np.mean(pges[:,42]))/np.std(pges[:,42]))
ax2.set_ylabel('$P\'_{ges}$')
ax2.legend(['Messstelle 70', 'Messstelle 43'])

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(image_folder+"standardization.pdf")
plt.close(fig1)

arima_results_df = pd.read_csv('timeseriesprediction/predictions/arima_prediction_10.csv', delimiter=';', names=['true', 'predict'])
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

fig1, (ax1) = plt.subplots(1, 1, figsize=(page_width, 4))
ax1.plot(n_datapoints, mse_arima_convergence)
ax1.set_ylabel('$MSE$')
ax1.set_xlabel('$n_{Datenpunkte}$')

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(image_folder+"arima_convergence.pdf")
plt.close(fig1)

lstm_single_df = pd.read_csv('timeseriesprediction/predictions/lstm_single_iteration_10.csv', delimiter=';', names=['r2', 'mse'])
mse_lstm = np.array(lstm_single_df['mse'])

fig1, ax1 = plt.subplots(1, 1, figsize=(page_width, 4))
ax1.plot(range(len(mse_lstm)), mse_lstm)
ax1.set_ylabel('$MSE$')
ax1.set_xlabel('$n_{Epochen}$')

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(image_folder+"lstm_convergence.pdf")
#plt.show()
plt.close(fig1)