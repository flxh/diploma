import scipy.io as sio
import pandas as pd
import numpy as np
import scipy.signal as sgn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ml.EpisodeCreator import EpisodeContainer
import pickle as pkl

load_file_path = '../../loadprofiles_1min.mat'
pv_file_path ='../../ihm-daten_20252.csv'

load_year_index = 44
radiation_year_index =4
episode_name = 'eval_episode8'

#1,0 11,2 17,3 25,4

MINUTES_PER_DAY = 24*60
# Build evaluation episode and safe as mat (MATLAB) and pkl (PYTHON)


def datetime2matlabdn(dt):
    mdn = dt + timedelta(days = 366)
    frac_seconds = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def build_dict_mat_file(ts_load, ts_pv, time):
    return {
        'P_ld': np.reshape(ts_load, (-1,1)).astype(np.float),
        'p_pv': np.reshape(ts_pv, (-1,1)) / 1000.,
        'time': np.reshape(time, (-1,1))
    }

mat_contents = sio.loadmat(load_file_path)
pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']

df = pd.read_csv(pv_file_path, names=['datetime', 'obb', 'tharandt'], header=0, delimiter=';')
df.fillna(0, inplace=True)

ts_th = np.array(df['tharandt'])
ts_th_1min = sgn.resample_poly(ts_th, 10,1)

ts_th_1min_years = np.reshape(ts_th_1min[:-(len(ts_th_1min)%(365*MINUTES_PER_DAY))], (-1, 365*MINUTES_PER_DAY))

ts_load_1min = pges[:, load_year_index]
ts_load_15min = np.mean(np.reshape(ts_load_1min, (-1,15)), axis=1)

dt0 = datetime(2010,1,1)
times = [datetime2matlabdn(dt0 + timedelta(minutes=i)) for i in range(365*MINUTES_PER_DAY)]

mat_dict = build_dict_mat_file(ts_load_1min, ts_th_1min_years[radiation_year_index], times)
sio.savemat(episode_name+'.mat', mat_dict)

plt.plot(range(MINUTES_PER_DAY*7), ts_load_1min[:MINUTES_PER_DAY*7])
plt.plot(range(MINUTES_PER_DAY*7), -ts_th_1min_years[0,:MINUTES_PER_DAY*7])
plt.show()

year_cycle = np.array([-np.cos(((x+MINUTES_PER_DAY*10)/(MINUTES_PER_DAY*365))*2*np.pi) for x in range(MINUTES_PER_DAY*365)])
buy_price_ts = np.array([1.]*len(ts_load_1min))
sell_price_ts = np.array([0.]*len(ts_load_1min))

py_episode = EpisodeContainer(ts_load_1min, -ts_th_1min_years[radiation_year_index], year_cycle, buy_price_ts, sell_price_ts)
pkl.dump(py_episode, open(episode_name+'.pkl', 'wb'))

print('exit')
