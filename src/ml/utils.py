from queue import Empty
import pandas as pd
import numpy as np
import scipy.signal as sgn
STEPS_PER_DAY = 96

def drain(q):
    while True:
        try:
            yield q.get_nowait()
        except Empty:  # on python 2 use Queue.Empty
            break

def load_irraditaion_data(path, day_start, day_end):
    df = pd.read_csv(path, names=['datetime', 'obb', 'tharandt'], header=0, delimiter=';')
    df.fillna(0, inplace=True)

    ts_obb = np.array(df['tharandt'])
    ts_obb_resampled = sgn.resample_poly(ts_obb, 2,3)
    ts_obb_resampled = ts_obb_resampled[:-(len(ts_obb_resampled)%(365*STEPS_PER_DAY))]
    ts_obb_resampled_years = np.reshape(ts_obb_resampled, (365*STEPS_PER_DAY, -1))

    ts_valid_range = ts_obb_resampled_years[day_start*STEPS_PER_DAY: day_end*STEPS_PER_DAY, :]

    return np.reshape(ts_valid_range, (-1,))


