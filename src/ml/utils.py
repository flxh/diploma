from queue import Empty
import pandas as pd
import numpy as np
import scipy.signal as sgn
STEPS_PER_DAY = 24*20


class ExponentialScheduler:
    def __init__(self, n_half, x0):
        self.n_half = n_half
        self.x0 = x0
        self.n = None

    def get_schedule_value(self):
        return self.x0 * 2**(-self.n / self.n_half)


class LinearScheduler:
    def __init__(self, yo, y1, x1):
        self.y0 = yo
        self.y1 = y1
        self.x1 = x1

        self.x = None

    def get_schedule_value(self ):
        return np.max([(self.y1 - self.y0)/self.x1 * self.x + self.y0, self.y1])

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
    ts_obb_resampled = sgn.resample_poly(ts_obb, 10,3)
    ts_obb_resampled = ts_obb_resampled[:-(len(ts_obb_resampled)%(365*STEPS_PER_DAY))]
    ts_obb_resampled_years = np.reshape(ts_obb_resampled, (-1, 365*STEPS_PER_DAY))

    ts_valid_range = ts_obb_resampled_years[:, day_start*STEPS_PER_DAY: day_end*STEPS_PER_DAY]

    return np.reshape(ts_valid_range, (-1,))


