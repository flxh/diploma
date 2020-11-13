from queue import Empty
import pandas as pd
import numpy as np
import scipy.signal as sgn

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

