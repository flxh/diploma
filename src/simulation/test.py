from simulation.Environment import Environment
from timeseriesprediction.utils import load_total_power_from_mat_file
import numpy as np
from ml.EpisodeCreator import EpisodeCreator
from multiprocessing import Queue

q = Queue(maxsize=10)

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p0 = power_vals[:,0]
irradtiation = [300]



for i in range(200):
    print(i)


