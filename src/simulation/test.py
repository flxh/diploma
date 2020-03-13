from simulation.Environment import Environment
from timeseriesprediction.utils import load_total_power_from_mat_file
import numpy as np

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p0 = power_vals[:,0]

for i in range(96*7, len(p0), 96*3):
    mean = np.mean(p0[i-96*7:i])
    print(mean)


e = Environment()
s = e.reset()
print(s)

for i in range(100):
    s = e.step(-1000)
    print(e.storage.stored_energy)
    print(s)
