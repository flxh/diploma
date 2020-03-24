from timeseriesprediction.utils import load_total_power_from_mat_file
from simulation.simulation_globals import TIME_STEP,JOULES_PER_KWH
import numpy as np
from sklearn.cluster import KMeans

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
energy = power_vals * TIME_STEP
summed_energy = np.sum(energy, axis=0)/JOULES_PER_KWH
energy_variance = np.var(energy, axis=0) / JOULES_PER_KWH

X = np.vstack((summed_energy, energy_variance)).swapaxes(0,1)

km = KMeans(n_clusters=8, verbose=True)
km.fit(X)

_, counts = np.unique(km.labels_, return_counts=True)

for i in range(8):
    print('Cluster ', i)
    print('Count   ', counts[i])
    print('Center  ', km.cluster_centers_[i])
    print('Indices ', np.nonzero(km.labels_ == i))
    print('\n\n')

print(km.cluster_centers_)


