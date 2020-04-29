from timeseriesprediction.utils import load_total_power_from_mat_file
from simulation.simulation_globals import TIME_STEP,JOULES_PER_KWH
import numpy as np
from sklearn.cluster import KMeans
import  pickle as pkl
import scipy.io as sio
import matplotlib.pyplot as plt

mat_contents = sio.loadmat('../../loadprofiles_1min.mat')
pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
power_vals = np.mean(np.reshape(pges, (-1,15,74)), axis=1) #zetiliche Auflüsung verringern

energy = power_vals * TIME_STEP
summed_energy = np.sum(energy, axis=0)/JOULES_PER_KWH
energy_variance = np.var(energy, axis=0) / JOULES_PER_KWH

X = np.vstack(((summed_energy-np.mean(summed_energy))/np.std(summed_energy), (energy_variance-np.mean(energy_variance))/np.std(energy_variance))).swapaxes(0,1)

km = KMeans(n_clusters=8, verbose=True)
km.fit(X)

_, counts = np.unique(km.labels_, return_counts=True)

clusters = []
for i in range(8):
    print('Cluster ', i)
    print('Count   ', counts[i])
    print('Center  ', km.cluster_centers_[i])
    print('Indices ', np.nonzero(km.labels_ == i))
    print('\n\n')

    clusters.append(X[np.nonzero(km.labels_ == i)])


fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
for c in clusters:
    ax1.scatter(c[:,0], c[:,1])
ax1.set_ylabel('Var(P)')
ax1.set_xlabel('Jahresverbrauch/kWh')
plt.show()

pkl.dump(clusters, open('clusters.pkl', 'wb'))
print(km.cluster_centers_)


