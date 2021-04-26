import pickle as pkl
import numpy as np#
from simulation.simulation_globals import KWP, STORAGE_EFFICIENCY, CAPACITY, CONVERTER_MAX_POWER, JOULES_PER_KWH
import matplotlib.pyplot as plt
from simulation.Environment import Environment

action_interval_min = 15
pv_lookback = 3*60
pv_lookahead = 15*60
lookahead_steps = 15*60//action_interval_min # 15 Std

def calculate_radiation_forcast(pv_ts):
    pv_days = pv_ts.reshape((-1,1440))
    pv_max10d = []
    for i in range(pv_days.shape[0]):
        pv_last_10d = pv_days[max(0, i-9):i+1, :]
        pv_max10d.extend(np.max(pv_last_10d, axis=0))
    pv_max10d = np.array(pv_max10d)
    day_mask = pv_ts > 0

    pv_day = pv_ts[day_mask]
    pv_max10d_day = pv_max10d[day_mask]
    E_pv = np.zeros(np.sum(day_mask))
    E_pv_max = np.zeros(np.sum(day_mask))

    for i in range(pv_lookback, len(pv_day)):
        E_pv[i] = np.sum(pv_day[i-pv_lookback:i])
        E_pv_max[i] = np.sum(pv_max10d_day[i-pv_lookback:i])
    ktf_day = E_pv/E_pv_max
    ktf_day = np.nan_to_num(ktf_day, nan=0.)
    ktf = np.zeros(len(pv_ts))
    ktf[day_mask] = ktf_day

    pv_max10d = np.append(pv_max10d, pv_max10d[-1440:])
    ktf_15min = np.mean(ktf.reshape((-1,15)), axis=1)
    pv_max10d_15min = np.mean(pv_max10d.reshape((-1,15)), axis=1)

    pv_prog = []
    for i in range(len(ktf_15min)):
        line = ktf_15min[i] * pv_max10d_15min[i:i+pv_lookahead//15]
        pv_prog.append(line)
    pv_prog = np.array(pv_prog)
    pv_prog[pv_prog<0] = 0

    return pv_prog

def calculate_load_forcast(load_ts):
    load_15min = np.mean(load_ts.reshape((-1,15)), axis=1)
    g1 = 1/np.exp(-0.1)*np.exp(-0.1*(np.arange(0,lookahead_steps)+1))
    g2 = 1-g1

    load_prog = []
    for i in range(len(load_15min)):
        # fill with zeros -> only use imidiate persistence when day back is not available
        load_day_back = load_15min[max(0, i-96):max(0, i-96+lookahead_steps)]
        load_day_back = [0]*(lookahead_steps-len(load_day_back)) +  list(load_day_back)
        load_day_back = np.array(load_day_back)
        line = g1*np.array([load_15min[i]]*lookahead_steps) + g2*load_day_back
        load_prog.append(line)

    return np.array(load_prog)

def calculate_battery_power(p_pv_prog, p_l_prog, soc):
    p_gflvir = np.arange(0.0, 0.5, 0.01) * KWP *1000
    p_diff = p_pv_prog-p_l_prog
    pv_diff_pos = p_diff.copy()
    pv_diff_pos[pv_diff_pos<0] = 0
    pv_diff_pos = np.tile(pv_diff_pos, (len(p_gflvir),1))

    diff_energy = np.sum(np.maximum(0, pv_diff_pos-p_gflvir.reshape((-1,1))) * STORAGE_EFFICIENCY * action_interval_min * 60, axis=1) - (1-soc) * CAPACITY
    best_boundary_idx = np.argmin(np.abs(diff_energy))

    p_b_pos = p_diff - p_gflvir[best_boundary_idx]
    p_b_pos[p_b_pos<0] = 0

    p_b = np.minimum(p_b_pos, p_diff)
    return p_b - p_diff

eval_episodes = []
with open('../../eval_episodes.pkl', 'rb') as file:
    while True:
        try:
            eval_episodes.append(pkl.load(file))
        except EOFError:
            break

for i in range(0,100):
    ep = eval_episodes[i]
    aux_infos = []
    env = Environment(96, ep, 60, sim_steps_per_action=15, balance_pdiff=False)
    state = env.reset()
    print(i)

    p_pv_for = calculate_radiation_forcast(ep.pv_ts) * KWP
    p_l_for = calculate_load_forcast(ep.load_ts)

    pbs = []
    for j in range(96, len(p_pv_for)):
        action = calculate_battery_power(p_pv_for[j], p_l_for[j], state[-1,0])
        pb = action[0]
        state, _ , done, aux = env.step([pb / 1500])
        aux_infos.append(aux)
        pbs.append(pb)
        if done:
            print('Done')

    #plt.plot(range(len(p_pv_for)-96), (p_pv_for[96:]-p_l_for[96:])[:,0])
    #plt.plot(range(len(p_pv_for)-96), pbs)


    with open('./prog_eval_no_bal/{}.csv'.format(i), 'a') as file:
        for ai in aux_infos:
            file.write(('{};'*len(ai)+'\n').format(*ai))



# erstelle Prognose PV und Last -> Matrix
# Schritte simulieren