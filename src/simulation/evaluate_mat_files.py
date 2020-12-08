import scipy.io as sio
from simulation.Environment import Environment
from simulation.simulation_globals import CONVERTER_MAX_POWER
import pickle as pkl

eval_episodes = []
with open('../../eval_episodes.pkl', 'rb') as file:
    while True:
        try:
            eval_episodes.append(pkl.load(file))
        except EOFError:
            break


if __name__ == '__main__':
    mat_file = sio.loadmat('p_soll.mat')
    P_schedule = mat_file['P_B_all']


    for i in range(0,50):
        aux_infos = []
        env = Environment(1, eval_episodes[i], 60, sim_steps_per_action=1)
        print(i)
        for p_schedule in P_schedule[i,0,:]:
            _,_,_,aux = env.step([p_schedule/CONVERTER_MAX_POWER])
            aux_infos.append(aux)

        aux_infos = aux_infos[1440:] #drop the first 1440 Steps because ML needs "warm up phase" = 96Steps * 15 Minutes
        for ai in aux_infos:
            with open('./mat_evaluations/{}_prio.csv'.format(i), 'a') as file:
                file.write(('{};'*len(ai)+'\n').format(*ai))