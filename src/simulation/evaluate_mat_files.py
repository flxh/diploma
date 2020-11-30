import scipy.io as sio
from simulation.Environment import Environment
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


    for i in range(100):
        aux_infos = []
        env = Environment(1, eval_episodes[i], 60, scale_actions=None)
        print(i)
        for p_schedule in P_schedule[i,1,:]:
            _,_,_,aux = env.step([p_schedule])
            aux_infos.append(aux)

        for ai in aux_infos:
            with open('./mat_evaluations/{}_prog.csv'.format(i), 'a') as file:
                file.write(('{};'*len(ai)+'\n').format(*ai))