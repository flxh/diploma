import tensorflow.compat.v1 as tf

from abc import abstractmethod
from multiprocessing import Manager
import sys
from datetime import datetime
import numpy as np
import random
from time import time
from multiprocessing import Process, Queue
import os
from ml.TransitionBuffer import Transition, TransitionBuffer
import pickle as pkl

from ml.EpisodeCreator import EpisodeLoader
from ml.TBTrainingLogger import TrainingSummaryCreator
from ml.PpoCore import Policy, StateValueApproximator
from simulation.Environment import Environment, INFO_HEADER
from ml.TBStateLogger import StateLogger

from traceback import print_exc

print(sys.argv)

random.seed(2)
np.random.seed(2)

#TODO TRANSFORMTHIS TO DICT
HP = {
    'BATCH_SIZE': 4096,
    'N_WORKERS': 256,
    'ENV_STEPS': 256,
    'TAIL_LEN': 96,
    'CELLS': 150,
    'GRADIENT_NORM': 10.,
    'KERNEL_REG': 0.,
    'BETA': 0.005,
    'LR_POLICY': 1e-4,
    'LR_VS': 2e-4,
    'EPSILON': 0.15,
    'GAMMA' : 0.998,
    'LAMBDA' : 0.9,
    'LOG_VAR_INIT' : -0.5,
    'K_EPOCHS' : 6,
    'K_EPOCHS_VS' : 4
}

DT_SIM_STEP = 60
SIM_STEPS_PER_ACTION = 15

QUEUE_SIZE = 10

load_model = False
tb_verbose = True

class Worker:
    def __init__(self):
        self.temp_buffer = TransitionBuffer()

        self.env = None
        self.state = None

        self.done = True

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abstractmethod
    def _next_episode(self):
        raise NotImplementedError

    @abstractmethod
    def _save_transition(self, s, a, r, sn, d, aux):
        raise NotImplementedError

    # TODO check if can be made private and called automatically
    def start_episode(self):
        episode_container = self._next_episode()
        self.env = Environment(HP['TAIL_LEN'], episode_container, DT_SIM_STEP, sim_steps_per_action=SIM_STEPS_PER_ACTION)

        self.temp_buffer.clear()

        try:
            self.state = self.env.reset()
            self.done = False
        except Exception as e:
            print_exc()
            print("Error during boot up phase - starting new episode")
            self.start_episode()

    def step(self, action):
        try:
            next_state, reward, done, aux_info = self.env.step(action)

        except Exception as e:
            print_exc()
            print("Error trying to step environment - starting new episode")
            self.done = True
            return

        if done:
            print('done')
            self.done = True

        self._save_transition(self.state, action, reward, next_state, done, aux_info)
        self.state = next_state


class TrainingWorker(Worker):
    def __init__(self, episode_queue):
        super().__init__()
        self.episode_queue = episode_queue

    def _next_episode(self):
        return self.episode_queue.get()

    def _save_transition(self, s, a, r, sn, d, aux):
        self.temp_buffer.append(Transition(state=s, action=a, reward=r, next_state=sn, done=d))


class EvaluationWorker(Worker):
    def __init__(self, episode):
        super().__init__()
        self.episode = episode

    def _next_episode(self):
        return self.episode

    def _save_transition(self, s, a, r, sn, d, aux):
        self.temp_buffer.append(Transition(reward=r, aux_info=aux))


def calculate_gae(transition_buffer, vs):
    '''
    weighs between TD(0) and TD(1) (normal advantage estimate)
    additionally weighs between 0 and inf. time horizon of empirical rewards  (see Form. 14 GAE)
    :param transitions:
    :param vs:
    :return:
    '''
    states = transition_buffer.states
    rewards = transition_buffer.rewards
    next_states = transition_buffer.next_states

    stacked_states = np.vstack((states, next_states))
    stacked_values, _ = vs.predict(stacked_states)

    state_values = stacked_values[:len(states)]
    next_state_values = stacked_values[len(states):]

    td_residuals = rewards + HP['GAMMA'] * next_state_values - state_values

    gae_values = []
    last_gea = 0

    # see GAE paper Schulman formula 15
    for tdr in reversed(td_residuals):
        gae = tdr + HP['LAMBDA'] * HP['GAMMA'] * last_gea
        gae_values.append(gae)
        last_gea = gae

    gae_values.reverse()
    return gae_values


class EvaluationSummaryCreator:
    def __init__(self, sess):
        self.sess = sess
        with tf.variable_scope("evaluation", reuse=True):
            self.rewards_ph = tf.placeholder(tf.float32, [None])
            self.reward_mean, var = tf.nn.moments(self.rewards_ph, axes=[0])

            self.info_ph = tf.placeholder(tf.float32, [None,len(INFO_HEADER)], name="info_ph")
            mean, variance = tf.nn.moments(self.info_ph, axes=[0])

            for i in range(len(INFO_HEADER)):
                name = INFO_HEADER[i]
                tf.summary.scalar("{}__{}_mean".format(i,name), mean[i])

            self.eval_rew_mean_summary = tf.summary.scalar("__eval_reward_mean", self.reward_mean)
            self.eval_rew_var_summary = tf.summary.scalar("__eval_reward_var", var)

            self.summary_op = tf.summary.merge_all(scope='evaluation')

    def create_summary(self, rewards, infos):
        summaries, _ = self.sess.run([self.summary_op, self.reward_mean], feed_dict={
            self.rewards_ph: rewards,
            self.info_ph: infos
        })
        return summaries


def worker_wrapper(worker, action):
    worker.step(action)
    return worker


if __name__ == '__main__':

    model_path = '/workspace/models/' # leave unchanged !
    tensor_board_path = '/workspace/tensorboard/' # used to store Tensorboard files
    temp_path = '/workspace/trajectories/' # used to save trajectory CSV files for each evaluation run
    runs_file = '/workspace/runs.csv'

    for p in [model_path, tensor_board_path, temp_path]:
        if not os.path.exists(p) or not os.path.isdir(p):
            raise IOError(f"Workspace dir {p} does not exist.\nIs /workspace mounted? Are subdirectories created?")

    RUN_ID = sys.argv[1]
    PARAM_SET = None
    if len(sys.argv) > 2:
        PARAM_SET = int(sys.argv[2])
        params = np.genfromtxt(f'{sys.argv[3]}', delimiter=';', skip_header=1)
        HP['GAMMA'] = params[PARAM_SET,0]
        HP['LR_POLICY'] = params[PARAM_SET,2]
        HP['LR_VS'] = params[PARAM_SET,2] * 2
        HP['BETA'] = params[PARAM_SET,1]

    for x in HP.items():
        print('{:.<16}{}'.format(*x))

    # initialization of the buffers and the episode queue
    m = Manager()
    episode_queue = m.Queue(maxsize=QUEUE_SIZE)
    training_buffer = TransitionBuffer()
    eval_buffer = TransitionBuffer()
    horizon_buffers = []

    print('Start episode loader')
    episode_loader = EpisodeLoader(episode_queue, os.path.abspath('../../training_episodes.pkl'))
    episode_proc = Process(target=episode_loader.fill_queue, name="episode_loader")
    episode_proc.start()

    eval_episodes = []
    with open('../../eval_episodes.pkl', 'rb') as file:
        while True:
            try:
                eval_episodes.append(pkl.load(file))
            except EOFError:
                break

    state_dim = 4
    action_dim = 1

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.31
    with tf.Session(config=config) as sess:
        now = datetime.now()

        vs = StateValueApproximator(sess, state_dim, HP['LR_VS'], HP['TAIL_LEN'], HP['CELLS'], HP['GAMMA'], HP['KERNEL_REG'])
        pol = Policy(sess, state_dim, action_dim, HP['LR_POLICY'], HP['TAIL_LEN'], HP['CELLS'],  HP['BETA'], HP['LOG_VAR_INIT'], HP['EPSILON'], HP['KERNEL_REG'], HP['GRADIENT_NORM'])
        state_logger = StateLogger(sess, ['SOC', 'LOAD', 'PV', 'CYCLE'], "state")
        state_logger_norm = StateLogger(sess,  ['SOC', 'LOAD', 'PV', 'CYCLE'], "norm_state")
        t_summary_creator = TrainingSummaryCreator(sess)

        if not os.path.isfile(runs_file):
            with open(runs_file, 'a') as file:
                file.write('RUN_ID'+(';{}'*len(HP)).format(*sorted(HP.keys())))
                file.write('\n')

        with open(runs_file, 'a') as file:
            file.write(f'{RUN_ID}'+(';{}'*len(HP)).format(*[HP[k] for k in sorted(HP.keys())]))
            file.write('\n')

        #experiment_name = "{}-a{}-lr{}-sr{}".format(now.strftime('%Y-%m-%dT%H%M%S'), BETA, LR_POLICY, SOC_REG_SCHEDULER.get_schedule_value())
        temp_folder = temp_path + f"{RUN_ID}"

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        writer = tf.summary.FileWriter(tensor_board_path+f"{RUN_ID}", sess.graph)
        # print("RUN META")
        print(RUN_ID)

        tr_workers = [TrainingWorker(episode_queue) for _ in range(HP['N_WORKERS'])]
        ev_workers = [EvaluationWorker(eep) for eep in eval_episodes[:64]]

        print('Workers created')
        ev_summ_creator = EvaluationSummaryCreator(sess)
        print('Evaluator created')

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        print('Models initialized')

        saver = tf.train.Saver()
        if load_model:
            saver.restore(sess, model_path+f"{RUN_ID}")

        n_transitions = 0
        i_policy_iter = 0

        print('Start Training')
        last_time = time()

        while True:
                for _ in range(HP['ENV_STEPS']):
                    states = []
                    for w in tr_workers + ev_workers:
                        if w.done: w.start_episode()
                        states.append(w.state)

                    determ_actions, actions, policy_iteration = pol.sample_action(states)

                    tr_actions = list(actions[:HP['N_WORKERS']])
                    ev_actions = list(determ_actions[HP['N_WORKERS']:])

                    for w,a in zip(tr_workers+ev_workers, tr_actions+ev_actions):
                        w.step(a)

                    for w in tr_workers:
                        if (w.done and w.temp_buffer) or len(w.temp_buffer) >= HP['ENV_STEPS']:
                            horizon_buffers.append(w.temp_buffer.copy())
                            w.temp_buffer.clear()

                eval_buffer.clear()
                for w in ev_workers:
                    eval_buffer += w.temp_buffer
                    w.temp_buffer.clear()

                total_transitions_training = 0

                # calculate gae values from transitions and extend training buffer
                for h_buf in horizon_buffers:
                    total_transitions_training += len(h_buf)
                    gae_vals = calculate_gae(h_buf, vs)
                    h_buf.add_gea_values(gae_vals)
                    training_buffer += h_buf

                print('Average horizon length ', total_transitions_training / len(horizon_buffers))
                horizon_buffers.clear()

                if len(training_buffer) >= HP['BATCH_SIZE']:
                    n_transitions += len(training_buffer)

                    # Indices : state:0 ; action:1 ; reward:2 ; next_state:3 ; gae:4

                    pol_summaries = None
                    v_summaries = None
                    batch_end = None

                    random.shuffle(training_buffer)

                    for i_epoch in range(HP['K_EPOCHS']):
                        for batch_start in range(0, len(training_buffer), HP['BATCH_SIZE']):
                            batch_end = min(batch_start + HP['BATCH_SIZE'], len(training_buffer))
                            batch = training_buffer[batch_start:batch_end]

                            i_policy_iter += 1

                            pol_summaries = pol.train_policy(
                                batch.states,
                                batch.actions,
                                batch.gae_values
                            )
                            writer.add_summary(pol_summaries, i_policy_iter)

                            if i_epoch < HP['K_EPOCHS_VS']:
                                v_summaries = vs.train(
                                    batch.states,
                                    batch.rewards,
                                    batch.next_states
                                )
                                writer.add_summary(v_summaries, i_policy_iter)

                    # LOGGING
                    if eval_buffer:
                        eval_summary = ev_summ_creator.create_summary(eval_buffer.rewards, eval_buffer.aux_info)
                        writer.add_summary(eval_summary, i_policy_iter)
                        cells = [i_policy_iter,RUN_ID, PARAM_SET] + ['']*100
                        cells[PARAM_SET+3] = np.mean(eval_buffer.rewards)
                        with open('/workspace/total_eval_summary.csv', 'a') as file:
                            file.write(';'.join([str(c) for c in cells])+'\n')

                    writer.add_summary(state_logger.log(training_buffer.states), i_policy_iter)
                    #writer.add_summary(state_logger_norm.log(norm_next_state), i_epochs)
                    time_now = time()
                    writer.add_summary(t_summary_creator.create_summary(training_buffer.rewards, time_now - last_time), n_transitions)
                    last_time = time_now
                    #SOC_REG_SCHEDULER.x = n_transitions

                    print('Total Transitions:  ', n_transitions)
                    #print('SOC regularization: ', SOC_REG_SCHEDULER.get_schedule_value())
                    print('Transitions wasted: ', len(training_buffer) - batch_end, ' / ', len(training_buffer))

                    print('Save model')
                    saver.save(sess, model_path+f"{RUN_ID}")

                    pol.update_old_policy()
                    training_buffer.clear()
