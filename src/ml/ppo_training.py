import tensorflow.compat.v1 as tf

from abc import abstractmethod
import sys
from datetime import datetime
import numpy as np
import random
from time import time
from multiprocessing import Process, Queue
import os
from ml.TransitionBuffer import Transition, TransitionBuffer
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from threading import Thread
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

# HYPERPARAMETERS
BATCH_SIZE = 6144
#BATCH_SIZE = 128

N_WORKERS = 256
#N_WORKERS = 16
ENV_STEPS = 256
#ENV_STEPS = 16

TAIL_LEN = 96
CELLS = 100
GRADIENT_NORM = 30.

#SOC_REG_SCHEDULER = LinearScheduler(0., 0., 30e6)
#SOC_REG_SCHEDULER.x = 0

SOC_REG = 0.

BETA = 0.0
LR_POLICY = 5e-5
LR_VS = 8e-5

# These hyperparameters can be left as they are
#KERNEL_REG = 1e-5
EPSILON = 0.2 # KL-Divergence Clipping as per recommendation
GAMMA = 0.995
LAMBDA = 0.98
LOG_VAR_INIT = -0.5

DT_WORKER_STEP = 60
WORKER_STEPS_PER_ACTION = 15

K_EPOCHS = 6
K_EPOCHS_VS = 4

QUEUE_SIZE = 10

load_model = False
tb_verbose = True

model_path = '/workspace/models/' # leave unchanged !
tensor_board_path = '/workspace/tensorboard/' # used to store Tensorboard files
temp_path = '/workspace/trajectories/' # used to save trajectory CSV files for each evaluation run
runs_file = '/workspace/runs.csv'

for p in [model_path, tensor_board_path, temp_path]:
    if not os.path.exists(p) or not os.path.isdir(p):
        raise IOError(f"Workspace dir {p} does not exist.\nIs /workspace mounted? Are subdirectories created?")

RUN_ID = sys.argv[1]
if len(sys.argv) > 2:
    params_set = int(sys.argv[2])
    params = np.genfromtxt(f'{sys.argv[3]}', delimiter=';', skip_header=1)
    SOC_REG = params[params_set,0]
    LR_POLICY = params[params_set,1]
    LR_VS = params[params_set,1] *1.5
    EPSILON = params[params_set,2]
    LAMBDA = params[params_set,3]
    GAMMA = params[params_set,4]

for x in zip(['SOC_REG','LR_POLICY','LR_VS', 'EPSILON', 'LAMBDA', 'GAMMA'],[SOC_REG,LR_POLICY,LR_VS, EPSILON, LAMBDA, GAMMA]):
    print('{:.<12}{}'.format(*x))


class Worker:
    def __init__(self):
        self.temp_buffer = TransitionBuffer()

        self.env = None
        self.state = None

        self.done = True

    @abstractmethod
    def _next_episode(self):
        raise NotImplementedError

    @abstractmethod
    def _save_transition(self, s, a, r, sn, d, aux):
        raise NotImplementedError

    def start_episode(self):
        episode_container = self._next_episode()
        self.env = Environment(TAIL_LEN, episode_container, DT_WORKER_STEP,soc_reward=SOC_REG, soc_initial=np.random.random(), sim_steps_per_action=WORKER_STEPS_PER_ACTION)

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

    td_residuals = rewards + GAMMA * next_state_values - state_values

    gae_values = []
    last_gea = 0

    # see GAE paper Schulman formula 15
    for tdr in reversed(td_residuals):
        gae = tdr + LAMBDA * GAMMA * last_gea
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


if __name__ == '__main__':
    # initialization of the buffers and the episode queue
    episode_queue = Queue(maxsize=QUEUE_SIZE)
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

        vs = StateValueApproximator(sess, state_dim, LR_VS, TAIL_LEN, CELLS, GAMMA, KERNEL_REG)
        pol = Policy(sess, state_dim, action_dim, LR_POLICY, TAIL_LEN, CELLS,  BETA, LOG_VAR_INIT, EPSILON, KERNEL_REG, GRADIENT_NORM)
        state_logger = StateLogger(sess, ['SOC', 'LOAD', 'PV', 'CYCLE'], "state")
        state_logger_norm = StateLogger(sess,  ['SOC', 'LOAD', 'PV', 'CYCLE'], "norm_state")
        t_summary_creator = TrainingSummaryCreator(sess)

        if not os.path.isfile(runs_file):
            with open(runs_file, 'a') as file:
                file.write(f'RUN_ID;SOC_REG;LR_POLICY;LR_VS;EPSILON;LAMBDA;GAMMA\n')

        with open(runs_file, 'a') as file:
            file.write(f'{RUN_ID};{SOC_REG};{LR_POLICY};{LR_VS};{EPSILON};{LAMBDA};{GAMMA}\n')

        #experiment_name = "{}-a{}-lr{}-sr{}".format(now.strftime('%Y-%m-%dT%H%M%S'), BETA, LR_POLICY, SOC_REG_SCHEDULER.get_schedule_value())
        temp_folder = temp_path + f"{RUN_ID}"

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        writer = tf.summary.FileWriter(tensor_board_path+f"{RUN_ID}", sess.graph)
        # print("RUN META")
        print(RUN_ID)

        tr_workers = [TrainingWorker(episode_queue) for _ in range(N_WORKERS)]
        #ev_workers = [EvaluationWorker(eep) for eep in eval_episodes[:64]]
        ev_workers = []
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
            for _ in range(ENV_STEPS):
                states = []
                for w in tr_workers + ev_workers:
                    if w.done: w.start_episode()
                    states.append(w.state)

                determ_actions, actions, policy_iteration = pol.sample_action(states)

                tr_actions = actions[:N_WORKERS]
                ev_actions = determ_actions[N_WORKERS:]

                for idx, w in enumerate(ev_workers):
                    w.step(ev_actions[idx])

                for idx, w in enumerate(tr_workers):
                    w.step(tr_actions[idx])
                    if (w.done and w.temp_buffer) or len(w.temp_buffer) >= ENV_STEPS:
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

            if len(training_buffer) >= BATCH_SIZE:
                n_transitions += len(training_buffer)

                # Indices : state:0 ; action:1 ; reward:2 ; next_state:3 ; gae:4

                pol_summaries = None
                v_summaries = None
                batch_end = None

                for i_epoch in range(K_EPOCHS):

                    random.shuffle(training_buffer)

                    for batch_start in range(0, len(training_buffer), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(training_buffer))
                        batch = training_buffer[batch_start:batch_end]

                        i_policy_iter += 1

                        pol_summaries = pol.train_policy(
                            batch.states,
                            batch.actions,
                            batch.gae_values
                        )
                        writer.add_summary(pol_summaries, i_policy_iter)

                        if i_epoch < K_EPOCHS_VS:
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
