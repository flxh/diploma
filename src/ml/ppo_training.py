import tensorflow as tf
from datetime import datetime
import numpy as np
import random
from time import time
from multiprocessing import Process, Queue
import os
import matplotlib
#matplotlib.use('Agg')
from threading import Thread

from ml.EpisodeCreator import EpisodeCreator
from ml.TBTrainingLogger import TrainingSummaryCreator
from ml.utils import drain
from ml.PpoCore import Policy, StateValueApproximator
from simulation.Environment import Environment, INFO_HEADER
from timeseriesprediction.utils import load_total_power_from_mat_file
from ml.utils import load_irraditaion_data
from ml.TBStateLogger import StateLogger

from traceback import print_exc


random.seed(2)
np.random.seed(2)

# HYPERPARAMETERS
BATCH_SIZE = 8192
#HORIZON = 16#64
#N_WORKERS_PERPROCESS = 64
#N_PROCESS = 8
#MIN_WORKERS_DONE = 8

N_WORKERS = 512
ENV_STEPS = 256

TAIL_LEN = 96
EPISODE_LEN = 29

SOC_REGULARIZATION = 4.0

BETA = 0.0
LR_POLICY = 3e-5
LR_VS = 1e-4

# These hyperparameters can be left as they are
KERNEL_REG = 1e-5
EPSILON = 0.2 # KL-Divergence Clipping as per recommendation
GAMMA = 0.9
LAMBDA = 0.95
LOG_VAR_INIT = -0.8

K_EPOCHS = 5
load_model = False
tb_verbose = True

model_path = "./model_safe/model_150_test"


class WorkerProcess(Process):

    def __init__(self, state_send_queue, action_receive_queue, workers, id):
        Process.__init__(self)
        self.state_send = state_send_queue
        self.action_receive = action_receive_queue

        self.workers = workers
        self.id = id
        self.current_policy_iteration = -1

    def run(self):
        while True:
            states = [w.state for w in self.workers]
            self.state_send.put((self.id, states))
            actions, policy_iteration = self.action_receive.get()

            # dropping old transitions collected from old policy iteration, because PPO assumes that all transitions
            # for training are collected under current policy
            if policy_iteration != self.current_policy_iteration:
                n_dropped = 0
                for w in self.workers:
                    n_dropped += len(w.temp_buffer)
                    w.temp_buffer = []
                print('{} transitions dropped from proc {}; {} -> {}'.format(n_dropped, self.id, self.current_policy_iteration, policy_iteration))
                self.current_policy_iteration = policy_iteration

            for a, w in zip(actions, self.workers):
                w.step(a)


class Worker:
    def __init__(self, episode_queue):
        self.temp_buffer = []
        self.episode_queue = episode_queue

        self.env = None
        self.agent_id = None
        self.state = None

        self.done = False

        self.start_episode()
        self.episode_reward = 0

    def start_episode(self):
        episode_container = self.episode_queue.get()
        self.env = Environment(TAIL_LEN, episode_container, soc_reward=SOC_REGULARIZATION)

        self.temp_buffer = []
        self.episode_reward = 0

        try:
            self.state = self.env.reset()
            self.done = False
        except Exception as e:
            print("Error during boot up phase - starting new episode")
            print_exc()
            self.start_episode()

    def step(self, action):
        try:
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward

        except Exception as e:
            print_exc()
            print("Error trying to step environment - starting new episode")
            self.start_episode()
            return None
        self.temp_buffer.append((self.state, action, reward, next_state))

        self.state = next_state

        if done:
            print('done')
            self.done = True


def calculate_gae(transitions, vs):
    '''
    weighs between TD(0) and TD(1) (normal advantage estimate)
    additionally weighs between 0 and inf. time horizon of empirical rewards  (see Form. 14 GAE)
    :param transitions:
    :param vs:
    :return:
    '''
    states = np.array([s[0] for s in transitions])
    rewards = np.reshape([s[2] for s in transitions], [-1 ,1])
    next_states = np.array([s[3] for s in transitions])

    state_values, _ = vs.predict(states)
    next_state_values, _ = vs.predict(next_states)

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


class Evaluation:
    def __init__(self, episode_container, pol, sess, writer, temp_folder):
        self.episode_container = episode_container
        self.pol = pol
        self.sess = sess
        self.writer = writer
        self.temp_folder = temp_folder

        self.n_evals = 0
        self.failed_attemps = 0

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

        self.single_evalutation(baseline_run=True)

    def single_evalutation(self, baseline_run=False):
        env = Environment(TAIL_LEN, self.episode_container)
        state = env.reset()
        done = False
        rewards = []
        infos = []

        while not done:
            if baseline_run:
                action = [0]
            else:
                actions, _, _ = self.pol.sample_action([state])
                action = actions[0]

            next_state, reward, done, info = env.step(action)
            rewards.append(reward)

            with open(self.temp_folder+'/eval_{}.csv'.format(self.n_evals), 'a') as file:
                file.write(('{};'*len(info)+'\n').format(*info))

            infos.append(info)
            state = next_state

        summaries, _ = self.sess.run([self.summary_op, self.reward_mean], feed_dict={
            self.rewards_ph: rewards,
            self.info_ph: np.array(infos)
        })
        self.writer.add_summary(summaries, self.n_evals)
        self.n_evals += 1

    def run_evaluation(self):
        while self.failed_attemps < 5:
            try:
                self.single_evalutation()
                self.failed_attemps = 0
            except Exception:
                self.failed_attemps += 1
                print_exc()
                print("Error during evaluation run")
        print('Five failed attempts. Exiting')


state_send_queue = Queue()
transition_queue = Queue()
episode_queue = Queue(maxsize=N_WORKERS)
action_receive_queues = {}
batch_buffer = []
horizon_buffer = []


load_data = load_total_power_from_mat_file('../../loadprofiles_1min.mat', 150, 240, [1,  3,  9, 14, 17, 20, 21, 25, 27, 29, 33, 39, 43, 44, 51, 57, 67, 73]) # multiple time series
irradiation_data = load_irraditaion_data('../../ihm-daten_20252.csv', 150, 240) *-1
assert len(load_data) * 5 == len(irradiation_data) * 18

# buy_price_data = [0.35]*96
# sell_price_data = [0.09]*96

buy_price_data = [1.]*96
sell_price_data = [0.]*96

print('start episode loader')
episode_loader = EpisodeCreator(episode_queue, load_data, irradiation_data, buy_price_data, sell_price_data, EPISODE_LEN)
episode_proc = Process(target=episode_loader.fill_queue, name="episode_loader")
episode_proc.start()

eval_episode = episode_loader.create_evaluation_episode(180)

state_dim = 3
action_dim = 1



with tf.Session() as sess:
    now = datetime.now()

    vs = StateValueApproximator(sess, state_dim, LR_VS, GAMMA, KERNEL_REG)
    pol = Policy(sess, state_dim, action_dim, LR_POLICY, BETA, LOG_VAR_INIT, EPSILON, KERNEL_REG)
    state_logger = StateLogger(sess, ['SOC', 'LOAD', 'PV'], "state")
    state_logger_norm = StateLogger(sess,  ['SOC', 'LOAD', 'PV'], "norm_state")
    t_summary_creator = TrainingSummaryCreator(sess)

    experiment_name = "{}-a:{}-lr:{}".format(now.strftime('%Y-%m-%dT%H:%M:%S'), BETA, LR_POLICY)
    temp_folder = "/tmp/{}".format(experiment_name)

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    writer = tf.summary.FileWriter("/home/florus/tb-diplom/"+experiment_name, sess.graph)
    print(experiment_name)

    '''
    print('build workers')
    for p_id in range(N_PROCESS):
        workers = [Worker(episode_queue, transition_queue) for _ in range(N_WORKERS_PERPROCESS)]

        action_receive_queue = Queue()
        action_receive_queues[p_id] = action_receive_queue
        wp = WorkerProcess(state_send_queue, action_receive_queue, workers, p_id)
        wp.start()
    '''

    workers = [Worker(episode_queue) for _ in range(N_WORKERS)]

    evaluator = Evaluation(eval_episode, pol, sess, writer, temp_folder)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver = tf.train.Saver()
    if load_model:
        saver.restore(sess, model_path)

    n_transitions = 0
    i_policy_iter = 0

    t_eval = Thread(target=evaluator.run_evaluation)
    t_eval.start()
    print('start training')
    last_time = time()
    while True:

        # collect states from worker processes
        process_order = []

        for _ in range(ENV_STEPS):
            states = [w.state for w in workers]

            _, actions, policy_iteration = pol.sample_action(states)

            for idx, w in enumerate(workers):
                w.step(actions[idx])
                if w.done:
                    horizon_buffer.append(w.temp_buffer)
                    w.temp_buffer =[]
                    w.start_episode()

        for w in workers:
            if w.temp_buffer:
                horizon_buffer.append(w.temp_buffer)
            w.temp_buffer = []
        '''
        for _ in range(MIN_WORKERS_DONE):
            q_item = state_send_queue.get()
            states.extend(q_item[1])
            process_order.append(q_item[0])

        for q_item in drain(state_send_queue):
            states.extend(q_item[1])
            process_order.append(q_item[0])

        # sample action
        _, actions, policy_iteration = pol.sample_action(states)

        # send actions to worker processes
        for idx, proc in enumerate(process_order):
            process_idx_start = idx*N_WORKERS_PERPROCESS
            process_idx_end = (idx+1)*N_WORKERS_PERPROCESS
            action_receive_queues[proc].put((actions[process_idx_start:process_idx_end], policy_iteration))
        '''

        total_transitions = 0

        # calculate gae values from transitions and extend training buffer
        for transitions in horizon_buffer:
            total_transitions += len(transitions)
            gae_vals = calculate_gae(transitions, vs)
            batch_buffer.extend(np.concatenate((transitions, gae_vals), axis=1))

        print('Average horizon length ', total_transitions / len(horizon_buffer))
        horizon_buffer = []

        if len(batch_buffer) >= BATCH_SIZE:
            n_transitions += len(batch_buffer)
            batch_buffer = np.array(batch_buffer)

            # Indexes : state:0 ; action:1 ; reward:2 ; next_state:3 ; gae:4

            pol_summaries = None
            v_summaries = None
            batch_end = None

            for _ in range(K_EPOCHS):
                for i_batch in range(int(np.ceil(len(batch_buffer) / BATCH_SIZE))):
                    batch_start = i_batch * BATCH_SIZE
                    batch_end = min((i_batch + 1) * BATCH_SIZE, len(batch_buffer))
                    i_policy_iter += 1

                    pol_summaries = pol.train_policy(
                        np.stack(batch_buffer[batch_start:batch_end, 0], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 1], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 4], axis=0)
                    )

                    v_summaries = vs.train(
                        np.stack(batch_buffer[batch_start:batch_end, 0], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 2], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 3], axis=0)
                    )

                    writer.add_summary(pol_summaries, i_policy_iter)
                    writer.add_summary(v_summaries, i_policy_iter)

            print(len(batch_buffer) - batch_end, ' / ', len(batch_buffer))

            writer.add_summary(state_logger.log(np.stack(batch_buffer[:, 3], axis=0)), i_policy_iter)
            #writer.add_summary(state_logger_norm.log(norm_next_state), i_epochs)
            time_now = time()
            writer.add_summary(t_summary_creator.create_summary(np.reshape(batch_buffer[:, 2], [-1]), time_now - last_time), n_transitions)
            last_time = time_now

            if i_policy_iter % 1000 < K_EPOCHS:
                saver.save(sess, model_path)

            pol.update_old_policy()
            batch_buffer = []
