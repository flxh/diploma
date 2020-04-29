import tensorflow.compat.v1 as tf

from datetime import datetime
import numpy as np
import random
from time import time
from multiprocessing import Process, Queue
import os
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from threading import Thread
import pickle as pkl

from ml.EpisodeCreator import EpisodeCreator
from ml.TBTrainingLogger import TrainingSummaryCreator
from ml.PpoCore import Policy, StateValueApproximator
from simulation.Environment import Environment, INFO_HEADER
from timeseriesprediction.utils import load_total_power_from_mat_file
from ml.utils import load_irraditaion_data, LinearScheduler
from ml.TBStateLogger import StateLogger

from traceback import print_exc


random.seed(2)
np.random.seed(2)

# HYPERPARAMETERS
BATCH_SIZE = 4096

N_WORKERS = 256
ENV_STEPS = 256

TAIL_LEN = 96

SOC_REGULARIZATION = 2.5
SOC_REG_SCHEDULER = LinearScheduler(2.5, 1.6, 16e6)
SOC_REG_SCHEDULER.x = 0

BETA = 0.0
LR_POLICY = 3e-5
LR_VS = 1e-4

# These hyperparameters can be left as they are
KERNEL_REG = 1e-5
EPSILON = 0.2 # KL-Divergence Clipping as per recommendation
GAMMA = 0.995
LAMBDA = 0.98
LOG_VAR_INIT = -0.5

DT_WORKER_STEP = 180
DT_EVAL_STEP = 60

WORKER_STEPS_PER_DAY = 20*24
EVAL_STEPS_PER_ACTION = 15
WORKER_STEPS_PER_ACTION = 5
WORKER_EPISODE_STEPS = 47 * WORKER_STEPS_PER_DAY

K_EPOCHS = 6
K_EPOCHS_VS = 4


load_model = False
tb_verbose = True

if load_model: print('LOADING MODEL')

model_path = './model_safe/model_150_test' # leave unchanged !

tensor_board_path = '/home/florus/tb-diplom/' # used to store Tensorboard files
temp_path = '/tmp/' # used to save trajectory CSV files for each evaluation run


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
        self.env = Environment(TAIL_LEN, episode_container, DT_WORKER_STEP,soc_reward=SOC_REG_SCHEDULER.get_schedule_value(), soc_initial=np.random.random(), sim_steps_per_action=WORKER_STEPS_PER_ACTION)

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
        print('starting evaluation run')
        env = Environment(TAIL_LEN, self.episode_container, DT_EVAL_STEP, sim_steps_per_action=EVAL_STEPS_PER_ACTION)
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


episode_queue = Queue(maxsize=N_WORKERS)
batch_buffer = []
horizon_buffer = []

load_data = load_total_power_from_mat_file('../../loadprofiles_1min.mat', 0, 365, [1, 11, 17, 25, 26, 27, 29, 44, 46, 47, 51, 54, 56, 57, 59, 60, 66, 67, 70, 71, 72, 73]) # multiple time series
irradiation_data = load_irraditaion_data('../../ihm-daten_20252.csv', 0, 365) *-1
assert len(load_data) * 5 == len(irradiation_data) * 22

year_cycle = [-np.cos(((x+WORKER_STEPS_PER_DAY*10)/(WORKER_STEPS_PER_DAY*365))*2*np.pi) for x in range(WORKER_STEPS_PER_DAY*365)]
buy_price_data = [1.]*WORKER_STEPS_PER_DAY
sell_price_data = [0.]*WORKER_STEPS_PER_DAY

print('start episode loader')
episode_loader = EpisodeCreator(episode_queue, load_data, irradiation_data, year_cycle, buy_price_data, sell_price_data)
episode_proc = Process(target=episode_loader.fill_queue, args=(WORKER_EPISODE_STEPS,), name="episode_loader")
episode_proc.start()

eval_episode = pkl.load(open('eval_episode.pkl', 'rb'))

plt.plot(range(2*365*WORKER_STEPS_PER_DAY), load_data[:2*365*WORKER_STEPS_PER_DAY])
plt.plot(range(2*365*WORKER_STEPS_PER_DAY), irradiation_data[:2*365*WORKER_STEPS_PER_DAY])
plt.show()

state_dim = 4
action_dim = 1

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
with tf.Session(config=config) as sess:
    now = datetime.now()

    vs = StateValueApproximator(sess, state_dim, LR_VS, GAMMA, KERNEL_REG)
    pol = Policy(sess, state_dim, action_dim, LR_POLICY, BETA, LOG_VAR_INIT, EPSILON, KERNEL_REG)
    state_logger = StateLogger(sess, ['SOC', 'LOAD', 'PV', 'CYCLE'], "state")
    state_logger_norm = StateLogger(sess,  ['SOC', 'LOAD', 'PV', 'CYCLE'], "norm_state")
    t_summary_creator = TrainingSummaryCreator(sess)

    experiment_name = "{}-a:{}-lr:{}-sr:{}".format(now.strftime('%Y-%m-%dT%H:%M:%S'), BETA, LR_POLICY, SOC_REGULARIZATION)
    temp_folder = temp_path + experiment_name

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    writer = tf.summary.FileWriter(tensor_board_path+experiment_name, sess.graph)
    print(experiment_name)

    workers = [Worker(episode_queue) for _ in range(N_WORKERS)]
    print('wokers created')
    evaluator = Evaluation(eval_episode, pol, sess, writer, temp_folder)
    print('evaluator created')

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print('models initialized')

    saver = tf.train.Saver()
    if load_model:
        saver.restore(sess, model_path)

    n_transitions = 0
    i_policy_iter = 0

    print('start training')
    t_eval = Thread(target=evaluator.run_evaluation)
    t_eval.start()
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

        total_transitions_training = 0

        # calculate gae values from transitions and extend training buffer
        for transitions in horizon_buffer:
            total_transitions_training += len(transitions)
            gae_vals = calculate_gae(transitions, vs)
            batch_buffer.extend(np.concatenate((transitions, gae_vals), axis=1))

        print('Average horizon length ', total_transitions_training / len(horizon_buffer))
        horizon_buffer = []

        if len(batch_buffer) >= BATCH_SIZE:
            n_transitions += len(batch_buffer)
            batch_buffer = np.array(batch_buffer)

            # Indexes : state:0 ; action:1 ; reward:2 ; next_state:3 ; gae:4

            pol_summaries = None
            v_summaries = None
            batch_end = None

            for i_epoch in range(K_EPOCHS):
                for i_batch in range(int(np.ceil(len(batch_buffer) / BATCH_SIZE))):
                    batch_start = i_batch * BATCH_SIZE
                    batch_end = min((i_batch + 1) * BATCH_SIZE, len(batch_buffer))
                    i_policy_iter += 1

                    pol_summaries = pol.train_policy(
                        np.stack(batch_buffer[batch_start:batch_end, 0], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 1], axis=0),
                        np.stack(batch_buffer[batch_start:batch_end, 4], axis=0)
                    )
                    writer.add_summary(pol_summaries, i_policy_iter)

                    if i_epoch < K_EPOCHS_VS:
                        v_summaries = vs.train(
                            np.stack(batch_buffer[batch_start:batch_end, 0], axis=0),
                            np.stack(batch_buffer[batch_start:batch_end, 2], axis=0),
                            np.stack(batch_buffer[batch_start:batch_end, 3], axis=0)
                        )
                        writer.add_summary(v_summaries, i_policy_iter)

            writer.add_summary(state_logger.log(np.stack(batch_buffer[:, 3], axis=0)), i_policy_iter)
            #writer.add_summary(state_logger_norm.log(norm_next_state), i_epochs)
            time_now = time()
            writer.add_summary(t_summary_creator.create_summary(np.reshape(batch_buffer[:, 2], [-1]), time_now - last_time), n_transitions)
            last_time = time_now
            SOC_REG_SCHEDULER.x = n_transitions

            print('Total Transitions:  ', n_transitions)
            print('SOC regularization: ', SOC_REG_SCHEDULER.get_schedule_value())
            print('Transitions wasted: ',len(batch_buffer) - batch_end, ' / ', len(batch_buffer))

            print('Save model')
            saver.save(sess, model_path)

            pol.update_old_policy()
            batch_buffer = []
