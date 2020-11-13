import tensorflow.compat.v1 as tf


from datetime import datetime
import numpy as np
import random
import pickle as pkl

from ml.PpoCore import Policy, StateValueApproximator
from simulation.Environment import Environment
from ml.utils import LinearScheduler

random.seed(2)
np.random.seed(2)

# HYPERPARAMETERS
BATCH_SIZE = 4096

N_WORKERS = 256
ENV_STEPS = 256

TAIL_LEN = 96

SOC_REG_SCHEDULER = LinearScheduler(3., 3., 30e6)
SOC_REG_SCHEDULER.x = 0

BETA = 0.0
LR_POLICY = 5e-5
LR_VS = 8e-5

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

eval_episode_name = 'eval_episode8'

model_path = './model_safe (copy)/model_150_test'

eval_episode = pkl.load(open('./{}.pkl'.format(eval_episode_name), 'rb'))

state_dim = 4
action_dim = 1

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
with tf.Session(config=config) as sess:
    now = datetime.now()

    vs = StateValueApproximator(sess, state_dim, LR_VS, GAMMA, KERNEL_REG)
    pol = Policy(sess, state_dim, action_dim, LR_POLICY, BETA, LOG_VAR_INIT, EPSILON, KERNEL_REG)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print('models initialized')

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print('starting evaluation run')
    env = Environment(TAIL_LEN, eval_episode, DT_EVAL_STEP, sim_steps_per_action=EVAL_STEPS_PER_ACTION)
    state = env.reset()
    done = False
    rewards = []
    infos = []
    steps =0

    while not done:
        steps +=1
        if steps % 1000 == 0:
            print(steps)
        actions, _, _ = pol.sample_action([state])
        action = actions[0]

        next_state, reward, done, info = env.step(action)
        rewards.append(reward)

        with open('./evaluations/{}.csv'.format(eval_episode_name), 'a') as file:
            file.write(('{};'*len(info)+'\n').format(*info))

        infos.append(info)
        state = next_state

    print(np.mean(rewards))