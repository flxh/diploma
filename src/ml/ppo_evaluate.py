import tensorflow.compat.v1 as tf


from datetime import datetime
import numpy as np
import random
import pickle as pkl

from ml.PpoCore import Policy, StateValueApproximator
from simulation.Environment import Environment
from ml.ppo_training import EvaluationWorker
from ml.utils import LinearScheduler

random.seed(2)
np.random.seed(2)

# HYPERPARAMETER
TAIL_LEN = 96
CELLS = 150
GRADIENT_NORM = 10.
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

model_path = '/workspace/models/6850676_0'

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
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
with tf.Session(config=config) as sess:

    pol = Policy(sess, state_dim, action_dim, LR_POLICY, TAIL_LEN, CELLS, BETA, LOG_VAR_INIT, EPSILON, KERNEL_REG, GRADIENT_NORM)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print('models initialized')

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print('starting evaluation run')
    eval_workers = [EvaluationWorker(ep) for ep in eval_episodes]

    for e in eval_workers:
        e.start_episode()

    steps = 0
    while not all([e.done for e in eval_workers]):
        steps +=1

        states = [e.state for e in eval_workers]

        actions, _, _ = pol.sample_action(states)

        for e, a in zip(eval_workers, actions):
            if e.done:
                continue
            e.step(a)

        if steps % 100 == 0 or all([e.done for e in eval_workers]):
            print(steps)

            for i, ew in enumerate(eval_workers):
                with open('./evaluations/{}.csv'.format(i), 'a') as file:
                    for ai in ew.temp_buffer.aux_info:
                        file.write(('{};'*len(ai)+'\n').format(*ai))
