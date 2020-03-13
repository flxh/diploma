import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARMA
import warnings
import pandas as pd
import os
warnings.simplefilter('ignore', FutureWarning)

class LSTMModel:
    def __init__(self, sess, sequence_len, n_features, n_cells, lr=1e-3):
        self.sess = sess

        self.input_sequences = tf.placeholder(tf.float32, [None, sequence_len, n_features])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        with tf.variable_scope('model'):
            lstm = tf.nn.rnn_cell.LSTMCell(n_cells)

            state = lstm.zero_state(tf.shape(self.input_sequences)[0], dtype=tf.float32)
            output = None
            for i in range(sequence_len):
                output, state = lstm(self.input_sequences[:, i], state)

            self.prediction_op = tf.layers.Dense(1, activation="linear")(output)

        self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

        self.loss_op = tf.reduce_mean((self.targets - self.prediction_op) ** 2)
        optimizer = tf.train.AdamOptimizer(lr)
        self.training_op = optimizer.minimize(self.loss_op)

    def train(self, X, y):
        loss, _, vars = self.sess.run([self.loss_op, self.training_op, self.variables], feed_dict={
            self.input_sequences: X,
            self.targets:y
        })
        return loss

    def predict(self, X):
        return self.sess.run(self.prediction_op, feed_dict={
            self.input_sequences: X
        })


BATCH_SIZE = 16000
SEQU_LEN = 100
EPOCHS = 500
CELLS = 30
FEATURES = 1
LR = 3.3e-4
SPLITS = 10

mat_contents = sio.loadmat('loadprofiles_1min.mat')
pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
pges = np.mean(np.reshape(pges, (-1,15,74)), axis=1)  #zetiliche Auflüsung verringern
p_diffs = np.diff(pges, axis=0)
# p_diffs = pges
stdx = np.std(p_diffs, axis=0) ##############

p_diffs_norm = (p_diffs - np.mean(p_diffs, axis=0)) / np.std(p_diffs, axis=0)

X = []
y = []
y_season=[]

season = 4*24

'''
ts = p_diffs_norm[:364*96, :].reshape((-1,))
with open('zero_change_prediction_10.csv', 'a') as file:
    for yt in ts:
        file.write('{};{}\n'.format(yt, 0))

k = TimeSeriesSplit(n_splits=SPLITS)

y_test_season = []
y_predict_season = []
for i in range(2000, 5000, 10):
    ts_train = ts[:i*96]
    ts_test = ts[i*96: (i+10)*96]

    model = np.mean(ts_train.reshape((96,-1)), axis=1)
    y_predict_season.extend(np.tile(model, 10))
    y_test_season.extend(ts_test)

with open('season_prediction_10.csv', 'a') as file:
    for yt, yp in zip(y_test_season, y_predict_season):
        file.write('{};{}\n'.format(yt, yp))

print('Season Model:')
print(mean_squared_error(y_test_season, y_predict_season))
print(r2_score(y_test_season, y_predict_season))
print(mean_absolute_error(y_test_season,y_predict_season))
print(np.mean(np.sign(y_test_season)==np.sign(y_predict_season)))


'''




ts = p_diffs_norm[:, 10].reshape((-1,))
for i in range(SEQU_LEN, len(ts)-1):
    X.append(ts[i-SEQU_LEN:i])
    y.append(ts[i+1])
    y_season.append(ts[i+1-season])

X = np.array(X).reshape((-1, SEQU_LEN, 1))
y = np.array(y).reshape((-1,1))
y_season = np.array(y_season).reshape((-1,1))
'''
print('dataset built')
print(len(ts))

y_predict_arima = []
y_test_arima = []

n_steps_predict = 5

for i in range(10000, len(ts), n_steps_predict):
    ts_train, ts_test = ts[:i], ts[i:i+n_steps_predict]

    model = ARMA(ts_train, order=(5,0))
    model_fit = model.fit(disp=False)
    prediction = model_fit.forecast(len(ts_test))

    y_test_arima.extend(ts_test)
    y_predict_arima.extend(prediction[0])
    print(r2_score(y_test_arima, y_predict_arima))
    print(mean_squared_error(y_test_arima, y_predict_arima))

    with open('arima_prediction_10.csv', 'a') as file:
        for k in range(len(ts_test)):
            file.write('{};{}\n'.format(ts_test[k], prediction[0][k]))

'''

with tf.Session() as sess:

    lstm = LSTMModel(sess, SEQU_LEN, FEATURES, CELLS, lr=LR)
    sess.run(tf.global_variables_initializer())

    k = TimeSeriesSplit(n_splits=SPLITS)

    with open('lstm_single_iteration_10.csv', 'a') as iterfile:
        split_number = 0
        for train_index, test_index in k.split(X):
            sess.run(tf.global_variables_initializer())
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_predict_season = y_season[test_index]

            split_number += 1
            print('Split No.:', split_number)
            iterfile.write('Split No.: {}'.format(split_number))


            y_predict = None
            for i_epoch in range(EPOCHS):
                for i in range(1, int(len(X_train)/BATCH_SIZE)+2):
                    X_batch = X_train[BATCH_SIZE*(i-1):min(len(X_train), BATCH_SIZE*i)]
                    y_batch = y_train[BATCH_SIZE*(i-1):min(len(X_train), BATCH_SIZE*i)]
                    loss = lstm.train(X_batch, y_batch)
                y_predict = lstm.predict(X_test).reshape((-1,))
                print(r2_score(y_test,y_predict), ';', mean_squared_error(y_test, y_predict))
                iterfile.write('{};{}\n'.format(r2_score(y_test,y_predict), mean_squared_error(y_test, y_predict)))
                #print(r2_score(y_test,np.zeros(len(y_test))), ' - ', np.mean(np.sign(y_predict_season)==np.sign(y_test)))

            y_test = y_test.reshape((-1,))
            #with open('lstm_prediction2.csv', 'a') as file:
            #    for i in range(len(y_predict)):
                    #file.write('{};{}\n'.format(y_test[i],y_predict[i]))


