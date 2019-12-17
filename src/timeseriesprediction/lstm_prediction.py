import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from timeseriesprediction.utils import get_power_differences, standardize_power_differences, load_total_power_from_mat_file
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

power_vals = load_total_power_from_mat_file('loadprofiles_1min.mat')
p_diffs = get_power_differences(power_vals)
p_diffs_norm = standardize_power_differences(p_diffs)

X = []
y = []

ts = p_diffs_norm[:, 10].reshape((-1,))
for i in range(SEQU_LEN, len(ts)-1):
    X.append(ts[i-SEQU_LEN:i])
    y.append(ts[i+1])

X = np.array(X).reshape((-1, SEQU_LEN, 1))
y = np.array(y).reshape((-1,1))


with tf.Session() as sess:

    lstm = LSTMModel(sess, SEQU_LEN, FEATURES, CELLS, lr=LR)
    sess.run(tf.global_variables_initializer())

    k = TimeSeriesSplit(n_splits=SPLITS)

    with open('lstm_single_iteration.csv', 'a') as iterfile:
        split_number = 0
        for train_index, test_index in k.split(X):
            sess.run(tf.global_variables_initializer())
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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

            y_test = y_test.reshape((-1,))
            with open('lstm_prediction.csv', 'a') as file:
                for i in range(len(y_predict)):
                    file.write('{};{}\n'.format(y_test[i],y_predict[i]))


