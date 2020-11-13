import tensorflow as tf
import tensorflow_probability as tfp

GRADIENT_NORM = 30
CELLS = 100
TAIL_LEN = 96

STATE_MEAN = tf.constant([ 1.6987270e-01,  3.3652806e+05, -4.7673375e+05, 0])

STATE_VAR = tf.constant([6.0812939e-02, 1.2899629e+11, 6.3592917e+11, 0.5])


class Policy:

    def __init__(self, sess, state_size, action_size, lr,
                 beta_entropy, log_var_init, epsilon, kernel_reg):
        self.sess = sess
        self.action_size = action_size
        self.kernel_reg = kernel_reg
        self.log_var_init = log_var_init

        self.policy_iteration = 0

        with tf.variable_scope("Policy"):
            # Platzhalter für die Eingaben in den Berechnungsgraph
            self.state_ph = tf.placeholder(tf.float32,
                                           [None, TAIL_LEN, state_size],
                                           name="state_ph")
            self.action_ph = tf.placeholder(tf.float32,
                                            [None, action_size],
                                            name="action_ph")
            self.advantage_ph = tf.placeholder(tf.float32, [None], name="adavantage_ph")

            self.norm_state = tf.nn.batch_normalization(self.state_ph,
                                                        STATE_MEAN,
                                                        STATE_VAR, None, None, 1e-12)

            self.norm_advantage = self.advantage_ph

            # Aktuelles Strategienetzwerk
            with tf.variable_scope("pi"):
                self.pi, self.mean_action = self._create_model(trainable=True,
                                                               input=self.norm_state,
                                                               n_cells=CELLS,
                                                               tail_len=TAIL_LEN)
                self.pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope="Policy/pi")
                self.sample_op = self.pi.sample()
                tf.summary.histogram('sampled_actions', self.sample_op)

            # Altes Strategienetzwerk der letzten Interation
            with tf.variable_scope("old_pi"):
                self.old_pi, _ = self._create_model(trainable=False,
                                                    input=self.norm_state,
                                                    n_cells=CELLS, tail_len=TAIL_LEN)
                self.old_pi_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/old_pi")

            # Kostenfunktion für die Optimierung
            with tf.variable_scope("loss"):
                prob_ratio = tf.exp(self.pi.log_prob(self.action_ph)
                                    - self.old_pi.log_prob(self.action_ph))
                tf.summary.histogram('prob_ratios', prob_ratio)

                surrogate = prob_ratio * self.norm_advantage
                clipped_surrogate = tf.minimum(surrogate,  tf.clip_by_value(
                    prob_ratio, 1.-epsilon, 1.+epsilon)*self.norm_advantage)

                self.pi_entropy = tf.reduce_mean(self.pi.entropy())
                self.surrogate = tf.reduce_mean(clipped_surrogate)

                tf.summary.scalar("entropy", self.pi_entropy)
                tf.summary.scalar('surrogate', self.surrogate)

                self.loss = -self.surrogate - beta_entropy * self.pi_entropy

                tf.summary.scalar("objective", self.loss)

            # Berechnung und Trimmen des Gradienten sowie Optimierer
            with tf.variable_scope("training"):
                self.gradients = tf.gradients(self.loss, self.pi_vars)
                self.gradients = [tf.clip_by_value(g, -1, 1) for g in self.gradients]
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, GRADIENT_NORM)
                grads = zip(self.gradients, self.pi_vars)
                optimizer = tf.train.RMSPropOptimizer(lr)

                self.optimize = optimizer.apply_gradients(grads)

                clipped_grads_tb = [tf.clip_by_value(g, -1e-8, 1e-8)
                                    for g in self.gradients]

            # Operation zum Kopieren der Modell Parameter in altes Strategienetzw.
            with tf.variable_scope("update_weights"):
                self.update_oldpi_op = [oldp.assign(p) for p,
                                            oldp in zip(self.pi_vars, self.old_pi_vars)]

        for g, v in zip(clipped_grads_tb, self.pi_vars):
            tf.summary.histogram(v.name.split(":")[0]+"_grad", g)

        self.summary_op = tf.summary.merge_all(scope="Policy")

    def _create_model(self, trainable, input, n_cells, tail_len):
        layer_names = ["lstm", "d1"]
        # Erstellung der LSTM-Schicht
        lstm = tf.nn.rnn_cell.LSTMCell(n_cells, name=layer_names[0],
                                       trainable=trainable)
        batch_size = tf.shape(input)[0]

        # Verknüpfung der sequentiellen Zeitschritte
        # mit definierter Länge
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        output = None
        for i in range(tail_len):
            output, state = lstm(input[:,i,:], state)

        # Ausgabe des Netzwerks = Mittelwert der Verteilung für das Sampling
        mu = tf.layers.Dense(self.action_size, activation="tanh", name=layer_names[1],
                             trainable=trainable,
                             kernel_initializer = tf.initializers.he_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                 scale=self.kernel_reg))(output)
        # log_sigma als trainierbarer Modellparameter
        log_sigma = tf.Variable(initial_value=tf.fill((self.action_size,),
                            self.log_var_init), trainable=False, name="log_sigma")

        distribution = tfp.distributions.MultivariateNormalDiag(loc=mu,
                                                    scale_diag=tf.exp(log_sigma))

        # Ausgaben für Tensorboard
        if trainable:
            tf.summary.histogram("log_sigma", log_sigma)
            tf.summary.histogram("mu", mu)
            mean_mu, var_mu = tf.nn.moments(tf.reshape(mu, (-1,)), axes=[0])
            tf.summary.scalar("mu_mean", mean_mu)
            tf.summary.scalar("mu_var", var_mu)

            for name in layer_names:
                with tf.variable_scope(name, reuse=True):
                    tf.summary.histogram("kernel", tf.get_variable("kernel"))
                    tf.summary.histogram("bias", tf.get_variable("bias"))

        return distribution, mu

    def sample_action(self, state):
        mean_action, sampled_action =  self.sess.run([self.mean_action, self.sample_op], feed_dict={
            self.state_ph: state
        })
        return mean_action, sampled_action, self.policy_iteration

    def train_policy(self, states, actions, advantages):
        _, summaries =self.sess.run([self.optimize, self.summary_op], feed_dict={
            self.state_ph:states,
            self.action_ph: actions,
            self.advantage_ph: advantages
        })
        self.policy_iteration += 1
        return summaries

    def update_old_policy(self):
        self.sess.run(self.update_oldpi_op)


class StateValueApproximator:
    def __init__(self, sess, state_size, lr, gamma, kernel_reg):
        self.kernel_reg = kernel_reg
        self.sess = sess

        self.gamma = tf.constant(gamma, dtype=tf.float32)

        with tf.variable_scope("V_s"):
            # Platzhalter für Eingabe in den Berechnungsgraph
            self.state_ph = tf.placeholder(tf.float32, [None, TAIL_LEN,state_size])
            self.next_state_ph = tf.placeholder(tf.float32, [None, TAIL_LEN,state_size])
            self.reward_ph = tf.placeholder(tf.float32, [None])

            # Normalisierung der Eingaben
            self.norm_state = tf.nn.batch_normalization(
                self.state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            self.norm_next_state = tf.nn.batch_normalization(
                self.next_state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)

            # Erstellen der neuronalen Netzwerke
            with tf.variable_scope("model", reuse=None):
                self.value_output = self._create_model(self.norm_state, CELLS, TAIL_LEN)
            with tf.variable_scope("model", reuse=True):
                self.next_value_output = self._create_model(
                    self.norm_next_state, CELLS, TAIL_LEN)

            self.variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/model")
            print(self.variables)

            # Kostenfunktion für die Optimierung
            with tf.variable_scope("loss"):
                self.diff = self.value_output \
                            - self.reward_ph - self.gamma*self.next_value_output
                self.loss = tf.reduce_mean(tf.square(self.diff))
                self.loss_summary = tf.summary.scalar(
                    "mean_error", tf.reduce_mean(tf.abs(self.diff)))
                self.mean_predict_summary = tf.summary.scalar(
                    "mean_prediction", tf.reduce_mean(self.value_output))

            # Gradientenberechnung, Trimmen und Optimierer
            with tf.variable_scope("training"):
                self.optimizer = tf.train.RMSPropOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.variables)
                self.grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
                self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, GRADIENT_NORM)

                grad_var_pairs = zip(self.clipped_grads, self.variables)

                self.optimize = self.optimizer.apply_gradients(grad_var_pairs)

        # Ausgaben für Tensorboard
        self.summaries = tf.summary.merge_all(scope="V_s/model")
        self.train_metrics_summaries = \
            tf.summary.merge([self.loss_summary, self.mean_predict_summary])

    def _create_model(self, input, n_cells, tail_len):
        layer_names = ["lstm", "d1"]
        # Erstellung der LSTM-Schicht
        lstm = tf.nn.rnn_cell.LSTMCell(n_cells, name=layer_names[0])
        batch_size = tf.shape(input)[0]

        # Verknüpfung der sequentiellen Zeitschritte
        # mit definierter Länge
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        output = None
        for i in range(tail_len):
            output, state = lstm(input[:,i,:], state)

        # Ausgabeschicht
        value_output = tf.layers.Dense(1, activation="linear", name=layer_names[1],
                                       kernel_initializer = tf.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           scale=self.kernel_reg))(output)

        # Ausgaben für Tensorboard
        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return value_output

    def get_summaries(self):
        return self.sess.run(self.summaries)

    def predict(self, states):
        return self.sess.run([self.value_output, self.norm_state] , feed_dict={
            self.state_ph: states
        })

    def train(self, states, rewards, next_states):
        summaries, _ = self.sess.run([self.train_metrics_summaries, self.optimize], feed_dict={
            self.state_ph: states,
            self.next_state_ph: next_states,
            self.reward_ph: rewards
        })
        return summaries

