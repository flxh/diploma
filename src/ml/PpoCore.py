import tensorflow as tf
import tensorflow_probability as tfp


STATE_MEAN = tf.constant([ 1.6987270e-01,  3.3652806e+05, -4.7673375e+05, 0])

STATE_VAR = tf.constant([6.0812939e-02, 1.2899629e+11, 6.3592917e+11, 0.5])

# TODO shared LSTM memory different heads

class Policy:

    def __init__(self, sess, state_size, action_size, lr, tail_len, cells, beta_entropy, log_var_init, epsilon, kernel_reg, gradient_norm):
        self.sess = sess
        self.action_size = action_size
        self.kernel_reg = kernel_reg
        self.log_var_init = log_var_init

        self.policy_iteration = 0

        with tf.variable_scope("Policy"):
            self.state_ph = tf.placeholder(tf.float32, [None, tail_len, state_size], name="state_ph")
            self.action_ph = tf.placeholder(tf.float32, [None, action_size], name="action_ph")
            self.advantage_ph = tf.placeholder(tf.float32, [None], name="adavantage_ph")

            self.norm_state = tf.nn.batch_normalization(self.state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            #self.norm_state = self.state_ph

            self.norm_advantage = self.advantage_ph
            with tf.variable_scope("adv_batch_norm"):
                adv_mean, adv_var = tf.nn.moments(self.advantage_ph, axes=[0])
                self.norm_advantage = tf.nn.batch_normalization(self.advantage_ph, adv_mean, adv_var, None, None, 1e-12)

            with tf.variable_scope("pi"):
                self.pi, self.mean_action = self._create_model(trainable=True, input=self.norm_state, n_cells=cells, tail_len=tail_len)
                self.pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/pi")
                self.sample_op = self.pi.sample()
                tf.summary.histogram('sampled_actions', self.sample_op)

            with tf.variable_scope("old_pi"):
                self.old_pi, _ = self._create_model(trainable=False, input=self.norm_state, n_cells=cells, tail_len=tail_len)
                self.old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/old_pi")

            with tf.variable_scope("loss"):
                prob_ratio = tf.exp(self.pi.log_prob(self.action_ph) - self.old_pi.log_prob(self.action_ph))
                tf.summary.histogram('prob_ratios', prob_ratio)

                surrogate = prob_ratio * self.norm_advantage
                clipped_surrogate = tf.minimum(surrogate,  tf.clip_by_value(prob_ratio, 1.-epsilon, 1.+epsilon)*self.norm_advantage)

                self.pi_entropy = tf.reduce_mean(self.pi.entropy())  # sum over action space then mean
                self.surrogate = tf.reduce_mean(clipped_surrogate)
                self.action_reg_loss = tf.reduce_mean(self.mean_action**2)

                tf.summary.scalar("entropy", self.pi_entropy)
                tf.summary.scalar('surrogate', self.surrogate)
                tf.summary.scalar('action_reg', self.action_reg_loss)

                self.loss = -self.surrogate - beta_entropy * self.pi_entropy + 0.033*self.action_reg_loss # maximise surrogate and entropy

                tf.summary.scalar("objective", self.loss)

            with tf.variable_scope("training"):
                self.gradients = tf.gradients(self.loss, self.pi_vars)
                #self.gradients = [tf.clip_by_value(g, -1, 1) for g in self.gradients]
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, gradient_norm)
                grads = zip(self.gradients, self.pi_vars)
                optimizer = tf.train.RMSPropOptimizer(lr)

                self.optimize = optimizer.apply_gradients(grads)

                clipped_grads_tb = [tf.clip_by_value(g, -1e-8, 1e-8) for g in self.gradients]

            with tf.variable_scope("update_weights"):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_vars, self.old_pi_vars)]

        for g, v in zip(clipped_grads_tb, self.pi_vars):
            tf.summary.histogram(v.name.split(":")[0]+"_grad", g)

        self.summary_op = tf.summary.merge_all(scope="Policy")

    def _create_model(self, trainable, input, n_cells, tail_len):
        layer_names = ["lstm", "d1"]

        lstm = tf.nn.rnn_cell.LSTMCell(n_cells, name=layer_names[0], trainable=trainable)
        batch_size = tf.shape(input)[0]

        state = lstm.zero_state(batch_size, dtype=tf.float32)
        output = None
        for i in range(tail_len):
            output, state = lstm(input[:,i,:], state)

        mu = tf.layers.Dense(self.action_size, activation="tanh", name=layer_names[1], trainable=trainable, kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(output)

        log_sigma = tf.Variable(initial_value=tf.fill((self.action_size,), self.log_var_init), trainable=trainable, name="log_sigma")
        #log_sigma = tf.constant([1.])

        distribution = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))

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
    def __init__(self, sess, state_size, lr, tail_len, cells, gamma, kernel_reg):
        self.kernel_reg = kernel_reg
        self.sess = sess

        self.gamma = tf.constant(gamma, dtype=tf.float32)

        with tf.variable_scope("V_s"):
            self.state_ph = tf.placeholder(tf.float32, [None, tail_len,state_size])
            self.next_state_ph = tf.placeholder(tf.float32, [None, tail_len,state_size])
            self.reward_ph = tf.placeholder(tf.float32, [None])

            self.norm_state = tf.nn.batch_normalization(self.state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            self.norm_next_state = tf.nn.batch_normalization(self.next_state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            #self.norm_state = self.state_ph

            with tf.variable_scope("model", reuse=None):
                self.value_output = self._create_model(self.norm_state, cells, tail_len)
            with tf.variable_scope("model", reuse=True):
                self.next_value_output = self._create_model(self.norm_next_state, cells, tail_len)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/model")
            print(self.variables)

            with tf.variable_scope("loss"):
                self.diff = self.value_output - self.reward_ph - self.gamma * tf.stop_gradient(self.next_value_output)
                self.loss = tf.reduce_mean(tf.square(self.diff))
                self.loss_summary = tf.summary.scalar("mean_error", tf.reduce_mean(tf.abs(self.diff)))
                self.mean_predict_summary = tf.summary.scalar("mean_prediction", tf.reduce_mean(self.value_output))

            with tf.variable_scope("training"):
                self.optimizer = tf.train.RMSPropOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.variables)
                # self.grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
                #self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, gradient_norm)

                grad_var_pairs = zip(self.grads, self.variables)

                self.optimize = self.optimizer.apply_gradients(grad_var_pairs)


        self.summaries = tf.summary.merge_all(scope="V_s/model")
        self.train_metrics_summaries = tf.summary.merge([self.loss_summary, self.mean_predict_summary])

    def _create_model(self, input, n_cells, tail_len):
        layer_names = ["lstm", "d1"]

        lstm = tf.nn.rnn_cell.LSTMCell(n_cells, name=layer_names[0])
        batch_size = tf.shape(input)[0]

        state = lstm.zero_state(batch_size, dtype=tf.float32)
        output = None
        for i in range(tail_len):
            output, state = lstm(input[:,i,:], state)

        value_output = tf.layers.Dense(1, activation="linear", name=layer_names[1], kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(output)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return tf.reshape(value_output, (-1,))

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

