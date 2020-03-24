import tensorflow as tf
import tensorflow_probability as tfp

GRADIENT_NORM = 5
CELLS = 100
TAIL_LEN = 96

STATE_MEAN = tf.constant([ 1.6987270e-01,  3.3652806e+05, -4.7673375e+05])

STATE_VAR = tf.constant([6.0812939e-02, 1.2899629e+11, 6.3592917e+11])


class Policy:

    def __init__(self, sess, state_size, action_size, lr, alpha_entropy,  epsilon, kernel_reg):
        self.sess = sess
        self.action_size = action_size
        self.kernel_reg = kernel_reg

        with tf.variable_scope("Policy"):
            self.state_ph = tf.placeholder(tf.float32, [None, TAIL_LEN, state_size], name="state_ph")
            self.action_ph = tf.placeholder(tf.float32, [None, action_size], name="action_ph")
            self.advantage_ph = tf.placeholder(tf.float32, [None, 1], name="adavantage_ph")

            self.norm_state = tf.nn.batch_normalization(self.state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            #self.norm_state = self.state_ph

            self.norm_advantage = self.advantage_ph
            #with tf.variable_scope("adv_batch_norm"):
                #adv_mean, adv_var = tf.nn.moments(self.advantage_ph, axes=[0])
                #self.norm_advantage = tf.nn.batch_normalization(self.advantage_ph, adv_mean, adv_var, None, None, 1e-12)

            with tf.variable_scope("pi"):
                self.pi, self.mean_action = self._create_model(trainable=True, input=self.norm_state, n_cells=CELLS, tail_len=TAIL_LEN)
                self.pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/pi")
                self.sample_op = self.pi.sample()

            with tf.variable_scope("old_pi"):
                self.old_pi, _ = self._create_model(trainable=False, input=self.norm_state, n_cells=CELLS, tail_len=TAIL_LEN)
                self.old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/old_pi")

            with tf.variable_scope("loss"):
                prob_ratio = self.pi.prob(self.action_ph) / self.old_pi.prob(self.action_ph)
                tf.summary.histogram('prob_ratios', prob_ratio)

                surrogate = prob_ratio * self.norm_advantage
                clipped_surrogate = tf.minimum(surrogate,  tf.clip_by_value(prob_ratio, 1.-epsilon, 1.+epsilon)*self.norm_advantage)

                probs = self.pi.prob(self.action_ph)
                self.pi_entropy =  tf.reduce_mean(-probs*tf.log(probs)) # sum over action space then mean


                self.surrogate = tf.reduce_mean(clipped_surrogate)

                tf.summary.scalar("entropy", self.pi_entropy)
                tf.summary.scalar('surrogate', self.surrogate)

                self.loss = -self.surrogate - alpha_entropy * self.pi_entropy # maximise surrogate and entropy

                tf.summary.scalar("objective", self.loss)

            with tf.variable_scope("training"):
                self.gradients = tf.gradients(self.loss, self.pi_vars)
                self.gradients = [tf.clip_by_value(g, -1, 1) for g in self.gradients]
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, GRADIENT_NORM)
                grads = zip(self.gradients, self.pi_vars)
                self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

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

        log_sigma = tf.Variable(initial_value=tf.fill((self.action_size,), 0.), trainable=trainable, name="log_sigma")

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
        return self.sess.run([self.mean_action, self.sample_op], feed_dict={
            self.state_ph: state
        })

    def train_policy(self, states, actions, advantages):
        _, summaries =self.sess.run([self.optimize, self.summary_op], feed_dict={
            self.state_ph:states,
            self.action_ph: actions,
            self.advantage_ph: advantages
        })
        return summaries

    def update_old_policy(self):
        self.sess.run(self.update_oldpi_op)


class StateValueApproximator:
    def __init__(self, sess, state_size, lr, kernel_reg):
        self.kernel_reg = kernel_reg
        self.sess = sess

        with tf.variable_scope("V_s"):
            self.v_target_ph = tf.placeholder(tf.float32, [None, 1])
            self.state_ph = tf.placeholder(tf.float32, [None, TAIL_LEN,state_size])

            self.norm_state = tf.nn.batch_normalization(self.state_ph, STATE_MEAN, STATE_VAR, None, None, 1e-12)
            #self.norm_state = self.state_ph

            with tf.variable_scope("model"):
                self.value_output = self._create_model(self.norm_state, CELLS, TAIL_LEN)

                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/model")

            with tf.variable_scope("loss"):
                self.diff = self.value_output - self.v_target_ph
                self.loss = tf.reduce_mean(tf.square(self.diff))
                self.loss_summary = tf.summary.scalar("mean_error", tf.reduce_mean(tf.abs(self.diff)))
                self.mean_predict_summary = tf.summary.scalar("mean_prediction", tf.reduce_mean(self.value_output))

            with tf.variable_scope("training"):
                self.optimizer = tf.train.AdamOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.variables)
                # self.grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
                #self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, GRADIENT_NORM)

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

        value_output = tf.layers.Dense(1, activation="linear", name=layer_names[1],kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(output)

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

    def train(self, states, v_targets):
        summaries, _ = self.sess.run([self.train_metrics_summaries, self.optimize], feed_dict={
            self.state_ph: states,
            self.v_target_ph: v_targets
        })
        return summaries

