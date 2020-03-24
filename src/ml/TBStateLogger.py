import tensorflow as tf


class StateLogger:
    def __init__(self, sess, names, scope_name):
        self.sess = sess

        with tf.variable_scope(scope_name):
            self.states = tf.placeholder(tf.float32, [None,None, len(names)], name="state_ph")
            mean, variance = tf.nn.moments(self.states, axes=[0,1])

            self.means, self.update_means = tf.metrics.mean_tensor(mean)
            self.vars, self.update_vars  = tf.metrics.mean_tensor(variance)

            for i in range(len(names)):
                name = names[i]
                tf.summary.scalar("{}__{}_mean".format(i,name), mean[i])
                tf.summary.scalar("{}__{}_var".format(i, name), variance[i])

            self.summary_op = tf.summary.merge_all(scope=scope_name)

    def log(self, states):
        summary, _, _, means, vars = self.sess.run([self.summary_op, self.update_means, self.update_vars, self.means, self.vars], feed_dict={
            self.states: states
        })

        print("Means: {}".format(means))
        print("Vars: {}".format(vars))

        return summary

