import tensorflow as tf


class TrainingSummaryCreator:
    def __init__(self, sess):
        self.sess = sess

        with tf.variable_scope("training", reuse=True):
            self.rewards_ph = tf.placeholder(tf.float32, [None])
            self.time_elapsed_ph = tf.placeholder(tf.float32, [])

            rew_mean, rew_variance = tf.nn.moments(self.rewards_ph, axes=[0])

            reward_summary = tf.summary.scalar("reward_mean", rew_mean)
            reward_var_summary = tf.summary.scalar("reward_var", rew_variance)
            steps_summary = tf.summary.scalar("speed",  tf.cast(tf.size(self.rewards_ph), tf.float32) / self.time_elapsed_ph)

            self.t_summaries = tf.summary.merge([reward_summary, steps_summary,  reward_var_summary])

    def create_summary(self, rewards,  time_elapsed):
        return self.sess.run(self.t_summaries, feed_dict={
            self.rewards_ph: rewards,
            self.time_elapsed_ph: time_elapsed

        })
