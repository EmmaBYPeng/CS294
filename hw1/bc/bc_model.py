import tensorflow as tf


class BC(object):
    def __init__(self, optimizer, lr):
        self.obs = tf.placeholder(tf.float32, [None, 11])
        self.actions = tf.placeholder(tf.float32, [None, 3])

        self.pred, self.parameters = self.network()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.saver = tf.train.Saver(var_list=tf.global_variables())

    def network(self):
        """Simple two layer neural network"""
        parameters = []
        with tf.variable_scope("fc1"):
            w_fc1 = tf.get_variable('Weight', [11, 128],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_fc1 = tf.get_variable('Bias', [128],
                                    initializer=tf.constant_initializer(0), dtype=tf.float32)
            z_fc1 = tf.add(tf.matmul(self.obs, w_fc1), b_fc1)
            a_fc1 = tf.nn.relu(z_fc1)
            parameters += [w_fc1, b_fc1]
        with tf.variable_scope("fc2"):
            w_fc2 = tf.get_variable('Weight', [128, 3],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_fc2 = tf.get_variable('Bias', [3],
                                    initializer=tf.constant_initializer(0), dtype=tf.float32)
            z_fc1 = tf.add(tf.matmul(a_fc1, w_fc2), b_fc2)
            parameters += [w_fc2, b_fc2]
        return z_fc1, parameters

    def get_optimizer(self, optimizer, lr):
        print("Using %s optimizer" % optimizer)
        if optimizer == "adam":
            return tf.train.AdamOptimizer(lr).minimize(self.loss)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(lr).minimize(self.loss)
        else:
            return tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

    def get_loss(self):
        loss = tf.reduce_mean(tf.pow(self.pred - self.actions, 2)) / 2
        return loss

    def step(self, sess, batch_x, batch_y=None, is_train=True):
        feed_dict = {self.obs: batch_x}
        if batch_y is not None:
            feed_dict[self.actions] = batch_y
        if is_train:
            _, pred, loss = sess.run([self.optimizer, self.pred, self.loss], feed_dict=feed_dict)
            return pred, loss
        else:
            if batch_y is not None:
                pred, loss = sess.run([self.pred, self.loss], feed_dict=feed_dict)
                return pred, loss
            else:
                pred = sess.run([self.pred], feed_dict=feed_dict)
                return pred


