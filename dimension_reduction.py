import tensorflow as tf
import numpy as np


class autoencoder():
    def __init__(self, fun_in, bottleneck):
        self.fun_in = fun_in
        self.bottleneck = bottleneck

        self.net_in = tf.placeholder(tf.float32, [None, fun_in])
        self.lr_tf = tf.placeholder(tf.float32)
        self.is_decode = tf.placeholder(tf.bool)
        self.cypher = tf.placeholder(tf.float32, [None, bottleneck])

        fc_init = tf.contrib.layers.xavier_initializer()
        # fc_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, dtype=tf.float32)
        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=self.fun_in/3*2, activation_fn=tf.nn.relu, weights_initializer=fc_init)
        fc_1 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=self.fun_in/2, activation_fn=tf.nn.relu, weights_initializer=fc_init)
        fc_bottleneck = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=bottleneck, activation_fn=tf.nn.relu, weights_initializer=fc_init)
        self.fc_bottleneck = tf.cond(self.is_decode, lambda: self.cypher, lambda: fc_bottleneck )
        fc_2 = tf.contrib.layers.fully_connected(inputs=self.fc_bottleneck, num_outputs=self.fun_in/2, activation_fn=tf.nn.relu, weights_initializer=fc_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=self.fun_in/3*2, activation_fn=tf.nn.relu, weights_initializer=fc_init)
        self.fc_out = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=self.fun_in, activation_fn=lambda i: i, weights_initializer=fc_init)

        self.loss = tf.reduce_mean(tf.square(self.net_in - self.fc_out))
        #self.percentage = 100*tf.reduce_mean(tf.abs(self.net_in - fc_out)/tf.sqrt(tf.reduce_sum(tf.square(self.net_in), axis=1)))
        self.opt = tf.train.AdamOptimizer(self.lr_tf).minimize(self.loss)

    def train(self, session, iteration, data, learning_rate):
        result = 0.0
        dummy_cypher = np.zeros((data.shape[0], self.bottleneck), dtype=np.float32)
        for i in xrange(iteration):
            result += session.run([self.opt, self.loss], feed_dict={self.net_in: data, self.lr_tf: learning_rate, self.is_decode: False, self.cypher: dummy_cypher})[1]/100.0
            if not i%100:
                print result
                result = 0.0

    def encode(self, session, data):
        dummy_cypher = np.zeros((data.shape[0], self.bottleneck), dtype=np.float32)
        return session.run(self.fc_bottleneck, feed_dict={self.net_in: data, self.is_decode: False, self.cypher: dummy_cypher})

    def decode_and_encode(self, session, data):
        dummy_cypher = np.zeros((data.shape[0], self.bottleneck), dtype=np.float32)
        return session.run([self.fc_out, self.fc_bottleneck], feed_dict={self.net_in: data, self.is_decode: False, self.cypher: dummy_cypher})

    def decode(self, session, data):
        dummy_input = np.zeros((data.shape[0], self.fun_in), dtype=np.float32)
        return session.run(self.fc_out, feed_dict={self.net_in: dummy_input, self.is_decode: True, self.cypher: data})
