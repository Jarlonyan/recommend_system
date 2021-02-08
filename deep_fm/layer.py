#coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers

class FM(tf.keras.layers.Layer):
    def __init__(self):
        super(FM, self).__init__()

    def call(self, input):
        square_of_sum = tf.math.pow(tf.math.reduce_sum(input, 1, keepdims=True), 2)
        sum_of_square = tf.math.reduce_sum(input*input, 1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5*tf.math.reduce_sum(cross_term, axis=2, keepdims=False)
        return tf.squeeze(corss_term, -1)

class DNN(tf.keras.layers.Layer):
    def __init__(self, inputs_dim, hiden_units, activation=tf.nn.relu, dropout_rate=0, use_bn=False):
        super(DNN, self).__init()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)
        self.use_bn = use_bn

        if len(hidden_units) == 0:
            raise ValueError('hidden_units is empty!')

        self.linears = [layers.Dense(hidden_units[i] for i in range(len(hidden_units))]
        if self.use_bn:
            self.bn = [nn.BatchNorm1d(hidden_units[i] for i in range(len(hidden_units))]

    def call(self, input):
        for i in range(len(self.linears)):
            fc = self.linears[i](input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            input = fc
        return input



