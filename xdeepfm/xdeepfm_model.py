#coding=utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import time
import conf
import tools

class xDeepFM(object):
    def __init__(self, single_size, numerical_size, multi_size):
        self.single_size, self.numerical_size, self.multi_size = single_size, numerical_size, multi_size
        self.field_size = self.single_size + self.numerical_size + self.multi_size
        self.embedding_length = self.field_size * conf.embedding_size

        self._init_placeholder()
        self._init_variable()
        self._init_model()

    def _init_placeholder(self):
        self.ph = {}
        self.ph['label'] = tf.placeholder(dtype=tf.int8, shape=[None, 2])
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        self.ph['value'] = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.single_size + self.numerical_size + self.multi_size])
        self.ph['single_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.single_size])
        self.ph['numerical_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.numerical_size])
        for s in conf.MULTI_FEATURES:
            self.ph['multi_index_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
            self.ph['multi_value_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None])
        if not conf.use_numerical_embedding:
            self.ph['numerical_value'] = tf.placeholder(dtype=tf.float32, shape=[None, self.numerical_size])

    def _init_variable(self):
        self.vr = {}
        self.vr['single_second_embedding'] = tf.get_variable(name='single_second_embedding',
                                                             shape=(10000, conf.embedding_size),
                                                             initializer=tf.glorot_uniform_initializer())
        self.vr['numerical_second_embedding'] = tf.get_variable(name='numerical_second_embedding',
                                                                shape=(10000, conf.embedding_size),
                                                                initializer=tf.glorot_uniform_initializer())
        for s in conf.MULTI_FEATURES:
            self.vr['multi_second_embedding_%s' % s] = tf.get_variable(name='multi_second_embedding_%s' % s,
                                                                       shape=(10000, conf.embedding_size),
                                                                       initializer=tf.glorot_uniform_initializer())

        self.vr['single_first_embedding'] = tf.get_variable(name='single_first_embedding',
                                                            shape=(10000, 1),
                                                            initializer=tf.glorot_uniform_initializer())
        self.vr['numerical_first_embedding'] = tf.get_variable(name='numerical_first_embedding',
                                                            shape=(10000, 1),
                                                            initializer=tf.glorot_uniform_initializer())
        for s in conf.MULTI_FEATURES:
            self.vr['multi_first_embedding_%s' % s] = tf.get_variable(name='multi_first_embedding_%s' % s,
                                                                      shape=(10000, 1),
                                                                      initializer=tf.glorot_uniform_initializer())
        # DNN part
        if conf.use_numerical_embedding:
            dnn_net = [self.embedding_length] + conf.dnn_net_size
        else:
            dnn_net = [self.embedding_length - self.numerical_size * conf.embedding_size + self.numerical_size] + conf.dnn_net_size
        for i in range(len(conf.dnn_net_size)):
            self.vr['W_%d' % i] = tf.get_variable(name='W_%d' % i, shape=[dnn_net[i], dnn_net[i + 1]], initializer=tf.glorot_uniform_initializer())
            self.vr['b_%d' % i] = tf.get_variable(name='b_%d' % i, shape=[dnn_net[i + 1]], initializer=tf.zeros_initializer())
        # output

    def _init_model(self):
        # first embedding
        first_single_result = tf.reshape(tf.nn.embedding_lookup(self.vr['single_first_embedding'], self.ph['single_index']),
                                         shape=[-1, self.single_size])
        first_numerical_result = tf.reshape(tf.nn.embedding_lookup(self.vr['numerical_first_embedding'], self.ph['numerical_index']),
                                         shape=[-1, self.numerical_size])
        if conf.MULTI_FEATURES:
            first_multi_result = []
            for s in conf.MULTI_FEATURES:
                temp_multi_result = tf.nn.embedding_lookup_sparse(self.vr['multi_first_embedding_%s' % s],
                                                                  tf.SparseTensor(indices=self.ph['multi_index_%s' % s],
                                                                                  values=self.ph['multi_value_%s' % s],
                                                                                  dense_shape=(conf.batch_size, conf.embedding_size)),
                                                                  None, combiner="sum")
                first_multi_result.append(temp_multi_result)
                first_multi_result = tf.concat(first_multi_result, axis=1)
            first_embedding_output = tf.concat([first_single_result, first_numerical_result, first_multi_result], axis=1)
        else:
            first_embedding_output = tf.concat([first_single_result, first_numerical_result], axis=1)

        y_first_order = tf.multiply(first_embedding_output, self.ph['value'])

        # second embedding
        second_single_result = tf.reshape(tf.nn.embedding_lookup(self.vr['single_second_embedding'], self.ph['single_index']),
                                          shape=[-1, conf.embedding_size * self.single_size])
        second_numerical_result = tf.reshape(tf.nn.embedding_lookup(self.vr['numerical_second_embedding'], self.ph['numerical_index']),
                                             shape=[-1, conf.embedding_size * self.numerical_size])
        if conf.MULTI_FEATURES:
            second_multi_result = []
            for s in conf.MULTI_FEATURES:
                temp_multi_result = tf.nn.embedding_lookup_sparse(self.vr['multi_second_embedding_%s' % s],
                                                                  tf.SparseTensor(indices=self.ph['multi_index_%s' % s],
                                                                                  values=self.ph['multi_value_%s' % s],
                                                                                  dense_shape=(conf.batch_size, conf.embedding_size)),
                                                                  None,
                                                                  combiner="sum")
                second_multi_result.append(temp_multi_result)
                second_multi_result = tf.concat(second_multi_result, axis=1)
            # DNN input
            self.DNN_input = tf.concat([second_single_result, second_multi_result], axis=1)
        else:
            self.DNN_input = tf.concat([second_single_result], axis=1)
        self.middle_fm_input = tf.concat([self.DNN_input, second_numerical_result], axis=1)
        if conf.use_numerical_embedding:
            self.DNN_input = tf.concat([self.DNN_input, second_numerical_result], axis=1)
        else:
            self.DNN_input = tf.concat([self.DNN_input, self.ph['numerical_value']], axis=1)
        self.shape = tf.shape(self.DNN_input)
        # second output
        second_FM_input = tf.reshape(self.middle_fm_input, shape=[-1, self.single_size + self.numerical_size + self.multi_size, conf.embedding_size])

        summed_features_emb = tf.reduce_sum(second_FM_input, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(second_FM_input)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

        dnn_output = self.DNN_input
        # DNN output
        for i in range(len(conf.dnn_net_size)):
            self.DNN_input = tf.add(tf.matmul(self.DNN_input, self.vr['W_%d' % i]), self.vr['b_%d' % i])
            self.DNN_input = tf.layers.batch_normalization(self.DNN_input,training=self.train_phase)
            dnn_output = tf.nn.relu(self.DNN_input)

        # CIN
        D = conf.embedding_size
        final_result = []
        final_len = 0
        field_nums = [self.field_size]
        if conf.MULTI_FEATURES:
            nn_input = tf.reshape(tf.concat([second_single_result, second_multi_result], axis=1), shape=[-1, self.field_size, conf.embedding_size])
        else:
            nn_input = tf.reshape(second_single_result, shape=[-1, self.field_size, conf.embedding_size])
        X0 = nn_input
        cin_layers = [X0]
        split_tensor_0 = tf.split(X0, D * [1], 2)
        for idx, Hk in enumerate(conf.cross_layer_size):
            now_tensor = tf.split(cin_layers[-1], D * [1], 2)
            # Hk x m
            dot_result_m = tf.matmul(split_tensor_0, now_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[D, -1, field_nums[0] * field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
            W = tf.get_variable(name="f_" + str(idx), shape=[1, field_nums[-1] * field_nums[0], Hk], dtype=tf.float32)
            curr_out = tf.nn.conv1d(dot_result, filters=W, stride=1, padding='VALID')
            b = tf.get_variable(name="f_b" + str(idx), shape=[Hk], dtype=tf.float32, initializer=tf.zeros_initializer())
            curr_out = tf.nn.relu(tf.nn.bias_add(curr_out, b))
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            if conf.cross_direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += Hk
                field_nums.append(int(Hk))
            else:
                if idx != len(conf.cross_layer_size) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(Hk / 2)], 1)
                    final_len += int(Hk / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += Hk

                field_nums.append(int(Hk / 2))
            final_result.append(direct_connect)
            cin_layers.append(next_hidden)
        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)
        w_nn_output1 = tf.get_variable(name='w_nn_output1', shape=[final_len, conf.cross_output_size], dtype=tf.float32)
        b_nn_output1 = tf.get_variable(name='b_nn_output1', shape=[conf.cross_output_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        CIN_out = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)

        # final output
        output_length = 0
        to_concat = []
        if conf.FM_layer:
            to_concat.append(y_first_order)
            to_concat.append(y_second_order)
            output_length += self.field_size + conf.embedding_size
        if conf.CIN_layer:
            to_concat.append(CIN_out)
            output_length += conf.cross_output_size
        if conf.DNN_layer:
            to_concat.append(dnn_output)
            output_length += conf.dnn_net_size[-1]

        output = tf.concat(to_concat, axis=1)

        self.vr['final_w'] = tf.get_variable(name='final_w', shape=[output_length, 2], initializer=tf.glorot_uniform_initializer())
        self.vr['final_b'] = tf.get_variable(name='final_b', shape=[2], initializer=tf.zeros_initializer())
        final_logits = tf.add(tf.matmul(output, self.vr['final_w']), self.vr['final_b'])
        self.softmax_output = tf.nn.softmax(final_logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph['label'], logits=final_logits))
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=conf.learning_rate).minimize(self.loss)

