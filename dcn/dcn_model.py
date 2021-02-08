#coding=utf-8

import tensorflow as tf
import conf

class DeepCrossNetwork(object):
    def __init__(self, single_size, numerical_size, multi_size)
        self.single_size, self.numerical_size, self.multi_size = single_size, numerical_size, multi_size
        self.field_size = single_size + numerical_size + multi_size

        self._init_placeholder()
        self._init_variable()
        self._init_model()

    def _init_placeholder(self):
        self.ph = {}
        self.ph['train_phase'] = tf.placeholder(tf.bool, name="train_phase")
        self.ph['label'] = tf.placeholder(dtype=tf.int8, shape=[None, 2])
        self.ph['value'] = tf.placeholder(dtype=tf.float32, shape=[None, self.field_size])
        self.ph['single_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.single_size])
        self.ph['numerical_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.numerical_size])
        if not conf.use_numerical_embedding:
            self.ph['numerical_value'] = tf.placeholder(dtype=tf.flota32, shape=[None, self.numerical_size])

        for s in conf.MULTI_FEATURES:
            self.ph['multi_index_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
            self.ph['multi_value_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None])

    def _init_variable(self):
        self.vr = {}
        self.vr['single_second_embedding'] = tf.get_variable(name='single_second_embedding',
                                                             shape=(10000, conf.embedding_size),
                                                             initializer=tf.glorot_uniform_initializer())

        self.vr['numerical_second_embedding'] = tf.get_variable(name='numerical_second_embedding',
                                                                shape=(10000, conf.embedding_size),
                                                                initializer=tf.glorot_uniform_initializer())
        for s in conf.MULTI_FEATURES:
            self.vr['multi_second_embedding_%s' % s] = tf.get_variable(name='multi_second_embedding_%s' % s),
                                                                       shape=(10000, conf.embedding_size),
                                                                       initializer=tf.glorot_uniform_initializer())


    def _init_model(self):
        self.graph = tf.graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
             
            #1. embedding layer
            self.embeddings = tf.reshape(tf.nn.embedding_lookup(self.vr["single_second_embedding"], self.ph['single_index']),  shape=[-1, conf.embedding_size * self.single_size])

            #2. deep network
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i,layer in enumerate(self.dnn_wides):
                self.y_deep = tf.add()


