#coding=utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from layer import FM, DNN

class Base(tf.keras.Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, sparse_emb_dim):
        super(Base, self).__init__()
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns
        self.sparse_feature_columns = list(filter(lambda x: isintance(x, SparseFeat), dnn_feature_columns)) if len(dn_feature_columns) else []

        self.multihot_sparse_feature_columns = list(filter(lambda x: isinstance(x, MultihotSparseFeat), dnn_feature_columns)) is dnn_feature_columns else []
        self.dense_feature_columns = list(filter(lambda x: isinstance(x,DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []

        self.embedding_dict = {feat.embedding_name: layers.Embedding(feat.dimensions, sparse_emb_dim, embeddings_initializer='normal') \
                                for feat in self.sparse_feature_columns + self.multihot_sparse_feature_columns}

        self.out_bias = tf.Variable(tf.zeros([1,]), trainable=True)

    def input_from_feature_columns(self, x):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](x[:, self.feature_index]) for feat in self.sparse_feature_columns]
        multihot_sparse_embedding_list = [self.embedding_dict[feat.embedding_name](x[:, self.feature_index[feat.name][0]: self.feature_index[feat.name][1]]) \
                                            for feat in self.multihot_sparse_feature_columns]
        dense_value_list = [x[:, self.feature_index[feat.name][0]: self.feature_index[feat.name[1]] for feat in self.dense_feature_columns]]
        return sparse_embedding_list, multihot_sparse_embedding_list, dense_value_list

class DeepFM(Base):
    def __init__(self, linear_feature_columns, dnn_feature_columns, sparse_emb_dim, dnn_layers, dropout_rate=0.5):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, sparse_emb_dim)
        self.fm = FM()
        self.dnn = tf.keras.Sequential([
            DNN(sum(map(lambda x: x.dimension, self.densee_feature_columns)) + len(self.sparse_feature_columns)*sparse_emb_dim, dnn_layers, dropout_rate=dropout_rate),
            layers.Dense(1, use_bias=False, activation='linear')]
        )

    def call(self, x):
        sparse_emb, multihot_emb, dense_emb = self.input_from_feature_columns(x)
        linear_sparse_logit = tf.reduce_sum(tf.concat(sparse_emb, axis=-1), axis=-1, keepdims=False)

        if len(dense_emb):
            linear_dense_logit = tf.matmul(tf.concat(dense_emb), axis=-1), self.weight)
            logit = tf.squeeze(linear_dense_logit + linear_dense_logit, -1)
            logit += tf.self.dnn(tf.concat([tf.squeeze(tf.concat(sparse_emb+multihot_emb, -1), 1), tf.concat(dense_emb, -1)], axis=-1))
        else:
            logit = tf.squeeze(linear_sparse_logit, -1)
            logit += self.dnn(tf.concat([tf.squeeze(x,1) for x in sparse_emb] + [tf.reshape(multihot_emb), [x.shape[0], -1])], 1)), 1)
            logit += self.fm(tf.concat(sparse_emb+multihot_emb, axis=1)

        pred = logit + self.out_bias
        return pred



