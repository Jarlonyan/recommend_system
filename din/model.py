import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.kears.regularizers import l2

from tools import *

class DIN(Model):
    def __init__(self, ):
        super(DIN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        self.embed_sparse_layers = [
            Embedding(input_dim=feat['feat_num'], 
                      input_length=1,
                      output_dim = feat['embed_dim'],
                      embeddings_initializer='random_uniform',
                      embedding_regularizer=l2(embed_reg))
            for feat in self.sparse_feature_columns if feat['feat'] in behavior_feature_list
        ]

        self.attention_layer(AttentionLayers)

        self.bn = BatchNormalization(trainable=True)

        self.dnese_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

        mask = tf.cast(tf.not_equal(seq_inputs[:,:,0], 0), dtype=tf.float32)

        



