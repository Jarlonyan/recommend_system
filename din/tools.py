import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense

class AttentionLayer(Layer):
    def __init__(self, atten_hidden_units, activation="prelu"):
        super(AttentionLayer, self).__init__()
        self.atten_dense = [Dense(unit, activation=activation) for unit in atten_hidden_units]
        self.atten_final_dense = Dense(1)

    def call(self, inputs):
        # query: candidate item (None, d*2), d is the dimension of embedding
        # key: hist items (None, seq_len, d*2)
        # value: hist items (None, seq_len, d*2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.title(q, multiples=[1, k.shape[1]])
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])
    
        #q,k,out product should concat
        info = tf.concat([q, k, q-k, q*k], axis=-1)

        for dense in self.atten_dense:
            info = dense(info)

        outputs = self.atten_final_dense(info) #(None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1) #(None, seq_len)

        paddings = tf.ones_like(outputs) * (-2**32 + 1) #(None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs) #(None, seq_len)

        #softmax
        outputs = tf.nn.softmax(logits=outputs)  #(None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1) #(None, 1, seq_len)

        outputs = tf.matmul(outputs, v) #(None, 1, d*2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.ad_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha*(1.0-x_p)*x + x_p * x


def attention(querys, keys, keys_len):
    # querys: [B, H]
    # keys: [B, T, H]
    # keys_len: [B]
    querys_hidden_units = querys.get_shape().as_list()[-1]
    querys = tf.tile(querys, [1, tf.shape(keys)[1]])
    querys = tf.reshape(querys, [-1, tf.shape(keys)[1], querys_hidden_units])
    din_all = tf.concat([querys, keys, querys - keys, querys * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer
