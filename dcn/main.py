#coding=utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import time
import conf
import tools
from dcn_model import DeepCrossNetwork

single_size, numerical_size, multi_size = len(conf.SINGLE_FEATURES), len(conf.NUMERICAL_FEATURES), len(conf.MULTI_FEATURES)
model = DeepCrossNetwork(single_size, numerical_size, multi_size)

train_data = tools.get_data(conf.train_instance_file)
valid_data = tools.get_data(conf.valid_instance_file)
test_data = tools.get_data(conf.test_instance_file)
valid_batch = tools.get_batch(valid_data, -1, single_size, numerical_size, multi_size, conf.batch_size, conf.use_numerical_embedding)

valid_dict = {
    model.ph['train_phase']: False,
    model.ph['label']: tools.get_label(valid_batch[0], 2),
    model.ph['single_index']: valid_batch[1],
    model.ph['numerical_index']: valid_batch[2],
    model.ph['numerical_value']: valid_batch[3],
    model.ph['value']: valid_batch[-1]
}
if conf.MULTI_FEATURES:
    for idx, s in enumerate(conf.MULTI_FEATURES):
        valid_dict[model.ph['multi_index_%s' % s]] = valid_batch[4]
        valid_dict[model.ph['multi_value_%s' % s]] = valid_batch[5]


def train(model):
    print('begin to train model......')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        all_data_length = len(train_data)
        global_step = 0
        print('total step:%d'%(conf.epochs * (int(all_data_length / conf.batch_size) + 1)))
        for i in range(conf.epochs):
            num_batchs = int(all_data_length / conf.batch_size) + 1
            for j in range(num_batchs):
                global_step  += 1
                now_batch = tools.get_batch(train_data, j, single_size, numerical_size, multi_size, conf.batch_size, conf.use_numerical_embedding)

                batch_dict = {
                                model.ph['train_phase']: True,
                                model.ph['label']: tools.get_label(now_batch[0], 2),
                                model.ph['single_index']: now_batch[1],
                                model.ph['numerical_index']: now_batch[2],
                                model.ph['numerical_value']: now_batch[3],
                                model.ph['value']: now_batch[-1]
                             }
                if conf.MULTI_FEATURES:
                    for idx,s in enumerate(conf.MULTI_FEATURES):
                        batch_dict[model.ph['multi_index_%s'%s]]= now_batch[4]
                        batch_dict[model.ph['multi_value_%s'%s]] = now_batch[5]

                _out, _loss, _ = sess.run([model.softmax_output, model.loss, model.optimizer], feed_dict=batch_dict)
                end = time.time()
                
                if global_step % 10 == 0:
                    _out2, _loss2, _ = sess.run([model.softmax_output, model.loss, model.optimizer], feed_dict=valid_dict)
                    print('step:',global_step,'train loss:',_loss,'valid loss:',_loss2,'valid_auc:', tools.auc_score(_out2, tools.get_label(valid_batch[0],2),2))
    #end-with

if __name__ == '__main__':
    train(model)



