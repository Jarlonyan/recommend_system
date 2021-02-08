#coding=utf-8

import tensorflow as tf
import argparse
import sys

from dataset import get_dataloader


if __name__ == '__main__':
    train_loader, test_loader, linear_feature_columns, dnn_feature_columns = get_dataloader(args.train_batch_size, args.test_batch_size)

    # loss
    optimizer = tf.keras.optimizers.Adam()
    loss_metric = tf.keras.metrics.Sum()
    auc_metric = tf.keras.metrics.AUC()

    model = DeepFM()



