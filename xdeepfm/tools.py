import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auc_score(preds, labels, label_size):
    preds = [x[label_size - 1] for x in preds]
    labels = [x[label_size - 1] for x in labels]
    roc_score = roc_auc_score(labels, preds)
    return roc_score


def get_data(data_dir):
    data = []
    with open(data_dir, 'r') as f:
        line = f.readline()
        while line:
            data.append(line)
            line = f.readline()
    return data


def get_conf():
    with open('data_conf.txt', 'r') as f:
        line = f.readline()
    line = line.split('\t')
    return int(line[0]), int(line[1]), int(line[2]), int(line[3])

def get_label(labels, label_size):
    final_label = []
    for v in labels:
        temp_label = [0] * label_size
        temp_label[v] = 1
        final_label.append(temp_label)
    return final_label

def get_batch(data, idx, single_size=0, numerical_size=0, multi_size=0, batch_size=0, use_numerical_embedding=False):
    if idx == -1:
        batch_data = data
    elif (idx + 1) * batch_size <= len(data):
        batch_data = data[idx*batch_size:(idx+1)*batch_size]
    else:
        batch_data = data[idx*batch_size:]
    final_label = []
    final_single_index = []
    final_numerical_value = []
    final_numerical_index = []
    final_multi_sparse_index = []
    final_multi_sparse_value = []
    final_value = []
    for idx, line in enumerate(batch_data):
        line_index = []
        line_value = []
        line_numerical_value = []
        line_data = line.split(',')
        final_label.append(int(line_data[0]))

        for i in range(1, 1 + single_size):
            single_pair = line_data[i].split(':')
            line_index.append(int(single_pair[0]))
            line_value.append(float(single_pair[1]))
        final_single_index.append(line_index)

        line_index = []
        if single_size + numerical_size:
            for i in range(1 + single_size, 1 + single_size + numerical_size):
                single_pair = line_data[i].split(':')
                if not use_numerical_embedding:
                    line_numerical_value.append(float(single_pair[1]))
                if float(single_pair[1]) == 0:
                    line_index.append(int(9999))
                    line_value.append(float(1))
                else:
                    line_index.append(int(single_pair[0]))
                    line_value.append(float(single_pair[1]))
        final_numerical_value.append(line_numerical_value)
        final_numerical_index.append(line_index)

        line_index = []
        total_length = 1 + single_size + numerical_size + multi_size
        if multi_size:
            for i in range(1 + single_size + numerical_size, total_length):
                single_pair = line_data[i].split(':')
                _multi = [int(x) for x in single_pair[0].split('|')]
                line_index.append(_multi)
                for v in _multi:
                    final_multi_sparse_index.append([idx, idx])
                    final_multi_sparse_value.append(v)
                line_value.append(float(single_pair[1]))
        final_value.append(line_value)
    #end-for
    return [final_label, final_single_index, final_numerical_index, final_numerical_value, final_multi_sparse_index, final_multi_sparse_value, final_value]

