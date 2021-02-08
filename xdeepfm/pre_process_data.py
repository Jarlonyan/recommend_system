#coding=utf-8
"""
    - 数据处理，数值型必须是float,离散型必须是int,多值离散是str中间用|隔开，eg. "1|2|3"
    - 暂时不能有缺失值

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# features
SINGLE_FEATURES = ['register_type', 'device_type']
NUMERICAL_FEATURES = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day', 'all_action_count', 'last_action', 'all_action_day', 'register_day']
MULTI_FEATURES = ['multi']

Num_embedding = True
Single_feature_frequency = 10
Multi_feature_frequency = 0


def get_dict(train, valid, test):
    global_emb_idx = 0
    num_dict = {}
    single_dict = {}
    multi_dict = {}
    backup_dict = {}

    if SINGLE_FEATURES:
        for s in SINGLE_FEATURES:
            #each field
            frequency_dict = {}
            current_dict = {}
            values = pd.concat([train, valid, test])[s]
            for v in values:
                if v in frequency_dict:
                    frequency_dict[v] += 1
                else:
                    frequency_dict[v] = 1
            for k,v in frequency_dict.items():
                if v > Single_feature_frequency:
                    current_dict[k] = global_emb_idx
                    global_emb_idx += 1
            single_dict[s] = current_dict
            backup_dict[s] = global_emb_idx
            global_emb_idx += 1

    if NUMERICAL_FEATURES and Num_embedding:
        for s in NUMERICAL_FEATURES:
            #each field
            num_dict[s] = global_emb_idx
            global_emb_idx += 1
            #for NaN
            backup_dict[s] = global_emb_idx
            global_emb_idx += 1

    if MULTI_FEATURES:
        #each field
        for s in MULTI_FEATURES:
            frequency_dict = {}
            current_dict = {}
            values = pd.concat([train, valid, test])[s]
            for vs in values:
                for v in vs.split('|'):
                    v = int(v)
                    if v in frequency_dict:
                        frequency_dict[v] += 1
                    else:
                        frequency_dict[v] = 1
            for k,v in frequency_dict.items():
                if v>Multi_feature_frequency:
                    current_dict[k] = global_emb_idx
                    global_emb_idx += 1
            multi_dict[s] = current_dict
            backup_dict[s] = global_emb_idx
    return global_emb_idx, num_dict, single_dict, multi_dict, backup_dict

def trans_raw_to_instance(data, num_dict, single_dict, multi_dict, backup_dict, instance_file):
    single_num = 0
    multi_num = 0
    with open(instance_file, 'w') as f:
        #label, index:value
        def instance_write_to_file(line):
            label = line['label']
            f.write(str(label)+',')
            for s in SINGLE_FEATURES:
                now_v = line[s] 
                if now_v in single_dict[s]:
                    now_idx = single_dict[s][now_v]
                else:
                    now_idx = backup_dict[s]
                f.write(str(now_idx)+':'+str(1)+',')

            for s in NUMERICAL_FEATURES:
                now_v = line[s]
                f.write(str(num_dict[s])+':'+str(now_v)+',')
                #single_num += 1

            for s in MULTI_FEATURES:
                now_v = line[s]
                if '|' not in now_v:
                    idxs = [now_v]
                else:
                    idxs = now_v.split('|')
                idxs = [x for x in idxs if int(x) in multi_dict[s]]
                if idxs:
                    f.write(str('|'.join(idxs))+':'+str(1)+',')
                else:
                    f.write(str(backup_dict[s])+':'+str(1)+',')
            f.write('\n') 
        #end-func
        data.apply(lambda x: instance_write_to_file(x), axis=1)
    #end-with

def pre_process_data():
    train = pd.read_csv('data/raw_train_data.csv', index_col=0)
    valid = pd.read_csv('data/raw_valid_data.csv', index_col=0)
    test = pd.read_csv('data/raw_test_data.csv', index_col=0)

    scalar = MinMaxScaler()
    all_data = pd.concat([train, valid, test])

    for s in NUMERICAL_FEATURES:
        scalar.fit(all_data[s].values.reshape(-1,1))
        train[s] = scalar.transform(train[s].values.reshape(-1,1))
        valid[s] = scalar.transform(valid[s].values.reshape(-1,1))
        test[s] = scalar.transform(test[s].values.reshape(-1,1))
    if (train.shape[1] == valid.shape[1] == test.shape[1]) is False:
        print 'error shape'
        return -1

    global_emb_idx, num_dict, single_dict, multi_dict, backup_dict = get_dict(train, valid, test)

    trans_raw_to_instance(train, num_dict, single_dict, multi_dict, backup_dict, 'data/train_instance.data') 
    trans_raw_to_instance(valid, num_dict, single_dict, multi_dict, backup_dict, 'data/valid_instance.data') 
    trans_raw_to_instance(test, num_dict, single_dict, multi_dict, backup_dict, 'data/test_instance.data') 


def main():
    pre_process_data()

if __name__ == '__main__':
    main()
