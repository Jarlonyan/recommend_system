

import argparse

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.01, help='learning_rate', type=float)
    parser.add_argument('--train_batch_size', default=100, type=int)
    parser.add_argument('--test_batch_size', default=10, type=int)

    #parser.add_argument('--dataset_dir', default='/data/private/Ad/ml-20m/np_prepro/', help='dataset path')
    parser.add_argument('--dataset_dir', default='aaa/', help='dataset path')


    args = parser.parse_args()
    return args

