#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import random
import os
import argparse
import logging
import time
import collections
import numpy as np
import faiss
import torch
import torch_npu
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PADR import prepare_data, get_train_data, train_AE

LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_dir", default=None, type=str, help="directory for dataset")
parser.add_argument("--val_dataset_dir", default="./validation/", type=str,
                    help="directory for save generated validation set")
parser.add_argument("--generate_val", default=False, type=bool, help="whether need to generate validation set")
parser.add_argument('--save_path', type=str, default=None, help='model save path while training')
parser.add_argument("--dim", default=512, type=int, choices=[96, 128, 200, 256, 512, 2048], help="dataset dim")
parser.add_argument("--npu", default=None, type=int, help="NPU setting")
parser.add_argument("--ratio", default=8, type=int, choices=[2, 4, 8, 16], help="dimension reduction rate")
parser.add_argument("--metric", default='L2', type=str, choices=['L2', 'IP'], help="distance metric")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer', 'test'],
                    help='program running mode')
parser.add_argument('--train_size', type=int, default=100000, help='size to sample training data')
parser.add_argument('--epochs', type=int, default=30, help='total epochs for training')
parser.add_argument('--train_batch_size', type=int, default=8192, help='batch size while training')
parser.add_argument('--infer_batch_size', type=int, default=128, help='batch size while inferring')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate while training')
parser.add_argument('--log_stride', type=int, default=500, help='step stride to print log')
parser.add_argument('--construct_neighbors', type=int, default=100, help='neighbors to construct dataset')
parser.add_argument('--queries_validation', type=int, default=1000, help='queries to construct validation set')
args = parser.parse_args()


def check_args():
    if not args.dataset_dir:
        raise Exception("Error: dataset dir is not set!")
    if not args.save_path:
        raise Exception("Error: save_path is not set! Please set save_path to save models after training!")
    if args.mode != "train":
        raise Exception("Mode not supported now! Please set train mode")
    if args.train_size <= 0 or args.queries_validation <= 0:
        raise Exception("Dat num for train and validation should be larger than 0!")
    if args.epochs <= 0:
        raise Exception("epochs should be larger than 0!")
    if args.train_batch_size <= 0 or args.infer_batch_size <= 0:
        raise Exception("train_batch_size and infer_batch_size should be larger than 0!")
    if args.learning_rate <= 0:
        raise Exception("learning_rate should be larger than 0!")
    if args.log_stride <= 0:
        raise Exception("log_stride should be larger than 0!")
    if args.construct_neighbors <= 0:
        raise Exception("construct_neighbors should be larger than 0!")


def main():
    check_args()
    # set device: cpu or npu
    device = "npu:{}".format(args.npu) if args.npu is not None else "cpu"
    if device != "cpu":
        torch.npu.set_device(device)

    logging.info("===> Start Loading Data!")

    # load dataset demo before training, modify here if you want to load your own dataset
    #####################################################################
    learn, base = get_train_data(args.dataset_dir, args.train_size)
    #####################################################################

    # generate validation set
    if args.generate_val:
        if not os.path.exists(args.val_dataset_dir):
            os.mkdir(args.val_dataset_dir, mode=0o750)
        prepare_data(args.val_dataset_dir, base, args.queries_validation)

    TrainParams = collections.namedtuple('TrainParams', ['save_path',
                                                         'data_learn',
                                                         'ratio',
                                                         'dim',
                                                         'val_dataset_dir',
                                                         'run_device',
                                                         'train_batch_size',
                                                         'learning_rate',
                                                         'epochs',
                                                         'log_stride',
                                                         'construct_neighbors',
                                                         'infer_batch_size'
                                                         ])

    train_params = TrainParams(args.save_path, learn, args.ratio, args.dim, args.val_dataset_dir, device,
                               args.train_batch_size, args.learning_rate, args.epochs, args.log_stride,
                               args.construct_neighbors, args.infer_batch_size)

    # train
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path, mode=0o750)
    train_AE(train_params)

if __name__ == "__main__":
    main()