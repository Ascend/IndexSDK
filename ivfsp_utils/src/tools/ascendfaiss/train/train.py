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
import argparse
from SP import SubSpaceClusterACL, read_file
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--nlist',
                        type=int,
                        required=True,
                        help="centroids number")
    parser.add_argument("--dim",
                        type=int,
                        required=True,
                        help="Feature dimension")
    parser.add_argument("--nonzero_num",
                        type=int,
                        required=True,
                        help="Non-zero dimension after dimension reduction")
    parser.add_argument("--num_iter",
                        type=int,
                        default=20,
                        help="train iteration num")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        required=True,
                        help="NPU device id")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32768,
                        help="code train batch size")
    parser.add_argument("--code_num",
                        type=int,
                        required=True,
                        help="codebook feature code access batch size")
    parser.add_argument("--ratio",
                        type=float,
                        default=1.0,
                        help="Sampling rate of the original feature code")
    parser.add_argument("--learn_data_path",
                        type=str,
                        required=True,
                        help="Feature code learn data file path")
    parser.add_argument("--codebook_output_dir",
                        type=str,
                        required=True,
                        help="Directory to which the codebook is output")
    parser.add_argument("--train_model_dir",
                        type=str,
                        required=True,
                        help="Directory where the train OM is located")
    return parser.parse_args()


def check_args(args):
    if args.nlist <= 0:
        raise Exception("nlist should be larger than 0!")
    if args.dim <= 0:
        raise Exception("dim should be larger than 0!")
    if args.nonzero_num <= 0:
        raise Exception("nonzero_num should be larger than 0!")
    if args.num_iter <= 0:
        raise Exception("num_iter should be larger than 0!")
    if args.batch_size <= 0:
        raise Exception("batch_size should be larger than 0!")
    if args.code_num <= 0:
        raise Exception("code_num should be larger than 0!")
    if args.ratio <= 0:
        raise Exception("ratio should be larger than 0!")


def run():
    args = get_args()
    check_args(args)
    k = args.nlist
    dim = args.dim
    nonzero_num = args.nonzero_num
    num_iter = args.num_iter
    device = args.device
    ratio = args.ratio
    learn_data_path = args.learn_data_path
    codebook_output_dir = args.codebook_output_dir
    train_model_dir = args.train_model_dir
    batch_size = args.batch_size
    update_codebook_batch_size = args.code_num
    feature_type = [-1, dim]
    learn = read_file(learn_data_path, dtype=np.float32, shape=feature_type)
    codebook = np.transpose(
        learn[np.random.choice(np.arange(len(learn)), k * nonzero_num, replace=False)])
    sp = SubSpaceClusterACL(dim=dim,
                            k=k,
                            block_size=nonzero_num,
                            model_dir=train_model_dir,
                            batch_size=batch_size,
                            update_codebook_batch_size=update_codebook_batch_size,
                            verbose=True,
                            eps=0.0,
                            device=device,
                            codebook=codebook)
    idx = np.random.choice(np.arange(len(learn)), int(ratio * len(learn)), replace=False)
    sp.train(base=learn[idx], num_iter=num_iter,
             codebook_dir=codebook_output_dir)
    sp.models.release_resource()
    pass

if __name__ == "__main__":
    run()