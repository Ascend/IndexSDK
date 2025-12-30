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
import time
import numpy as np
import faiss
import ascendfaiss


def log_wrapper(func):
    def wrapper():
        print('start testing %s ...' % func.__name__)
        func()
        print('--------' * 10)
    return wrapper


def create_ascend_index_ivfsq(devices, d, nlist, metric, by_residual=True, cluster_parameter=None):
    # set running devices
    dev = ascendfaiss.IntVector()
    for i in devices:
        dev.push_back(i)

    # set config(device config and clustering config)
    config = ascendfaiss.AscendIndexIVFSQConfig(dev)
    if cluster_parameter is not None:
        config.cp = cluster_parameter

    # create ivfsq index
    ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
                                                      metric, by_residual, config)
    return ascend_index_ivfsq


def data_to_int8(sq, data):
    codes = sq.compute_codes(data)
    codes_int8 = codes.astype("int32") - 128
    return codes_int8.astype("int8")


def train_kmeans(x, ncentroids):
    niter = 10
    verbose = True
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)
    return kmeans.centroids


def random_feature(dim, feature_count, seed=7777777, normalize=True):
    print("generate: dim %d, feature num: %d, seed: %d" % (dim, feature_count, seed))
    np.random.seed(seed)
    ts = time.time()
    xb = np.random.random((feature_count, dim)).astype('float32')
    if normalize:
        xb_norm = np.linalg.norm(xb, ord=2, axis=1, keepdims=True)
        xb = xb / xb_norm
        print("normalize features")
    te = time.time()
    print("generate used: %.4f s" % (te - ts))
    return xb
