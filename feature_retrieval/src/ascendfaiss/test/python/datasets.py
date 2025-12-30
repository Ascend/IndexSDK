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
import os
import sys
import time
import numpy as np


def sift_file_read(fname):
    # read as ivecs, fvecs need to view as float32
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1M(path='./sift1M', scale=False, normalize=False):
    """
    scale_factor: 1.0 / 128.0  # scale is pre-processing to avoid fp16 overflow
    """
    if not os.path.exists(os.path.join(path, "sift_learn.fvecs")) or \
            not os.path.exists(os.path.join(path, "sift_base.fvecs")) or \
            not os.path.exists(os.path.join(path, "sift_query.fvecs")) or \
            not os.path.exists(os.path.join(path, "sift_groundtruth.ivecs")):
        raise Exception(f"path {path} invalid, please check it")

    start_time = time.time()
    print("Loading sift1M from %s..." % path, end='', file=sys.stderr)
    xt = sift_file_read(os.path.join(path, "sift_learn.fvecs")).view('float32')
    xb = sift_file_read(os.path.join(path, "sift_base.fvecs")).view('float32')
    xq = sift_file_read(os.path.join(path, "sift_query.fvecs")).view('float32')
    gt = sift_file_read(os.path.join(path, "sift_groundtruth.ivecs"))
    print("done", file=sys.stderr)

    if scale:
        scale_factor = 1.0 / 128.0
        print("xt, xb, xq scaled by %f" % scale_factor)
        xt = xt * scale_factor
        xb = xb * scale_factor
        xq = xq * scale_factor

    if normalize:
        print("normalize features")
        xt_norm = np.linalg.norm(xt, ord=2, axis=1, keepdims=True)
        xt = xt / xt_norm
        xb_norm = np.linalg.norm(xb, ord=2, axis=1, keepdims=True)
        xb = xb / xb_norm
        xq_norm = np.linalg.norm(xq, ord=2, axis=1, keepdims=True)
        xq = xq / xq_norm

    print("[%.3f] Reading data from SIFT1M" % (time.time() - start_time))
    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    # noqa: E741
    _, indices = index.search(xq, k)
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (indices[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
