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
import numpy as np
import faiss
import ascendfaiss
from common import log_wrapper


def test(index, xt, xb, xq, k, display):
    index.train(xt)
    index.add(xb)

    cpu_distance, cpu_labels = index.search(xq, k)
    print("CPU Top %d of first %d queries:" % (k, display))
    print(cpu_labels[:display])
    print(cpu_distance[:display])

    devices = ascendfaiss.IntVector()
    for i in range(1):
        devices.push_back(i)

    ascend_index = ascendfaiss.index_cpu_to_ascend(devices, index)
    distance, labels = ascend_index.search(xq, k)
    print("Ascend Top %d of first %d queries:" % (k, display))
    print(labels[:display])
    print(distance[:display])

    assert labels[0][0] == cpu_labels[0][0]


@log_wrapper
def test_ivfpq():
    # dims
    d = 512
    quantizer = faiss.IndexFlatL2(d)

    # database size
    nb = 50000
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')

    nq = 1
    xq = xb[:nq, :]

    # L1 ivf list number
    nlist = 16384
    # PQ M
    m = 64
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.nprobe = 64

    k = 10
    # show results number
    display = 10
    test(index, xb, xb, xq, k, display)


@log_wrapper
def test_ivfsq():
    d = 128
    quantizer = faiss.IndexFlatL2(d)

    nb = 50000
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')

    nq = 1
    xq = xb[:nq, :]

    nlist = 16384
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit)
    index.nprobe = 64

    k = 10
    display = 10
    test(index, xb, xb, xq, k, display)


if __name__ == '__main__':
    test_ivfpq()
    test_ivfsq()
