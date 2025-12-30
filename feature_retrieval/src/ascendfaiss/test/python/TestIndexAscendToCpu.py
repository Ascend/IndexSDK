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
import ascendfaiss
from common import log_wrapper


def test(ascend_index, xt, xb, xq, k, display):
    ascend_index.train(xt)
    ascend_index.add(xb)

    distance, labels = ascend_index.search(xq, k)
    print("Ascend Top %d of first %d queries:" % (k, display))
    print(labels[:display])
    print(distance[:display])

    cpu_index = ascendfaiss.index_ascend_to_cpu(ascend_index)
    distance, cpu_labels = cpu_index.search(xq, k)
    print("CPU Top %d of first %d queries:" % (k, display))
    print(labels[:display])
    print(distance[:display])

    assert labels[0][0] == cpu_labels[0][0]


@log_wrapper
def test_index():
    d = 512        # dims
    nlist = 16384  # L1 ivf list number
    m = 64         # PQ M
    nb = 50000     # database size
    nq = 1         # query size
    k = 10         # topk
    display = 10   # show results number

    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xq = xb[:nq, :]

    dev_pq = ascendfaiss.IntVector()
    dev_pq.push_back(0)

    config_pq = ascendfaiss.AscendIndexIVFPQConfig(dev_pq)
    ascend_index_ivfpq = ascendfaiss.AscendIndexIVFPQ(
        d, nlist, m, 8, ascendfaiss.METRIC_L2, config_pq)

    test(ascend_index_ivfpq, xb, xb, xq, k, display)

    dev_sq = ascendfaiss.IntVector()
    dev_sq.push_back(1)

    config_sq = ascendfaiss.AscendIndexIVFSQConfig(dev_sq)
    ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(
        d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
        ascendfaiss.METRIC_L2, True, config_sq)

    test(ascend_index_ivfsq, xb, xb, xq, k, display)


if __name__ == '__main__':
    test_index()
