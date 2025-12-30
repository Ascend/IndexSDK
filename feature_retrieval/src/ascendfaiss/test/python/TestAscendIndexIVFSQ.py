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


d = 128      # vector dims
nb = 100000  # databse size
nq = 1       # query size
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = xb[:nq, :]

nlist = 16384  # L1 IVFList size

# set running devices
dev = ascendfaiss.IntVector()
for i in range(1, 4):
    dev.push_back(i)
config = ascendfaiss.AscendIndexIVFSQConfig(dev)
# create ivfsq index
ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(
    d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
    ascendfaiss.METRIC_L2, True, config)
ascend_index_ivfsq.verbose = True

# train
ascend_index_ivfsq.train(xb)

# add database
ascend_index_ivfsq.add(xb)

# search topk results
k = 10
display = 10
distance, labels = ascend_index_ivfsq.search(xq, k)
print("Top %d of first %d queries:" % (k, display))
print(labels[:display])
assert labels[0][0] == 0

# remove vector
ids_remove = ascendfaiss.IDSelectorRange(0, 1)
ids_remove_batch = labels[0][:int(k / 2)].copy()

print("Remove top1")
num_removed = ascend_index_ivfsq.remove_ids(ids_remove)
print("Removed num:", num_removed)
assert num_removed == 1
distance, labels = ascend_index_ivfsq.search(xq, k)
print("Top %d of first %d queries after remove top1:" % (k, display))
print(labels[:display])

print("Remove top5", ids_remove_batch)
num_removed = ascend_index_ivfsq.remove_ids(ids_remove_batch)
print("Removed num:", num_removed)
assert num_removed == int(k / 2) - 1
distance, labels = ascend_index_ivfsq.search(xq, k)
print("Top %d of first %d queries after delete top5:" % (k, display))
print(labels[:display])
