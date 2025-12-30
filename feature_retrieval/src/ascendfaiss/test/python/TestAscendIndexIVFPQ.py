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


d = 128      # 向量维度
nb = 100000  # 底库大小
nq = 1       # 查询向量个数
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = xb[:nq, :]

nlist = 2048  # L1聚类中心数
m = 32        # PQ分段数

# 指定参与运算的Device
dev = ascendfaiss.IntVector()
for i in range(1):
    dev.push_back(i)
config = ascendfaiss.AscendIndexIVFPQConfig(dev)
# 创建IVFPQ索引，PQ编码长度为8bit，L2距离作为相似度标准
ascend_index_ivfpq = ascendfaiss.AscendIndexIVFPQ(
    d, nlist, m, 8, ascendfaiss.METRIC_L2, config)

# 训练
ascend_index_ivfpq.train(xb)
# 建库
ascend_index_ivfpq.add(xb)
# 查找Top K个最近向量
k = 10
display = 10
D, INDEX = ascend_index_ivfpq.search(xq, k)
print("Top %d of first %d queries:" % (k, display))
print(INDEX[:display])

# 删除底库
ids_remove = ascendfaiss.IDSelectorRange(0, 1)
ids_remove_batch = INDEX[0][:int(k / 2)].copy()

print("Remove top1")
num_removed = ascend_index_ivfpq.remove_ids(ids_remove)
print("Removed num:", num_removed)
assert num_removed == 1
D, INDEX = ascend_index_ivfpq.search(xq, k)
print("Top %d of first %d queries after remove top1:" % (k, display))
print(INDEX[:display])

print("Remove top5", ids_remove_batch)
num_removed = ascend_index_ivfpq.remove_ids(ids_remove_batch)
print("Removed num:", num_removed)
assert num_removed == int(k / 2) - 1
D, INDEX = ascend_index_ivfpq.search(xq, k)
print("Top %d of first %d queries after delete top5:" % (k, display))
print(INDEX[:display])
