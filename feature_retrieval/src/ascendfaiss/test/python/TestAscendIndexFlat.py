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

d = 512      # 向量维度
nb = 100000  # 底库大小
nq = 10      # 查询向量个数
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = xb[:nq, :]

# 指定参与运算的Device
dev = ascendfaiss.IntVector()
dev.push_back(0)
config = ascendfaiss.AscendIndexFlatConfig(dev)
# 创建Flat索引
ascend_index_flat = ascendfaiss.AscendIndexFlatL2(d, config)

# 训练
ascend_index_flat.train(xb)
# 建库
ascend_index_flat.add(xb)
# 查找Top K个最近向量
k = 10
display = 10
_, indices = ascend_index_flat.search(xq, k)
print("Top %d of first %d queries:" % (k, display))
print(indices[:display])

# 删除底库
ids_remove = ascendfaiss.IDSelectorRange(0, 1)
ids_remove_batch = indices[0][:int(k / 2)].copy()
print("Remove top1")
num_removed = ascend_index_flat.remove_ids(ids_remove)
print("Removed num:", num_removed)
assert num_removed == 1
_, indices = ascend_index_flat.search(xq, k)
print("Top %d of first %d queries after remove top1:" % (k, display))
print(indices[:display])

# reset
ascend_index_flat.reset()

# cpu to ascend
cpu_index_flat = faiss.IndexFlatL2(d)
cpu_index_flat.add(xb)
dev = ascendfaiss.IntVector()
dev.push_back(1)
ascend_index_flat = ascendfaiss.index_cpu_to_ascend(dev, cpu_index_flat)
_, indices = ascend_index_flat.search(xq, k)
print("[CpuToAscend] Top %d of first %d queries:" % (k, display))
print(indices[:display])

# ascend to cpu
cpu_index_flat = ascendfaiss.index_ascend_to_cpu(ascend_index_flat)
_, indices = ascend_index_flat.search(xq, k)
print("[AscendToCpu] Top %d of first %d queries:" % (k, display))
print(indices[:display])
