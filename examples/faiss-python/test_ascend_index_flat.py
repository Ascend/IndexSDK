# -------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All Rights Reserved.
#
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import logging
import numpy as np
import faiss
import ascendfaiss

DIM = 512  # dimension
BASE = 100000  # database size
QUERY = 100  # BASE of queries
np.random.seed(1234)  # make reproducible
xb = np.random.random((BASE, DIM)).astype('float32')
xb[:, 0] += np.arange(BASE) / 1000.0
xq = xb[:QUERY, :]

# 指定参与运算的device
dev = ascendfaiss.IntVector()
dev.push_back(0)
config = ascendfaiss.AscendIndexFlatConfig(dev)

# 创建index
ascend_index_flat = ascendfaiss.AscendIndexFlatL2(DIM, config)

# 添加底库
ascend_index_flat.add(xb)

logging.basicConfig(level=logging.INFO)
# 查找topk
k = 10
distances, indices = ascend_index_flat.search(xq, k)
logging.info("indices: %s", indices)
logging.info("distances: %s", distances)

# 删除底库
ids_remove = faiss.IDSelectorRange(0, 1)
ids_remove_batch = indices[0][: int(k / 2)].copy()
num_removed = ascend_index_flat.remove_ids(ids_remove)

# reset
ascend_index_flat.reset()

# cpu to ascend
cpu_index_flat = faiss.IndexFlatL2(DIM)
cpu_index_flat.add(xb)

dev = ascendfaiss.IntVector()
dev.push_back(0)
ascend_index_flat = ascendfaiss.index_cpu_to_ascend(dev, cpu_index_flat)

_, indices = ascend_index_flat.search(xq, k)


# ascend to cpu
cpu_index_flat = ascendfaiss.index_ascend_to_cpu(ascend_index_flat)
cpu_index_flat.d = ascend_index_flat.d
cpu_index_flat.ntotal = ascend_index_flat.ntotal

_, indices = cpu_index_flat.search(xq, k)
logging.info("after search indices: %s", indices)
