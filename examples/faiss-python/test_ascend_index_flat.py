# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

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
