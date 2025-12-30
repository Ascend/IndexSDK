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
import ascendfaiss
import ascendargs as args
import datasets


start_time = time.time()
base, query, train, ground_truth = datasets.load_sift1M(args.data_home, scale=False, normalize=args.need_norm)

dim = base.shape[1]    # vector dims
nlist = 2048  # L1 IVFList size
m = 32        # Number of sub-quantizers per vector

# create ivfpq index
devices = args.devices
dev = ascendfaiss.IntVector()
for i in devices:
    dev.push_back(i)

config = ascendfaiss.AscendIndexIVFPQConfig(dev)
print("[%.3f] Init AscendIndexIVFPQ, dim = %d, sub_quantizers = %d, metric_tpye = %s, byResidual = %s, devices = %s"
      % (time.time() - start_time, dim, m, args.metric_name, str(args.by_residual), str(devices)))

ascend_index_ivfpq = ascendfaiss.AscendIndexIVFPQ(dim, nlist, m, 8, ascendfaiss.METRIC_L2, config)

# train
ascend_index_ivfpq.train(train)

# add database
ascend_index_ivfpq.add(base)

# search topk results
k = 1000

for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    ascendfaiss.AscendParameterSpace().set_index_parameter(
        ascend_index_ivfpq, 'nprobe', i)
    t, r = datasets.evaluate(ascend_index_ivfpq, query, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0), r.get(1000, -1.0)))
