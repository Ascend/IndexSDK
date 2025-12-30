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
base, query, train, ground_truth = \
    datasets.load_sift1M(args.data_home, scale=args.need_scale, normalize=args.need_norm)

dim = base.shape[1]    # vector dims
nlist = 2048         # L1 IVFList size

# set running devices
devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
print("[%.3f] Init AscendIndexIVFFlat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
config = ascendfaiss.AscendIndexIVFFlatConfig(dev)
# create ivfsq index
ascend_index_ivfflat = ascendfaiss.AscendIndexIVFFlat(
    dim, nlist, args.metric_type, config)
ascend_index_ivfflat.verbose = True

# train
ascend_index_ivfflat.train(train)

# add database
nb = base.shape[0]   # database size
ascend_index_ivfflat.reserveMemory(nb)
ascend_index_ivfflat.add(base)

# search topk results
k = 1000
for i in [1, 2, 4, 8, 16, 32, 64]:
    ascendfaiss.AscendParameterSpace().set_index_parameter(
        ascend_index_ivfflat, 'nprobe', i)
    t, r = datasets.evaluate(ascend_index_ivfflat, query, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0), r.get(1000, -1.0)))
