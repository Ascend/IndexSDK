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
from common import create_ascend_index_ivfsq


start_time = time.time()
base, query, train, ground_truth = \
    datasets.load_sift1M(args.data_home, scale=args.need_scale, normalize=args.need_norm)

# set ClusteringParameters
cp = ascendfaiss.ClusteringParameters()
cp.niter = 16
cp.max_points_per_centroid = 256

# create ivfsq index
dim = base.shape[1]    # vector dims
_NLIST = 8192         # L1 IVFList size
devices = args.devices
print("[%.3f] Init AscendIndexIVFSQ, dim = %d, metric_tpye = %s, byResidual = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(args.by_residual), str(devices)))
ascend_index_ivfsq = create_ascend_index_ivfsq(devices, dim, _NLIST, args.metric_type, args.by_residual, cp)
ascend_index_ivfsq.verbose = True

# train
ascend_index_ivfsq.train(train)

# add database
nb = base.shape[0]   # database size
ascend_index_ivfsq.reserveMemory(nb)
ascend_index_ivfsq.add(base)

# search topk results
k = 1000

for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    ascendfaiss.AscendParameterSpace().set_index_parameter(
        ascend_index_ivfsq, 'nprobe', i)
    t, r = datasets.evaluate(ascend_index_ivfsq, query, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0), r.get(1000, -1.0)))
