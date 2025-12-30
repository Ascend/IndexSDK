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
import faiss
import ascendfaiss
import ascendargs as args
import common as utils
import datasets


start_time = time.time()
base, query, train, ground_truth = datasets.load_sift1M(args.data_home, scale=args.need_scale)

dim = train.shape[1]
sq = faiss.ScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
sq.train(train)

ncentroids = 1024

centroids_data = utils.train_kmeans(train, ncentroids)
centroids_int8 = utils.data_to_int8(sq, centroids_data)

base_int8 = utils.data_to_int8(sq, base)
query_int8 = utils.data_to_int8(sq, query)
train_int8 = utils.data_to_int8(sq, train)

devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexInt8IVFFlatConfig(dev)
print("[%.3f] Init AscendIndexInt8IVFFlat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, "COS" if args.metric_name == "IP" else "L2", str(devices)))
ascend_index = ascendfaiss.AscendIndexInt8IVFFlat(dim, ncentroids, args.metric_type, config)

ascend_index.updateCentroids(centroids_int8)
# add database
nb = base_int8.shape[0]
ascend_index.reserveMemory(nb)
ascend_index.add(base_int8)

# search topk results
k = 1000
for i in [1, 4, 8, 16, 32, 64]:
    ascend_index.setNumProbes(i)
    t, r = datasets.evaluate(ascend_index, query_int8, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0), r.get(1000, -1.0)))
