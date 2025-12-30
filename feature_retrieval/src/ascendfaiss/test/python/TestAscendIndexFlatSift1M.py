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
import datasets
import ascendfaiss
import ascendargs as args


start_time = time.time()
base, query, train, ground_truth = \
    datasets.load_sift1M(args.data_home, scale=args.need_scale, normalize=args.need_norm)

devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexFlatConfig(dev)

dim = base.shape[1]
print("[%.3f] Init AscendIndexFlat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
index = ascendfaiss.AscendIndexFlat(dim, args.metric_type, config)

print("[%.3f] Start building database, size = %d" % (
      time.time() - start_time, base.shape[0]))
index.add(base)

K = 100
print("[%.3f] Start searching, query num = %d, topk = %d" % (
      time.time() - start_time, query.shape[0], K))

t, r = datasets.evaluate(index, query, ground_truth, K)
print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f" %
      (1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0)))
