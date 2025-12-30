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

# set running devices
devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexSQConfig(dev)

# create sq index
dim = base.shape[1]    # vector dims
print("[%.3f] Init AscendIndexSQ, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
ascend_index_sq = ascendfaiss.AscendIndexSQ(dim, ascendfaiss.ScalarQuantizer.QT_8bit,
                                            args.metric_type, config)
ascend_index_sq.verbose = True

# train
ascend_index_sq.train(train)

# add database
ascend_index_sq.add(base)

# search topk results
k = 1000

t, r = datasets.evaluate(ascend_index_sq, query, ground_truth, k)
print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
      (1000.0 / t, r.get(1, -1.0), r.get(10, -1.0), r.get(100, -1.0), r.get(1000, -1.0)))
