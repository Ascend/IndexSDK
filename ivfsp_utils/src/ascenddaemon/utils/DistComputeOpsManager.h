/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */


#ifndef ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED
#define ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED

#include <map>
#include <vector>
#include <memory>
#include <ascenddaemon/utils/AscendOperator.h>

namespace ascendSearch {
enum class IndexTypeIdx {
    ITI_FLAT_IP = 0,
    ITI_FLAT_L2,
    ITI_INT8_COS,
    ITI_INT8_L2,
    ITI_INT8_APPROX_L2,
    ITI_SQ_CID_FILTER,
    ITI_SQ_DIST_IP,
    ITI_SQ_DIST_DIM64_IP,
    ITI_SQ_DIST_MASK_IP,
    ITI_SQ_DIST_MASK_DIM64_IP,
    ITI_SQ_DIST_L2,
    ITI_SQ_DIST_DIM64_L2,
    ITI_SQ_DIST_MASK_L2,
    ITI_SQ_DIST_MASK_DIM64_L2,
    ITI_TOPK_FLAT,
    ITI_TOPK_MULTISEARCH,
    ITI_TOPK_IVF,
    ITI_MAX
};

class DistComputeOpsManager {
public:
    static DistComputeOpsManager &getInstance()
    {
        static DistComputeOpsManager distCompOpsManger;
        return distCompOpsManger;
    }

    DistComputeOpsManager() {}

    ~DistComputeOpsManager() {}

    void initialize()
    {
        for (int i = 0; i < static_cast<int>(IndexTypeIdx::ITI_MAX); ++i) {
            std::map<int, std::unique_ptr<AscendOperator>> ops;
            distComputeOps.push_back(std::move(ops));
        }
    }

    void uninitialize()
    {
        distComputeOps.clear();
    }

    std::map<int, std::unique_ptr<AscendOperator>> &getDistComputeOps(IndexTypeIdx type)
    {
        return distComputeOps[static_cast<int>(type)];
    }

    std::vector<std::map<int, std::unique_ptr<AscendOperator>>> distComputeOps;
};
}
#endif // ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED