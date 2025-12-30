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


#ifndef THREAD_SAFE_DIST_COMPUTE_OPS_MANAGER_H
#define THREAD_SAFE_DIST_COMPUTE_OPS_MANAGER_H

#include <mutex>
#include "DistComputeOpsManager.h"

namespace ascend {
class ThreadSafeDistComputeOpsManager : public DistComputeOpsManager {
public:
    ThreadSafeDistComputeOpsManager() {}

    ~ThreadSafeDistComputeOpsManager() override {}

    ThreadSafeDistComputeOpsManager(const ThreadSafeDistComputeOpsManager&) = delete;

    ThreadSafeDistComputeOpsManager& operator=(const ThreadSafeDistComputeOpsManager&) = delete;

    void initialize() override;

    void uninitialize() override;

    std::map<OpsMngKey, std::unique_ptr<AscendOperator>> &getDistComputeOps(IndexTypeIdx type) override;

    APP_ERROR resetOp(const std::string &opTypeName,
                      IndexTypeIdx indexType,
                      OpsMngKey &opsKey,
                      const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                      const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output) override;

    APP_ERROR runOp(IndexTypeIdx indexType,
                    OpsMngKey &opsKey,
                    const std::vector<const AscendTensorBase *> &input,
                    const std::vector<const AscendTensorBase *> &output,
                    aclrtStream stream) override;

private:
    mutable std::mutex mtx;
};
}
#endif // THREAD_SAFE_DIST_COMPUTE_OPS_MANAGER_H