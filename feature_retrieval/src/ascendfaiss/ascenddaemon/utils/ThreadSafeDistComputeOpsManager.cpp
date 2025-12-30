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


#include "ThreadSafeDistComputeOpsManager.h"

namespace ascend {
void ThreadSafeDistComputeOpsManager::initialize()
{
    std::lock_guard<std::mutex> lock(mtx);
    DistComputeOpsManager::initialize();
}

void ThreadSafeDistComputeOpsManager::uninitialize()
{
    std::lock_guard<std::mutex> lock(mtx);
    DistComputeOpsManager::uninitialize();
}

std::map<OpsMngKey, std::unique_ptr<AscendOperator>> &ThreadSafeDistComputeOpsManager::getDistComputeOps(
    IndexTypeIdx type)
{
    std::lock_guard<std::mutex> lock(mtx);
    return DistComputeOpsManager::getDistComputeOps(type);
}

APP_ERROR ThreadSafeDistComputeOpsManager::resetOp(const std::string &opTypeName,
    IndexTypeIdx indexType,
    OpsMngKey &opsKey,
    const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
    const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output)
{
    std::lock_guard<std::mutex> lock(mtx);
    return DistComputeOpsManager::resetOp(opTypeName, indexType, opsKey, input, output);
}

APP_ERROR ThreadSafeDistComputeOpsManager::runOp(IndexTypeIdx indexType,
    OpsMngKey &opsKey,
    const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output,
    aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mtx);
    return DistComputeOpsManager::runOp(indexType, opsKey, input, output, stream);
}
}