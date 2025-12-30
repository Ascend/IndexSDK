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


#ifndef OCK_HCPS_HANLDER_HMM_HELPER_H
#define OCK_HCPS_HANLDER_HMM_HELPER_H
#include <cstdint>
#include <memory>
#include <deque>
#include <vector>
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/log/OckLogger.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
namespace ock {
namespace hcps {
namespace handler {
namespace helper {

const uint64_t FRAGMENT_SIZE_THRESHOLD = 2ULL * 1024ULL * 1024ULL;
const uint64_t MIN_INCREASE_MEMORY_BYTESIZE = 4ULL * 1024ULL * 1024ULL * 1024ULL;

std::shared_ptr<hmm::OckHmmHMObject> MakeHmo(hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes,
    hmm::OckHmmMemoryAllocatePolicy policy, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> MakeHostHmo(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode);
std::deque<std::shared_ptr<hmm::OckHmmHMObject>> MakeHostHmoDeque(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, uint64_t count, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> MakeDeviceHmo(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode);
std::deque<std::shared_ptr<hmm::OckHmmHMObject>> MakeDeviceHmoDeque(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, uint64_t count, hmm::OckHmmErrorCode &errorCode);

std::shared_ptr<hmm::OckHmmHMObject> MakeHostHmo(
    OckHeteroHandler &handler, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> MakeDeviceHmo(
    OckHeteroHandler &handler, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode);

/*
@brief 拷贝HMO到Host， 如果srcHmo本身在HOST侧，直接返回srcHmo, 否则新建一个HMO并复制好数据返回
*/
std::shared_ptr<hmm::OckHmmHMObject> CopyToHostHmo(
    OckHeteroHandler &handler, std::shared_ptr<hmm::OckHmmHMObject> srcHmo, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> CopyToHostHmo(
    OckHeteroHandler &handler, hmm::OckHmmHMObject &srcHmo, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<OckHeteroStreamBase> MakeStream(OckHeteroHandler &handler, hmm::OckHmmErrorCode &errorCode,
    OckDevStreamType streamType = OckDevStreamType::AI_NULL);

/**
 * @brief FunctionName       	    判断是否需要增加内存，调用 HMM 内存增量接口
 * @incByteSize                     需要增加的 host 内存量
 * @return                          无需增加内存和增加内存成功时: HMM_SUCCESS
 */
hmm::OckHmmErrorCode UseIncBindMemory(OckHeteroHandler &handler, uint64_t incByteSize, std::string incLocationName);
hmm::OckHmmErrorCode UseIncBindMemory(std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> hmmMgr, uint64_t incByteSize,
    std::string incLocationName);

void MergeMultiHMObjectsToOne(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, std::shared_ptr<hmm::OckHmmHMObject> hmObject,
    hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> MergeMultiHMObjectsToHost(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, hmm::OckHmmErrorCode &errorCode);
std::shared_ptr<hmm::OckHmmHMObject> MergeMultiHMObjectsToDevice(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, hmm::OckHmmErrorCode &errorCode);
}  // namespace helper
}  // namespace handler
}  // namespace hcps
}  // namespace ock
#endif