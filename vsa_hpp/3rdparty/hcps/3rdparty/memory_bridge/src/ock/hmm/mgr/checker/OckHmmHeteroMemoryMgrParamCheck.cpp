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

#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"
#include "ock/log/OckLogger.h"
namespace ock {
namespace hmm {

OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckMalloc(const uint64_t byteSize)
{
    if (byteSize <= conf::OckSysConf::HmmConf().minMallocBytes ||
        byteSize > conf::OckSysConf::HmmConf().maxMallocBytes) {
        OCK_HMM_LOG_ERROR("parameter byteSize(" << byteSize << ") is out of range (" <<
                          conf::OckSysConf::HmmConf().minMallocBytes << ", " <<
                          conf::OckSysConf::HmmConf().maxMallocBytes << "].");
        return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckAlloc(const uint64_t hmoBytes)
{
    // 校验hmoBytes是否超出规定取值范围
    if (hmoBytes <= conf::OckSysConf::HmmConf().minAllocHmoBytes ||
        hmoBytes > conf::OckSysConf::HmmConf().maxAllocHmoBytes) {
        OCK_HMM_LOG_ERROR("parameter hmoBytes(" << hmoBytes << ") is out of range (" <<
                          conf::OckSysConf::HmmConf().minAllocHmoBytes << ", " <<
                          conf::OckSysConf::HmmConf().maxAllocHmoBytes << "].");
        return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckFree(std::shared_ptr<OckHmmHMObject> hmo)
{
    if (hmo.get() == nullptr) {
        OCK_HMM_LOG_ERROR("the hmo is a nullptr!");
        return HMM_ERROR_INPUT_PARAM_EMPTY;
    }
    auto retCode = CheckHmoIsValid(*hmo);
    if (retCode != HMM_SUCCESS) {
        return retCode;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckCopy(const OckHmmHMObject &dstHMO, const uint64_t dstOffset,
    const OckHmmHMObject &srcHMO, const uint64_t srcOffset, const size_t length)
{
    auto retCode = CheckCopyHmo(dstHMO, srcHMO);
    if (retCode != HMM_SUCCESS) {
        return retCode;
    }
    retCode = CheckHmoIsValid(dstHMO);
    if (retCode != HMM_SUCCESS) {
        return retCode;
    }
    retCode = CheckHmoIsValid(srcHMO);
    if (retCode != HMM_SUCCESS) {
        return retCode;
    }
    if (dstOffset >= dstHMO.GetByteSize()) {
        OCK_HMM_LOG_ERROR("dstOffset(" << dstOffset << ") >= dstHMO.size(" << dstHMO.GetByteSize() << ")");
        return HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE;
    }
    if (dstOffset + length > dstHMO.GetByteSize()) {
        OCK_HMM_LOG_ERROR("dstOffset(" << dstOffset << ") + length(" << length << ") > dstHMO.size(" <<
            dstHMO.GetByteSize() << ")");
        return HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE;
    }
    if (srcOffset >= srcHMO.GetByteSize()) {
        OCK_HMM_LOG_ERROR("srcOffset(" << srcOffset << ") >= srcHMO.size(" << srcHMO.GetByteSize() << ")");
        return HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE;
    }
    if (srcOffset + length > srcHMO.GetByteSize()) {
        OCK_HMM_LOG_ERROR("srcOffset(" << srcOffset << ") + length(" << length << ") > srcHMO.size(" <<
            srcHMO.GetByteSize() << ")");
        return HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckCopyHmo(const OckHmmHMObject &dstHMO,
    const OckHmmHMObject &srcHMO)
{
    OckHmmDeviceId dstDeviceId = OckHmmHMOObjectIDGenerator::ParseDeviceId(dstHMO.GetId());
    OckHmmDeviceId srcDeviceId = OckHmmHMOObjectIDGenerator::ParseDeviceId(srcHMO.GetId());
    if (dstDeviceId != srcDeviceId) {
        OCK_HMM_LOG_ERROR("The dstHMO " << dstHMO << " and the srcHMO " << srcHMO << " belong to different devices!");
        return HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckHmoIsValid(const OckHmmHMObject &hmo)
{
    OckHmmDeviceId deviceId = OckHmmHMOObjectIDGenerator::ParseDeviceId(hmo.GetId());
    if (!OckHmmHMOObjectIDGenerator::Valid(hmo.GetId(), deviceId, hmo.Addr())) {
        OCK_HMM_LOG_ERROR("the hmo " << hmo << " is invalid!");
        return HMM_ERROR_HMO_OBJECT_INVALID;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckGetBuffer(OckHmmHeteroMemoryLocation location, uint64_t offset,
    uint64_t length, const OckHmmHMObjectExt &hmo, OckHmmSubMemoryAllocDispatcher &allocDispatcher)
{
    // 在这里实现参数检查
    uint64_t hmoBytes = hmo.GetByteSize();
    if (offset >= hmoBytes) {
        OCK_HMM_LOG_ERROR("offset(" << offset << ") >= hmo.size(" << hmoBytes << ")");
        return HMM_ERROR_EXEC_INVALID_INPUT_PARAM;
    }
    if (offset + length > hmoBytes) {
        OCK_HMM_LOG_ERROR("offset(" << offset << ") + length(" << length << ") > hmo.size(" << hmoBytes << ")");
        return HMM_ERROR_EXEC_INVALID_INPUT_PARAM;
    }
    if (length == 0) {
        OCK_HMM_LOG_ERROR("buffer length is zero, which is not allowed!");
        return HMM_ERROR_EXEC_INVALID_INPUT_PARAM;
    }
    uint64_t usedBytes = allocDispatcher.SwapAlloc(location)->GetUsedInfo(hmoBytes)->usedBytes;
    uint64_t leftBytes = allocDispatcher.SwapAlloc(location)->GetUsedInfo(hmoBytes)->leftBytes;
    if ((length > usedBytes + leftBytes) && (location != hmo.Location())) {
        OCK_HMM_LOG_ERROR("buffer length(" << length << ") > maxSwapCapacity(" << usedBytes + leftBytes << ")");
        return HMM_ERROR_SWAP_SPACE_NOT_ENOUGH;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckGetUsedInfo(const uint64_t fragThreshold)
{
    if (fragThreshold < conf::OckSysConf::HmmConf().minFragThreshold ||
        fragThreshold > conf::OckSysConf::HmmConf().maxFragThreshold) {
        OCK_HMM_LOG_ERROR("fragThreshold(" << fragThreshold << ") is out of range [" <<
                          conf::OckSysConf::HmmConf().minFragThreshold << ", " <<
                          conf::OckSysConf::HmmConf().maxFragThreshold << "].");
        return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    return HMM_SUCCESS;
}
OckHmmErrorCode OckHmmHeteroMemoryMgrParamCheck::CheckGetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds)
{
    if (maxGapMilliSeconds < conf::OckSysConf::HmmConf().minMaxGapMilliSeconds ||
        maxGapMilliSeconds > conf::OckSysConf::HmmConf().maxMaxGapMilliSeconds) {
        OCK_HMM_LOG_ERROR("maxGapMilliSeconds(" << maxGapMilliSeconds << ") is out of the range [" <<
                                           conf::OckSysConf::HmmConf().minMaxGapMilliSeconds << ", " <<
                                           conf::OckSysConf::HmmConf().maxMaxGapMilliSeconds << "]!");
        return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    return HMM_SUCCESS;
}
}  // namespace hmm
}  // namespace ock