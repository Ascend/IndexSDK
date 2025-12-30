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


#include <unistd.h>
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
using namespace hmm;
class OckVsaAnnCreateParamImpl : public OckVsaAnnCreateParam {
public:
    OckVsaAnnCreateParamImpl(hmm::OckHmmDeviceId deviceIdx, uint64_t maxFeatureNum, uint32_t tokenCount,
        cpu_set_t cpuSets, uint32_t extKeyAttributeByteSize, uint32_t extKeyAttributeBlockSize = 262144UL,
        double distThreshold = 0.8, uint32_t blockSize = 262144UL, uint32_t groupSize = 64UL,
        uint32_t sliceSize = 64UL, double firstNeighborCellThreshold = 0.9,
        double secondNeighborCellThreshold = 0.8)
        : maxFeatureRowCount(maxFeatureNum),
          deviceId(deviceIdx),
          tokenNum(tokenCount),
          cpuSet(cpuSets),
          extKeyAttrByteSize(extKeyAttributeByteSize),
          extKeyAttrBlockSize(extKeyAttributeBlockSize),
          distanceThreshold(distThreshold),
          blockRowCount(blockSize),
          groupBlockCount(groupSize),
          sliceRowCount(sliceSize),
          firstClassNeighborCellThreshold(firstNeighborCellThreshold),
          secondClassNeighborCellThreshold(secondNeighborCellThreshold)
    {}

    uint64_t MaxFeatureRowCount(void) const override
    {
        return maxFeatureRowCount;
    }

    hmm::OckHmmDeviceId DeviceId(void) const override
    {
        return deviceId;
    }

    uint32_t TokenNum(void) const override
    {
        return tokenNum;
    }

    const cpu_set_t &CpuSet(void) const override
    {
        return cpuSet;
    }

    uint32_t ExtKeyAttrByteSize(void) const override
    {
        return extKeyAttrByteSize;
    }
    uint32_t ExtKeyAttrBlockSize(void) const override
    {
        return extKeyAttrBlockSize;
    }

    uint32_t BlockRowCount(void) const override
    {
        return blockRowCount;
    }

    /*
    @brief 每个组的最大Block数量(64), 必须是16的倍数, GroupBlockCount * BlockRowCount 不能超过419430400
    */
    uint32_t GroupBlockCount(void) const override
    {
        return groupBlockCount;
    }

    /*
    @brief 每个Slice的数据行数(256)，必须是64的倍数
    */
    uint32_t SliceRowCount(void) const override
    {
        return sliceRowCount;
    }

    /*
    @brief 算法计算用：每个Group的数据行数 = GroupBlockCount * BlockRowCount 必须小于等于419430400, 能整除 ExtKeyAttrBlockSize
    */
    uint32_t GroupRowCount(void) const override
    {
        return groupBlockCount * blockRowCount;
    }

    virtual uint32_t MaxGroupCount(void) const override
    {
        return static_cast<uint32_t>(utils::SafeDivUp(maxFeatureRowCount, GroupRowCount()));
    }

    /*
    @brief 每个Group中的Slice数(65536)， 必须是64的倍数
    */
    uint32_t GroupSliceCount(void) const override
    {
        return blockRowCount * groupBlockCount / sliceRowCount;
    }
    /*
    @brief 距离阈值，小于该阈值的距离全部舍弃
    */
    double DistanceThreshold(void) const override
    {
        return distanceThreshold;
    }
    double FirstClassNeighborCellThreshold(void) const override
    {
        return firstClassNeighborCellThreshold;
    }
    double SecondClassNeighborCellThreshold(void) const override
    {
        return secondClassNeighborCellThreshold;
    }
    void SetFirstClassNeighborCellThreshold(double firstThreshold) override
    {
        if (firstThreshold > 1UL) {
            OCK_HCPS_LOG_ERROR("firstThreshold[" << firstThreshold << "] cannot exceed 1");
            return;
        }
        firstClassNeighborCellThreshold = firstThreshold;
    }
    void SetSecondClassNeighborCellThreshold(double secondThreshold) override
    {
        if (secondThreshold > 1UL) {
            OCK_HCPS_LOG_ERROR("secondThreshold[" << secondThreshold << "] cannot exceed 1");
            return;
        }
        secondClassNeighborCellThreshold = secondThreshold;
    }
    /*
    @brief 拷贝一份OckVsaAnnCreateParam，除输入的maxFeatureRowCount参数外，其余参数完全一致
    */
    std::shared_ptr<OckVsaAnnCreateParam> Copy(uint64_t maxFeatureNum) const override
    {
        return OckVsaAnnCreateParam::Create(cpuSet, deviceId, maxFeatureNum, tokenNum, extKeyAttrByteSize,
            extKeyAttrBlockSize);
    }

    static void CheckDeviceIdExists(OckVsaErrorCode &retCode, const hmm::OckHmmDeviceId deviceId);
    static void CheckCpuIdExists(OckVsaErrorCode &retCode, const cpu_set_t &cpuSet);
    template <typename T>
    static void CheckParamRange(OckVsaErrorCode &retCode, std::pair<T, T> range, T value, std::string paramName,
        OckVsaErrorCode errorCode);
    template <typename T>
    static void CheckParamMultiple(OckVsaErrorCode &retCode, const T valueBase, T value, std::string paramName,
        OckVsaErrorCode errorCode);

private:
    const uint64_t maxFeatureRowCount;
    const hmm::OckHmmDeviceId deviceId;
    const uint32_t tokenNum;
    const cpu_set_t cpuSet;
    const uint32_t extKeyAttrByteSize;
    const uint32_t extKeyAttrBlockSize;
    const double distanceThreshold;
    const uint32_t blockRowCount;
    const uint32_t groupBlockCount;
    const uint32_t sliceRowCount;
    double firstClassNeighborCellThreshold;
    double secondClassNeighborCellThreshold;
};

void OckVsaAnnCreateParamImpl::CheckDeviceIdExists(OckVsaErrorCode &retCode, const hmm::OckHmmDeviceId deviceId)
{
    uint32_t deviceCount = 0;
    auto ret = aclrtGetDeviceCount(&deviceCount);
    if (ret == 0 && deviceId >= deviceCount) {
        OCK_HCPS_LOG_ERROR("the deviceId(" << deviceId << ") not exists!");
        retCode = VSA_ERROR_DEVICE_NOT_EXISTS;
    }
}
void OckVsaAnnCreateParamImpl::CheckCpuIdExists(OckVsaErrorCode &retCode, const cpu_set_t &cpuSet)
{
    uint32_t cpuCount = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_CONF));
    for (uint32_t i = cpuCount; i < __CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuSet) != 0) {
            OCK_HCPS_LOG_ERROR("there's a cpuId(" << i << ") in cpuSet that does not exist in Physical environment!");
            retCode = VSA_ERROR_CPUID_NOT_EXISTS;
        }
    }
}
template <typename T>
void OckVsaAnnCreateParamImpl::CheckParamRange(OckVsaErrorCode &retCode, std::pair<T, T> range, T value,
    std::string paramName, OckVsaErrorCode errorCode)
{
    if (retCode != VSA_SUCCESS) {
        return;
    }
    if (value < range.first || value > range.second) {
        OCK_HCPS_LOG_ERROR(paramName << "(" << value << ") is out of range, which range is [" << range.first << ", " <<
            range.second << "].");
        retCode = errorCode;
    }
}
template <typename T>
void OckVsaAnnCreateParamImpl::CheckParamMultiple(OckVsaErrorCode &retCode, const T valueBase, T value,
    std::string paramName, OckVsaErrorCode errorCode)
{
    if (retCode != VSA_SUCCESS) {
        return;
    }
    if (utils::SafeMod(value, valueBase) != 0) {
        OCK_HCPS_LOG_ERROR(paramName << "(" << value << ") is not a multiple of " << valueBase);
        retCode = errorCode;
    }
}

OckVsaErrorCode OckVsaAnnCreateParam::CheckValid(const OckVsaAnnCreateParam &param)
{
    auto retCode = VSA_SUCCESS;
    // MaxFeatureRowCount: range[16777216, 432537600] , 256的倍数
    OckVsaAnnCreateParamImpl::CheckParamRange(retCode,
        std::make_pair(npu::MIN_MAX_FEATURE_ROW_COUNT, npu::MAX_MAX_FEATURE_ROW_COUNT), param.MaxFeatureRowCount(),
        "maxFeatureRowCount", VSA_ERROR_MAX_FEATURE_ROW_COUNT_OUT);
    OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, npu::MULTIPLE_MAX_FEATURE_ROW_COUNT,
        param.MaxFeatureRowCount(), "maxFeatureRowCount", VSA_ERROR_MAX_FEATURE_ROW_COUNT_DIVISIBLE);
    // DeviceId
    OckVsaAnnCreateParamImpl::CheckDeviceIdExists(retCode, param.DeviceId());
    // TokenNum: (0, 300000], (0, TokenNum() - 1]
    OckVsaAnnCreateParamImpl::CheckParamRange(retCode, std::make_pair(npu::MIN_TOKEN_NUM, npu::MAX_TOKEN_NUM),
        param.TokenNum(), "tokenNum", VSA_ERROR_TOKEN_NUM_OUT);
    // CpuSet: 必须指定CPU核, cpu有效核数必选大于等于4
    OckVsaAnnCreateParamImpl::CheckCpuIdExists(retCode, param.CpuSet());
    OckVsaAnnCreateParamImpl::CheckParamRange(retCode,
        std::make_pair(npu::MIN_CPU_SET_NUM, sysconf(_SC_NPROCESSORS_CONF)),
        static_cast<int64_t>(CPU_COUNT(&param.CpuSet())), "cpuNum", VSA_ERROR_CPUID_OUT);

    if (param.ExtKeyAttrByteSize() != 0 || param.ExtKeyAttrBlockSize() != 0) {
        // ExtKeyAttrByteSize: [0, 22]
        OckVsaAnnCreateParamImpl::CheckParamRange(retCode, std::make_pair(1U, npu::MAX_EXTKEYATTR_BYTESIZE),
            param.ExtKeyAttrByteSize(), "extKeyAttrByteSize", VSA_ERROR_EXTKEYATTR_BYTE_SIZE_OUT);
        // ExtKeyAttrBlockSize: [262144, GroupBlockCount * BlockRowCount]
        OckVsaAnnCreateParamImpl::CheckParamRange(retCode,
            std::make_pair(npu::MIN_EXTKEYATTR_BLOCKSIZE, param.GroupBlockCount() * param.BlockRowCount()),
            param.ExtKeyAttrBlockSize(), "extKeyAttrBlockSize", VSA_ERROR_EXTKEYATTR_BLOCK_SIZE_OUT);
        // ExtKeyAttrBlockSize需要被GroupRowCount整除
        OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, param.ExtKeyAttrBlockSize(), param.GroupRowCount(),
            "groupRowCount", VSA_ERROR_GROUP_ROW_COUNT_DIVISIBLE);
    }

    // SliceRowCount: 64的倍数
    OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, npu::MULTIPLE_SLICE_ROW_COUNT, param.SliceRowCount(),
        "sliceRowCount", VSA_ERROR_SLICE_ROW_COUNT_DIVISIBLE);
    // BlockRowCount: SliceRowCount的倍数
    OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, param.SliceRowCount(), param.BlockRowCount(), "blockRowCount",
        VSA_ERROR_BLOCK_ROW_COUNT_DIVISIBLE);
    // GroupBlockCount: 16的倍数
    OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, npu::MULTIPLE_GROUP_BLOCK_COUNT, param.GroupBlockCount(),
        "groupBlockCount", VSA_ERROR_GROUP_BLOCK_COUNT_DIVISIBLE);
    // GroupRowCount: [0, 419430400]
    OckVsaAnnCreateParamImpl::CheckParamRange(retCode, std::make_pair(0U, npu::MAX_GROUP_ROW_COUNT),
        param.GroupRowCount(), "groupRowCount", VSA_ERROR_GROUP_ROW_COUNT_OUT);
    // MaxGroupCount: [0, 100]
    OckVsaAnnCreateParamImpl::CheckParamRange(retCode, std::make_pair(0U, npu::MAX_MAX_GROUP_COUNT),
        param.MaxGroupCount(), "maxGroupCount", VSA_ERROR_MAX_GROUP_COUNT_OUT);
    // GroupSliceCount: 64的倍数
    OckVsaAnnCreateParamImpl::CheckParamMultiple(retCode, npu::MULTIPLE_GROUP_SLICE_COUNT, param.GroupSliceCount(),
        "groupSliceCount", VSA_ERROR_GROUP_SLICE_COUNT_DIVISIBLE);
    return retCode;
}
std::shared_ptr<OckVsaAnnCreateParam> OckVsaAnnCreateParam::Create(const cpu_set_t &cpuSet,
    hmm::OckHmmDeviceId deviceId, uint64_t maxFeatureRowCount, uint32_t tokenNum, uint32_t extKeyAttrsByteSize,
    uint32_t extKeyAttrBlockSize)
{
    return std::make_shared<OckVsaAnnCreateParamImpl>(deviceId, maxFeatureRowCount, tokenNum, cpuSet,
        extKeyAttrsByteSize, extKeyAttrBlockSize);
}

std::ostream &operator << (std::ostream &os, const OckVsaAnnCreateParam &param)
{
    return os << "{'maxFeatureRowCount': " << param.MaxFeatureRowCount() << ", 'deviceId': " << param.DeviceId() <<
        ", 'tokenNum': " << param.TokenNum() << ", 'cpuSet': " << param.CpuSet() << ", 'extKeyAttrByteSize': " <<
        param.ExtKeyAttrByteSize() << ", 'extKeyAttrBlockSize': " << param.ExtKeyAttrBlockSize() <<
        ", 'blockRowCount': " << param.BlockRowCount() << ", 'groupBlockCount': " << param.GroupBlockCount() <<
        ", 'sliceRowCount': " << param.SliceRowCount() << ", 'groupRowCount': " << param.GroupRowCount() <<
        ", 'maxGroupCount': " << param.MaxGroupCount() << ", 'groupSliceCount': " << param.GroupSliceCount() <<
        ", 'distanceThreshold': " << param.DistanceThreshold() << ", 'firstClassThreshold': " <<
        param.FirstClassNeighborCellThreshold() << ", 'secondClassThreshold': " <<
        param.SecondClassNeighborCellThreshold() << "}";
}
std::ostream &operator<<(std::ostream &os, const cpu_set_t &cpuSet)
{
    return hmm::operator<<(os, cpuSet);
}
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock