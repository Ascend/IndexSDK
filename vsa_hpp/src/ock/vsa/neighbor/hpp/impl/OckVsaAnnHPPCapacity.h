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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_CAPACITY_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_CAPACITY_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait> class OckVsaAnnHPPCapacity {
public:
    static uint64_t HostSpace(const OckVsaAnnCreateParam &param);
    static uint64_t DeviceSpace(const OckVsaAnnCreateParam &param);
    static uint64_t HPPKernelDeviceSpace(const OckVsaAnnCreateParam &param);
    static uint64_t HostSwapSpace(const OckVsaAnnCreateParam &param);
    static uint64_t DeviceSwapSpace(const OckVsaAnnCreateParam &param);

private:
    static uint64_t FeatureByteSizeInHost(const OckVsaAnnCreateParam &param);
    static uint64_t ExtFeatureByteSizeInHost(const OckVsaAnnCreateParam &param);
    static uint64_t ObjIdMapByteSizeInHost(const OckVsaAnnCreateParam &param);
    static uint64_t FeatureByteSizeInDevice(const OckVsaAnnCreateParam &param);
    static uint64_t DeviceTempDataSpace(const OckVsaAnnCreateParam &param);
};
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::DeviceSwapSpace(
    const OckVsaAnnCreateParam &param)
{
    return param.BlockRowCount() * sizeof(DataT) * DimSizeT * 8ULL; // 支持8个block的swap空间
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::HostSwapSpace(
    const OckVsaAnnCreateParam &param)
{
    return param.BlockRowCount() * sizeof(DataT) * DimSizeT * 16ULL; // 支持16个block的swap空间
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::HostSpace(
    const OckVsaAnnCreateParam &param)
{
    // 特征空间+扩展特征空间+ObjID空间(视代码实现情况修改)
    return FeatureByteSizeInHost(param) + ExtFeatureByteSizeInHost(param) + ONE_GROUP_SPACE;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::DeviceSpace(
    const OckVsaAnnCreateParam &param)
{
    return FeatureByteSizeInDevice(param) + DeviceTempDataSpace(param);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::DeviceTempDataSpace(
    const OckVsaAnnCreateParam &param)
{
    // 预留最多1个组的底库空间或4G作为临时变量用
    return std::max((uint64_t)(param.GroupRowCount() * sizeof(DataT) * DimSizeT), ONE_GROUP_SPACE);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::ObjIdMapByteSizeInHost(
    const OckVsaAnnCreateParam &param)
{
    // 28的由来： 外部ID_内部ID --> 16字节， 内部ID-->外部ID --> 8字节， TokenId-->内部ID列表 4字节
    // 32字节的由来： 每个空bucket占32个字节
    return param.MaxFeatureRowCount() * 28ULL + IDX_MAP_BUCKET_NUMBER * 32ULL;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::FeatureByteSizeInDevice(
    const OckVsaAnnCreateParam &param)
{
    uint64_t maxFeatureRowCount = param.MaxFeatureRowCount() + param.BlockRowCount(); // 底库数据多预留1个BloCK
    return maxFeatureRowCount * NormTypeByteSizeT +                                   // Norm数据10UL
        maxFeatureRowCount * npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::KeyAttrByteSize() +
        maxFeatureRowCount * param.ExtKeyAttrByteSize() + // 扩展属性数据
        maxFeatureRowCount / __CHAR_BIT__ +               // mask存储空间
        maxFeatureRowCount * sizeof(DataT) * DimSizeT;    // 底库数据
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::HPPKernelDeviceSpace(
    const OckVsaAnnCreateParam &param)
{
    uint64_t maxFeatureRowCount = param.MaxFeatureRowCount();
    uint64_t ret = maxFeatureRowCount * NormTypeByteSizeT + // Norm数据10UL
        maxFeatureRowCount * npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::KeyAttrByteSize() +
        maxFeatureRowCount * param.ExtKeyAttrByteSize() + // 扩展属性数据
        maxFeatureRowCount * __CHAR_BIT__; // mask存储空间，按照底库数的8倍来，相当于bs=64
    return ret;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::ExtFeatureByteSizeInHost(
    const OckVsaAnnCreateParam &param)
{
    // 每条数据放大两倍
    using ExtFeatureT = hcps::algo::OckBitSetWithPos<EXT_FEATURE_SCALE_OUT_BITS * DimSizeT, DimSizeT>;
    return param.GroupRowCount() * sizeof(ExtFeatureT);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::FeatureByteSizeInHost(
    const OckVsaAnnCreateParam &param)
{
    // 最大空间+4G(因为有一个Group从NPU迁移到HOST时，需要占用临时空间)
    uint64_t maxFeatureRowCount = param.MaxFeatureRowCount() + param.GroupRowCount();
    if (param.MaxFeatureRowCount() > 0) {
        maxFeatureRowCount = std::max(maxFeatureRowCount,
            (uint64_t)(MAX_FEATURE_GROUP_IN_DEVICE + 1UL) * (uint64_t)param.GroupRowCount());
    }
    return maxFeatureRowCount * NormTypeByteSizeT + // Norm数据
                                                    // 10UL 修改成 npu::OckVsaAnnNpuIndex<DataT, DimSizeT,
                                                    // NormTypeByteSizeT, KeyTrait>::KeyAttrByteSize()
        maxFeatureRowCount * npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::KeyAttrByteSize() +
        maxFeatureRowCount * param.ExtKeyAttrByteSize() + // 扩展属性数据
        maxFeatureRowCount / __CHAR_BIT__ +               // mask存储空间
        maxFeatureRowCount * sizeof(DataT) * DimSizeT;    // 底库数据
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif