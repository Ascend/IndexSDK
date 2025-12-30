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


#ifndef OCK_VSA_ANN_CREATE_PARAM_H
#define OCK_VSA_ANN_CREATE_PARAM_H
#include <cstdint>
#include <memory>
#include <thread>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
namespace ock {
namespace vsa {
namespace neighbor {
class OckVsaAnnCreateParam {
public:
    virtual ~OckVsaAnnCreateParam() noexcept = default;

    /*
    @brief 支持的最大特征数, 部分空间需要提前分配，因此需要设置本Search的最大特征数
    最小值为 262144 * 64 必须是256的倍数
    MaxFeatureRowCount 必须小于 432537600
    */
    virtual uint64_t MaxFeatureRowCount(void) const = 0;

    /*
    @brief 使用的NPU设备
    */
    virtual hmm::OckHmmDeviceId DeviceId(void) const = 0;
    /*
    @brief Token数目， 必须大于0，小于3.0e5，真实的tokenId值为(0, TokenNum() - 1]
    */
    virtual uint32_t TokenNum(void) const = 0;
    /*
    @brief 能够使用的CPU核, 必须指定CPU核, cpu有效核数必选大于等于4
    */
    virtual const cpu_set_t &CpuSet(void) const = 0;
    /*
    @brief 用户自定义属性的字节数, [0, 22]
    */
    virtual uint32_t ExtKeyAttrByteSize(void) const = 0;
    /*
    @brief 用户自定义属性的block的行数, [262144, GroupRowCount]，通常是262144的倍数, 能被 GroupRowCount 整除
    */
    virtual uint32_t ExtKeyAttrBlockSize(void) const = 0;
    /*
    @brief 每个Block的数据行数(262144)，必须是SliceRowCount的倍数。 当前这个值受算子影响为定值
    */
    virtual uint32_t BlockRowCount(void) const = 0;
    /*
    @brief 每个组的最大Block数量(64), 必须是16的倍数, GroupBlockCount * BlockRowCount 不能超过419430400
    */
    virtual uint32_t GroupBlockCount(void) const = 0;
    /*
    @brief 每个Slice的数据行数(256)，必须是64的倍数
    */
    virtual uint32_t SliceRowCount(void) const = 0;
    /*
    @brief 算法计算用：每个Group的数据行数 = GroupBlockCount *  BlockRowCount 必须小于等于419430400, 能整除
    ExtKeyAttrBlockSize
    */
    virtual uint32_t GroupRowCount(void) const = 0;
    /*
    @brief 算法计算用：最大的组数 =ceil(MaxFeatureRowCount / GroupRowCount)，必须小于100
    */
    virtual uint32_t MaxGroupCount(void) const = 0;
    /*
    @brief 每个Group中的Slice数(65536)， 必须是64的倍数
    */
    virtual uint32_t GroupSliceCount(void) const = 0;
    /*
    @brief 距离阈值，小于该阈值的距离全部舍弃
    */
    virtual double DistanceThreshold(void) const = 0;
    /*
    @brief 一等邻区阈值，大于该阈值的距离全部选择
    */
    virtual double FirstClassNeighborCellThreshold(void) const = 0;
    /*
    @brief 二等邻区阈值，小于该阈值的距离全部舍弃
    */
    virtual double SecondClassNeighborCellThreshold(void) const = 0;
    /*
    @brief 设置一等邻区阈值，小于该阈值的距离全部舍弃
    */
    virtual void SetFirstClassNeighborCellThreshold(double firstThreshold) = 0;
    /*
    @brief 设置二等邻区阈值，小于该阈值的距离全部舍弃
    */
    virtual void SetSecondClassNeighborCellThreshold(double secondThreshold) = 0;
    /*
    @brief 拷贝一份OckVsaAnnCreateParam，除输入的maxFeatureRowCount参数外，其余参数完全一致
    */
    virtual std::shared_ptr<OckVsaAnnCreateParam> Copy(uint64_t maxFeatureRowCount) const = 0;
    /*
    @brief 检查参数合法性
    */
    static OckVsaErrorCode CheckValid(const OckVsaAnnCreateParam &param);

    static std::shared_ptr<OckVsaAnnCreateParam> Create(
            const cpu_set_t &cpuSet,
            hmm::OckHmmDeviceId deviceId,
            uint64_t maxFeatureRowCount = 16777216ULL,
            uint32_t tokenNum = 2500UL,
            uint32_t extKeyAttrsByteSize = 22UL,
            uint32_t extKeyAttrBlockSize = 262144UL);
};

std::ostream &operator << (std::ostream &os, const OckVsaAnnCreateParam &param);
std::ostream &operator << (std::ostream &os, const cpu_set_t &cpuSet);
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif