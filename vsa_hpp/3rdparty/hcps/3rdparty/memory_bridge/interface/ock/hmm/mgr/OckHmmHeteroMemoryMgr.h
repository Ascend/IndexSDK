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


#ifndef OCK_MEMORY_BRIDGE_MULTI_DEVICE_HETERO_MEMROY_MGR_H
#define OCK_MEMORY_BRIDGE_MULTI_DEVICE_HETERO_MEMROY_MGR_H
#include <memory>
#include <thread>
#include <vector>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
#include "ock/hmm/mgr/OckHmmMemorySpecification.h"
#include "ock/hmm/mgr/OckHmmTrafficStatisticsInfo.h"
#include "ock/hmm/mgr/OckHmmMemoryPool.h"

namespace ock {
namespace hmm {
struct OckHmmPureDeviceInfo {
    OckHmmDeviceId deviceId{ 65535U };
    cpu_set_t cpuSet{};
    OckHmmMemorySpecification memorySpec{};
};

struct OckHmmDeviceInfo {
    OckHmmDeviceId deviceId{ 65535U };
    cpu_set_t cpuSet{};
    uint32_t transferThreadNum = {0U};
    OckHmmMemorySpecification memorySpec{};
};

std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfo &data);
using OckHmmDeviceInfoVec = std::vector<OckHmmDeviceInfo>;
std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfoVec &data);

std::ostream &operator<<(std::ostream &os, const OckHmmPureDeviceInfo &data);
using OckHmmPureDeviceInfoVec = std::vector<OckHmmPureDeviceInfo>;
std::ostream &operator<<(std::ostream &os, const OckHmmPureDeviceInfoVec &data);

std::ostream &operator<<(std::ostream &os, const cpu_set_t &cpuSet);

class OckHmmHeteroMemoryMgrBase : public OckHmmMemoryPool {
public:
    virtual ~OckHmmHeteroMemoryMgrBase() noexcept = default;
    /*
    @brief HMO对象的分配。单设备同时支持最多MaxHMOCountPerDevice(10万)个HMO对象在线。
    @return：1. 当内存不足时， 返回的std::shared_ptr<HMObject>可能为空
             2. 调用Free接口主动释放HMOObject指向的内存块，如果不主动释放，HMM模块将本Class析构的时候释放相应内存块。
             3. 只有OckHmmErrorCode为Success时，返回的HMO对象才是有效的
    */
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(
        uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy = OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) = 0;
    /*
    @brief 释放HMO对象。
    */
    virtual void Free(std::shared_ptr<OckHmmHMObject> hmo) = 0;
    /*
    @brief 两个HMO对象的数据之间的互相拷贝
    @dstHMO HMO对象必须是本Alloc分配的对象
    @dstOffset 目标位置的起始位置
    @srcHMO HMO对象必须是本Alloc分配的对象
    @srcOffset 源数据的起始位置
    @length 要拷贝的数据长度
    */
    virtual OckHmmErrorCode CopyHMO(
        OckHmmHMObject &dstHMO, uint64_t dstOffset, OckHmmHMObject &srcHMO, uint64_t srcOffset, size_t length) = 0;
    /*
   @brief 获取内存使用信息
   @param fragThreshold 内存碎片认定标准。内存长度低于fragThreshold的连续内存被认为是碎片。
          使用的时候，通常将fragThreshold设置为HMO的大小规格
   */
    virtual std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold) const = 0;
    /*
    @brief 每次获取都返回上次至本次获取期间的流量统计信息
    1. 获取完成后会清零当前计数
    */
    virtual std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds = 10) = 0;
};
/*
@brief 单卡管理对象
*/
class OckHmmSingleDeviceMgr : public OckHmmHeteroMemoryMgrBase {
public:
    virtual ~OckHmmSingleDeviceMgr() noexcept = default;
    /*
    @brief 返回内存规格信息
    */
    virtual const OckHmmMemorySpecification &GetSpecific(void) const = 0;
    /*
    @brief 返回cpuSet信息 HMM模块内部线程都bind在对应的cpuSet上。
    */
    virtual const cpu_set_t &GetCpuSet(void) const = 0;
    /*
    @brief 返回DeviceID。
    */
    virtual OckHmmDeviceId GetDeviceId(void) const = 0;
    /* *
     * @brief 在 allocType 指示的位置新增 byteSize 大小的内存分配
     * @allocType 新增内存的位置
     * @byteSize 新增内存的大小, 最小4G, 最大100G
     */
    virtual OckHmmErrorCode IncBindMemory(OckHmmHeteroMemoryLocation allocType, uint64_t byteSize,
        uint32_t timeout = 0) = 0;
};
class OckHmmMultiDeviceHeteroMemoryMgrBase {
public:
    virtual ~OckHmmMultiDeviceHeteroMemoryMgrBase() noexcept = default;
    /*
    @brief 获取内存使用信息
    @param fragThreshold 内存碎片认定标准。内存长度低于fragThreshold的连续内存被认为是碎片。
           使用的时候，通常将fragThreshold设置为HMO的大小规格
    */
    virtual std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(
        uint64_t fragThreshold, OckHmmDeviceId deviceId) const = 0;
    /*
    @brief 每次获取都返回上次至本次获取期间的流量统计信息
    1. 获取完成后会清零当前计数
    2. 当deviceId为INVALID_DEVICE_ID时， 返回多设备的汇总信息
    3. 当指定设备不存在时，返回空
    */
    virtual std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(
        OckHmmDeviceId deviceId, uint32_t maxGapMilliSeconds = 10) = 0;
    /*
    @brief 返回cpuset信息 HMM模块内部线程都bind在对应的cpuset上。
    @return 当无deviceId对应的cpuset时，返回空
    */
    virtual const cpu_set_t *GetCpuSet(OckHmmDeviceId deviceId) const = 0;
    
    /*
    @brief 指定设备分配HMO对象。单设备同时支持最多MaxHMOCountPerDevice(10万)个HMO对象在线。
    @return：1. 当内存不足时， 返回的std::unique_ptr<HMObject>可能为空
             2. 调用Free接口主动释放HMOObject指向的内存块，如果不主动释放，HMM模块将本Class析构的时候释放相应内存块。
             3. 只有OckHmmErrorCode为Success时，返回的HMO对象才是有效的
    */
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(OckHmmDeviceId deviceId,
        uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy = OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) = 0;
};
/*
@brief 多卡管理对象，不同卡间的HOST内存不共享
*/
class OckHmmComposeDeviceMgr : public OckHmmMultiDeviceHeteroMemoryMgrBase, public OckHmmHeteroMemoryMgrBase {
public:
    using OckHmmHeteroMemoryMgrBase::Alloc;
    using OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc;
    using OckHmmHeteroMemoryMgrBase::GetUsedInfo;
    using OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo;
    using OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo;
    using OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo;

    virtual ~OckHmmComposeDeviceMgr() noexcept = default;
    /*
    @brief 返回内存规格信息
    @return 当无deviceId对应的内存规格时，返回空
    */
    virtual const OckHmmMemorySpecification *GetSpecific(OckHmmDeviceId deviceId) const = 0;
};

/*
@brief 多卡管理对象，不同卡间的内存共享
*/
class OckHmmShareDeviceMgr : public OckHmmHeteroMemoryMgrBase {
public:
    virtual ~OckHmmShareDeviceMgr() noexcept = default;
    /*
    @brief 返回Device的内存规格信息
    @return 当无deviceId对应的内存规格时，返回空
    */
    virtual const OckHmmMemorySpecification *GetDeviceSpecific(OckHmmDeviceId deviceId) const = 0;
    /*
    @brief 返回Host内存规格信息
    */
    virtual const OckHmmMemorySpecification *GetHostSpecific(void) const = 0;
};
std::ostream &operator<<(std::ostream &os, const cpu_set_t &cpuSet);
}  // namespace hmm
}  // namespace ock
#endif