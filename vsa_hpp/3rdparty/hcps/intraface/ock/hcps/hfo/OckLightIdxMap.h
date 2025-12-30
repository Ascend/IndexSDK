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


#ifndef OCK_HCPS_HFO_HETERO_LIGHT_IDX_MAP_H
#define OCK_HCPS_HFO_HETERO_LIGHT_IDX_MAP_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/hcps/hfo/OckIdxMapMgr.h"
#include "ock/hcps/hfo/OckOneSideIdxMap.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"

namespace ock {
namespace hcps {
namespace hfo {
/*
@brief: 特别地， 我们认为内部idx是我们能控制的，能从0开始数的连续的数字
而外部的idx是我们不能控制的，是由用户生成的idx。 本OckLightIdxMap基于此设计。
实现一对数据大概使用24字节的数据空间(包括正向和反向映射)
*/
class OckLightIdxMap : public OckIdxMapMgr {
public:
    virtual void SetIdxMap(uint64_t innerIdx, uint64_t outterIdx) = 0;
    virtual void SetRemoved(uint64_t outterIdx) = 0;
    virtual void SetRemovedByInnerId(uint64_t innerIdx) = 0;
    virtual void Delete(uint64_t outterIdx) = 0;
    virtual void DeleteByInnerId(uint64_t innerIdx) = 0;
    // BatchDelete和Delete的区别是增加了尾部拷贝动作和批处理
    virtual void BatchDeleteByInnerId(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        uint32_t count) = 0;
    virtual uint64_t InnerValidSize() = 0;
    virtual bool InOutterMap(uint64_t outterIdx) = 0;
    /*
    @brief 按照内部idx，删除前count条数据。
    特别地：
    1) innerToOutterMap中的数据只打INVALID标记，不做删除处理
    2) outterToInnerMap中的数据也只打INVALID标记，不做删除处理
    */
    virtual void RemoveFrontByInner(uint64_t count) = 0;
    virtual std::shared_ptr<OckHeteroOperatorGroupQueue> CreateRemoveFrontByInnerOps(uint64_t count) = 0;
    /*
    @brief 从指定内部idx位置开始，删除count条数据
    */
    virtual void BatchRemoveByInner(uint64_t innerStartIdx, uint64_t count) = 0;
    /*
    @brief 根据外部Idx列表软删除数据
    */
    virtual std::shared_ptr<OckHeteroOperatorGroupQueue> CreateSetRemovedOps(
        uint64_t count, const uint64_t *outterIdx) = 0;
    /*
    @brief 根据内部Idx的起始值与要删除的数目软删除数据
    */
    virtual std::shared_ptr<OckHeteroOperatorGroupQueue> CreateSetRemovedOps(
        uint64_t count, uint64_t innerStartIdx) = 0;
    /*
    @brief 增加数据的算子
    @param count 增加的数据条数
    @param outterIdx 增加的数据的外部Id的起始位置
    @param innerStartIdx 增加的数据的内部Id的起始值
    */
    virtual std::shared_ptr<OckHeteroOperatorGroup> CreateAddDatasOps(
        uint64_t count, const uint64_t *outterIdx, uint64_t innerStartIdx) = 0;
    /*
    @brief 根据inner 开始位置 获取连续的count条外部idx。
    */
    virtual std::vector<uint64_t> GetOutterIdxs(uint64_t innerStartIdx, uint64_t count) const = 0;
    /*
    @brief 根据inner 开始位置 获取连续的count条外部idx。存放在pOutDataAddr开头的地址中，调用者确保pOutDataAddr的空间足够
    */
    virtual void GetOutterIdxs(uint64_t innerStartIdx, uint64_t count, uint64_t *pOutDataAddr) const = 0;
    virtual void GetOutterIdxs(uint64_t innerStartIdx, uint64_t count, OckOneSideIdxMap &outData) const = 0;
    virtual void GetOutterIdxs(uint64_t* innerStartIdx, uint64_t count, uint64_t *pOutDataAddr) const = 0;

    virtual std::ostream &Print(std::ostream &os) const = 0;
    /*
    @brief 如本类的说明，该类支持的inner idx要求是连续的，需要提前告知idx的大小
    */
    static std::shared_ptr<OckLightIdxMap> Create(uint64_t maxCount, hmm::OckHmmMemoryPool &memPool,
                                                  uint64_t bucketCount = 4000000ULL);
};
std::ostream &operator<<(std::ostream &os, const OckLightIdxMap &idxMap);
}  // namespace hfo
}  // namespace hcps
}  // namespace ock
#endif
