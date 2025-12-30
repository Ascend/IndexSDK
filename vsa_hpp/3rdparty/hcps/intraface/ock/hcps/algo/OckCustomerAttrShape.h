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


#ifndef OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_H
#define OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_H
#include <cstdint>
#include <memory>
#include <ostream>

namespace ock {
namespace hcps {
namespace algo {
/*
@brief 用户自定义属性的分形处理, 一个用户自定义属性有m个子属性，每个子属性都只有一个字节
    OckCustomerAttrShape 管理的是多组分形后的数据。每一组我们称之为一个block
    我们定义attr[m,n]为第m个属性的第n条数据，分形后的内存分布情况为
    block1: attr[0,0],attr[0,1],...,attr[0,n],attr[1,0],...,attr[m,n]
    block2: attr[0,0],attr[0,1],...,attr[0,n],attr[1,0],...,attr[m,n]
    block3: attr[0,0],attr[0,1],...,attr[0,n],attr[1,0],...,attr[m,n]
    ...
    每个block的大小是相同的
@param _IdMapT 为idMap映射的类  _IdMapT具备uint64_t类型的pos属性
*/
template <typename _IdMapT> class OckCustomerAttrShape {
public:
    /*
    @param addr 分形数据的起始地址
    @param attrCount 用户自定义属性的字节数(m)
    @param blockCount block数目
    @param blockRowCount 每个block中的数据条数(n)
    */
    OckCustomerAttrShape(uint8_t *address, uint32_t attributeCount, uint32_t blockNum, uint64_t blockRowNum);

    /*
    @brief 将other数据按照idMap重排，
    调用者保障rowCount正确(地址空间不越界), 通常attrCount的大小≥0.此处追求高性能
    代码应该确保0好使，同时实现attrCount=22时的高性能(此时：blockRowCount = 262144)
    用户调用此接口时， 会利用blockId做多线程调度，用户会保证每个线程访问的地址不会冲突
    @param other 代表数据源
    @param idMap 是id映射表，重排后的第i条数据来源于other的第idMap[i]->pos条(其中pos为uint64_t类型)
    @param blockId 表示this的数据位置，代表需要计算[blockRowCount*blockId, blockRowCount*blockId + blockRowCount)
          之间的数据
    */
    void CopyFrom(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId);
    void CopyFromZeroAttr(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId);
    void CopyFromOneAttr(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId);

    // Boost需确保blockRowCount为16的倍数
    // 两种写法分别在ARM和X86上比general提升约30%性能
    void CopyFromBoost(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId);
    void CopyFromGeneral(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId);

private:
    uint8_t *addr;
    const uint32_t attrCount;
    const uint32_t blockCount;
    const uint64_t blockRowCount;
    const uint64_t blockSize;
};
} // namespace algo
} // namespace hcps
} // namespace ock
#include "ock/hcps/algo/impl/OckCustomerAttrShapeImpl.h"
#endif // OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_H