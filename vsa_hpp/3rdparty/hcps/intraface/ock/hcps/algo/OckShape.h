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


#ifndef OCK_HCPS_ALGO_SHAPE_H
#define OCK_HCPS_ALGO_SHAPE_H
#include <cstdint>
#include <memory>
#include <ostream>

namespace ock {
namespace hcps {
namespace algo {

/*
@param DataTemp 数据类型
@param DimSizeData 数据维度
@param CubeAlignTemp Cube的数据对齐行数
@param CubeBandWidthBytesTemp Cube的每个通道的寄存器大小(32字节)
*/
template <typename DataTemp = int8_t, uint64_t DimSizeData = 256ULL, uint64_t CubeAlignTemp = 16ULL,
    uint64_t CubeBandWidthBytesTemp = 32ULL>
class OckShape {
public:
    OckShape(uintptr_t address, uint64_t byteCount, uint64_t usedRowCount = 0);
    void AddData(const DataTemp *pSrc);
    void AddData(const DataTemp *pSrc, uint64_t rowCount);
    void GetData(uint64_t rowId, DataTemp *pOut) const;
    void Restore(DataTemp *pOut) const;
    /*
    @brief 拷贝数据
    */
    void CopyFrom(uint64_t curRowId, const OckShape<DataTemp, DimSizeData> &other, uint64_t otherRowId);
    /*
    @brief 一次增加一条，需要注意不是分形块起始位置问题
    */
    void AddFrom(const OckShape<DataTemp, DimSizeData> &other, uint64_t otherRowId);
    /*
    @brief 一次增加多条向量
    */
    void AddSegment(const OckShape<DataTemp, DimSizeData> &other, uint64_t fromRowId, uint64_t rowCount);
    /*
    @brief 获取一段数据，且确保每次增加都满足最小分形块大小, 内部实现可以直接copy就可以了。
    @fromRowId 能整除最小分形块的行数
    @rowCount 能整除最小分形块的行数
    @dstData 存放目标数据的起始地址
    @maxDstDataSize dstData的内存大小
    */
    void GetSegment(uint64_t fromRowId, uint64_t rowCount, DataTemp *dstData, uint64_t maxDstDataSize);

    uint64_t ValidateRowCount(void) const;

    uint64_t ByteSize(void) const;

    uintptr_t Addr(void) const;

    template <typename DataT, uint64_t DimSize, uint64_t CubeAlign, uint64_t CubeBandWidthBytes>
    friend std::ostream &operator << (std::ostream &, const OckShape<DataT, DimSize, CubeAlign, CubeBandWidthBytes> &);

private:
    uint64_t validateRowCount;
    uint64_t subDim;
    uint64_t byteSize;
    uintptr_t addr;
    uint64_t halfCubeBandWidthBytes;
    uint64_t subNum;
    uintptr_t currentAddr;
};
} // namespace algo
} // namespace hcps
} // namespace ock
#include "ock/hcps/algo/impl/OckShapeImpl.h"
#endif // OCK_HCPS_ALGO_SHAPE_H
