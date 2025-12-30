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


#include "index_custom/IndexIVFSQCIPAicpu.h"

#include <algorithm>

#include "index/IndexIVFSQIPAicpu.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
IndexIVFSQCIPAicpu::IndexIVFSQCIPAicpu(int numList, int dimIn, int dimOut, int nprobes, int64_t resourceSize)
    : IndexIVFSQIPAicpu(numList, dimIn, false, nprobes, resourceSize), dimIn(dimIn), dimOut(dimOut)
{
    ASCEND_THROW_IF_NOT(dimIn % CUBE_ALIGN == 0);
    ASCEND_THROW_IF_NOT(dimOut != 0);
    ASCEND_THROW_IF_NOT(dimOut % CUBE_ALIGN == 0);
    // For reserveMemory
    this->bytesPerVector = dimOut;
}

IndexIVFSQCIPAicpu::~IndexIVFSQCIPAicpu() {}

void IndexIVFSQCIPAicpu::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
    AscendTensor<float16_t, DIMS_1> &trainedDiff)
{
    this->dims = dimOut;
    IndexIVFSQ::updateTrainedValue(trainedMin, trainedDiff);
    this->dims = dimIn;
}

void IndexIVFSQCIPAicpu::updateCompressValue(AscendTensor<float, DIMS_2> &compressValue,
    AscendTensor<int, DIMS_2> &compressIndex)
{
    int compressValueDim0 = compressValue.getSize(0);
    int compressValueDim1 = compressValue.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(compressValueDim0 == (this->dimIn / this->dimOut) && compressValueDim1 == this->dimOut,
        "compress value's shape invalid.(%d, %d)", compressValueDim0, compressValueDim1);
    int compressIndexDim0 = compressIndex.getSize(0);
    int compressIndexDim1 = compressIndex.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(compressIndexDim1 == (this->dimIn / this->dimOut) && compressIndexDim0 == this->dimOut,
        "compress info value's shape invalid.(%d, %d)", compressIndexDim0, compressIndexDim1);

    vcompressValue = std::move(compressValue);
    vcompressIndex = std::move(compressIndex);
}

int IndexIVFSQCIPAicpu::getDim() const
{
    return dimOut;
}

int IndexIVFSQCIPAicpu::getDimIn() const
{
    return dimIn;
}

void IndexIVFSQCIPAicpu::changeToDimOut()
{
    this->dims = dimOut;
}

void IndexIVFSQCIPAicpu::changeToDimIn()
{
    this->dims = dimIn;
}
} // ascend
