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


#ifndef ASCEND_INDEXIVFSQC_IP_AICPU_INCLUDED
#define ASCEND_INDEXIVFSQC_IP_AICPU_INCLUDED

#include "index/IndexIVFSQIPAicpu.h"

namespace ascend {
class IndexIVFSQCIPAicpu : public IndexIVFSQIPAicpu {
public:
    IndexIVFSQCIPAicpu(int numList, int dimIn, int dimOut, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFSQCIPAicpu();

    void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
        AscendTensor<float16_t, DIMS_1> &trainedDiff) override;

    void updateCompressValue(AscendTensor<float, DIMS_2> &compressValue, AscendTensor<int, DIMS_2> &compressIndex);
 
    int getDim() const;

    int getDimIn() const;

    void changeToDimOut();

    void changeToDimIn();

protected:
    int dimIn;
    int dimOut;

    AscendTensor<float, DIMS_2> vcompressValue;
    AscendTensor<int, DIMS_2> vcompressIndex;
};
} // namespace ascend
#endif // ASCEND_INDEXIVFSQC_IP_AICPU_INCLUDED
