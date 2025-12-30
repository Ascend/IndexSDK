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


#ifndef ASCEND_PCAR_REDUCTION_INCLUDED
#define ASCEND_PCAR_REDUCTION_INCLUDED

#include <vector>
#include "ascend/custom/IReduction.h"

namespace faiss {
namespace ascend {
class PcarReductionImpl;

class PcarReduction : public IReduction {
public:
    PcarReduction(int dimIn, int dimOut, float eigenPower, bool randomRotation);

    virtual ~PcarReduction();

    void train(idx_t n, const float *x) const override;

    void reduce(idx_t n, const float *x, float *res) const override;

    // AscendIndex object is NON-copyable
    PcarReduction(const PcarReduction &) = delete;
    PcarReduction &operator = (const PcarReduction &) = delete;

protected:
    std::shared_ptr<PcarReductionImpl> pcarReductionImpl = nullptr;
};
} // ascend
} // faiss
#endif
