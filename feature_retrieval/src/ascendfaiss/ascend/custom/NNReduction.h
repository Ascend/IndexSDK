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


#ifndef ASCEND_NN_REDUCTION_INCLUDED
#define ASCEND_NN_REDUCTION_INCLUDED

#include <vector>
#include "ascend/custom/IReduction.h"

namespace faiss {
namespace ascend {
class NNReductionImpl;

class NNReduction : public IReduction {
public:
    NNReduction(std::vector<int> deviceList, const char *model, uint64_t modelSize);

    virtual ~NNReduction();

    void train(idx_t n, const float *x) const override;

    void reduce(idx_t n, const float *x, float *res) const override;

    // AscendIndex object is NON-copyable
    NNReduction(const NNReduction &) = delete;

    NNReduction &operator = (const NNReduction &) = delete;

protected:
    std::shared_ptr<NNReductionImpl> nnReductionImpl = nullptr;
};
} // ascend
} // faiss
#endif