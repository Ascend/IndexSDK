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


#ifndef ASCEND_IREDUCTION_INCLUDED
#define ASCEND_IREDUCTION_INCLUDED

#include <vector>
#include <faiss/Index.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
class IReduction {
public:
    virtual ~IReduction() = default;

    virtual void train(idx_t n, const float *x) const = 0;

    virtual void reduce(idx_t n, const float *x, float *res) const = 0;
};

struct ReductionConfig {
    // PCAR
    inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation)
        : dimIn(dimIn), dimOut(dimOut), eigenPower(eigenPower), randomRotation(randomRotation) {}

    // NN
    inline ReductionConfig(std::vector<int> deviceList, const char *model, uint64_t modelSize)
        : deviceList(deviceList), model(model), modelSize(modelSize) {}

    // PCAR
    int dimIn;
    int dimOut;
    float eigenPower;
    // random rotation after PCA
    bool randomRotation;

    // NN降维
    std::vector<int> deviceList;
    const char *model;
    uint64_t modelSize;
};

IReduction *CreateReduction(std::string typeName, const ReductionConfig &config);
} // ascend
} // faiss
#endif