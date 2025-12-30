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

#include "IReduction.h"

#include <algorithm>

#include "ascend/custom/NNReduction.h"
#include "ascend/custom/PcarReduction.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
const std::vector<std::string> TYPES = { "NN", "PCAR" };

IReduction *CreateReduction(std::string typeName, const ReductionConfig &config)
{
    APP_LOG_INFO("CreateReduction operation started.\n");
    FAISS_THROW_IF_NOT_MSG(std::find(TYPES.begin(), TYPES.end(), typeName) != TYPES.end(),
        "Unsupported typeName, not in {NN, PCAR}.");
    if (typeName == "NN") {
        return new NNReduction(config.deviceList, config.model, config.modelSize);
    } else {
        return new PcarReduction(config.dimIn, config.dimOut, config.eigenPower, config.randomRotation);
    }
    APP_LOG_INFO("CreateReduction operation finished.\n");
};
} // ascend
} // faiss
