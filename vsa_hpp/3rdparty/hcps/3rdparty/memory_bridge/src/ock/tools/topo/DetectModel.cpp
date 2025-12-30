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

#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/DetectModel.h"

namespace ock {
namespace tools {
namespace topo {

bool FromString(DetectModel &result, const std::string &data)
{
    std::string dataUpCase = utils::ToUpper(data);
    if (dataUpCase == "PARALLEL") {
        result = DetectModel::PARALLEL;
        return true;
    }
    if (dataUpCase == "SERIAL") {
        result = DetectModel::SERIAL;
        return true;
    }
    return false;
}
std::ostream &operator<<(std::ostream &os, DetectModel model)
{
    switch (model) {
        case DetectModel::PARALLEL:
            os << "PARALLEL";
            break;
        case DetectModel::SERIAL:
            os << "SERIAL";
            break;
        case DetectModel::UNKNOWN:
        default:
            os << "UNKNOWN";
            break;
    }
    return os;
}
}  // namespace topo
}  // namespace tools
}  // namespace ock