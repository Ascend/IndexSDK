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


#ifndef HCPS_OCKMODELPATH_H
#define HCPS_OCKMODELPATH_H
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
namespace ock {
namespace hcps {

class OckModelPath {
public:
    virtual ~OckModelPath() noexcept = default;

    virtual void SetPath(const std::string &path) = 0;

    virtual std::string Path(void) const = 0;

    static OckModelPath &Instance(void);
};

std::ostream &operator<<(std::ostream &os, const OckModelPath &modelPath);
}  // namespace hcps
}  // namespace ock
#endif // HCPS_OCKMODELPATH_H
