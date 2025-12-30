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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_FREE_PARAM_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_FREE_PARAM_H
#include <ostream>
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
namespace ock {
namespace acladapter {

class OckMemoryFreeParam : public OckAsyncTaskParamBase {
public:
    virtual ~OckMemoryFreeParam() noexcept = default;
    explicit OckMemoryFreeParam(void *address, hmm::OckHmmHeteroMemoryLocation memLocation);

    OckMemoryFreeParam(const OckMemoryFreeParam &) = delete;
    OckMemoryFreeParam &operator=(const OckMemoryFreeParam &) = delete;

    void *Addr(void);
    const void *Addr(void) const;
    hmm::OckHmmHeteroMemoryLocation Location(void) const;

private:
    void *addr;
    hmm::OckHmmHeteroMemoryLocation location;
};

std::ostream &operator<<(std::ostream &os, const OckMemoryFreeParam &param);
}  // namespace acladapter
}  // namespace ock
#endif