; /*
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

#ifndef OCK_MEMORY_BRIDGE_OCK_GET_BUFFER_PARAM_H
#define OCK_MEMORY_BRIDGE_OCK_GET_BUFFER_PARAM_H
#include <ostream>
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"
#include "ock/hmm/mgr/OckHmmHMObjectExt.h"
#
namespace ock {
namespace hmm {

class OckGetBufferParam : public acladapter::OckAsyncTaskParamBase {
public:
    virtual ~OckGetBufferParam() noexcept = default;
    explicit OckGetBufferParam(
        std::shared_ptr<OckHmmSubMemoryAlloc> memoryAlloc, uint64_t bufferOffset, uint64_t size,
            OckHmmHMObjectExt &srcHmoObj);

    std::shared_ptr<OckHmmSubMemoryAlloc> MemAlloc(void);
    uint64_t Offset(void) const;
    uint64_t Length(void) const;
    OckHmmHMObjectExt &SrcHmo(void);
    const OckHmmHMObjectExt &SrcHmo(void) const;

private:
    std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc;
    uint64_t offset;
    uint64_t length;
    OckHmmHMObjectExt &srcHmo;
};
std::ostream &operator<<(std::ostream &os, const OckGetBufferParam &param);
}  // namespace hmm
}  // namespace ock
#endif