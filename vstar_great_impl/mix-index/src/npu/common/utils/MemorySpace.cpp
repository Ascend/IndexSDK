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

#include <npu/common/utils/AscendAssert.h>
#include <npu/common/utils/SocUtils.h>
#include <npu/common/utils/MemorySpace.h>
#include <npu/common/utils/AscendUtils.h>

#include "acl/acl.h"

namespace ascendSearchacc {
namespace {
const size_t BYTE_OFFSET = 32;
const bool ASCEND_310_SOC = ascendSearchacc::SocUtils::GetInstance().IsAscend310();
}  // namespace

void AllocMemorySpaceV(MemorySpace space, void **p, size_t sizeTmp)
{
    if (!ASCEND_310_SOC) {
        sizeTmp = sizeTmp > BYTE_OFFSET ? sizeTmp - BYTE_OFFSET : sizeTmp;
    }

    switch (space) {
        case MemorySpace::DEVICE: {
            aclError err = aclrtMalloc(p, sizeTmp, ACL_MEM_MALLOC_NORMAL_ONLY);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE, "failed to aclrtMalloc %zu bytes (error %d)\n", sizeTmp,
                                    static_cast<int>(err));
            break;
        }
        case MemorySpace::DEVICE_HUGEPAGE: {
            aclError err = aclrtMalloc(p, sizeTmp, ACL_MEM_MALLOC_HUGE_FIRST);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE, "failed to aclrtMalloc %zu bytes (error %d)\n", sizeTmp,
                                    static_cast<int>(err));
            break;
        }
        default:
            ASCEND_THROW_MSG("Unsupported memoryspace type\n");
    }
}

void FreeMemorySpace(MemorySpace space, void *p)
{
    VALUE_UNUSED(space);
    (void)aclrtFree(p);
}
}  // namespace ascendSearchacc
