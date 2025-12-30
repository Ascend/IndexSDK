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


#include "ascenddaemon/utils/MemorySpace.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/SocUtils.h"

#include "acl/acl.h"

namespace ascend {
const size_t BYTE_OFFSET = 32;

void AllocMemorySpaceV(MemorySpace space, void** const p, size_t size)
{
    switch (space) {
        case MemorySpace::DEVICE: {
            aclError err = aclrtMalloc(p, size, ACL_MEM_MALLOC_NORMAL_ONLY);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE,
                                    "failed to aclrtMalloc %zu bytes (error %d)\n",
                                    size, (int)err);
            break;
        }
        case MemorySpace::DEVICE_HUGEPAGE: {
            aclError err = aclrtMalloc(p, size, ACL_MEM_MALLOC_HUGE_FIRST);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE,
                                    "failed to aclrtMalloc %zu bytes (error %d)\n",
                                    size, (int)err);
            break;
        }
        default:
            ASCEND_THROW_MSG("Unsupported memoryspace type\n");
    }
}

void FreeMemorySpace(MemorySpace, void *p)
{
    (void) aclrtFree(p);
}
}  // namespace ascend
