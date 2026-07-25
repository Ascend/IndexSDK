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

#include "acl/acl.h"
#include "ascenddaemon/utils/MemDebug.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/SocUtils.h"

namespace ascend
{
const size_t BYTE_OFFSET = 32;

namespace
{
void AllocWithDebug(MemorySpace space, void **const p, size_t size, aclrtMemMallocPolicy policy)
{
    int deviceId = -1;
    (void)aclrtGetDevice(&deviceId);

    size_t freeBefore = 0;
    size_t totalBefore = 0;
    if (MemDebugEnabled())
    {
        (void)QueryHbm(&freeBefore, &totalBefore);
        (void)NoteAlloc(space, size, deviceId, freeBefore, totalBefore);
    }

    aclError err = aclrtMalloc(p, size, policy);
    if (err != ACL_ERROR_NONE)
    {
        if (MemDebugEnabled())
        {
            DumpAllocRingOnFailure(space, size, static_cast<int>(err), deviceId);
        }
        ASCEND_THROW_FMT(
            "failed to aclrtMalloc %zu bytes (error %d) space=%s device=%d "
            "HBM_free_before=%zu HBM_total_before=%zu\n",
            size, static_cast<int>(err), MemorySpaceName(space), deviceId, freeBefore, totalBefore);
    }
}
}  // namespace

void AllocMemorySpaceV(MemorySpace space, void **const p, size_t size)
{
    switch (space)
    {
        case MemorySpace::DEVICE:
        {
            AllocWithDebug(space, p, size, ACL_MEM_MALLOC_NORMAL_ONLY);
            break;
        }
        case MemorySpace::DEVICE_HUGEPAGE:
        {
            AllocWithDebug(space, p, size, ACL_MEM_MALLOC_HUGE_FIRST);
            break;
        }
        default:
            ASCEND_THROW_MSG("Unsupported memoryspace type\n");
    }
}

void FreeMemorySpace(MemorySpace, void *p) { (void)aclrtFree(p); }
}  // namespace ascend
