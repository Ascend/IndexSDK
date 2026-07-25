/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "ascenddaemon/utils/MemDebug.h"

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "acl/acl.h"
#include "common/utils/LogUtils.h"

namespace ascend
{
namespace
{

constexpr size_t kRingCapacity = 64;
constexpr size_t kDefaultEvery = 64;

struct AllocRecord
{
    uint64_t seq{0};
    size_t size{0};
    MemorySpace space{MemorySpace::DEVICE};
    int deviceId{-1};
    size_t freeBefore{0};
    size_t totalBefore{0};
};

std::atomic<int> gEnabled{-1};  // -1 unset, 0 off, 1 on
std::atomic<size_t> gEvery{0};  // 0 unset
std::atomic<uint64_t> gAllocSeq{0};

std::mutex gRingMu;
AllocRecord gRing[kRingCapacity];
size_t gRingCount{0};
size_t gRingNext{0};

bool ParseEnabled()
{
    const char *env = std::getenv("ASCENDFAISS_MEM_DEBUG");
    if (env == nullptr || env[0] == '\0')
    {
        return false;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "FALSE") != 0 && std::strcmp(env, "OFF") != 0;
}

size_t ParseEvery()
{
    const char *env = std::getenv("ASCENDFAISS_MEM_DEBUG_EVERY");
    if (env == nullptr || env[0] == '\0')
    {
        return kDefaultEvery;
    }
    char *end = nullptr;
    errno = 0;
    unsigned long v = std::strtoul(env, &end, 10);
    if (end == env || errno == ERANGE || v == 0)
    {
        return kDefaultEvery;
    }
    return static_cast<size_t>(v);
}

void MemDebugPrintImpl(const char *fmt, va_list args)
{
    va_list args2;
    va_copy(args2, args);
    std::vfprintf(stderr, fmt, args);
    std::fflush(stderr);
    char buf[1024];
    int n = std::vsnprintf(buf, sizeof(buf), fmt, args2);
    va_end(args2);
    if (n > 0)
    {
        APP_LOG_INFO("%s", buf);
    }
}

}  // namespace

bool MemDebugEnabled()
{
    int cached = gEnabled.load(std::memory_order_relaxed);
    if (cached < 0)
    {
        bool on = ParseEnabled();
        gEnabled.store(on ? 1 : 0, std::memory_order_relaxed);
        return on;
    }
    return cached == 1;
}

size_t MemDebugEvery()
{
    size_t cached = gEvery.load(std::memory_order_relaxed);
    if (cached == 0)
    {
        size_t every = ParseEvery();
        gEvery.store(every, std::memory_order_relaxed);
        return every;
    }
    return cached;
}

void MemDebugPrintf(const char *fmt, ...)
{
    if (!MemDebugEnabled())
    {
        return;
    }
    va_list args;
    va_start(args, fmt);
    MemDebugPrintImpl(fmt, args);
    va_end(args);
}

const char *MemorySpaceName(MemorySpace space)
{
    switch (space)
    {
        case MemorySpace::DEVICE:
            return "DEVICE";
        case MemorySpace::DEVICE_HUGEPAGE:
            return "DEVICE_HUGEPAGE";
        default:
            return "UNKNOWN";
    }
}

bool QueryHbm(size_t *freeBytes, size_t *totalBytes)
{
    if (freeBytes == nullptr || totalBytes == nullptr)
    {
        return false;
    }
    *freeBytes = 0;
    *totalBytes = 0;
    aclError err = aclrtGetMemInfo(ACL_HBM_MEM, freeBytes, totalBytes);
    return err == ACL_ERROR_NONE;
}

void LogHbm(const char *tag, int deviceId)
{
    if (!MemDebugEnabled())
    {
        return;
    }
    int curDev = -1;
    (void)aclrtGetDevice(&curDev);
    if (deviceId >= 0 && deviceId != curDev)
    {
        (void)aclrtSetDevice(deviceId);
    }
    size_t freeB = 0;
    size_t totalB = 0;
    bool ok = QueryHbm(&freeB, &totalB);
    int reportDev = deviceId >= 0 ? deviceId : curDev;
    if (ok)
    {
        MemDebugPrintf("[MemDebug] %s device=%d HBM free=%zu (%.2f GiB) total=%zu (%.2f GiB)\n",
                       tag == nullptr ? "" : tag, reportDev, freeB, freeB / (1024.0 * 1024.0 * 1024.0), totalB,
                       totalB / (1024.0 * 1024.0 * 1024.0));
    }
    else
    {
        MemDebugPrintf("[MemDebug] %s device=%d aclrtGetMemInfo(ACL_HBM_MEM) failed\n", tag == nullptr ? "" : tag,
                       reportDev);
    }
    if (deviceId >= 0 && deviceId != curDev && curDev >= 0)
    {
        (void)aclrtSetDevice(curDev);
    }
}

uint64_t NoteAlloc(MemorySpace space, size_t size, int deviceId, size_t freeBefore, size_t totalBefore)
{
    uint64_t seq = gAllocSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!MemDebugEnabled())
    {
        return seq;
    }

    {
        std::lock_guard<std::mutex> lock(gRingMu);
        gRing[gRingNext] = AllocRecord{seq, size, space, deviceId, freeBefore, totalBefore};
        gRingNext = (gRingNext + 1) % kRingCapacity;
        if (gRingCount < kRingCapacity)
        {
            ++gRingCount;
        }
    }

    size_t every = MemDebugEvery();
    if (every > 0 && (seq % every == 0 || size <= 4096))
    {
        MemDebugPrintf("[MemDebug] alloc seq=%llu size=%zu space=%s device=%d HBM_free_before=%zu HBM_total=%zu\n",
                       static_cast<unsigned long long>(seq), size, MemorySpaceName(space), deviceId, freeBefore,
                       totalBefore);
    }
    return seq;
}

void DumpAllocRingOnFailure(MemorySpace space, size_t size, int err, int deviceId)
{
    size_t freeB = 0;
    size_t totalB = 0;
    bool ok = QueryHbm(&freeB, &totalB);

    // Always print failure details to stderr even if APP_LOG is filtered.
    std::fprintf(stderr,
                 "[MemDebug] aclrtMalloc FAILED size=%zu space=%s device=%d err=%d "
                 "HBM_free=%zu (%.2f GiB) HBM_total=%zu (%.2f GiB) queryOk=%d allocSeq=%llu\n",
                 size, MemorySpaceName(space), deviceId, err, freeB, freeB / (1024.0 * 1024.0 * 1024.0), totalB,
                 totalB / (1024.0 * 1024.0 * 1024.0), ok ? 1 : 0,
                 static_cast<unsigned long long>(gAllocSeq.load(std::memory_order_relaxed)));
    std::fflush(stderr);
    APP_LOG_ERROR(
        "[MemDebug] aclrtMalloc FAILED size=%zu space=%s device=%d err=%d "
        "HBM_free=%zu HBM_total=%zu queryOk=%d allocSeq=%llu\n",
        size, MemorySpaceName(space), deviceId, err, freeB, totalB, ok ? 1 : 0,
        static_cast<unsigned long long>(gAllocSeq.load(std::memory_order_relaxed)));

    std::lock_guard<std::mutex> lock(gRingMu);
    std::fprintf(stderr, "[MemDebug] dumping last %zu alloc records (oldest->newest):\n", gRingCount);
    if (gRingCount == 0)
    {
        std::fflush(stderr);
        return;
    }
    size_t start = (gRingCount < kRingCapacity) ? 0 : gRingNext;
    for (size_t i = 0; i < gRingCount; ++i)
    {
        const AllocRecord &r = gRing[(start + i) % kRingCapacity];
        std::fprintf(stderr, "[MemDebug]   seq=%llu size=%zu space=%s device=%d freeBefore=%zu totalBefore=%zu\n",
                     static_cast<unsigned long long>(r.seq), r.size, MemorySpaceName(r.space), r.deviceId, r.freeBefore,
                     r.totalBefore);
    }
    std::fflush(stderr);
}

}  // namespace ascend
