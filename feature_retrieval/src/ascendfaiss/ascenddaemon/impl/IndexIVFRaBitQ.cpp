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

#include "ascenddaemon/impl/IndexIVFRaBitQ.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/CoarseCenterVerify.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/MemDebug.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend
{
namespace
{
enum class RabitqSelectorFilterStage
{
    AUTO,
    PRE,
    POST,
    BOTH
};

RabitqSelectorFilterStage GetRabitqSelectorFilterStage()
{
    const char *env = std::getenv("IVFRABITQ_SELECTOR_FILTER_STAGE");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0)
    {
        return RabitqSelectorFilterStage::AUTO;
    }
    if (std::strcmp(env, "pre") == 0 || std::strcmp(env, "prefilter") == 0)
    {
        return RabitqSelectorFilterStage::PRE;
    }
    if (std::strcmp(env, "post") == 0 || std::strcmp(env, "postfilter") == 0)
    {
        return RabitqSelectorFilterStage::POST;
    }
    if (std::strcmp(env, "both") == 0)
    {
        return RabitqSelectorFilterStage::BOTH;
    }
    APP_LOG_WARNING("Invalid IVFRABITQ_SELECTOR_FILTER_STAGE=%s, use auto instead.\n", env);
    return RabitqSelectorFilterStage::AUTO;
}

bool IsRabitqL2PrefilterKernelSupported()
{
    const auto &soc = faiss::ascend::SocUtils::GetInstance();
    return !soc.IsAscend310P() && !soc.IsAscendA5();
}

float CpuL2Sq(const float *a, const float *b, int dim)
{
    float sum = 0.0f;
    for (int j = 0; j < dim; ++j)
    {
        const float d = a[j] - b[j];
        sum += d * d;
    }
    return sum;
}

void CpuTopkProbeIds(const float *dists, int numLists, int k, std::vector<int64_t> &outIds)
{
    std::vector<std::pair<float, int>> pairs(static_cast<size_t>(numLists));
    for (int i = 0; i < numLists; ++i)
    {
        pairs[static_cast<size_t>(i)] = {dists[i], i};
    }
    const int topk = std::min(k, numLists);
    std::partial_sort(pairs.begin(), pairs.begin() + topk, pairs.end());
    outIds.resize(static_cast<size_t>(topk));
    for (int i = 0; i < topk; ++i)
    {
        outIds[static_cast<size_t>(i)] = pairs[static_cast<size_t>(i)].second;
    }
}

int CountProbeOverlap(const std::vector<int64_t> &a, const int64_t *b, int nprobe)
{
    std::unordered_set<int64_t> setA(a.begin(), a.end());
    int overlap = 0;
    for (int i = 0; i < nprobe; ++i)
    {
        if (setA.count(b[i]) != 0)
        {
            ++overlap;
        }
    }
    return overlap;
}

void VerifyL1DistAndProbes(int numLists, int dims, int nprobe, const float *npuDists, const float *rotateQuery,
                           const float *rotateCentroids, const int64_t *npuProbeIds)
{
    std::vector<float> cpuDists(static_cast<size_t>(numLists));
    for (int i = 0; i < numLists; ++i)
    {
        cpuDists[static_cast<size_t>(i)] =
            CpuL2Sq(rotateQuery, rotateCentroids + static_cast<size_t>(i) * static_cast<size_t>(dims), dims);
    }

    const int boundaryIds[] = {8191, 8192, 8193, numLists - 3, numLists - 2, numLists - 1};
    for (int id : boundaryIds)
    {
        if (id < 0 || id >= numLists)
        {
            continue;
        }
        const float cpuVal = cpuDists[static_cast<size_t>(id)];
        const float npuVal = npuDists[id];
        const float absErr = std::fabs(cpuVal - npuVal);
        fprintf(stderr, "[IVFRaBitQ] L1 dist golden id=%d cpu=%.6f npu=%.6f absErr=%.6f\n", id, cpuVal, npuVal, absErr);
    }

    std::vector<int64_t> cpuProbeIds;
    CpuTopkProbeIds(cpuDists.data(), numLists, nprobe, cpuProbeIds);
    const int overlap = CountProbeOverlap(cpuProbeIds, npuProbeIds, nprobe);
    int tile1Max = std::min(8192, numLists);
    int cpuTile2 = 0;
    for (int64_t pid : cpuProbeIds)
    {
        if (pid >= tile1Max)
        {
            ++cpuTile2;
        }
    }
    fprintf(stderr, "[IVFRaBitQ] L1 probe overlap q0: nprobe=%d overlap=%d jaccard=%.3f cpu_tile2=%d\n", nprobe,
            overlap, static_cast<double>(overlap) / static_cast<double>(nprobe), cpuTile2);
}

// The GROUPED window width, resolved from the device rather than configured. It has to match what the
// TopK kernel can actually drain in parallel, which is min(GetCPUNum(ctx), nq): a window wider than that
// just delays every query's finish, and a narrower one leaves threads idle. The min(nq) half is applied by
// BuildBlockSlotPlan, so only the AI CPU count is resolved here.
//
// ACL reports the device's AI CPU count, not the per-context count the kernel sees, so on a card shared by
// several index instances this reads high. That is safe in both directions -- blockSlots is an explicit
// bijection the kernel reads back, so the width only ever costs parallelism, never correctness -- and a
// too-wide window degenerates to the per-block round-robin that LEGACY already represents.
//
// Returns false when the device cannot answer (no device set on this thread yet, an ACL version without the
// attribute, a simulator), leaving `width` untouched so the caller keeps whatever it had. Reporting failure
// rather than silently substituting the constant is what lets the caller retry later: the index is often
// constructed before aclrtSetDevice runs, and a one-shot resolve would pin the fallback for the process's
// lifetime even after the device became available. The return code is checked rather than the value: a
// failed query leaves `aicpuNum` untouched, and trusting it would size every window at 0.
bool TryResolveTopkGroupSize(int64_t &width)
{
    int32_t deviceId = -1;
    if (aclrtGetDevice(&deviceId) != ACL_ERROR_NONE)
    {
        return false;
    }

    int32_t aicpuNum = 0;
    aclError ret = aclrtGetDeviceCapability(deviceId, ACL_DEV_ATTR_AICPU_CORE_NUM, &aicpuNum);
    if (ret != ACL_ERROR_NONE || aicpuNum <= 0)
    {
        APP_LOG_WARNING(
            "aclrtGetDeviceCapability(ACL_DEV_ATTR_AICPU_CORE_NUM) returned %d (value %d) on device %d; "
            "keeping the TopK group window at %ld\n",
            static_cast<int>(ret), aicpuNum, deviceId, static_cast<long>(width));
        return false;
    }
    width = static_cast<int64_t>(aicpuNum);
    return true;
}

int64_t GetLayoutModeEnv(const char *name, int64_t defVal)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return defVal;
    }
    if (std::strcmp(raw, "legacy") == 0)
    {
        return aicpu::RABITQ_LAYOUT_LEGACY;
    }
    if (std::strcmp(raw, "group") == 0 || std::strcmp(raw, "grouped") == 0)
    {
        return aicpu::RABITQ_LAYOUT_GROUPED;
    }
    return defVal;
}

// Read a boolean env switch. Unset or empty keeps defaultOn; "0" is off and anything else is on. Shared by
// the two top-k path toggles so their ON/off reporting cannot drift from what was passed to the kernel in
// the attr slot.

// Read a boolean env switch. Unset or empty keeps defaultOn; exactly "0" is off and anything else is on
// ("00", "0x1", "0abc" are all ON -- leading zeros do not count). Shared by the two top-k path toggles so
// their ON/off reporting cannot drift from what was passed to the kernel in the attr slot. The comparison
// is exact rather than first-character because the docs say "0 = off, anything else = on", and a prefix
// test would silently disable the switch for values the docs describe as ON.
bool IsEnvSwitchOn(const char *name, bool defaultOn)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
    {
        return defaultOn;
    }
    return std::strcmp(env, "0") != 0;
}

// Reservoir chunk size B from env IVFRABITQ_TOPK_REFILL_CHUNK, returned as the log2 the kernel attr
// carries (0 = unset, use the kernel's default). Accepts the VALUE, not the exponent -- "4096", not "12" --
// because the value is what the comments, the logs and the cost model all talk about; making the operator
// convert in their head is how a run gets mislabelled.
//
// Rejects rather than clamps, in all three failure modes (unparseable, not a power of two, out of range).
// This knob exists to answer "does raising B help?", and that question is only answerable if the B that
// ran is the B that was asked for. Silently substituting a legal neighbour would produce a measurement
// attributed to the wrong input, which is worse than no measurement: it is a wrong one that looks right.
// On rejection we warn loudly and fall back to the default, so a typo costs a baseline run, not a crash.
int64_t GetRefillChunkLog2()
{
    const char *env = std::getenv("IVFRABITQ_TOPK_REFILL_CHUNK");
    if (env == nullptr || env[0] == '\0')
    {
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    const long long value = std::strtoll(env, &end, 10);
    const bool parsed = (errno == 0 && end != env && *end == '\0');
    // Power of two, and strictly positive. value & (value - 1) is the standard test; it is only meaningful
    // once value > 0, hence the ordering.
    const bool isPow2 = parsed && value > 0 && (value & (value - 1)) == 0;
    int64_t log2Value = 0;
    if (isPow2)
    {
        while ((1LL << log2Value) < value)
        {
            ++log2Value;
        }
    }
    if (!isPow2 || log2Value < aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_MIN ||
        log2Value > aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_MAX)
    {
        APP_LOG_WARNING(
            "Invalid IVFRABITQ_TOPK_REFILL_CHUNK=%s: expected a power of two in [%lld, %lld]. "
            "Falling back to the default %lld.\n",
            env, 1LL << aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_MIN, 1LL << aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_MAX,
            static_cast<long long>(aicpu::TOPK_IVF_RABITQ_REFILL_DEFAULT));
        return 0;
    }
    return log2Value;
}

// Largest id in `indices`, or 0 for an empty range. Used to track the index-wide id ceiling that licenses
// the reservoir's uint32 candidate ids; see maxStoredId_.
idx_t MaxId(const idx_t *indices, size_t numVecs)
{
    if (indices == nullptr || numVecs == 0)
    {
        return 0;
    }
    return *std::max_element(indices, indices + numVecs);
}
}  // namespace

// Decides which absolute grid slot each block goes to: the i-th block emitted owns slot i, and the slot is
// written back into `slots` at [query][i-th block of that query] so the kernel can find it again. `group` is
// how many consecutive queries stay in flight together -- group=1 emits one query's blocks before touching
// the next, which is exactly the query-major legacy order.
// Kept out of the anonymous namespace so the UT can assert the plan and fillDisOpInputDataByBlock agree; the
// bijection between the two is silent when broken and only shows up as quietly lower recall.
void BuildBlockSlotPlan(const std::vector<int64_t> &blockNumPerQ, size_t maxBlocksPerQ, int64_t group, int64_t *slots)
{
    const size_t batch = blockNumPerQ.size();
    const size_t window = std::min<size_t>(batch, group > 0 ? static_cast<size_t>(group) : 1);
    std::vector<int64_t> emitted(batch, 0);
    int64_t slot = 0;
    for (size_t winStart = 0; winStart < batch; winStart += window)
    {
        const size_t winEnd = std::min(batch, winStart + window);
        bool progressed = true;
        while (progressed)
        {
            progressed = false;
            for (size_t qIdx = winStart; qIdx < winEnd; ++qIdx)
            {
                if (emitted[qIdx] >= blockNumPerQ[qIdx])
                {
                    continue;
                }
                slots[qIdx * maxBlocksPerQ + static_cast<size_t>(emitted[qIdx])] = slot++;
                emitted[qIdx]++;
                progressed = true;
            }
        }
    }
}

const int IVF_RABITQ_BLOCK_SIZE = 16384 * 2;
// Tail allocation granularity, in vectors. A list grows its last segment in these steps instead of
// reserving a whole IVF_RABITQ_BLOCK_SIZE segment up front, so a sparse index stays small.
const size_t IVF_RABITQ_TAIL_GROWTH = IVF_RABITQ_BLOCK_SIZE / 8;
const int SCAN_BIT = 8;
const int LUT_NUM = std::pow(2, SCAN_BIT);
IndexIVFRaBitQ::IndexIVFRaBitQ(int numList, int dim, int nprobes, const std::string &metricType, int64_t resourceSize,
                               int quantBits)
    : IndexIVF(numList, dim * 4, dim, nprobes, resourceSize), blockSize(IVF_RABITQ_BLOCK_SIZE), quantBits_(quantBits)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);
    ASCEND_THROW_IF_NOT_MSG(quantBits_ == 1 || quantBits_ == 2 || quantBits_ == 4,
                            "RaBitQ quantBits must be 1, 2, or 4");
    ASCEND_THROW_IF_NOT_MSG(quantBits_ == 1 || metricType == "L2", "Multi-bit RaBitQ POC only supports L2");
    ASCEND_THROW_IF_NOT_MSG(quantBits_ != 2 || dim <= 512,
                            "2-bit RaBitQ POC supports dimensions up to 512 (query LUT UB limit)");
    ASCEND_THROW_IF_NOT_MSG(quantBits_ != 4 || dim <= 256,
                            "4-bit RaBitQ POC supports dimensions up to 256 (query LUT UB limit)");
    ASCEND_THROW_IF_NOT_MSG(quantBits_ == 1 || (!faiss::ascend::SocUtils::GetInstance().IsAscendA5() &&
                                                !faiss::ascend::SocUtils::GetInstance().IsAscend310P()),
                            "Multi-bit RaBitQ POC only supports the A3/Cube distance path");
    ASCEND_THROW_IF_NOT_MSG(metricType == "L2" || metricType == "IP", "Unsupported metric type!");
    if (metricType == "L2")
    {
        metric = MetricType::METRIC_L2;
    }
    else
    {
        metric = MetricType::METRIC_INNER_PRODUCT;
    }
    isTrained = false;
    searchBatchSizes = {64, 32, 16, 8, 4, 2, 1};  // supported batch size
    listArena_ = std::make_shared<DeviceMemArena>();
    initListStorage();
    originCentroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    originCentroidsOnDevice->resize(numList * dims);
    centroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsOnDevice->resize(numList * dims);
    OrthogonalMatrixOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    OrthogonalMatrixOnDevice->resize(dims * dims);
    CentroidL2OnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    CentroidL2OnDevice->resize(numList);
    LUTMatrixOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    LUTMatrixOnDevice->resize(SCAN_BIT * LUT_NUM * quantBits_);
    CentroidLUTOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    CentroidLUTOnDevice->resize(numList * getCodePartBytes() * LUT_NUM);
    uploadLUTMatrix();
    distResultOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    distResultSpace = CORE_NUM * static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    distResultOnDevice->resize(distResultSpace);
    // The window is the device's AI CPU count: that is how many of a window's queries the TopK kernel can
    // actually drain in parallel, so widening it only delays every finish. See TryResolveTopkGroupSize.
    // The index is routinely constructed before this thread has a device set, so a failure here is normal
    // and not fatal: the constant stands in until the first search retries the query (see
    // EnsureTopkGroupSizeResolved).
    // LEGACY is kept only as the A/B baseline; it is the same fill order as group=1, so it needs no
    // separate path beyond the kernel's contiguous addressing shortcut.
    topkLayoutMode_ = GetLayoutModeEnv("IVFRABITQ_TOPK_LAYOUT", aicpu::RABITQ_LAYOUT_GROUPED);
    topkGroupSize_ = aicpu::TOPK_IVF_RABITQ_AICPU_NUM;
    topkGroupSizeResolved_ = TryResolveTopkGroupSize(topkGroupSize_);
    if (topkLayoutMode_ == aicpu::RABITQ_LAYOUT_LEGACY)
    {
        topkGroupSize_ = 1;
        // Nothing left to resolve: LEGACY emits one query's blocks at a time regardless of the AI CPU count.
        topkGroupSizeResolved_ = true;
    }
    auto ret = resetL1TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1TopkOp failed!");
    ret = resetL1DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1DistOp failed!");
    ret = resetL2DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2DistOp failed!");
    ret = resetL2TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2TopkOp failed!");
    ret = resetCenterRotateL2Op();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetCenterRotateL2Op failed!");
    ret = resetCenterLUTOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetCenterLUTOp failed!");
    ret = resetIndexRotateL2Op();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetIndexRotateL2Op failed!");
    ret = resetIndexCodeAndPreComputeOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetIndexCodeAndPreComputeOp failed!");
    ret = resetQueryRotateL2Op();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetQueryRotateL2Op failed!");
    ret = resetQueryLUTOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetQueryLUTOp failed!");
}

namespace
{
// Rolls a list's segment geometry back to its committed length unless the write is explicitly marked
// complete. ensureListCapacity grows segments_ up front, but every step after it -- three H2D copies, two
// stream syncs, the id copy -- can return early, and none of them commit a new length. Without this the
// list is left with segmentCount == ceil(newLen / blockSize) while listLen still holds the old value, so
// sum(segmentLength) exceeds listLen: getListIds / getListRawCodes / CollectDeletedSlots then read slots
// that were never written, and the host-side copies size their buffers from listLen and overflow them.
//
// Rolling back the segment count (rather than only fixing up lengths) also releases the segments the
// aborted write had just allocated, so a failing add does not permanently inflate the list's footprint.
class ListCapacityGuard
{
   public:
    ListCapacityGuard(IndexIVFRaBitQ::ListStorage &list, size_t blockSize) : list_(list), blockSize_(blockSize) {}

    ~ListCapacityGuard()
    {
        if (committed_)
        {
            return;
        }
        const size_t segments = utils::divUp(list_.listLen, blockSize_);
        list_.codes.resizeSegments(segments);
        list_.indexL1.resizeSegments(segments);
        list_.indexL2.resizeSegments(segments);
        list_.ids.resizeSegments(segments);
    }

    ListCapacityGuard(const ListCapacityGuard &) = delete;
    ListCapacityGuard &operator=(const ListCapacityGuard &) = delete;

    // Call once the four commitLength() calls and listLen have been updated consistently.
    void commit() { committed_ = true; }

   private:
    IndexIVFRaBitQ::ListStorage &list_;
    size_t blockSize_;
    bool committed_{false};
};
}  // namespace

void IndexIVFRaBitQ::initListStorage()
{
    ASCEND_THROW_IF_NOT_MSG(listArena_ != nullptr, "listArena_ is null");

    // Base IndexIVF's deviceListData/deviceListIndices stay unused: RaBitQ keeps codes, factors and ids
    // in lists_ instead.
    lists_.clear();
    lists_.reserve(static_cast<size_t>(numLists));
    // elementsPerSlot for codes is the quantised row width, so it has to follow quantBits_ -- a hardcoded
    // (dims + 7) / 8 would size every slot for 1-bit and truncate 2/4-bit codes.
    const size_t codePartBytes = getCodePartBytes();
    for (int i = 0; i < numLists; ++i)
    {
        lists_.emplace_back(static_cast<size_t>(blockSize), IVF_RABITQ_TAIL_GROWTH, codePartBytes, listArena_);
    }
}

IndexIVFRaBitQ::~IndexIVFRaBitQ()
{
    // Segments hold raw pointers into listArena_, so drop them before the arena member is destroyed.
    lists_.clear();
}

APP_ERROR IndexIVFRaBitQ::reset()
{
    // Must precede the arena reset: the segments being dropped point into its slabs.
    lists_.clear();

    if (listArena_ != nullptr)
    {
        listArena_->Reset();
    }
    else
    {
        listArena_ = std::make_shared<DeviceMemArena>();
    }
    initListStorage();
    maxListLength = 0;
    this->ntotal = 0;
    maxStoredId_ = 0;

    // 聚类中心、正交矩阵、LUT 和质心 L2 是训练结果，reset 后 add 仍要使用，不能释放。
    if (distResultOnDevice != nullptr)
    {
        distResultOnDevice->clear();
        distResultSpace = 0;
    }
    ClearCachedFilterPayload();
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::ClearCachedFilterPayload()
{
    if (cachedFilterPayload != nullptr)
    {
        cachedFilterPayload->clear();
    }
    cachedFilterSrc = nullptr;
    cachedFilterBytes = 0;
    cachedFilterMode = 0;
    cachedFilterNegate = 0;
    cachedFilterAux0 = 0;
    cachedFilterAux1 = 0;
    cachedFilterGeneration = 0;
    if (cachedPrefixFilterPayload != nullptr)
    {
        cachedPrefixFilterPayload->clear();
    }
    cachedPrefixFilterSrc = nullptr;
    cachedPrefixFilterBytes = 0;
    cachedPrefixFilterGeneration = 0;
}

APP_ERROR IndexIVFRaBitQ::EnsureFilterPayloadOnDevice(const RabitqIdFilterHost *idFilter, int64_t &selPtr)
{
    selPtr = 0;
    APPERR_RETURN_IF(idFilter == nullptr, APP_ERR_OK);
    const size_t payloadBytes = idFilter->payloadBytes();
    APPERR_RETURN_IF(payloadBytes == 0, APP_ERR_OK);

    const void *src = idFilter->payloadSrc();
    APPERR_RETURN_IF_NOT_LOG(src != nullptr, APP_ERR_INVALID_PARAM, "id filter payload is empty");

    constexpr size_t kRoaringAlign = aicpu::RABITQ_ROARING_FROZEN_ALIGN;
    const bool needAlign = (idFilter->mode == aicpu::RABITQ_ID_FILTER_ROARING);
    const size_t allocBytes = needAlign ? (payloadBytes + kRoaringAlign) : payloadBytes;

    const bool cacheHit = (cachedFilterPayload != nullptr) && (cachedFilterPayload->size() >= allocBytes) &&
                          (cachedFilterSrc == src) && (cachedFilterBytes == payloadBytes) &&
                          (cachedFilterMode == idFilter->mode) && (cachedFilterNegate == idFilter->negate) &&
                          (cachedFilterAux0 == idFilter->aux0) && (cachedFilterAux1 == idFilter->aux1) &&
                          (cachedFilterGeneration == idFilter->generation);
    auto alignedSelPtr = [&]() -> int64_t
    {
        uint8_t *base = cachedFilterPayload->data();
        if (!needAlign)
        {
            return static_cast<int64_t>(reinterpret_cast<uintptr_t>(base));
        }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
        const size_t off = (kRoaringAlign - (addr % kRoaringAlign)) % kRoaringAlign;
        return static_cast<int64_t>(reinterpret_cast<uintptr_t>(base + off));
    };
    if (cacheHit)
    {
        selPtr = alignedSelPtr();
        return APP_ERR_OK;
    }

    if (cachedFilterPayload == nullptr)
    {
        cachedFilterPayload = std::make_unique<DeviceVector<uint8_t>>();
    }
    if (cachedFilterPayload->size() < allocBytes)
    {
        cachedFilterPayload->resize(allocBytes, true);
    }
    uint8_t *dst = cachedFilterPayload->data();
    if (needAlign)
    {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
        const size_t off = (kRoaringAlign - (addr % kRoaringAlign)) % kRoaringAlign;
        dst += off;
    }
    auto retCopy = aclrtMemcpy(dst, payloadBytes, src, payloadBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(retCopy == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy id filter payload to device failed %d",
                             retCopy);

    cachedFilterSrc = src;
    cachedFilterBytes = payloadBytes;
    cachedFilterMode = idFilter->mode;
    cachedFilterNegate = idFilter->negate;
    cachedFilterAux0 = idFilter->aux0;
    cachedFilterAux1 = idFilter->aux1;
    cachedFilterGeneration = idFilter->generation;
    selPtr = alignedSelPtr();
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::EnsurePrefixFilterPayloadOnDevice(const RabitqIdFilterHost *idFilter, int64_t &selPtr)
{
    selPtr = 0;
    APPERR_RETURN_IF(idFilter == nullptr || idFilter->sortedPrefixPayload.empty(), APP_ERR_OK);

    const void *src = idFilter->sortedPrefixPayload.data();
    const size_t payloadBytes = idFilter->sortedPrefixPayload.size();
    const bool cacheHit = (cachedPrefixFilterPayload != nullptr) &&
                          (cachedPrefixFilterPayload->size() >= payloadBytes) && (cachedPrefixFilterSrc == src) &&
                          (cachedPrefixFilterBytes == payloadBytes) &&
                          (cachedPrefixFilterGeneration == idFilter->generation);
    if (cacheHit)
    {
        selPtr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(cachedPrefixFilterPayload->data()));
        return APP_ERR_OK;
    }

    if (cachedPrefixFilterPayload == nullptr)
    {
        cachedPrefixFilterPayload = std::make_unique<DeviceVector<uint8_t>>();
    }
    if (cachedPrefixFilterPayload->size() < payloadBytes)
    {
        cachedPrefixFilterPayload->resize(payloadBytes, true);
    }
    auto retCopy =
        aclrtMemcpy(cachedPrefixFilterPayload->data(), payloadBytes, src, payloadBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(retCopy == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "copy prefix id filter payload to device failed %d", retCopy);

    cachedPrefixFilterSrc = src;
    cachedPrefixFilterBytes = payloadBytes;
    cachedPrefixFilterGeneration = idFilter->generation;
    selPtr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(cachedPrefixFilterPayload->data()));
    return APP_ERR_OK;
}

// Grows the four vectors of one list to newLen slots. Not transactional on its own: the caller must pair
// it with a ListCapacityGuard so a failure between here and commitLength cannot leave segmentCount and
// listLen disagreeing (see the guard's comment).
void IndexIVFRaBitQ::ensureListCapacity(ListStorage &list, size_t newLen, aclrtStream stream)
{
    list.codes.ensureCapacity(newLen, stream);
    list.indexL1.ensureCapacity(newLen, stream);
    list.indexL2.ensureCapacity(newLen, stream);
    list.ids.ensureCapacity(newLen, stream);
}

size_t IndexIVFRaBitQ::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    return lists_[listId].listLen;
}

void IndexIVFRaBitQ::uploadLUTMatrix()
{
    APP_LOG_INFO("IndexIVFRaBitQ  create LUTMatrix started\n");
    const int lutColumns = LUT_NUM * quantBits_;
    std::vector<float> LUTMatrix(SCAN_BIT * lutColumns, 0);
    for (int i = 0; i < SCAN_BIT; i++)
    {
        int mask = (1 << i);
        for (int j = 0; j < LUT_NUM; j++)
        {
            if (j & mask)
            {
                // The matmul emits adjacent 256-entry tables for bit planes
                // ordered from MSB to LSB (e.g. 4-bit weights 8, 4, 2, 1).
                for (int plane = 0; plane < quantBits_; ++plane)
                {
                    LUTMatrix[i * lutColumns + plane * LUT_NUM + j] = static_cast<float>(1 << (quantBits_ - 1 - plane));
                }
            }
        }
    }

    auto ret = aclrtMemcpy(LUTMatrixOnDevice->data(), LUTMatrixOnDevice->size() * sizeof(float), LUTMatrix.data(),
                           LUTMatrix.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to aclrtMemcpy LUTMatrix, result is: %d\n", ret);
}

APP_ERROR IndexIVFRaBitQ::updateCoarseCenterImpl(std::vector<float> &centerData)
{
    APPERR_RETURN_IF_NOT_FMT(centerData.size() == static_cast<size_t>(dims) * static_cast<size_t>(numLists),
                             APP_ERR_INVALID_PARAM, "the centerData size is %zu, not dims(%d) * numLists(%d) %d",
                             centerData.size(), dims, numLists, dims * numLists);

    using coarseCenterVerify::IsVerifyEnabled;
    using coarseCenterVerify::LogHostSampleRows;
    using coarseCenterVerify::LogMemcpyAudit;
    using coarseCenterVerify::LogStageDone;
    using coarseCenterVerify::VerifyCentroidL2SamplesD2H;
    using coarseCenterVerify::VerifyFullD2H;
    using coarseCenterVerify::VerifySamplesD2H;

    const size_t centroidBytes = static_cast<size_t>(dims) * static_cast<size_t>(numLists) * sizeof(float);
    if (IsVerifyEnabled())
    {
        LogMemcpyAudit("enter", numLists, dims, dims * static_cast<int>(sizeof(float)),
                       centerData.size() * sizeof(float), originCentroidsOnDevice->size() * sizeof(float),
                       centroidBytes);
        LogHostSampleRows("enter", centerData, numLists, dims);
    }

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float, DIMS_2> originCentroids(mem, {numLists, dims}, stream);
    // 聚类中心传到npu
    auto ret = aclrtMemcpy(originCentroids.data(), originCentroids.getSizeInBytes(), centerData.data(),
                           dims * numLists * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to aclrtMemcpy CoarseCenter, result is: %d\n", ret);
    VerifySamplesD2H("originCentroids_temp", centerData, originCentroids.data(), numLists, dims);

    ret = aclrtMemcpy(originCentroidsOnDevice->data(), originCentroidsOnDevice->size() * sizeof(float),
                      centerData.data(), dims * numLists * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to aclrtMemcpy CoarseCenter, result is: %d\n", ret);
    VerifySamplesD2H("originCentroidsOnDevice", centerData, originCentroidsOnDevice->data(), numLists, dims);

    // c = Pcr, ||cr||^2 预计算
    AscendTensor<int32_t, DIMS_1> vectorSizeVec(mem, {1}, stream);
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    AscendTensor<float, DIMS_1> centroidsl2(CentroidL2OnDevice->data(), {numLists});
    std::vector<int32_t> vectorSize(1, 0);
    vectorSize[0] = numLists;
    ret = aclrtMemcpy(vectorSizeVec.data(), sizeof(int32_t), vectorSize.data(), sizeof(int32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy vectorSize to device failed %d", ret);
    runCenterRotateL2Op(originCentroids, vectorSizeVec, rotatematrixDev, centroidsDev, centroidsl2, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    VerifySamplesD2H("centroidsOnDevice_rotated", centerData, centroidsOnDevice->data(), numLists, dims, false);
    VerifyFullD2H("centroidsOnDevice_rotated_full", centerData, centroidsOnDevice->data(), numLists, dims, false);
    VerifyCentroidL2SamplesD2H("CentroidL2", CentroidL2OnDevice->data(), numLists);

    // <x, c> LUT预计算
    AscendTensor<float, DIMS_2> centroidsrotate(centroidsOnDevice->data(), {numLists * dims / SCAN_BIT, SCAN_BIT});
    AscendTensor<float, DIMS_2> lutmatrixDev(LUTMatrixOnDevice->data(), {SCAN_BIT, LUT_NUM * quantBits_});
    AscendTensor<float, DIMS_2> centroidslut(CentroidLUTOnDevice->data(),
                                             {numLists * dims / SCAN_BIT, LUT_NUM * quantBits_});
    runCenterLUTOp(centroidsrotate, lutmatrixDev, centroidslut, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    LogStageDone("LUT precompute");

    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::resetCenterRotateL2Op()
{
    auto CenterRotateL2OpReset = [&](std::unique_ptr<AscendOperator> &op)
    {
        AscendOpDesc desc("RotateAndL2AtFP32");
        std::vector<int64_t> codesShape({numLists, dims});
        std::vector<int64_t> sizeShape({1});
        std::vector<int64_t> matrixShape({dims, dims});

        std::vector<int64_t> rotateResultShape({numLists, dims});
        std::vector<int64_t> l2ResultShape({numLists});
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRaBitQRotateL2Ops = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(CenterRotateL2OpReset(ivfRaBitQRotateL2Ops), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                             "IVFRaBitQ Rotate L2 FP32 distance op init failed");
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runCenterRotateL2Op(AscendTensor<float, DIMS_2> &centroid,
                                         AscendTensor<int32_t, DIMS_1> &vectorSize, AscendTensor<float, DIMS_2> &matrix,
                                         AscendTensor<float, DIMS_2> &rotateCentroid,
                                         AscendTensor<float, DIMS_1> &centroidl2, aclrtStream stream)
{
    AscendOperator *op = ivfRaBitQRotateL2Ops.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateCentroid.data(), rotateCentroid.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(centroidl2.data(), centroidl2.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetCenterLUTOp()
{
    auto CenterLUTOpReset = [&](std::unique_ptr<AscendOperator> &op)
    {
        AscendOpDesc desc("MatmulAtFP32");
        std::vector<int64_t> codesShape({numLists * dims / SCAN_BIT, SCAN_BIT});
        std::vector<int64_t> matrixShape({SCAN_BIT, LUT_NUM * quantBits_});

        std::vector<int64_t> lutResultShape({numLists * dims / SCAN_BIT, LUT_NUM * quantBits_});
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, lutResultShape.size(), lutResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfCenterLUTOps = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(CenterLUTOpReset(ivfCenterLUTOps), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                             "IVFRaBitQ Center LUT FP32 op init failed");
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runCenterLUTOp(AscendTensor<float, DIMS_2> &centroid, AscendTensor<float, DIMS_2> &matrix,
                                    AscendTensor<float, DIMS_2> &centroidlut, aclrtStream stream)
{
    AscendOperator *op = ivfCenterLUTOps.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(centroidlut.data(), centroidlut.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    ClearCachedFilterPayload();
    auto &list = lists_[listId];
    const size_t oldLen = list.listLen;
    const size_t newLen = oldLen + numVecs;
    if (MemDebugEnabled())
    {
        static std::atomic<uint64_t> addLogSeq{0};
        uint64_t seq = addLogSeq.fetch_add(1, std::memory_order_relaxed) + 1;
        size_t every = MemDebugEvery();
        if (every > 0 && (seq % every == 0 || numVecs >= 100000))
        {
            MemDebugPrintf("[MemDebug] IndexIVFRaBitQ::addVectors listId=%d numVecs=%zu prevLen=%zu\n", listId, numVecs,
                           oldLen);
            LogHbm("IndexIVFRaBitQ_addVectors_before", -1);
        }
    }

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    ensureListCapacity(list, newLen, stream);
    // Any early return between here and the commitLength block below rolls the segment count back.
    ListCapacityGuard capacityGuard(list, static_cast<size_t>(blockSize));
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    AscendTensor<float, DIMS_2> centroidDev(centroidsOnDevice->data() + listId * dims, {1, dims});
    AscendTensor<float, DIMS_1> centroidl2Dev(CentroidL2OnDevice->data() + listId, {1});
    AscendTensor<int32_t, DIMS_1> vectorSizeVec(mem, {1}, stream);
    AscendTensor<float, DIMS_2> indexOriginVec(mem, {blockSize, dims}, stream);
    AscendTensor<float, DIMS_2> indexRotateVec(mem, {blockSize, dims}, stream);
    AscendTensor<float, DIMS_1> indexL2Vec(mem, {blockSize}, stream);
    std::vector<int32_t> vectorSize(1, 0);
    size_t inputOffset = 0;
    // One iteration per segment the input spans: a tile never crosses a segment boundary, so every op
    // writes into a single contiguous allocation.
    while (inputOffset < numVecs)
    {
        const size_t slot = oldLen + inputOffset;
        const size_t count =
            std::min(numVecs - inputOffset, static_cast<size_t>(blockSize) - slot % static_cast<size_t>(blockSize));
        // 原始索引向量传到npu
        auto ret = aclrtMemcpy(indexOriginVec.data(), count * dims * sizeof(float), codes + inputOffset * dims,
                               count * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy index to device failed %d", ret);
        vectorSize[0] = static_cast<int32_t>(count);
        ret = aclrtMemcpy(vectorSizeVec.data(), sizeof(int32_t), vectorSize.data(), sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy vectorSize to device failed %d", ret);
        AscendTensor<float, DIMS_1> indexesl2(list.indexL2.slot(slot), {blockSize});
        AscendTensor<float, DIMS_1> indexesl1(list.indexL1.slot(slot), {blockSize});
        // Row width is getCodePartBytes(), not (dims + 7) / 8: multi-bit quantisation stores quantBits_
        // planes per dimension, so the 1-bit formula would under-size every row by that factor.
        AscendTensor<uint8_t, DIMS_2> indexCodes(list.codes.slot(slot),
                                                 {blockSize, static_cast<int>(getCodePartBytes())});
        // o = Por, ||or||^2 预计算
        runIndexRotateL2Op(indexOriginVec, vectorSizeVec, rotatematrixDev, indexRotateVec, indexL2Vec, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
                                 ret);
        runIndexCodeAndPreComputeOp(vectorSizeVec, indexRotateVec, indexL2Vec, centroidDev, centroidl2Dev, indexCodes,
                                    indexesl2, indexesl1, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
                                 ret);
        ret = aclrtMemcpy(list.ids.slot(slot), count * sizeof(idx_t), indices + inputOffset, count * sizeof(idx_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy ids to device failed %d", ret);
        inputOffset += count;
    }
    list.codes.commitLength(newLen);
    list.indexL1.commitLength(newLen);
    list.indexL2.commitLength(newLen);
    list.ids.commitLength(newLen);
    list.listLen = newLen;
    capacityGuard.commit();
    // From newLen, and after the commit: getListLength() reads list.listLen, which only takes its new value
    // on the line above, so computing this any earlier would fold in the pre-append length and leave
    // maxListLength short of what this call just stored. Mirrors addEncodedVectors.
    maxListLength = std::max(maxListLength, static_cast<int>(newLen));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    IndexIVF::ntotal += numVecs;
    maxStoredId_ = std::max(maxStoredId_, MaxId(indices, numVecs));
    if (MemDebugEnabled() && listArena_ != nullptr)
    {
        static std::atomic<uint64_t> afterAddLogSeq{0};
        uint64_t seq = afterAddLogSeq.fetch_add(1, std::memory_order_relaxed) + 1;
        size_t every = MemDebugEvery();
        if (every > 0 && (seq % every == 0 || numVecs >= 100000))
        {
            MemDebugPrintf(
                "[MemDebug] IndexIVFRaBitQ::addVectors done listId=%d numVecs=%zu listLen=%zu "
                "codeSegments=%zu codeBytes=%zu arenaReserved=%zu slabs=%zu\n",
                listId, numVecs, newLen, list.codes.segmentCount(), list.codes.allocatedBytes(),
                listArena_->TotalReservedBytes(), listArena_->SlabCount());
            LogHbm("IndexIVFRaBitQ_addVectors_after", -1);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::resetIndexRotateL2Op()
{
    auto IndexRotateL2OpReset = [&](std::unique_ptr<AscendOperator> &op)
    {
        AscendOpDesc desc("RotateAndL2AtFP32");
        std::vector<int64_t> codesShape({blockSize, dims});
        std::vector<int64_t> sizeShape({1});
        std::vector<int64_t> matrixShape({dims, dims});

        std::vector<int64_t> rotateResultShape({blockSize, dims});
        std::vector<int64_t> l2ResultShape({blockSize});
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRaBitQIndexRotateL2Ops = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(IndexRotateL2OpReset(ivfRaBitQIndexRotateL2Ops), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                             "IVFRaBitQ Rotate L2 FP32 distance op init failed");
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runIndexRotateL2Op(AscendTensor<float, DIMS_2> &index, AscendTensor<int32_t, DIMS_1> &vectorSize,
                                        AscendTensor<float, DIMS_2> &matrix, AscendTensor<float, DIMS_2> &rotateIndex,
                                        AscendTensor<float, DIMS_1> &indexl2, aclrtStream stream)
{
    AscendOperator *op = ivfRaBitQIndexRotateL2Ops.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(index.data(), index.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateIndex.data(), rotateIndex.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetIndexCodeAndPreComputeOp()
{
    auto IndexCodeAndPreComputeOpReset = [&](std::unique_ptr<AscendOperator> &op)
    {
        AscendOpDesc desc("IndexCodeAndPrecompute");
        std::vector<int64_t> sizeShape({1});
        std::vector<int64_t> indexShape({blockSize, dims});
        std::vector<int64_t> indexl2Shape({blockSize});
        std::vector<int64_t> centroidShape({1, dims});
        std::vector<int64_t> centroidl2Shape({1});

        std::vector<int64_t> indexCodeShape({blockSize, static_cast<int64_t>(getCodePartBytes())});
        std::vector<int64_t> l2ResultShape({blockSize});
        std::vector<int64_t> l1ResultShape({blockSize});
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, indexl2Shape.size(), indexl2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, centroidShape.size(), centroidShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, centroidl2Shape.size(), centroidl2Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT8, indexCodeShape.size(), indexCodeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l1ResultShape.size(), l1ResultShape.data(), ACL_FORMAT_ND);
        if (this->metric == MetricType::METRIC_L2)
        {
            aclopSetAttrInt(desc.opAttr, "metric_type", 0);
        }
        else if (this->metric == MetricType::METRIC_INNER_PRODUCT)
        {
            aclopSetAttrInt(desc.opAttr, "metric_type", 1);
        }
        aclopSetAttrInt(desc.opAttr, "quant_bits", quantBits_);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRaBitQIndexCodeAndPreComputeOps = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(IndexCodeAndPreComputeOpReset(ivfRaBitQIndexCodeAndPreComputeOps),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRaBitQ Index PreCompute FP32 op init failed");
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runIndexCodeAndPreComputeOp(
    AscendTensor<int32_t, DIMS_1> &vectorSize, AscendTensor<float, DIMS_2> &index, AscendTensor<float, DIMS_1> &indexl2,
    AscendTensor<float, DIMS_2> &centroid, AscendTensor<float, DIMS_1> &centroidl2,
    AscendTensor<uint8_t, DIMS_2> &indexCodes, AscendTensor<float, DIMS_1> &l2Result,
    AscendTensor<float, DIMS_1> &l1Result, aclrtStream stream)
{
    AscendOperator *op = ivfRaBitQIndexCodeAndPreComputeOps.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(index.data(), index.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidl2.data(), centroidl2.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(indexCodes.data(), indexCodes.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(l2Result.data(), l2Result.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(l1Result.data(), l1Result.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetQueryRotateL2Op()
{
    auto queryRotateL2OpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("RotateAndL2AtFP32");
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> sizeShape({1});
        std::vector<int64_t> matrixShape({dims, dims});

        std::vector<int64_t> rotateResultShape({batch, dims});
        std::vector<int64_t> l2ResultShape({batch});
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes)
    {
        queryRotateL2Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(queryRotateL2OpReset(queryRotateL2Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L1 distance op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runQueryRotateL2Op(int batch, AscendTensor<float, DIMS_2> &queries,
                                        AscendTensor<int32_t, DIMS_1> &vectorSize, AscendTensor<float, DIMS_2> &matrix,
                                        AscendTensor<float, DIMS_2> &rotateQueries,
                                        AscendTensor<float, DIMS_1> &queryl2, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (queryRotateL2Ops.find(batch) != queryRotateL2Ops.end())
    {
        op = queryRotateL2Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateQueries.data(), rotateQueries.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(queryl2.data(), queryl2.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetQueryLUTOp()
{
    auto queryLUTOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("MatmulAtFP32");
        std::vector<int64_t> queryShape({batch * dims / SCAN_BIT, SCAN_BIT});
        std::vector<int64_t> matrixShape({SCAN_BIT, LUT_NUM * quantBits_});

        std::vector<int64_t> lutResultShape({batch * dims / SCAN_BIT, LUT_NUM * quantBits_});
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, lutResultShape.size(), lutResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes)
    {
        queryLUTOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(queryLUTOpReset(queryLUTOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "IVFRaBitQ Query LUT FP32 op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runQueryLUTOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &matrix,
                                   AscendTensor<float, DIMS_2> &querieslut, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (queryLUTOps.find(batch) != queryLUTOps.end())
    {
        op = queryLUTOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(querieslut.data(), querieslut.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0{1, batch, numLists};
        std::vector<int64_t> shape1{1, batch, numLists / IVF_RABITQ_BURST_LEN * 2};
        std::vector<int64_t> shape2{1, CORE_NUM, SIZE_ALIGN};
        std::vector<int64_t> shape3{1, CORE_NUM, FLAG_SIZE};
        std::vector<int64_t> shape4{aicpu::TOPK_FLAT_ATTR_IDX_COUNT};
        std::vector<int64_t> shape5{batch, 0};

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes)
    {
        topkFp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkFp32[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "topk op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runL1TopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                                 AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                                 AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                                 AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (topkFp32.find(batch) != topkFp32.end())
    {
        op = topkFp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    AscendTensor<float, DIMS_3> distsTopk(dists.data(), {1, batch, numLists});
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(), {1, batch, numLists / IVF_RABITQ_BURST_LEN * 2});
    AscendTensor<uint32_t, DIMS_3> sizesTopk(sizes.data(), {1, CORE_NUM, SIZE_ALIGN});
    AscendTensor<uint16_t, DIMS_3> flagsTopk(flags.data(), {1, CORE_NUM, FLAG_SIZE});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(distsTopk.data(), distsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdistsTopk.data(), vmdistsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizesTopk.data(), sizesTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flagsTopk.data(), flagsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetL1DistOp()
{
    auto l1DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> codesShape({numLists, dims});
        std::vector<int64_t> codesSqrSum({numLists});
        std::vector<int64_t> distResultShape({batch, numLists});
        std::vector<int64_t> minResultShape({batch, numLists / IVF_RABITQ_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({CORE_NUM, 16});
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesSqrSum.size(), codesSqrSum.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes)
    {
        l1DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DisOpReset(l1DistFp32Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L1 distance op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                                 AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                                 AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                                 aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l1DistFp32Ops.find(batch) != l1DistFp32Ops.end())
    {
        op = l1DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    AscendTensor<float, DIMS_1> centroidsSqrSum(CentroidL2OnDevice->data(), {numLists});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsDev.data(), centroidsDev.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsSqrSum.data(), centroidsSqrSum.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFRaBitQ::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("TopkIvfRabitqFp32");
        std::vector<int64_t> shape0{0, IVF_RABITQ_BLOCK_SIZE};
        std::vector<int64_t> shape1{0, (IVF_RABITQ_BLOCK_SIZE + IVF_RABITQ_BURST_LEN - 1) / IVF_RABITQ_BURST_LEN * 2};
        std::vector<int64_t> shape2{0, 1};
        std::vector<int64_t> shape3{0, 1};
        std::vector<int64_t> shape4{batch};
        std::vector<int64_t> shape5{0, 32};
        std::vector<int64_t> shape6{aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT};
        std::vector<int64_t> shapeSlots{batch, 0};

        std::vector<int64_t> shape7{batch, 0};

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shapeSlots.size(), shapeSlots.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape7.size(), shape7.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, shape7.size(), shape7.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes)
    {
        topkL2Fp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkL2Fp32[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "l2 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runL2TopkOp(
    int batch, AscendTensor<float, DIMS_2, size_t> &distResult, AscendTensor<float, DIMS_2, size_t> &vmdistResult,
    AscendTensor<int64_t, DIMS_2, size_t> &ids, AscendTensor<uint32_t, DIMS_2, size_t> &sizes,
    AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQ, AscendTensor<uint16_t, DIMS_2, size_t> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<int64_t, DIMS_2, size_t> &blockSlots,
    AscendTensor<float, DIMS_2, size_t> &outdists, AscendTensor<uint64_t, DIMS_2, size_t> &outlabel, aclrtStream stream)
{
    std::vector<const AscendTensorBase *> input{&distResult,   &vmdistResult, &ids,   &sizes,
                                                &blockNumPerQ, &flags,        &attrs, &blockSlots};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    AscendOperator *op = nullptr;
    if (topkL2Fp32.find(batch) != topkL2Fp32.end())
    {
        op = topkL2Fp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    for (auto tensor : input)
    {
        topkOpInput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    for (auto tensor : output)
    {
        topkOpOutput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

bool IndexIVFRaBitQ::l2DisOpReset(std::unique_ptr<AscendOperator> &op, int64_t batch)
{
    AscendOpDesc desc;
    std::vector<int64_t> queryl2Shape({batch});
    std::vector<int64_t> querylutShape({batch * static_cast<int64_t>(getCodePartBytes()), LUT_NUM});
    std::vector<int64_t> centroidslutShape({numLists * static_cast<int64_t>(getCodePartBytes()), LUT_NUM});
    std::vector<int64_t> queryidShape({CORE_NUM});
    std::vector<int64_t> centroidsidShape({CORE_NUM});
    std::vector<int64_t> centroidsl2Shape({CORE_NUM});
    std::vector<int64_t> codesShape({IVF_RABITQ_BLOCK_SIZE, static_cast<int64_t>(getCodePartBytes())});
    std::vector<int64_t> offsetShape({CORE_NUM});
    std::vector<int64_t> sizeShape({CORE_NUM});
    std::vector<int64_t> indexl2Shape({IVF_RABITQ_BLOCK_SIZE});
    std::vector<int64_t> indexl1Shape({IVF_RABITQ_BLOCK_SIZE});
    std::vector<int64_t> indexesl2offsetShape({CORE_NUM});
    std::vector<int64_t> indexesl1offsetShape({CORE_NUM});
    std::vector<int64_t> idsShape({CORE_NUM});
    std::vector<int64_t> distFilterAttrsShape({5});
    std::vector<int64_t> distResultShape({CORE_NUM, IVF_RABITQ_BLOCK_SIZE});
    std::vector<int64_t> mixResultShape(
        {CORE_NUM, (IVF_RABITQ_BLOCK_SIZE + IVF_RABITQ_BURST_LEN - 1) / IVF_RABITQ_BURST_LEN * 2});
    std::vector<int64_t> flagShapeShape({CORE_NUM, 32});
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5())
    {
        desc.setOpName("DistanceIVFRabitqL2FP32Simt");
    }
    else
    {
        desc.setOpName("DistanceIVFRabitqL2FP32");
    }
    desc.addInputTensorDesc(ACL_FLOAT, queryl2Shape.size(), queryl2Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, querylutShape.size(), querylutShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, centroidslutShape.size(), centroidslutShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, queryidShape.size(), queryidShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, centroidsidShape.size(), centroidsidShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, centroidsl2Shape.size(), centroidsl2Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, indexl2Shape.size(), indexl2Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, indexl1Shape.size(), indexl1Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, indexesl2offsetShape.size(), indexesl2offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, indexesl1offsetShape.size(), indexesl1offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT64, idsShape.size(), idsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT64, distFilterAttrsShape.size(), distFilterAttrsShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT, mixResultShape.size(), mixResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
    if (this->metric == MetricType::METRIC_L2)
    {
        aclopSetAttrInt(desc.opAttr, "metric_type", 0);
    }
    else if (this->metric == MetricType::METRIC_INNER_PRODUCT)
    {
        aclopSetAttrInt(desc.opAttr, "metric_type", 1);
    }
    if (!faiss::ascend::SocUtils::GetInstance().IsAscendA5())
    {
        aclopSetAttrInt(desc.opAttr, "quant_bits", quantBits_);
    }
    op.reset();
    op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    return op->init();
}

APP_ERROR IndexIVFRaBitQ::resetL2DistOp()
{
    for (auto batch : searchBatchSizes)
    {
        ivfRaBitQL2DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l2DisOpReset(ivfRaBitQL2DistOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "l2 dist op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::runL2DistOp(
    AscendTensor<float, DIMS_1> &queryL2Vec, AscendTensor<float, DIMS_2> &subQuerylut,
    AscendTensor<float, DIMS_2, size_t> &centroidslut, AscendTensor<uint32_t, DIMS_1, size_t> &subQueryid,
    AscendTensor<uint32_t, DIMS_1, size_t> &subCentroidsid, AscendTensor<float, DIMS_1, size_t> &subCentroidsl2,
    AscendTensor<uint8_t, DIMS_2, size_t> &codeVec, AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
    AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize, AscendTensor<float, DIMS_1, size_t> &indexl2,
    AscendTensor<float, DIMS_1, size_t> &indexl1, AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl2Offset,
    AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl1Offset, AscendTensor<int64_t, DIMS_1, size_t> &subIds,
    AscendTensor<int64_t, DIMS_1> &distFilterAttrs, AscendTensor<float, DIMS_2, size_t> &subDis,
    AscendTensor<float, DIMS_2, size_t> &subVcMaxDis, AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag,
    aclrtStream stream)
{
    // subQuerylut keeps the physical matmul shape [batch*d/8, 256*bits].
    // The distance op reinterprets the same bytes as [batch*codeBytes, 256].
    int batch = static_cast<int>(subQuerylut.getSize(0) / (dims / SCAN_BIT));

    AscendOperator *op = nullptr;
    if (ivfRaBitQL2DistOps.find(batch) != ivfRaBitQL2DistOps.end())
    {
        op = ivfRaBitQL2DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queryL2Vec.data(), queryL2Vec.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subQuerylut.data(), subQuerylut.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidslut.data(), centroidslut.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subQueryid.data(), subQueryid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subCentroidsid.data(), subCentroidsid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subCentroidsl2.data(), subCentroidsl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(codeVec.data(), codeVec.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subOffset.data(), subOffset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subBaseSize.data(), subBaseSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl1.data(), indexl1.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subIndexl2Offset.data(), subIndexl2Offset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subIndexl1Offset.data(), subIndexl1Offset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subIds.data(), subIds.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(distFilterAttrs.data(), distFilterAttrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(subDis.data(), subDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subVcMaxDis.data(), subVcMaxDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subOpFlag.data(), subOpFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

void IndexIVFRaBitQ::fillDisOpInputDataByBlock(
    size_t batch, size_t coreNum, AscendTensor<uint32_t, DIMS_2, size_t> &queryidHostVec,
    AscendTensor<uint32_t, DIMS_2, size_t> &centroidsidHostVec, AscendTensor<float, DIMS_2, size_t> &centroidsl2HostVec,
    AscendTensor<uint32_t, DIMS_2, size_t> &baseSizeHostVec, AscendTensor<uint64_t, DIMS_2, size_t> &offsetHostVec,
    AscendTensor<uint64_t, DIMS_2, size_t> &indexl2OffsetHostVec,
    AscendTensor<uint64_t, DIMS_2, size_t> &indexl1OffsetHostVec, AscendTensor<int64_t, DIMS_2, size_t> &idsHostVec,
    AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQHostVec, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
    AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, const int64_t *slotPlan, size_t maxBlocksPerQ)
{
    for (size_t qIdx = 0; qIdx < batch; qIdx++)
    {
        int64_t blockNumPerQ = 0;
        size_t emitted = 0;
        // Bound by the actual probe width of this request, not the member nprobe: a request may override it,
        // and the slot plan was built against the same width.
        for (size_t tIdx = 0; tIdx < static_cast<size_t>(l1TopNprobeIndicesHost.getSize(1)); tIdx++)
        {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][tIdx].value();
            float centerl2 = l1TopNprobeDistsHost[qIdx][tIdx].value();
            const auto &list = lists_[listId];
            size_t blockNum = list.codes.segmentCount();
            blockNumPerQ += blockNum;
            for (size_t segIdx = 0; segIdx < blockNum; segIdx++)
            {
                // The slot this block occupies in the grid is dictated by the layout plan, not by the order the
                // probe walk reaches it: the plan interleaves queries so a window of them finishes together.
                const int64_t slot = slotPlan[qIdx * maxBlocksPerQ + emitted];
                const size_t iter = static_cast<size_t>(slot) / coreNum;
                const size_t cIdx = static_cast<size_t>(slot) % coreNum;
                emitted++;

                queryidHostVec[iter][cIdx].value(qIdx);
                centroidsidHostVec[iter][cIdx].value(listId);
                centroidsl2HostVec[iter][cIdx].value(centerl2);
                baseSizeHostVec[iter][cIdx].value(list.codes.segmentLength(segIdx));
                idsHostVec[iter][cIdx].value(reinterpret_cast<int64_t>(list.ids.segmentData(segIdx)));

                // The base pointers passed to the op are nullptr, so each offset is the segment's own
                // absolute device address. Unlike IVFFlat's single contiguous blockList, RaBitQ segments
                // are independent allocations, so a min base would buy nothing.
                offsetHostVec[iter][cIdx].value(reinterpret_cast<uint64_t>(list.codes.segmentData(segIdx)));
                indexl2OffsetHostVec[iter][cIdx].value(reinterpret_cast<uint64_t>(list.indexL2.segmentData(segIdx)));
                indexl1OffsetHostVec[iter][cIdx].value(reinterpret_cast<uint64_t>(list.indexL1.segmentData(segIdx)));
            }
        }
        blockNumPerQHostVec[qIdx].value(blockNumPerQ);
    }
}

// Retries the device query that the constructor could not answer. An index is commonly built before the
// calling thread has run aclrtSetDevice, and resolving only once would pin the compile-time fallback for
// the life of the process even after a device became available. Once resolved the flag latches, so the
// steady-state search path costs one bool test; a device that never answers retries once per call, which
// is two ACL queries against a search that stages thousands of blocks.
void IndexIVFRaBitQ::EnsureTopkGroupSizeResolved()
{
    if (topkGroupSizeResolved_)
    {
        return;
    }
    if (TryResolveTopkGroupSize(topkGroupSize_))
    {
        topkGroupSizeResolved_ = true;
        APP_LOG_INFO("TopK group window resolved to %ld after the device became available\n",
                     static_cast<long>(topkGroupSize_));
    }
}

APP_ERROR IndexIVFRaBitQ::fillDisOpInputData(
    int k, size_t batch, size_t totalBlockNum, size_t coreNum, AscendTensor<uint64_t, DIMS_2, size_t> &offset,
    AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset, AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
    AscendTensor<uint32_t, DIMS_2, size_t> &queryid, AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
    AscendTensor<float, DIMS_2, size_t> &centroidsl2, AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
    AscendTensor<int64_t, DIMS_2, size_t> &ids, AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQ,
    AscendTensor<int64_t, DIMS_2, size_t> &blockSlots, AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost,
    const std::vector<int64_t> &blockNumPlan, size_t maxBlocksPerQ, aicpu::RabitqIdFilterMode selMode, int64_t selPtr,
    int64_t selAux0, int64_t selAux1, int64_t selNegate)
{
    size_t iterNum = (totalBlockNum + coreNum - 1) / coreNum;
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    std::vector<uint64_t> offsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> offsetHostVec(offsetHost.data(), {iterNum, coreNum});
    std::vector<uint64_t> indexl2OffsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl2OffsetHostVec(indexl2OffsetHost.data(), {iterNum, coreNum});
    std::vector<uint64_t> indexl1OffsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl1OffsetHostVec(indexl1OffsetHost.data(), {iterNum, coreNum});
    std::vector<uint32_t> queryidHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> queryidHostVec(queryidHost.data(), {iterNum, coreNum});
    std::vector<uint32_t> centroidsidHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> centroidsidHostVec(centroidsidHost.data(), {iterNum, coreNum});
    std::vector<float> centroidsl2Host(iterNum * coreNum, 0);
    AscendTensor<float, DIMS_2, size_t> centroidsl2HostVec(centroidsl2Host.data(), {iterNum, coreNum});
    std::vector<uint32_t> baseSizeHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> baseSizeHostVec(baseSizeHost.data(), {iterNum, coreNum});
    std::vector<int64_t> idsHost(iterNum * coreNum, 0);
    AscendTensor<int64_t, DIMS_2, size_t> idsHostVec(idsHost.data(), {iterNum, coreNum});
    std::vector<int64_t> blockNumPerQHost(batch, 0);
    AscendTensor<int64_t, DIMS_1, size_t> blockNumPerQHostVec(blockNumPerQHost.data(), {batch});

    // -1 marks a slot a query does not own: rows are sized to the widest query, so shorter queries leave a
    // tail. The kernel stops at blocknums[q] and never reads it, but a plain 0 there would alias grid slot 0.
    std::vector<int64_t> blockSlotsHost(batch * maxBlocksPerQ, -1);
    EnsureTopkGroupSizeResolved();
    BuildBlockSlotPlan(blockNumPlan, maxBlocksPerQ, topkGroupSize_, blockSlotsHost.data());

    fillDisOpInputDataByBlock(batch, coreNum, queryidHostVec, centroidsidHostVec, centroidsl2HostVec, baseSizeHostVec,
                              offsetHostVec, indexl2OffsetHostVec, indexl1OffsetHostVec, idsHostVec,
                              blockNumPerQHostVec, l1TopNprobeIndicesHost, l1TopNprobeDistsHost, blockSlotsHost.data(),
                              maxBlocksPerQ);
    auto ret = aclrtMemcpy(offset.data(), offset.getSizeInBytes(), offsetHostVec.data(), offsetHostVec.getSizeInBytes(),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy offset to device failed %d", ret);
    ret = aclrtMemcpy(indexl2offset.data(), indexl2offset.getSizeInBytes(), indexl2OffsetHostVec.data(),
                      indexl2OffsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy indexl2offset to device failed %d", ret);
    ret = aclrtMemcpy(indexl1offset.data(), indexl1offset.getSizeInBytes(), indexl1OffsetHostVec.data(),
                      indexl1OffsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy indexl1offset to device failed %d", ret);
    ret = aclrtMemcpy(queryid.data(), queryid.getSizeInBytes(), queryidHostVec.data(), queryidHostVec.getSizeInBytes(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy queryid to device failed %d", ret);
    ret = aclrtMemcpy(centroidsid.data(), centroidsid.getSizeInBytes(), centroidsidHostVec.data(),
                      centroidsidHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy centroidsid to device failed %d", ret);
    ret = aclrtMemcpy(centroidsl2.data(), centroidsl2.getSizeInBytes(), centroidsl2HostVec.data(),
                      centroidsl2HostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy centroidsl2 to device failed %d", ret);
    ret = aclrtMemcpy(baseSize.data(), baseSize.getSizeInBytes(), baseSizeHostVec.data(),
                      baseSizeHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy basesize to device failed %d", ret);
    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(), idsHostVec.data(), idsHostVec.getSizeInBytes(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy ids to device failed %d", ret);
    ret = aclrtMemcpy(blockNumPerQ.data(), blockNumPerQ.getSizeInBytes(), blockNumPerQHostVec.data(),
                      blockNumPerQHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy blockNumPerQ to device failed %d", ret);
    ret = aclrtMemcpy(blockSlots.data(), blockSlots.getSizeInBytes(), blockSlotsHost.data(),
                      blockSlotsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy blockSlots to device failed %d", ret);

    fillL2TopkOpInputData(k, batch, coreNum, attrs, selMode, selPtr, selAux0, selAux1, selNegate);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::fillL2TopkOpInputData(int k, size_t batch, size_t coreNum,
                                                AscendTensor<int64_t, DIMS_1> &attrs, aicpu::RabitqIdFilterMode selMode,
                                                int64_t selPtr, int64_t selAux0, int64_t selAux1, int64_t selNegate)
{
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT, 0);
    if (this->metric == MetricType::METRIC_L2)
    {
        attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_ASC_IDX] = 1;
    }
    else if (this->metric == MetricType::METRIC_INNER_PRODUCT)
    {
        attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_ASC_IDX] = 0;
    }
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_BURST_LEN_IDX] = IVF_RABITQ_BURST_LEN;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_BLOCK_NUM_IDX] = 0;
    // QUERY_NUM carries batch in its low bits plus the optional-path toggles in high bits. The kernel has
    // never read this slot (nq comes from the tensor shapes), but keeping batch in the low bits means the
    // slot still reads correctly as batch for anyone who starts using it. The envs are read HERE, on the
    // host: AICPU kernels run in a separate device-side execution environment and do not inherit the host's
    // environ, so std::getenv inside the kernel always returns null. Both toggles exist so the two top-k
    // variants can be A/B'd WITHIN one deployment: measuring across redeploys has a noise floor (die
    // asymmetry drift, warm-up) of the same order as the effect itself.
    int64_t queryNumAttr = static_cast<int64_t>(batch);
    // Reservoir: opt-OUT (default on). "=0" restores the historical k-heap, which is the control arm.
    //
    // Gated additionally on every stored id fitting in uint32. The kernel's reservoir stores each candidate
    // as {float dist; uint32_t id} -- 8B with no padding, against 16B for an int64 id -- which halves the
    // buffer and the cache lines a refill walks. The gate is on maxStoredId_, the running maximum of what
    // the two ingest paths actually appended, and NOT on ntotal: ntotal is a count, the ids are caller
    // labels, and nothing in addVectors/addEncodedVectors ties the two together. A caller passing sparse or
    // otherwise non-ordinal labels keeps ntotal small while pushing ids past 2^32, and gating on ntotal
    // would hand those to a path that truncates them to their low 32 bits -- returning a wrong label that
    // still looks like a legal one, with no error anywhere.
    // The check belongs on the host because the kernel cannot make it: each ComputeQuery call sees only its
    // own block's baseSize, never the index-wide id range.
    // Past the bound we fall back to the k-heap path, which keeps full-width ids -- the reservoir is a
    // performance option, so declining it is always safe, whereas truncating would silently return wrong
    // labels.
    const bool idsFitUint32 = (maxStoredId_ <= static_cast<idx_t>(UINT32_MAX));
    if (IsEnvSwitchOn("IVFRABITQ_TOPK_RESERVOIR", true) && idsFitUint32)
    {
        queryNumAttr |= aicpu::TOPK_IVF_RABITQ_RESERVOIR_FLAG;
    }
    // Packed introsort in Reorder: opt-OUT (default on). "=0" restores the in-place heap drain, which is
    // the control arm rather than the treatment.
    if (IsEnvSwitchOn("IVFRABITQ_TOPK_PACKED_SORT", true))
    {
        queryNumAttr |= aicpu::TOPK_IVF_RABITQ_PACKED_SORT_FLAG;
    }
    // Reservoir chunk size B, as log2 in bits 36-39. Only meaningful on the reservoir path, but it is set
    // unconditionally: the kernel ignores it when the reservoir is off, and making the two envs interact
    // here would mean IVFRABITQ_TOPK_REFILL_CHUNK silently did nothing depending on another variable's
    // value. 0 (env unset or rejected) leaves the bits clear, which the kernel reads as "use the default".
    queryNumAttr |= (GetRefillChunkLog2() & aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_MASK)
                    << aicpu::TOPK_IVF_RABITQ_REFILL_LOG2_SHIFT;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_QUERY_NUM_IDX] = queryNumAttr;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_CORE_NUM_IDX] = static_cast<int64_t>(coreNum);
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_SEL_MODE_IDX] = static_cast<int64_t>(selMode);
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_SEL_PTR_IDX] = selPtr;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_SEL_AUX0_IDX] = selAux0;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_SEL_AUX1_IDX] = selAux1;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_SEL_NEGATE_IDX] = selNegate;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_LAYOUT_MODE_IDX] = topkLayoutMode_;
    auto ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(), attrsVec.data(), attrsVec.size() * sizeof(int64_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy attrs to device failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::fillL1TopkOpInputData(AscendTensor<int64_t, DIMS_1> &attrsInput, int probeK)
{
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = probeK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_RABITQ_BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::callL2DistanceOp(
    size_t batch, size_t totalBlockNum, size_t coreNum, size_t vcMaxLen, AscendTensor<float, DIMS_1> &queryL2Vec,
    AscendTensor<float, DIMS_2> &queryLutVec, AscendTensor<float, DIMS_2, size_t> &centroidsLutVec,
    AscendTensor<uint32_t, DIMS_2, size_t> &queryid, AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
    AscendTensor<float, DIMS_2, size_t> &centroidsl2, AscendTensor<uint64_t, DIMS_2, size_t> &offset,
    AscendTensor<uint32_t, DIMS_2, size_t> &baseSize, AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
    AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset, AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
    AscendTensor<float, DIMS_2, size_t> &disVec, AscendTensor<float, DIMS_2, size_t> &vcMaxDisVec,
    AscendTensor<uint8_t, DIMS_2, size_t> &codeVec, AscendTensor<float, DIMS_1, size_t> &Indexl2,
    AscendTensor<float, DIMS_1, size_t> &Indexl1, AscendTensor<int64_t, DIMS_2, size_t> &ids,
    AscendTensor<int64_t, DIMS_1> &distFilterAttrs, aclrtStream &stream)
{
    VALUE_UNUSED(batch);
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    size_t iterNum = (totalBlockNum + coreNum - 1) / coreNum;
    for (size_t iter = 0; iter < iterNum; iter++)
    {
        AscendTensor<uint32_t, DIMS_1, size_t> subQueryid(queryid[iter].data(), {coreNum});
        AscendTensor<uint32_t, DIMS_1, size_t> subCentroidsid(centroidsid[iter].data(), {coreNum});
        AscendTensor<float, DIMS_1, size_t> subCentroidsl2(centroidsl2[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subOffset(offset[iter].data(), {coreNum});
        AscendTensor<uint32_t, DIMS_1, size_t> subBaseSize(baseSize[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subIndexl2Offset(indexl2offset[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subIndexl1Offset(indexl1offset[iter].data(), {coreNum});
        AscendTensor<int64_t, DIMS_1, size_t> subIds(ids[iter].data(), {coreNum});
        AscendTensor<uint16_t, DIMS_2, size_t> subOpFlag(opFlag[iter].data(), {coreNum, 32});
        AscendTensor<float, DIMS_2, size_t> subDis(disVec[iter].data(), {coreNum, ivfRabitqBlockSize});
        AscendTensor<float, DIMS_2, size_t> subVcMaxDis(vcMaxDisVec[iter].data(), {coreNum, vcMaxLen});

        runL2DistOp(queryL2Vec, queryLutVec, centroidsLutVec, subQueryid, subCentroidsid, subCentroidsl2, codeVec,
                    subOffset, subBaseSize, Indexl2, Indexl1, subIndexl2Offset, subIndexl1Offset, subIds,
                    distFilterAttrs, subDis, subVcMaxDis, subOpFlag, stream);
    }
}

size_t IndexIVFRaBitQ::getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                                     float *distances, idx_t *labels) const
{
    size_t maxLen = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++)
    {
        for (size_t probId = 0; probId < static_cast<size_t>(l1TopNprobeIndicesHost.getSize(1)); probId++)
        {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][probId].value();
            size_t listNum = lists_[listId].listLen;
            maxLen = maxLen > listNum ? maxLen : listNum;
        }
    }
    size_t topk = static_cast<size_t>(k);
    if (maxLen == 0)
    {
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < topk; j++)
            {
                distances[i * topk + j] = std::numeric_limits<float>::min();
                labels[i * topk + j] = -1;  // 使用-1表示无效ID
            }
        }
    }
    return maxLen;
}

size_t IndexIVFRaBitQ::getTotalBlockNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                                        float *distances, idx_t *labels) const
{
    size_t totalBlockNum = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++)
    {
        for (size_t probId = 0; probId < static_cast<size_t>(l1TopNprobeIndicesHost.getSize(1)); probId++)
        {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][probId].value();
            size_t blockNum = lists_[listId].codes.segmentCount();
            totalBlockNum += blockNum;
        }
    }
    size_t topk = static_cast<size_t>(k);
    if (totalBlockNum == 0)
    {
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < topk; j++)
            {
                distances[i * topk + j] = std::numeric_limits<float>::min();
                labels[i * topk + j] = -1;  // 使用-1表示无效ID
            }
        }
    }
    return totalBlockNum;
}

size_t IndexIVFRaBitQ::getBlockNumPerQ(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       std::vector<int64_t> &blockNumPerQ) const
{
    blockNumPerQ.assign(batch, 0);
    size_t maxBlocksPerQ = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++)
    {
        int64_t blockNum = 0;
        // Must use the same probe width as the emit loop in fillDisOpInputDataByBlock, which is this
        // request's width and not the member nprobe: a narrower plan would truncate blockSlots rows.
        for (size_t probId = 0; probId < static_cast<size_t>(l1TopNprobeIndicesHost.getSize(1)); probId++)
        {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][probId].value();
            blockNum += static_cast<int64_t>(lists_[listId].codes.segmentCount());
        }
        blockNumPerQ[qIdx] = blockNum;
        maxBlocksPerQ = std::max(maxBlocksPerQ, static_cast<size_t>(blockNum));
    }
    // A zero-width tensor has no valid device allocation, and a batch where every query probes only empty
    // lists is already short-circuited by getTotalBlockNum -- keep at least one column so the shape is legal.
    return std::max<size_t>(maxBlocksPerQ, 1);
}

void IndexIVFRaBitQ::resizeDistResult(size_t iterNum, size_t coreNum, size_t ivfRabitqBlockSize)
{
    size_t disSpace = iterNum * coreNum * ivfRabitqBlockSize;
    if (disSpace > this->distResultSpace)
    {
        distResultOnDevice->resize(disSpace);
        this->distResultSpace = disSpace;
    }
}

APP_ERROR IndexIVFRaBitQ::searchImplL2(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_1> &queryL2,
                                       AscendTensor<float, DIMS_2> &queriesLut,
                                       AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, int k, float *distances,
                                       idx_t *labels, const RabitqIdFilterHost *idFilter)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    size_t batch = static_cast<size_t>(queries.getSize(0));
    size_t coreNum = static_cast<size_t>(CORE_NUM);
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    size_t totalBlockNum = getTotalBlockNum(batch, l1TopNprobeIndicesHost, k, distances, labels);
    APPERR_RETURN_IF_NOT_LOG(totalBlockNum != 0, APP_ERR_OK, "all nprobe cluster is empty by L1 search !!!");
    size_t iterNum = (totalBlockNum + coreNum - 1) / coreNum;
    AscendTensor<uint16_t, DIMS_2, size_t> opFlag(mem, {iterNum, coreNum * 32}, stream);
    (void)opFlag.zero();
    AscendTensor<uint64_t, DIMS_2, size_t> offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl2offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl1offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> queryid(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> centroidsid(mem, {iterNum, coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> centroidsl2(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> baseSize(mem, {iterNum, coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> centroidsLutVec(
        CentroidLUTOnDevice->data(),
        {static_cast<size_t>(numLists) * getCodePartBytes(), static_cast<size_t>(LUT_NUM)});
    // Base pointers are nullptr: the kernel only computes base + offset and never dereferences base, so
    // each per-segment offset carries the segment's absolute device address (see fillDisOpInputDataByBlock).
    AscendTensor<uint8_t, DIMS_2, size_t> codeVec(nullptr, {ivfRabitqBlockSize, getCodePartBytes()});
    AscendTensor<float, DIMS_1, size_t> Indexl2(nullptr, {ivfRabitqBlockSize});
    AscendTensor<float, DIMS_1, size_t> Indexl1(nullptr, {ivfRabitqBlockSize});
    size_t vcMaxLen = static_cast<size_t>((ivfRabitqBlockSize + IVF_RABITQ_BURST_LEN - 1) / IVF_RABITQ_BURST_LEN * 2);
    resizeDistResult(iterNum, coreNum, ivfRabitqBlockSize);
    AscendTensor<float, DIMS_2, size_t> disVec(distResultOnDevice->data(), {iterNum, coreNum * ivfRabitqBlockSize});
    AscendTensor<int64_t, DIMS_2, size_t> ids(mem, {iterNum, coreNum}, stream);
    AscendTensor<int64_t, DIMS_1, size_t> blocknumPerQ(mem, {batch}, stream);
    AscendTensor<float, DIMS_2, size_t> vcMaxDisVec(mem, {iterNum, coreNum * vcMaxLen}, stream);
    AscendTensor<int64_t, DIMS_1> attrs(mem, {aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT}, stream);
    AscendTensor<int64_t, DIMS_1> distFilterAttrs(mem, {5}, stream);

    aicpu::RabitqIdFilterMode filterMode = aicpu::RABITQ_ID_FILTER_NONE;
    int64_t filterAux0 = 0;
    int64_t filterAux1 = 0;
    int64_t filterNegate = 0;
    int64_t requestedDistSelMode = aicpu::RABITQ_ID_FILTER_NONE;
    if (idFilter != nullptr && idFilter->mode != aicpu::RABITQ_ID_FILTER_NONE)
    {
        filterMode = static_cast<aicpu::RabitqIdFilterMode>(idFilter->mode);
        filterAux0 = idFilter->aux0;
        filterAux1 = idFilter->aux1;
        filterNegate = idFilter->negate;
        if (idFilter->mode == aicpu::RABITQ_ID_FILTER_BITMAP)
        {
            requestedDistSelMode = aicpu::RABITQ_ID_FILTER_BITMAP;
        }
        else if (idFilter->mode == aicpu::RABITQ_ID_FILTER_RANGE)
        {
            requestedDistSelMode = aicpu::RABITQ_ID_FILTER_RANGE;
        }
        else if (idFilter->mode == aicpu::RABITQ_ID_FILTER_SORTED && !idFilter->sortedPrefixPayload.empty())
        {
            requestedDistSelMode = aicpu::RABITQ_ID_FILTER_SORTED_PREFIX;
        }
    }

    const bool hasFilter = (filterMode != aicpu::RABITQ_ID_FILTER_NONE);
    const bool canUseL2Prefilter =
        hasFilter && requestedDistSelMode != aicpu::RABITQ_ID_FILTER_NONE && IsRabitqL2PrefilterKernelSupported();
    const RabitqSelectorFilterStage filterStage = GetRabitqSelectorFilterStage();
    APPERR_RETURN_IF_NOT_LOG(filterStage != RabitqSelectorFilterStage::PRE || !hasFilter || canUseL2Prefilter,
                             APP_ERR_INVALID_PARAM,
                             "IVFRABITQ_SELECTOR_FILTER_STAGE=pre requires an L2 prefilter path, which is not "
                             "supported on 310P/SIMT");

    bool useTopkFilter = false;
    bool useDistFilter = false;
    switch (filterStage)
    {
        case RabitqSelectorFilterStage::PRE:
            useDistFilter = canUseL2Prefilter;
            break;
        case RabitqSelectorFilterStage::POST:
            useTopkFilter = hasFilter;
            break;
        case RabitqSelectorFilterStage::BOTH:
            useDistFilter = canUseL2Prefilter;
            useTopkFilter = hasFilter;
            break;
        case RabitqSelectorFilterStage::AUTO:
        default:
            useDistFilter = canUseL2Prefilter;
            useTopkFilter = hasFilter && !useDistFilter;
            break;
    }

    aicpu::RabitqIdFilterMode topkSelMode = aicpu::RABITQ_ID_FILTER_NONE;
    int64_t topkSelPtr = 0;
    int64_t topkSelAux0 = 0;
    int64_t topkSelAux1 = 0;
    int64_t topkSelNegate = 0;
    int64_t distSelMode = aicpu::RABITQ_ID_FILTER_NONE;
    int64_t distSelPtr = 0;
    int64_t distSelAux0 = 0;
    int64_t distSelAux1 = 0;
    int64_t distSelNegate = 0;

    if (useTopkFilter)
    {
        topkSelMode = filterMode;
        topkSelAux0 = filterAux0;
        topkSelAux1 = filterAux1;
        topkSelNegate = filterNegate;
        if (topkSelMode == aicpu::RABITQ_ID_FILTER_SORTED || topkSelMode == aicpu::RABITQ_ID_FILTER_BITMAP ||
            topkSelMode == aicpu::RABITQ_ID_FILTER_ROARING)
        {
            auto retFilter = EnsureFilterPayloadOnDevice(idFilter, topkSelPtr);
            APPERR_RETURN_IF_NOT_FMT(retFilter == APP_ERR_OK, retFilter, "ensure id filter payload on device failed %d",
                                     retFilter);
        }
    }

    if (useDistFilter)
    {
        distSelMode = requestedDistSelMode;
        distSelNegate = filterNegate;
        distSelAux0 = filterAux0;
        distSelAux1 = filterAux1;
        if (distSelMode == aicpu::RABITQ_ID_FILTER_BITMAP)
        {
            if (topkSelPtr != 0)
            {
                distSelPtr = topkSelPtr;
            }
            else
            {
                auto retFilter = EnsureFilterPayloadOnDevice(idFilter, distSelPtr);
                APPERR_RETURN_IF_NOT_FMT(retFilter == APP_ERR_OK, retFilter,
                                         "ensure id filter payload on device failed %d", retFilter);
            }
        }
        else if (distSelMode == aicpu::RABITQ_ID_FILTER_SORTED_PREFIX)
        {
            auto retPrefixFilter = EnsurePrefixFilterPayloadOnDevice(idFilter, distSelPtr);
            APPERR_RETURN_IF_NOT_FMT(retPrefixFilter == APP_ERR_OK, retPrefixFilter,
                                     "ensure prefix id filter payload on device failed %d", retPrefixFilter);
        }
    }
    std::vector<int64_t> distFilterAttrHost(5, 0);
    if (distSelMode != aicpu::RABITQ_ID_FILTER_NONE)
    {
        distFilterAttrHost[0] = distSelMode;
        distFilterAttrHost[1] = distSelPtr;
        distFilterAttrHost[2] = distSelAux0;
        distFilterAttrHost[3] = distSelAux1;
        distFilterAttrHost[4] = distSelNegate;
    }
    auto retCopyDistFilterAttrs =
        aclrtMemcpy(distFilterAttrs.data(), distFilterAttrs.getSizeInBytes(), distFilterAttrHost.data(),
                    distFilterAttrHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(retCopyDistFilterAttrs == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "copy dist filter attrs to device failed %d", retCopyDistFilterAttrs);

    std::vector<int64_t> blockNumPlan;
    size_t maxBlocksPerQ = getBlockNumPerQ(batch, l1TopNprobeIndicesHost, blockNumPlan);
    AscendTensor<int64_t, DIMS_2, size_t> blockSlots(mem, {batch, maxBlocksPerQ}, stream);
    fillDisOpInputData(k, batch, totalBlockNum, coreNum, offset, indexl2offset, indexl1offset, queryid, centroidsid,
                       centroidsl2, baseSize, ids, blocknumPerQ, blockSlots, attrs, l1TopNprobeIndicesHost,
                       l1TopNprobeDistsHost, blockNumPlan, maxBlocksPerQ, topkSelMode, topkSelPtr, topkSelAux0,
                       topkSelAux1, topkSelNegate);
    AscendTensor<float, DIMS_2, size_t> outDist(mem, {batch, static_cast<size_t>(k)}, stream);
    AscendTensor<idx_t, DIMS_2, size_t> outLabel(mem, {batch, static_cast<size_t>(k)}, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    runL2TopkOp(batch, disVec, vcMaxDisVec, ids, baseSize, blocknumPerQ, opFlag, attrs, blockSlots, outDist, outLabel,
                streamAicpu);
    callL2DistanceOp(batch, totalBlockNum, coreNum, vcMaxLen, queryL2, queriesLut, centroidsLutVec, queryid,
                     centroidsid, centroidsl2, offset, baseSize, indexl2offset, indexl1offset, opFlag, disVec,
                     vcMaxDisVec, codeVec, Indexl2, Indexl1, ids, distFilterAttrs, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "sync aicpu stream failed %d", ret);
    ret = aclrtMemcpy(distances, batch * k * sizeof(float), outDist.data(), outDist.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy distances to host failed %d", ret);
    ret = aclrtMemcpy(labels, batch * k * sizeof(idx_t), outLabel.data(), outLabel.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy outLabel to host failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::searchImplL1(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &rotateQueries,
                                       AscendTensor<float, DIMS_1> &queryL2, AscendTensor<float, DIMS_2> &queriesLut,
                                       AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, int probeK)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int n = queries.getSize(0);
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    AscendTensor<int32_t, DIMS_1> vectorSizeVec(mem, {1}, stream);
    std::vector<int32_t> vectorSize(1, 0);
    vectorSize[0] = n;
    auto ret = aclrtMemcpy(vectorSizeVec.data(), sizeof(int32_t), vectorSize.data(), sizeof(int32_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy vectorSize to device failed %d", ret);
    runQueryRotateL2Op(n, queries, vectorSizeVec, rotatematrixDev, rotateQueries, queryL2, stream);

    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    // L1 dist op output: dists / vmdists / opFlag
    AscendTensor<float, DIMS_2> dists(mem, {n, numLists}, stream);
    int minDistSize = numLists / IVF_RABITQ_BURST_LEN * 2;
    AscendTensor<float, DIMS_2> vmdists(mem, {n, minDistSize}, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE}, stream);
    opFlag.zero();
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, {aicpu::TOPK_FLAT_ATTR_IDX_COUNT}, stream);
    fillL1TopkOpInputData(attrsInput, probeK);
    AscendTensor<float, DIMS_2> l1TopNprobeDists(mem, {n, probeK}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, {n, probeK}, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    // run l1 distance calculation
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    runL1DistOp(n, rotateQueries, centroidsDev, dists, vmdists, opFlag, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu stream failed: %i\n",
                             ret);
    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(), l1TopNprobeIndices.data(),
                      l1TopNprobeIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    ret = aclrtMemcpy(l1TopNprobeDistsHost.data(), l1TopNprobeDistsHost.getSizeInBytes(), l1TopNprobeDists.data(),
                      l1TopNprobeDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    const char *debugEnv = std::getenv("IVFRABITQ_DEBUG_L1_PROBE");
    const bool l1ProbeDebug = debugEnv != nullptr && debugEnv[0] != '\0' && debugEnv[0] != '0';
    const bool l1DistVerify = []()
    {
        const char *verifyEnv = std::getenv("IVFRABITQ_VERIFY_L1_DIST");
        return verifyEnv != nullptr && verifyEnv[0] != '\0' && verifyEnv[0] != '0';
    }();
    const bool probeStatsOnly = l1ProbeDebug && debugEnv != nullptr && std::strcmp(debugEnv, "stats") == 0;
    const bool probeFull = l1ProbeDebug && debugEnv != nullptr && std::strcmp(debugEnv, "full") == 0;

    auto dumpProbeStats = [&]()
    {
        const int tile1Max = std::min(8192, numLists);
        int inTile1 = 0;
        int inTile2 = 0;
        int minId = numLists;
        int maxId = 0;
        for (int t = 0; t < probeK; ++t)
        {
            const int64_t id = l1TopNprobeIndicesHost[0][t].value();
            if (id < tile1Max)
            {
                ++inTile1;
            }
            else
            {
                ++inTile2;
            }
            minId = std::min(minId, static_cast<int>(id));
            maxId = std::max(maxId, static_cast<int>(id));
        }
        fprintf(stderr, "[IVFRaBitQ] L1 probe stats q0: nprobe=%d in[0,%d)=%d in[%d,%d)=%d min=%d max=%d\n", probeK,
                tile1Max, inTile1, tile1Max, numLists, inTile2, minId, maxId);
    };

    if (l1ProbeDebug)
    {
        const int probeDump = std::min(probeK, 8);
        fprintf(stderr, "[IVFRaBitQ] L1 probes nlist=%d n=%d first %d probes q0:", numLists, n, probeDump);
        for (int t = 0; t < probeDump; ++t)
        {
            fprintf(stderr, " %ld", static_cast<long>(l1TopNprobeIndicesHost[0][t].value()));
        }
        fprintf(stderr, "\n");
    }

    if (probeStatsOnly)
    {
        dumpProbeStats();
    }

    if (l1DistVerify || probeFull)
    {
        const size_t distRowBytes = static_cast<size_t>(numLists) * sizeof(float);
        std::vector<float> npuDists(static_cast<size_t>(numLists));
        std::vector<float> rotateQuery(static_cast<size_t>(dims));
        std::vector<float> rotateCentroids(static_cast<size_t>(numLists) * static_cast<size_t>(dims));
        ret = aclrtMemcpy(npuDists.data(), distRowBytes, dists.data(), distRowBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "L1 verify dist D2H failed %d", ret);
        ret = aclrtMemcpy(rotateQuery.data(), rotateQuery.size() * sizeof(float), rotateQueries.data(),
                          static_cast<size_t>(dims) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "L1 verify query D2H failed %d", ret);
        ret = aclrtMemcpy(rotateCentroids.data(), rotateCentroids.size() * sizeof(float), centroidsDev.data(),
                          centroidsDev.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "L1 verify centroids D2H failed %d", ret);

        std::vector<int64_t> npuProbeIds(static_cast<size_t>(probeK));
        for (int t = 0; t < probeK; ++t)
        {
            npuProbeIds[static_cast<size_t>(t)] = l1TopNprobeIndicesHost[0][t].value();
        }
        VerifyL1DistAndProbes(numLists, dims, probeK, npuDists.data(), rotateQuery.data(), rotateCentroids.data(),
                              npuProbeIds.data());

        if (probeFull)
        {
            dumpProbeStats();
        }
    }

    // <x, q> LUT预计算
    AscendTensor<float, DIMS_2> queriesfastscan(rotateQueries.data(), {n * dims / SCAN_BIT, SCAN_BIT});
    AscendTensor<float, DIMS_2> lutmatrixDev(LUTMatrixOnDevice->data(), {SCAN_BIT, LUT_NUM * quantBits_});
    runQueryLUTOp(n, queriesfastscan, lutmatrixDev, queriesLut, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::searchWithBatch(int n, const float *x, int k, float *distances, idx_t *labels,
                                          const float *srcIndexes, const RabitqIdFilterHost *idFilter, int searchNprobe)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    const int probeK = searchNprobe > 0 ? searchNprobe : nprobe;
    AscendTensor<float, DIMS_2> queries(mem, {n, dims}, stream);
    AscendTensor<float, DIMS_2> rotateQueries(mem, {n, dims}, stream);
    AscendTensor<float, DIMS_1> queryL2(mem, {n}, stream);
    AscendTensor<float, DIMS_2> queriesLut(mem, {n * dims / SCAN_BIT, LUT_NUM * quantBits_}, stream);
    auto ret =
        aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, n * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy query error %d", ret);
    std::vector<int64_t> l1TopNprobeIndicesVec(n * probeK, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), {n, probeK});
    std::vector<float> l1TopNprobeDistsVec(n * probeK, 0);
    AscendTensor<float, DIMS_2> l1TopNprobeDistsHost(l1TopNprobeDistsVec.data(), {n, probeK});

    ret =
        searchImplL1(queries, rotateQueries, queryL2, queriesLut, l1TopNprobeIndicesHost, l1TopNprobeDistsHost, probeK);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfrabitq L1 search failed! %d", ret);

    if (srcIndexes == nullptr)
    {
        ret = searchImplL2(rotateQueries, queryL2, queriesLut, l1TopNprobeIndicesHost, l1TopNprobeDistsHost, k,
                           distances, labels, idFilter);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfrabitq L2 search failed! %d", ret);
        return APP_ERR_OK;
    }
    std::vector<float> topkdist(n * k, 0);
    std::vector<idx_t> topklabel(n * k, 0);
    ret = searchImplL2(rotateQueries, queryL2, queriesLut, l1TopNprobeIndicesHost, l1TopNprobeDistsHost, k,
                       topkdist.data(), topklabel.data(), idFilter);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfrabitq L2 search failed! %d", ret);
    refine(n, x, k, distances, labels, topkdist.data(), topklabel.data(), srcIndexes);
    return APP_ERR_OK;
}

size_t IndexIVFRaBitQ::getCodeSize() const
{
    size_t codePartBytes = getCodePartBytes();  // 压缩码部分
    size_t factorBytes = 2 * sizeof(float);     // L2因子 + L1因子，各4字节
    return codePartBytes + factorBytes;
}

APP_ERROR IndexIVFRaBitQ::getListIds(int listId, std::vector<int64_t> &ids) const
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM, "listId %d out of range [0, %d)",
                             listId, numLists);
    const auto &list = lists_[listId];
    size_t len = list.listLen;
    ids.resize(len);
    if (len == 0)
    {
        return APP_ERR_OK;
    }

    size_t copied = 0;
    for (size_t segmentId = 0; segmentId < list.ids.segmentCount(); ++segmentId)
    {
        // Clamp to what the destination can still take. The buffer is sized from listLen, while the loop
        // walks segmentLength(), and the two only agree once a write has committed its new length; bounding
        // the copy here means a disagreement costs a short read rather than a heap overflow.
        size_t count = std::min(list.ids.segmentLength(segmentId), len - copied);
        if (count == 0)
        {
            break;
        }
        size_t bytes = count * sizeof(idx_t);
        auto ret =
            aclrtMemcpy(ids.data() + copied, bytes, list.ids.segmentData(segmentId), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy failed for list %d IDs, ret=%d",
                                 listId, ret);
        copied += count;
    }
    APPERR_RETURN_IF_NOT_FMT(copied == len, APP_ERR_INNER_ERROR, "copied %zu ids, expected %zu", copied, len);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::getListRawCodes(int listId, std::vector<uint8_t> &rawCodes) const
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM, "listId %d out of range [0, %d)",
                             listId, numLists);

    const auto &list = lists_[listId];
    size_t numVecs = list.listLen;
    size_t codeSizePerVec = getCodeSize();
    size_t totalBytes = numVecs * codeSizePerVec;
    rawCodes.resize(totalBytes);

    if (numVecs == 0)
    {
        return APP_ERR_OK;
    }

    // 同步默认流，确保之前的所有设备操作已完成
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto retSync = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_LOG(retSync == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream failed before getListRawCodes");

    // 布局：[所有压缩码][所有 L2 因子][所有 L1 因子]
    size_t codePartBytesPerVec = getCodePartBytes();
    size_t codePartTotal = numVecs * codePartBytesPerVec;
    size_t l2Offset = codePartTotal;
    size_t l1Offset = l2Offset + numVecs * sizeof(float);

    size_t copied = 0;
    size_t vectorOffset = 0;
    for (size_t segmentId = 0; segmentId < list.codes.segmentCount(); ++segmentId)
    {
        // Clamp to the vectors still expected. rawCodes is sized from listLen while the loop walks
        // segmentLength(), and the three destination regions are packed back to back, so an unclamped
        // count would not just overrun the buffer -- it would also write the L2/L1 regions past their
        // own extents. Bounding it here turns any disagreement into a short read caught by the check
        // below, instead of silent heap corruption.
        size_t count = std::min(list.codes.segmentLength(segmentId), numVecs - vectorOffset);
        if (count == 0)
        {
            break;
        }
        size_t codeBytes = count * codePartBytesPerVec;
        size_t factorBytes = count * sizeof(float);
        auto ret = aclrtMemcpy(rawCodes.data() + copied, codeBytes, list.codes.segmentData(segmentId), codeBytes,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtMemcpy failed for list %d codes, ret=%d", listId, ret);
        ret = aclrtMemcpy(rawCodes.data() + l2Offset + vectorOffset * sizeof(float), factorBytes,
                          list.indexL2.segmentData(segmentId), factorBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtMemcpy failed for L2 factors of list %d, ret=%d", listId, ret);
        ret = aclrtMemcpy(rawCodes.data() + l1Offset + vectorOffset * sizeof(float), factorBytes,
                          list.indexL1.segmentData(segmentId), factorBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtMemcpy failed for L1 factors of list %d, ret=%d", listId, ret);
        copied += codeBytes;
        vectorOffset += count;
    }
    APPERR_RETURN_IF_NOT_FMT(copied == codePartTotal, APP_ERR_INNER_ERROR,
                             "Copied %zu bytes, expected %zu for code part", copied, codePartTotal);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::getCentroids(std::vector<float> &centroids) const
{
    size_t sz = numLists * dims;
    centroids.resize(sz);

    if (!originCentroidsOnDevice || originCentroidsOnDevice->size() < sz)
    {
        return APP_ERR_INNER_ERROR;
    }

    auto ret = aclrtMemcpy(centroids.data(), sz * sizeof(float), originCentroidsOnDevice->data(), sz * sizeof(float),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy failed for centroids, ret=%d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::addEncodedVectors(int listId, size_t numVecs, const uint8_t *encodedCodes,
                                            const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    auto &list = lists_[listId];
    const size_t oldLen = list.listLen;
    const size_t newLen = oldLen + numVecs;

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    ensureListCapacity(list, newLen, stream);
    // Any early return between here and the commitLength block below rolls the segment count back.
    ListCapacityGuard capacityGuard(list, static_cast<size_t>(blockSize));

    size_t codePartBytes = getCodePartBytes();  // 压缩码部分字节数
    size_t factorBytes = sizeof(float);         // 每个因子字节数

    // 输入布局：[所有压缩码][所有 L2 因子][所有 L1 因子]
    const uint8_t *allCodesPtr = encodedCodes;
    const uint8_t *allL2Ptr = encodedCodes + numVecs * codePartBytes;
    const uint8_t *allL1Ptr = allL2Ptr + numVecs * factorBytes;

    size_t inputOffset = 0;
    while (inputOffset < numVecs)
    {
        const size_t slot = oldLen + inputOffset;
        const size_t count =
            std::min(numVecs - inputOffset, static_cast<size_t>(blockSize) - slot % static_cast<size_t>(blockSize));
        size_t tileCodeBytes = count * codePartBytes;
        size_t tileFactorBytes = count * factorBytes;

        auto ret = aclrtMemcpy(list.codes.slot(slot), tileCodeBytes, allCodesPtr + inputOffset * codePartBytes,
                               tileCodeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy codes failed for list %d, ret=%d",
                                 listId, ret);

        ret = aclrtMemcpy(list.indexL2.slot(slot), tileFactorBytes, allL2Ptr + inputOffset * factorBytes,
                          tileFactorBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy L2 factors failed for list %d, ret=%d",
                                 listId, ret);

        ret = aclrtMemcpy(list.indexL1.slot(slot), tileFactorBytes, allL1Ptr + inputOffset * factorBytes,
                          tileFactorBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy L1 factors failed for list %d, ret=%d",
                                 listId, ret);

        ret = aclrtMemcpy(list.ids.slot(slot), count * sizeof(idx_t), indices + inputOffset, count * sizeof(idx_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy ids failed for list %d, ret=%d", listId,
                                 ret);

        inputOffset += count;
    }

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream failed after addEncodedVectors, ret=%d", ret);

    list.codes.commitLength(newLen);
    list.indexL1.commitLength(newLen);
    list.indexL2.commitLength(newLen);
    list.ids.commitLength(newLen);
    list.listLen = newLen;
    capacityGuard.commit();
    // Kept after the commit to match addVectors: on a failure before this point the guard rolls the segment
    // count back, so maxListLength must not have been raised for a write that never landed.
    maxListLength = std::max(maxListLength, static_cast<int>(newLen));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    IndexIVF::ntotal += numVecs;
    maxStoredId_ = std::max(maxStoredId_, MaxId(indices, numVecs));

    return APP_ERR_OK;
}

template <typename T, typename C>
void UpdateHeap(float *dists, T *label, int64_t len, int64_t index, C &&cmp)
{
    int64_t l = 0;
    int64_t r = 0;
    int64_t m = 0;
    while (true)
    {
        l = 2 * index + 1;  // 2 * index + 1 to find left subnode
        r = 2 * index + 2;  // 2 * index + 2 to find right subnode
        m = index;
        if (l < len && cmp(dists[l], dists[m]))
        {
            m = l;
        }
        if (r < len && cmp(dists[r], dists[m]))
        {
            m = r;
        }
        if (m != index)
        {
            std::swap(dists[m], dists[index]);
            std::swap(label[m], label[index]);
            index = m;
        }
        else
        {
            break;
        }
    }
}

float fvec_L2sqr(const float *x, const float *y, size_t d)
{
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++)
    {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

void fvec_L2sqr_by_idx(float *__restrict dis, const float *x, const float *y, const idx_t *__restrict ids, size_t d,
                       size_t nx, size_t ny)
{
#pragma omp parallel for
    for (size_t j = 0; j < nx; j++)
    {
        const idx_t *__restrict idsj = ids + j * ny;
        const float *xj = x + j * d;
        float *__restrict disj = dis + j * ny;
        for (size_t i = 0; i < ny; i++)
        {
            if (idsj[i] < 0)
            {
                disj[i] = INFINITY;
            }
            else
            {
                disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
            }
        }
    }
}

void IndexIVFRaBitQ::refine(int n, const float *x, int k, float *distances, idx_t *labels, float *topkdist,
                            idx_t *topklabel, const float *srcIndexes)
{
    fvec_L2sqr_by_idx(topkdist, x, srcIndexes, topklabel, dims, n, k);

    std::fill_n(labels, n * k, 0xffffffffffffffff);  // 0xffffffffffffffff为无效label
    std::fill_n(distances, n * k, 0x7f7fffff);       // 小端排序模式初始化为0x7f7fffff，即最大值
    auto cmp = [](float a, float b) -> bool { return a > b; };
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
    {
        const idx_t offset = i * k;
        float *outdists = distances + offset;
        float *srcdist = topkdist + offset;
        idx_t *outlabels = labels + offset;
        idx_t *srclabel = topklabel + offset;
        for (int64_t j = 0; j < k; ++j)
        {
            if (cmp(outdists[0], srcdist[j]))
            {
                outdists[0] = srcdist[j];
                outlabels[0] = srclabel[j];
                UpdateHeap(outdists, outlabels, k, 0, cmp);
            }
        }
        for (int64_t j = k - 1; j >= 1; --j)
        {
            std::swap(outdists[0], outdists[j]);
            std::swap(outlabels[0], outlabels[j]);
            UpdateHeap(outdists, outlabels, j, 0, cmp);
        }
    }
}

APP_ERROR IndexIVFRaBitQ::searchImpl(int n, const float *x, int k, float *distances, idx_t *labels,
                                     const float *srcIndexes, const RabitqIdFilterHost *idFilter, int searchNprobe)
{
    APP_ERROR ret = APP_ERR_OK;
    if (n == 1 || searchBatchSizes.empty())
    {
        return searchWithBatch(n, x, k, distances, labels, srcIndexes, idFilter, searchNprobe);
    }
    size_t size = searchBatchSizes.size();
    int64_t searched = 0;
    for (size_t i = 0; i < size; i++)
    {
        int64_t batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize)
        {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++)
            {
                ret = searchWithBatch(batchSize, x + searched * dims, k, distances + searched * k,
                                      labels + searched * k, srcIndexes, idFilter, searchNprobe);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }
    for (int64_t i = searched; i < n; i++)
    {
        ret =
            searchWithBatch(1, x + i * dims, k, distances + i * k, labels + i * k, srcIndexes, idFilter, searchNprobe);
        APPERR_RETURN_IF(ret, ret);
    }
    return APP_ERR_OK;
}

void IndexIVFRaBitQ::compactList(int listId, std::vector<size_t> &deletedSlots)
{
    auto &list = lists_[listId];
    std::sort(deletedSlots.begin(), deletedSlots.end());
    deletedSlots.erase(std::unique(deletedSlots.begin(), deletedSlots.end()), deletedSlots.end());
    if (deletedSlots.empty())
    {
        return;
    }
    const size_t oldLen = list.listLen;
    ASCEND_THROW_IF_NOT(deletedSlots.size() <= oldLen);
    ASCEND_THROW_IF_NOT(deletedSlots.back() < oldLen);
    const size_t newLen = oldLen - deletedSlots.size();

    // Deleted slots below newLen are holes to fill; surviving slots at or above newLen are the sources
    // that move into them. Both sets have the same size, so this is a pairwise move, not a shift.
    std::unordered_set<size_t> deletedLookup(deletedSlots.begin(), deletedSlots.end());
    std::vector<size_t> holes;
    for (size_t slot : deletedSlots)
    {
        if (slot < newLen)
        {
            holes.push_back(slot);
        }
    }
    std::vector<size_t> sources;
    for (size_t slot = newLen; slot < oldLen; ++slot)
    {
        if (deletedLookup.find(slot) == deletedLookup.end())
        {
            sources.push_back(slot);
        }
    }
    ASCEND_THROW_IF_NOT(holes.size() == sources.size());

    const size_t codeBytes = list.codes.elementsPerSlot() * sizeof(uint8_t);
    for (size_t i = 0; i < holes.size(); ++i)
    {
        auto ret = aclrtMemcpy(list.codes.slot(holes[i]), codeBytes, list.codes.slot(sources[i]), codeBytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        ret = aclrtMemcpy(list.indexL1.slot(holes[i]), sizeof(float), list.indexL1.slot(sources[i]), sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        ret = aclrtMemcpy(list.indexL2.slot(holes[i]), sizeof(float), list.indexL2.slot(sources[i]), sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        ret = aclrtMemcpy(list.ids.slot(holes[i]), sizeof(idx_t), list.ids.slot(sources[i]), sizeof(idx_t),
                          ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
    }

    const size_t segmentCount = utils::divUp(newLen, static_cast<size_t>(blockSize));
    list.codes.resizeSegments(segmentCount);
    list.indexL1.resizeSegments(segmentCount);
    list.indexL2.resizeSegments(segmentCount);
    list.ids.resizeSegments(segmentCount);
    list.codes.commitLength(newLen);
    list.indexL1.commitLength(newLen);
    list.indexL2.commitLength(newLen);
    list.ids.commitLength(newLen);
    list.listLen = newLen;
}

// Collect the slots of `listId` whose id satisfies `isDeleted`, scanning one segment at a time.
template <typename Pred>
static void CollectDeletedSlots(const IndexIVFRaBitQ::ListStorage &list, size_t blockSize, Pred &&isDeleted,
                                std::vector<size_t> &deletedSlots)
{
    std::vector<idx_t> segIds;
    size_t scanned = 0;
    for (size_t segId = 0; segId < list.ids.segmentCount(); ++segId)
    {
        // Clamped to listLen: the buffer below is sized from `count` so it cannot overflow, but an
        // uncommitted growth leaves sum(segmentLength) above listLen, and scanning those slots would feed
        // never-written memory to the predicate and queue bogus slots for deletion.
        const size_t count = std::min(list.ids.segmentLength(segId), list.listLen - scanned);
        if (count == 0)
        {
            // Either an empty segment or listLen is already fully accounted for; nothing follows either way
            // because segments are walked in slot order.
            break;
        }
        scanned += count;
        segIds.resize(count);
        auto ret = aclrtMemcpy(segIds.data(), count * sizeof(idx_t), list.ids.segmentData(segId), count * sizeof(idx_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        for (size_t j = 0; j < count; ++j)
        {
            if (isDeleted(segIds[j]))
            {
                deletedSlots.push_back(segId * blockSize + j);
            }
        }
    }
}

size_t IndexIVFRaBitQ::removeIds(const ascend::IDSelector &sel)
{
    ClearCachedFilterPayload();
    size_t removeCntAll = 0;
#pragma omp parallel for reduction(+ : removeCntAll) num_threads(CommonUtils::GetThreadMaxNums())
    for (int id = 0; id < numLists; id++)
    {
        std::vector<size_t> deletedSlots;
        CollectDeletedSlots(
            lists_[id], static_cast<size_t>(blockSize), [&sel](idx_t vecId) { return sel.is_member(vecId); },
            deletedSlots);
        if (deletedSlots.empty())
        {
            continue;
        }
        compactList(id, deletedSlots);
        removeCntAll += deletedSlots.size();
    }
    this->ntotal -= removeCntAll;
    return removeCntAll;
}

APP_ERROR IndexIVFRaBitQ::removeIds(size_t numVecs, const idx_t *indices)
{
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    ClearCachedFilterPayload();
    std::unordered_set<idx_t> deleteSet(indices, indices + numVecs);
    size_t removeCntAll = 0;
    for (int id = 0; id < numLists; id++)
    {
        std::vector<size_t> deletedSlots;
        CollectDeletedSlots(
            lists_[id], static_cast<size_t>(blockSize),
            [&deleteSet](idx_t vecId) { return deleteSet.find(vecId) != deleteSet.end(); }, deletedSlots);
        if (deletedSlots.empty())
        {
            continue;
        }
        compactList(id, deletedSlots);
        removeCntAll += deletedSlots.size();
    }
    this->ntotal -= removeCntAll;
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
                                     AscendTensor<float16_t, DIMS_2> &outDistance,
                                     AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(k);
    VALUE_UNUSED(outDistance);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRaBitQ::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
                                      AscendTensor<float16_t, DIMS_2> &maxDistances,
                                      AscendTensor<int64_t, DIMS_2> &maxIndices)
{
    VALUE_UNUSED(pageId);
    VALUE_UNUSED(pageNum);
    VALUE_UNUSED(queries);
    VALUE_UNUSED(maxDistances);
    VALUE_UNUSED(maxIndices);
    return APP_ERR_OK;
}

}  // namespace ascend
