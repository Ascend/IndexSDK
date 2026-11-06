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

#include "topk_ivf_rabitq_fp32_cpu_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_shared_def.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "roaring.h"

namespace
{
const char *TOPK_IVF_RABITQ_FP32 = "TopkIvfRabitqFp32";
constexpr int kBitmapPrefetchWindow = 16;

struct BitmapCandidate
{
    float dist;
    int64_t id;
};

bool BitmapBitIsSet(const uint8_t *bitmap, int64_t nbits, int64_t id)
{
    const uint64_t uid = static_cast<uint64_t>(id);
    if (uid >= static_cast<uint64_t>(nbits))
    {
        return false;
    }
    return ((bitmap[uid >> 3] >> (uid & 7u)) & 1u) != 0;
}

void PrefetchBitmapByte(const uint8_t *bitmap, int64_t nbits, int64_t id)
{
    const uint64_t uid = static_cast<uint64_t>(id);
    if (uid >= static_cast<uint64_t>(nbits))
    {
        return;
    }
    __builtin_prefetch(bitmap + (uid >> 3), 0, 0);
}

// Below this k, the heap's k*12B working set fits in AICPU L1D (32KB), so the in-place heap drain's
// sift-down walk hits cache and the packed sort's pack+sort+scatter passes would cost more than they save.
// 2048 -> ~24KB, the largest power-of-two k that stays comfortably inside L1D alongside the scratch.
// At k=5000 the set is ~60KB and the drain is measurably off-cache: reorder came in at ~348us before this
// change, i.e. ~70ns per drained entry for what is nominally an O(log k) sift.
constexpr int64_t PACKED_SORT_MIN_K = 2048;

// How many candidates the reservoir accepts beyond k before compacting back down to the best k.
// B trades two costs against each other. Larger B amortises the O(k) compaction over more accepts, but it
// also lets the pruning cutoff go staler, so more bursts survive the burst-level test and candSeen rises.
//
// On hardware at k=5000 / 200 blocks / burstLen=64, a query does ~21.5k accepts over ~235k scanned
// candidates (~9% accept rate), i.e. ~8 refills at B=2048. Buffer is (k+B)*8B = ~55KB since RabitqCand
// carries a uint32 id (it was 16B and ~110KB while the id was an int64 -- see the struct's comment).
//
// A joint fit of per-block timings on blockTime ~ a*candSeen + b*heapPush + c*burstSkip + d*refills
// (two dies x 197 blocks, R^2 0.85/0.88) puts a refill at 62.6 +/- 2.7 us -- about 500us per query, ~20%
// of blockScan, and the two dies agree to within 0.6 sigma. Per-candidate scan is 7.2 ns and a marginal
// accept only 7.1 ns, so refill is the second-largest term after the scan loop itself.
//
// Raising B is the direct lever on this term, but it does not scale the way the refill count does: a
// refill selects over k+B entries, so the per-refill cost grows as B grows. Modelled against the measured
// 62.6us: B=4096 -> 4 refills, ~327us (-180us); B=8192 -> 2 refills, ~237us (-270us), at 103KB per worker
// thread. Worth trying, but not measured on hardware yet, and the staler cutoff pushes candSeen (a
// 7.2 ns/candidate term over ~235k candidates) the other way -- which is why B stays at 2048 until an A/B
// says otherwise.
//
// This is the DEFAULT, not the only value: IVFRABITQ_TOPK_REFILL_CHUNK overrides it (host reads the env
// and passes log2(B) in QUERY_NUM bits 36-39; see refillChunk_). The env exists precisely because the two
// terms above pull in opposite directions -- that is a question for an A/B on hardware, and requiring a
// recompile to ask it is how the wrong answer stays in place.
//
// The value itself lives in kernel_shared_def.h, so the member initialiser and this fallback cannot drift
// apart. This alias keeps the REASONING (measurement-specific and kernel-internal) next to the code it
// governs.
constexpr int64_t REFILL_CHUNK_DEFAULT = aicpu::TOPK_IVF_RABITQ_REFILL_DEFAULT;

// The ONLY burstLen the reservoir path accepts; DoCompute refuses to activate the reservoir for anything
// else and the historical k-heap path runs instead. A compile-time constant rather than a use of the
// burstLen_ member in the per-candidate loop, which is the whole point: with a literal bound the compiler
// can fully unroll the inner loop and drop the loop counter, which it cannot do against a member it must
// reload. 64 is also the only step that matches the producer: distance_ivf_rabitq_l2_fp32 reduces at a
// hardcoded mask of 64 and strides its per-core output by ceil(blockSize/64)*2 floats. Gating rather than
// rejecting keeps this change invisible to any caller that still sends a different burstLen.
constexpr int64_t BURST_LEN = 64;

// Below this k the k-heap's working set is L1-resident and its sift-downs are cheap, so the reservoir's
// buffer + nth_element passes would cost more than they save. Same threshold rationale as
// PACKED_SORT_MIN_K above.
constexpr int64_t RESERVOIR_MIN_K = 2048;

// Map a float's bits to a uint32 whose ascending integer order matches the desired float order, so entries
// can be packed into a uint64 and sorted with no comparator. Negatives get all bits flipped and
// non-negatives just the sign bit set, which makes integer order match ascending float order across the
// sign boundary; `descending` then inverts the whole key. Shared by Reorder and DrainReservoir — the
// flip is subtle enough that a second hand-written copy would eventually drift out of sync.
// (-0.0 and 0.0 map to different keys, but float comparison calls them equal, so either order is valid.)
inline uint32_t PackOrderKey(float dist, bool descending)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &dist, sizeof(bits));
    bits = (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
    return descending ? ~bits : bits;
}

// Exact inverse of PackOrderKey: recovers the original float from the order key. Lets the sort's output be
// reconstructed from the keys alone, so the distance array is never re-read during scatter.
inline float UnpackOrderKey(uint32_t bits, bool descending)
{
    if (descending)
    {
        bits = ~bits;
    }
    bits = (bits & 0x80000000u) != 0 ? (bits & 0x7fffffffu) : ~bits;
    float dist = 0.0f;
    std::memcpy(&dist, &bits, sizeof(dist));
    return dist;
}
}  // namespace

namespace aicpu
{
uint32_t TopkIvfRabitqfP32CpuKernel::Compute(CpuKernelContext &ctx)
{
    Inputs inputs;
    Outputs outputs;
    auto ret = GetInOutAndCheck(ctx, inputs, outputs);
    if (ret != KERNEL_STATUS_OK)
    {
        KERNEL_LOG_ERROR("Failed to get inputs or outputs");
        return ret;
    }

    ret = CheckInputShapes(inputs);
    if (ret != KERNEL_STATUS_OK)
    {
        KERNEL_LOG_ERROR("Failed to check input shapes");
        return ret;
    }

    blockOffset_.resize(nq_ + 1);
    KernelTensor<int64_t> blocknums(inputs.blocknums);
    int64_t offset = 0;
    maxBlocksPerQ_ = 0;
    for (int64_t qidx = 0; qidx < nq_; qidx++)
    {
        blockOffset_[qidx] = offset;
        int64_t thisBlockNum = *(blocknums.GetSubTensorDim0(qidx));
        offset += thisBlockNum;
        maxBlocksPerQ_ = std::max(maxBlocksPerQ_, thisBlockNum);
    }
    blockOffset_[nq_] = offset;

    ret = UpdateInOutShape(inputs, outputs);
    if (ret != KERNEL_STATUS_OK)
    {
        KERNEL_LOG_ERROR("Failed to update in/out shape");
        return ret;
    }

    // Must run after UpdateInOutShape, which is what resolves maxBlocksPerQ_ into the table's row stride,
    // and before InitTopkHeap, so a rejected plan leaves the outputs untouched rather than half-initialised.
    ret = ValidateBlockSlots(inputs);
    if (ret != KERNEL_STATUS_OK)
    {
        KERNEL_LOG_ERROR("Failed to validate blockSlots");
        return ret;
    }

    InitTopkHeap(outputs);
    selRoaringView_ = nullptr;
    if (selMode_ == RABITQ_ID_FILTER_ROARING && selRoaringBuf_ != nullptr && selAux0_ > 0)
    {
        selRoaringView_ = roaring_bitmap_frozen_view(selRoaringBuf_, static_cast<size_t>(selAux0_));
        if (selRoaringView_ == nullptr)
        {
            KERNEL_LOG_ERROR("Failed to create roaring frozen view, bytes=%ld", selAux0_);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    auto funcLess = [](float a, float b) -> bool { return a < b; };
    auto funcGreater = [](float a, float b) -> bool { return a > b; };

#ifdef AICPU_UTEST
    uint32_t core = 1;
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
#endif

    auto computeFunc = [&](size_t start, size_t end)
    {
        (void)end;  // end is unuseful in this function
        if (asc_ != 0)
        {
            // put greatest one to top of heap
            DoCompute(core, start, inputs, outputs, funcGreater);
        }
        else
        {
            // put least one to top of heap
            DoCompute(core, start, inputs, outputs, funcLess);
        }
    };

#ifdef AICPU_UTEST
    computeFunc(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif

    if (selRoaringView_ != nullptr)
    {
        roaring_bitmap_free(const_cast<roaring_bitmap_t *>(static_cast<const roaring_bitmap_t *>(selRoaringView_)));
        selRoaringView_ = nullptr;
    }

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfRabitqfP32CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs,
                                                      Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfRabitqfP32CpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.ids = ctx.Input(INPUT_NUM2);
    inputs.size = ctx.Input(INPUT_NUM3);
    inputs.blocknums = ctx.Input(INPUT_NUM4);
    inputs.opflag = ctx.Input(INPUT_NUM5);
    inputs.attr = ctx.Input(INPUT_NUM6);
    inputs.blockSlots = ctx.Input(INPUT_NUM7);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[ids] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.blocknums, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[blocknums] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[attr] failed");
    KERNEL_CHECK_NULLPTR(inputs.blockSlots, KERNEL_STATUS_PARAM_INVALID, "Get input[7], name[blockSlots] failed");
    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

    KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
                    ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][vmdists] is %s",
                    ShapeToString(inputs.vmdists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[2][ids] is %s", ShapeToString(inputs.ids->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[3][size] is %s",
                    ShapeToString(inputs.size->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[4][blocknums] is %s",
                    ShapeToString(inputs.blocknums->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[5][opflag] is %s",
                    ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[6][attr] is %s",
                    ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[7][blockSlots] is %s",
                    ShapeToString(inputs.blockSlots->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfRabitqfP32CpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvfRabitqfP32CpuKernel CheckInputShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeIds = inputs.ids->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeBlocknums = inputs.blocknums->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();
    auto shapeBlockSlots = inputs.blockSlots->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[0][indists] must be 2");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[1][vmdists] must be 2");
    KERNEL_CHECK_TRUE(shapeIds->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[2][ids] must be 2");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[3][size] must be 2");
    KERNEL_CHECK_TRUE(shapeBlocknums->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[4][blocknums] must be 1");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[6][attr] must be 1");
    KERNEL_CHECK_TRUE(shapeBlockSlots->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[7][blockSlots] must be 2");

    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM1);
    nq_ = shapeBlocknums->GetDimSize(INPUT_NUM0);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVF_RABITQ_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID,
                      "Num of attrs must be %d", TOPK_IVF_RABITQ_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_IVF_RABITQ_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_IVF_RABITQ_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_IVF_RABITQ_ATTR_BURST_LEN_IDX);
    core_ = *(attr + TOPK_IVF_RABITQ_ATTR_CORE_NUM_IDX);
    selMode_ = static_cast<RabitqIdFilterMode>(*(attr + TOPK_IVF_RABITQ_ATTR_SEL_MODE_IDX));
    selNegate_ = *(attr + TOPK_IVF_RABITQ_ATTR_SEL_NEGATE_IDX);
    selAux0_ = *(attr + TOPK_IVF_RABITQ_ATTR_SEL_AUX0_IDX);
    selAux1_ = *(attr + TOPK_IVF_RABITQ_ATTR_SEL_AUX1_IDX);
    const int64_t selPtr = *(attr + TOPK_IVF_RABITQ_ATTR_SEL_PTR_IDX);
    layoutMode_ = *(attr + TOPK_IVF_RABITQ_ATTR_LAYOUT_MODE_IDX);
    selSorted_ = nullptr;
    selBitmap_ = nullptr;
    selRoaringBuf_ = nullptr;

    // Optional-path toggles ride high bits of the QUERY_NUM slot (the host reads the envs and sets them).
    // They cannot be read from the environment here: AICPU kernels execute in a separate device-side
    // environment that does not inherit the host process's environ, so std::getenv would always return null.
    const int64_t queryNumAttr = *(attr + TOPK_IVF_RABITQ_ATTR_QUERY_NUM_IDX);
    reservoirEnabled_ = (queryNumAttr & TOPK_IVF_RABITQ_RESERVOIR_FLAG) != 0;
    packedSortEnabled_ = (queryNumAttr & TOPK_IVF_RABITQ_PACKED_SORT_FLAG) != 0;
    // REFILL_CHUNK override, carried as log2(B) in bits 36-39. Zero means the host sent nothing (or is an
    // older host that does not know this field), in which case the compiled-in default stands. An
    // out-of-range non-zero value is rejected rather than clamped: it can only come from a deliberate env
    // setting, and silently running a different B than was asked for would corrupt the very A/B the env
    // exists to serve.
    const int64_t refillLog2 = (queryNumAttr >> TOPK_IVF_RABITQ_REFILL_LOG2_SHIFT) & TOPK_IVF_RABITQ_REFILL_LOG2_MASK;
    if (refillLog2 != 0)
    {
        KERNEL_CHECK_TRUE(
            refillLog2 >= TOPK_IVF_RABITQ_REFILL_LOG2_MIN && refillLog2 <= TOPK_IVF_RABITQ_REFILL_LOG2_MAX,
            KERNEL_STATUS_PARAM_INVALID, "refill log2 must be in [%ld, %ld], got %ld",
            static_cast<int64_t>(TOPK_IVF_RABITQ_REFILL_LOG2_MIN),
            static_cast<int64_t>(TOPK_IVF_RABITQ_REFILL_LOG2_MAX), refillLog2);
        refillChunk_ = 1LL << refillLog2;
    }
    else
    {
        refillChunk_ = REFILL_CHUNK_DEFAULT;
    }
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && core_ > 0 && nq_ > 0, KERNEL_STATUS_PARAM_INVALID,
                      "Value of asc, k, bustLen, core, nq must ge 0");
    KERNEL_CHECK_TRUE(layoutMode_ >= 0 && layoutMode_ < RABITQ_LAYOUT_MODE_COUNT, KERNEL_STATUS_PARAM_INVALID,
                      "Value of layoutMode must be in [0, %d)", RABITQ_LAYOUT_MODE_COUNT);
    KERNEL_CHECK_TRUE(selMode_ == RABITQ_ID_FILTER_NONE || selMode_ == RABITQ_ID_FILTER_RANGE ||
                          selMode_ == RABITQ_ID_FILTER_SORTED || selMode_ == RABITQ_ID_FILTER_BITMAP ||
                          selMode_ == RABITQ_ID_FILTER_ROARING,
                      KERNEL_STATUS_PARAM_INVALID, "Unsupported selMode %ld", static_cast<int64_t>(selMode_));
    KERNEL_CHECK_TRUE(selNegate_ == 0 || selNegate_ == 1, KERNEL_STATUS_PARAM_INVALID, "selNegate must be 0 or 1");

    switch (selMode_)
    {
        case RABITQ_ID_FILTER_NONE:
        case RABITQ_ID_FILTER_RANGE:
            break;
        case RABITQ_ID_FILTER_SORTED:
            KERNEL_CHECK_TRUE(selAux0_ >= 0, KERNEL_STATUS_PARAM_INVALID, "selAux0 (sorted count) must ge 0");
            if (selAux0_ > 0)
            {
                KERNEL_CHECK_TRUE(selPtr != 0, KERNEL_STATUS_PARAM_INVALID,
                                  "selPtr must be non-null for sorted filter");
                selSorted_ = reinterpret_cast<const int64_t *>(selPtr);
            }
            break;
        case RABITQ_ID_FILTER_BITMAP:
            KERNEL_CHECK_TRUE(selAux0_ >= 0 && selAux0_ % 8 == 0, KERNEL_STATUS_PARAM_INVALID,
                              "selAux0 (bitmap bit count) must be a multiple of 8");
            if (selAux0_ > 0)
            {
                KERNEL_CHECK_TRUE(selPtr != 0, KERNEL_STATUS_PARAM_INVALID,
                                  "selPtr must be non-null for bitmap filter");
                selBitmap_ = reinterpret_cast<const uint8_t *>(selPtr);
            }
            break;
        case RABITQ_ID_FILTER_ROARING:
            KERNEL_CHECK_TRUE(selAux0_ >= 0, KERNEL_STATUS_PARAM_INVALID, "selAux0 (roaring frozen bytes) must ge 0");
            if (selAux0_ > 0)
            {
                KERNEL_CHECK_TRUE(selPtr != 0, KERNEL_STATUS_PARAM_INVALID,
                                  "selPtr must be non-null for roaring filter");
                KERNEL_CHECK_TRUE((static_cast<uint64_t>(selPtr) & (RABITQ_ROARING_FROZEN_ALIGN - 1U)) == 0,
                                  KERNEL_STATUS_PARAM_INVALID,
                                  "selPtr must be 32-byte aligned for roaring frozen view");
                selRoaringBuf_ = reinterpret_cast<const char *>(selPtr);
            }
            break;
        default:
            KERNEL_LOG_ERROR("Unsupported selMode %ld", static_cast<int64_t>(selMode_));
            return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfRabitqfP32CpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfRabitqfP32CpuKernel UpdateInOutShape begin");

    int64_t allBlock = (blockOffset_[nq_] + core_ - 1) / core_ * core_;

    auto shapeIndists = inputs.indists->GetTensorShape();
    std::vector<int64_t> dimIndists = shapeIndists->GetDimSizes();
    dimIndists[INPUT_NUM0] = allBlock;
    shapeIndists->SetDimSizes(dimIndists);

    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    std::vector<int64_t> dimVmdists = shapeVmdists->GetDimSizes();
    dimVmdists[INPUT_NUM0] = allBlock;
    shapeVmdists->SetDimSizes(dimVmdists);

    auto shapeIds = inputs.ids->GetTensorShape();
    std::vector<int64_t> dimIds = shapeIds->GetDimSizes();
    dimIds[INPUT_NUM0] = allBlock;
    shapeIds->SetDimSizes(dimIds);

    auto shapeSize = inputs.size->GetTensorShape();
    std::vector<int64_t> dimSize = shapeSize->GetDimSizes();
    dimSize[INPUT_NUM0] = allBlock;
    shapeSize->SetDimSizes(dimSize);

    auto shapeOpFlag = inputs.opflag->GetTensorShape();
    std::vector<int64_t> dimOpFlag = shapeOpFlag->GetDimSizes();
    dimOpFlag[INPUT_NUM0] = allBlock;
    shapeOpFlag->SetDimSizes(dimOpFlag);

    // blockSlots' row length is declared dynamic in the op desc and the framework leaves it at 0, so
    // KernelTensor would derive a zero stride and every row would alias row 0. It has to be resolved from
    // blocknums here, before any KernelTensor over it is constructed.
    //
    // The row width cannot come from the op desc instead: resetL2TopkOp builds the op per entry of
    // searchBatchSizes at init time, and maxBlocksPerQ depends on how many blocks this request's L1 probe
    // result actually selects, which is not known until the search runs. So the host and this function each
    // compute the same number independently (host: getBlockNumPerQ; here: max over blocknums), and the only
    // defence against the two drifting apart is to check them against each other -- hence the assert below
    // rather than a silent overwrite. A mismatch means the host sized the device buffer to one width while
    // the kernel is about to stride it by another, i.e. every row past the first reads at the wrong offset.
    auto shapeBlockSlots = inputs.blockSlots->GetTensorShape();
    std::vector<int64_t> dimBlockSlots = shapeBlockSlots->GetDimSizes();
    const int64_t declaredRows = dimBlockSlots[INPUT_NUM0];
    const int64_t declaredDim1 = dimBlockSlots[INPUT_NUM1];
    // 0 is the legal "framework left it dynamic" value; anything else must agree with what we derived.
    KERNEL_CHECK_TRUE(declaredDim1 == 0 || declaredDim1 == maxBlocksPerQ_, KERNEL_STATUS_PARAM_INVALID,
                      "blockSlots declared row width %ld disagrees with maxBlocksPerQ %ld derived from blocknums",
                      declaredDim1, maxBlocksPerQ_);
    KERNEL_CHECK_TRUE(declaredRows == nq_, KERNEL_STATUS_PARAM_INVALID,
                      "blockSlots declared rows %ld must equal nq %ld", declaredRows, nq_);
    dimBlockSlots[INPUT_NUM0] = nq_;
    dimBlockSlots[INPUT_NUM1] = maxBlocksPerQ_;
    shapeBlockSlots->SetDimSizes(dimBlockSlots);

    auto shapeOutdists = outputs.outdists->GetTensorShape();
    std::vector<int64_t> dimOutdists;
    dimOutdists.push_back(nq_);
    dimOutdists.push_back(k_);
    shapeOutdists->SetDimSizes(dimOutdists);

    auto shapeOutlabels = outputs.outlabels->GetTensorShape();
    std::vector<int64_t> dimOutlabels;
    dimOutlabels.push_back(nq_);
    dimOutlabels.push_back(k_);
    shapeOutlabels->SetDimSizes(dimOutlabels);
    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfRabitqfP32CpuKernel::ValidateBlockSlots(const Inputs &inputs) const
{
    // LEGACY never consults the table -- SlotOf returns blockOffset_[q] + i, which is correct by
    // construction -- so there is nothing to validate and the whole pass is skipped.
    if (layoutMode_ == RABITQ_LAYOUT_LEGACY)
    {
        return KERNEL_STATUS_OK;
    }

    const int64_t totalBlock = blockOffset_[nq_];
    if (totalBlock <= 0)
    {
        return KERNEL_STATUS_OK;
    }

    const int64_t *slots = static_cast<const int64_t *>(inputs.blockSlots->GetData());
    KERNEL_CHECK_NULLPTR(slots, KERNEL_STATUS_PARAM_INVALID, "blockSlots data is null");
    const int64_t *blocknums = static_cast<const int64_t *>(inputs.blocknums->GetData());
    KERNEL_CHECK_NULLPTR(blocknums, KERNEL_STATUS_PARAM_INVALID, "blocknums data is null");

    // One bit per grid slot. The table is the only thing that says which block a query reads, so the cost
    // here is O(totalBlock) on a few thousand entries once per call, against a per-candidate inner loop that
    // runs over millions -- it does not register.
    //
    // In-range plus injective is already the full bijection, with no third check needed: totalBlock is by
    // construction the sum of blocknums (Compute accumulates it into blockOffset_[nq_]), so this loop visits
    // exactly totalBlock entries. totalBlock distinct values drawn from [0, totalBlock) must exhaust the
    // range, so surjectivity -- "no staged block goes unread" -- follows by pigeonhole rather than by
    // counting.
    std::vector<bool> seen(static_cast<size_t>(totalBlock), false);
    for (int64_t qidx = 0; qidx < nq_; qidx++)
    {
        const int64_t thisBlockNum = blocknums[qidx];
        KERNEL_CHECK_TRUE(thisBlockNum >= 0 && thisBlockNum <= maxBlocksPerQ_, KERNEL_STATUS_PARAM_INVALID,
                          "blocknums[%ld]=%ld outside [0, maxBlocksPerQ=%ld]", qidx, thisBlockNum, maxBlocksPerQ_);
        for (int64_t blkidx = 0; blkidx < thisBlockNum; blkidx++)
        {
            const int64_t slot = slots[qidx * maxBlocksPerQ_ + blkidx];
            // In range. Without this the indexing in RunQueries walks off the staged distance grid into
            // whatever follows it, and the candidates it finds there look perfectly valid.
            KERNEL_CHECK_TRUE(slot >= 0 && slot < totalBlock, KERNEL_STATUS_PARAM_INVALID,
                              "blockSlots[%ld][%ld]=%ld outside [0, %ld)", qidx, blkidx, slot, totalBlock);
            // Injective. An in-range duplicate is the failure a range check cannot see: one block's
            // candidates get scanned twice while another's are never scanned at all, and the only symptom
            // is a recall number that is quietly too low.
            KERNEL_CHECK_TRUE(!seen[static_cast<size_t>(slot)], KERNEL_STATUS_PARAM_INVALID,
                              "blockSlots[%ld][%ld]=%ld is a duplicate slot", qidx, blkidx, slot);
            seen[static_cast<size_t>(slot)] = true;
        }
    }

    return KERNEL_STATUS_OK;
}

void TopkIvfRabitqfP32CpuKernel::InitTopkHeap(Outputs &outputs) const
{
    float *outdists = static_cast<float *>(outputs.outdists->GetData());
    int64_t *outlabels = static_cast<int64_t *>(outputs.outlabels->GetData());
    std::fill_n(outlabels, nq_ * k_, 0xffffffffffffffff);  // 0xffffffffffffffff为无效label
    if (asc_ != 0)
    {
        std::fill_n(outdists, nq_ * k_, std::numeric_limits<float>::max());  // 小端排序模式初始化为最大值
    }
    else
    {
        std::fill_n(outdists, nq_ * k_, std::numeric_limits<float>::lowest());  // 大端排序模式初始化为最小值
    }
}

template <typename C, typename Pred>
void TopkIvfRabitqfP32CpuKernel::RunQueries(size_t tcnt, size_t tid, KernelTensor<float> &indists,
                                            KernelTensor<float> &vmdists, KernelTensor<int64_t> &ids,
                                            KernelTensor<uint32_t> &size, KernelTensor<int64_t> &blocknums,
                                            KernelTensor<uint16_t> &opflag, KernelTensor<int64_t> &blockSlots,
                                            KernelTensor<float> &outdists, KernelTensor<int64_t> &outlabels,
                                            std::vector<uint64_t> &scratch, RabitqReservoir &res, C &&cmp,
                                            Pred &&isSelected)
{
    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt)
    {
        int64_t thisBlockNum = *(blocknums.GetSubTensorDim0(qidx));
        // Reset per query: the buffer allocation is reused, the contents are not. thresh starts at the
        // accept-all sentinel — the extreme that every real distance compares better than under cmp.
        res.count = 0;
        res.everFull = false;
        res.thresh = (asc_ != 0) ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
        for (int64_t blkidx = 0; blkidx < thisBlockNum; blkidx++)
        {
            int64_t slot = SlotOf(qidx, blkidx, blockSlots);
            if (slot < 0)
            {
                continue;
            }
            if (*(size.GetSubTensorDim0(slot)) == 0)
            {
                continue;
            }
            auto flagPtr = opflag.GetSubTensorDim0(slot);
            WAITING_FLAG_READY(*(flagPtr), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            ComputeQuery(qidx, slot, indists, vmdists, ids, size, outdists, outlabels, res, cmp, isSelected);
        }
        // Both exits produce the same thing — the best k, sorted into the output tensors — but from
        // different containers: the reservoir path sorts its flat buffer, the heap path drains the heap that
        // already lives in the output slots.
        if (res.active)
        {
            DrainReservoir(qidx, outdists, outlabels, res, scratch, cmp);
        }
        else
        {
            Reorder(qidx, outdists, outlabels, scratch, cmp);
        }
    }
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float> indists(inputs.indists);
    KernelTensor<float> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<int64_t> blocknums(inputs.blocknums);
    KernelTensor<uint16_t> opflag(inputs.opflag);
    KernelTensor<int64_t> blockSlots(inputs.blockSlots);

    KernelTensor<float> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    // Scratch for the packed introsort, allocated once per DoCompute call (i.e. once per worker thread, not
    // per query) and reused across this thread's queries — a per-query allocation would show up as malloc
    // traffic in the very path being optimized. Used by BOTH exits: Reorder (heap path) and DrainReservoir
    // (reservoir path) sort the same packed uint64 representation. Left empty when it does not apply, which
    // makes both fall back to their comparator-based sort:
    //   - k_ < PACKED_SORT_MIN_K: a k*12B working set that small stays L1-resident, so the heap drain's
    //     random walk is cheap and packing would be pure overhead.
    //   - k_ > UINT32_MAX: the low 32 bits of a packed entry hold the source slot and would overflow.
    //   - packedSortEnabled_ off (host env IVFRABITQ_TOPK_PACKED_SORT=0): the A/B control arm.
    std::vector<uint64_t> reorderScratch;
    if (packedSortEnabled_ && k_ >= PACKED_SORT_MIN_K && k_ <= static_cast<int64_t>(UINT32_MAX))
    {
        reorderScratch.resize(static_cast<size_t>(k_));
    }
    // Reservoir buffer, allocated once per DoCompute (per worker thread) and reused across this thread's
    // queries — allocating per query would put malloc traffic in the path being optimised. Disabled for
    // small k, where the k-heap's working set is already L1-resident, and for any burstLen other than
    // BURST_LEN, which the reservoir's literal-bound inner loop assumes.
    RabitqReservoir res;
    res.active = reservoirEnabled_ && k_ >= RESERVOIR_MIN_K && burstLen_ == BURST_LEN;
    if (res.active)
    {
        res.buf.resize(static_cast<size_t>(k_ + refillChunk_));
    }

    const bool negate = selNegate_ != 0;
    switch (selMode_)
    {
        case RABITQ_ID_FILTER_NONE:
            RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                       reorderScratch, res, cmp, [](int64_t) { return true; });
            break;
        case RABITQ_ID_FILTER_RANGE:
        {
            const int64_t lo = selAux0_;
            const int64_t hi = selAux1_;
            if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp, [lo, hi](int64_t id) { return id < lo || id >= hi; });
            }
            else
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp, [lo, hi](int64_t id) { return id >= lo && id < hi; });
            }
            break;
        }
        case RABITQ_ID_FILTER_SORTED:
        {
            const int64_t *sorted = selSorted_;
            const int64_t n = selAux0_;
            if (sorted == nullptr || n <= 0)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp, [negate](int64_t) { return negate; });
            }
            else if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp,
                           [sorted, n](int64_t id) { return !std::binary_search(sorted, sorted + n, id); });
            }
            else
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp,
                           [sorted, n](int64_t id) { return std::binary_search(sorted, sorted + n, id); });
            }
            break;
        }
        case RABITQ_ID_FILTER_BITMAP:
        {
            if (selBitmap_ == nullptr)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp, [negate](int64_t) { return negate; });
            }
            else
            {
                // Heap path only, whatever reservoirEnabled_ says: this variant inlines its own scan around a
                // prefetch window over the bitmap, so there is no accept hook for the reservoir to take over.
                // Not a correctness gap, just an unoptimised combination — the heap result is the same result.
                RunBitmapQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists,
                                 outlabels, reorderScratch, cmp);
            }
            break;
        }
        case RABITQ_ID_FILTER_ROARING:
        {
            const roaring_bitmap_t *roaringView = static_cast<const roaring_bitmap_t *>(selRoaringView_);
            if (roaringView == nullptr)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp, [negate](int64_t) { return negate; });
            }
            else if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp,
                           [roaringView](int64_t id)
                           {
                               if (id < 0 || static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max())
                               {
                                   return true;
                               }
                               return roaring_bitmap_contains(roaringView, static_cast<uint32_t>(id)) == 0;
                           });
            }
            else
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           reorderScratch, res, cmp,
                           [roaringView](int64_t id)
                           {
                               if (id < 0 || static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max())
                               {
                                   return false;
                               }
                               return roaring_bitmap_contains(roaringView, static_cast<uint32_t>(id)) != 0;
                           });
            }
            break;
        }
        default:
            RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                       reorderScratch, res, cmp, [](int64_t) { return false; });
            break;
    }
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::RunBitmapQueries(size_t tcnt, size_t tid, KernelTensor<float> &indists,
                                                  KernelTensor<float> &vmdists, KernelTensor<int64_t> &ids,
                                                  KernelTensor<uint32_t> &size, KernelTensor<int64_t> &blocknums,
                                                  KernelTensor<uint16_t> &opflag, KernelTensor<int64_t> &blockSlots,
                                                  KernelTensor<float> &outdists, KernelTensor<int64_t> &outlabels,
                                                  std::vector<uint64_t> &scratch, C &&cmp)
{
    const bool negate = selNegate_ != 0;
    const uint8_t *bitmap = selBitmap_;
    const int64_t nbits = selAux0_;
    BitmapCandidate window[kBitmapPrefetchWindow];
    int nwin = 0;

    auto drain = [&](float *heapDists, int64_t *heapLabels)
    {
        for (int i = 0; i < nwin; ++i)
        {
            if (!cmp(heapDists[0], window[i].dist))
            {
                continue;
            }
            const bool bit = BitmapBitIsSet(bitmap, nbits, window[i].id);
            const bool selected = negate ? !bit : bit;
            if (!selected)
            {
                continue;
            }
            heapDists[0] = window[i].dist;
            heapLabels[0] = window[i].id;
            UpdateHeap(heapDists, heapLabels, k_, 0, cmp);
        }
        nwin = 0;
    };

    auto push = [&](float *heapDists, int64_t *heapLabels, float dist, int64_t id)
    {
        PrefetchBitmapByte(bitmap, nbits, id);
        window[nwin].dist = dist;
        window[nwin].id = id;
        ++nwin;
        if (nwin == kBitmapPrefetchWindow)
        {
            drain(heapDists, heapLabels);
        }
    };

    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt)
    {
        nwin = 0;
        float *heapDists = outdists.GetSubTensorDim0(qidx);
        int64_t *heapLabels = outlabels.GetSubTensorDim0(qidx);
        int64_t thisBlockNum = *(blocknums.GetSubTensorDim0(qidx));
        for (int64_t blkidx = 0; blkidx < thisBlockNum; ++blkidx)
        {
            int64_t slot = SlotOf(qidx, blkidx, blockSlots);
            if (slot < 0)
            {
                continue;
            }
            if (*(size.GetSubTensorDim0(slot)) == 0)
            {
                continue;
            }
            auto flagPtr = opflag.GetSubTensorDim0(slot);
            WAITING_FLAG_READY(*(flagPtr), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

            float *blockDists = indists.GetSubTensorDim0(slot);
            float *blockVm = vmdists.GetSubTensorDim0(slot);
            int64_t *idBase = reinterpret_cast<int64_t *>(*(ids.GetSubTensorDim0(slot)));
            int64_t ntotal = static_cast<int64_t>(*(size.GetSubTensorDim0(slot)));
            int64_t idx = 0;
            int64_t burstSize = ntotal / burstLen_;
            for (int64_t i = 0; i < burstSize; ++i)
            {
                if (!cmp(heapDists[0], blockVm[i * 2]))
                {
                    idx += burstLen_;
                    continue;
                }
                for (int64_t j = 0; j < burstLen_ && idx < ntotal; ++j, ++idx)
                {
                    if (!cmp(heapDists[0], blockDists[idx]))
                    {
                        continue;
                    }
                    push(heapDists, heapLabels, blockDists[idx], *(idBase + idx));
                }
            }
            while (idx < ntotal)
            {
                if (cmp(heapDists[0], blockDists[idx]))
                {
                    push(heapDists, heapLabels, blockDists[idx], *(idBase + idx));
                }
                ++idx;
            }
        }
        drain(heapDists, heapLabels);
        Reorder(qidx, outdists, outlabels, scratch, cmp);
    }
}

int64_t TopkIvfRabitqfP32CpuKernel::SlotOf(int64_t qidx, int64_t blkidx, KernelTensor<int64_t> &blockSlots) const
{
    // Under LEGACY the host fills the grid query-major, so a query's blocks are contiguous and the prefix sum
    // already names them -- the indirection table is not even staged. Any other layout scatters them, and the
    // table is the only way back from (query, i-th block of that query) to a grid slot.
    if (layoutMode_ == RABITQ_LAYOUT_LEGACY)
    {
        return static_cast<int64_t>(blockOffset_[qidx]) + blkidx;
    }
    if (blkidx >= maxBlocksPerQ_)
    {
        return -1;
    }
    // No bound test on the value read back: ValidateBlockSlots has already proved, for this call, that every
    // slot a query will ask for is in [0, totalBlock) and that no two asks collide. Re-checking per block
    // would put a compare and a member load in a loop that runs once per block of every query, to re-derive
    // a fact that cannot have changed since the table was last written.
    return *(blockSlots.GetSubTensorDim1(qidx, blkidx));
}

template <typename C, typename Pred>
void TopkIvfRabitqfP32CpuKernel::ComputeQuery(int64_t qidx, int64_t blkidx, KernelTensor<float> &indistsTensor,
                                              KernelTensor<float> &vmdistsTensor, KernelTensor<int64_t> &idsTensor,
                                              KernelTensor<uint32_t> &sizeTensor, KernelTensor<float> &outdistsTensor,
                                              KernelTensor<int64_t> &outlabelsTensor, RabitqReservoir &res, C &&cmp,
                                              Pred &&isSelected)
{
    float *indists = indistsTensor.GetSubTensorDim0(blkidx);
    float *vmdists = vmdistsTensor.GetSubTensorDim0(blkidx);
    int64_t *ids = idsTensor.GetSubTensorDim0(blkidx);
    uint32_t *size = sizeTensor.GetSubTensorDim0(blkidx);
    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    int64_t ntotal = static_cast<int64_t>(*size);
    int64_t idx = 0;
    int64_t *id = reinterpret_cast<int64_t *>(*ids);

    if (res.active)
    {
        ComputeQueryReservoir(indists, vmdists, id, ntotal, res, cmp, isSelected);
        return;
    }

    int64_t burstSize = ntotal / burstLen_;
    for (int64_t i = 0; i < burstSize; ++i)
    {
        if (!cmp(outdists[0], vmdists[i * 2]))
        {  // vmdists[i*2] is dists, vmdists[i*2+1] is label
            // skip one burst
            idx += burstLen_;
            continue;
        }
        for (int64_t j = 0; j < burstLen_ && idx < ntotal; ++j, ++idx)
        {
            if (!cmp(outdists[0], indists[idx]))
            {
                continue;
            }
            const int64_t candId = *(id + idx);
            if (!isSelected(candId))
            {
                continue;
            }
            outdists[0] = indists[idx];
            outlabel[0] = candId;
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }
    }
    while (idx < ntotal)
    {
        if (cmp(outdists[0], indists[idx]))
        {
            const int64_t candId = *(id + idx);
            if (isSelected(candId))
            {
                outdists[0] = indists[idx];
                outlabel[0] = candId;
                UpdateHeap(outdists, outlabel, k_, 0, cmp);
            }
        }
        ++idx;
    }
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::RefillReservoir(RabitqReservoir &res, C &&cmp)
{
    if (res.count <= k_)
    {
        // Fewer than k candidates so far: everything stays and the cutoff must remain accept-all.
        // Tightening it to the worst of `count` here would be wrong -- it would start rejecting candidates
        // that still belong in a top-k that is not yet full.
        return;
    }
    RabitqCand *data = res.buf.data();
    // cmp(x, y) means "x is worse than y" in heap terms: for ascending output cmp is funcGreater, and the
    // larger distance is the worse one. Better-first ordering is therefore cmp with the arguments
    // REVERSED — cmp(a, b) would order worst-first and make nth_element select the worst k.
    auto betterFirst = [&cmp](const RabitqCand &a, const RabitqCand &b) { return cmp(b.dist, a.dist); };
    // Partition so [0, k) holds the best k. O(count) average, and the k-th best lands exactly on
    // data[k_-1], which is the new cutoff.
    std::nth_element(data, data + (k_ - 1), data + res.count, betterFirst);
    res.count = k_;
    res.thresh = data[k_ - 1].dist;
    res.everFull = true;
}

template <typename C, typename Pred>
void TopkIvfRabitqfP32CpuKernel::ComputeQueryReservoir(const float *indists, const float *vmdists, const int64_t *id,
                                                       int64_t ntotal, RabitqReservoir &res, C &&cmp, Pred &&isSelected)
{
    // The k-heap pays ~log2(k) pointer-chasing levels (~12 at k=5000) per accepted candidate, over a
    // working set too large for L1D. Here an accept is one sequential store into a flat buffer; the O(k)
    // selection cost is deferred to refill and amortised over refillChunk_ accepts.
    //
    // Consequence: the cutoff driving pruning is only refreshed at refill boundaries, not after every
    // accept. It is always *loose* relative to the true k-th best — it is the k-th best as of an earlier
    // moment, and that value only improves — so pruning stays conservative: a loose cutoff admits
    // candidates the heap would have rejected, but never rejects one the heap would have kept. The final
    // top-k is therefore identical to the heap path; only the amount of scanning differs, bounded by
    // whatever improvement lands inside one refill window.
    RabitqCand *buf = res.buf.data();
    const int64_t bufCap = k_ + refillChunk_;

    // One accepted candidate: append, and compact when the buffer is full. Shared by the burst loop and the
    // ragged tail below so the two bookkeeping steps cannot drift apart between them.
    //
    // The ID filter is applied HERE rather than in the scan loops, for the same reason the id load itself
    // lives here: both are only needed for a candidate that already beat the distance cutoff (~9% of those
    // scanned at k=5000), and hoisting either into the loops would put them on the other 91% too. Filtering
    // at accept also keeps the reservoir exact under a filter — a rejected id never enters the buffer, so it
    // can never be selected by a refill nor appear in the drained top-k. For the unfiltered mode the
    // predicate is a constant-true lambda that inlines away to nothing.
    //
    // The cast to uint32_t is the one lossy step in the reservoir path, and it is explicit so it reads as a
    // decision rather than a warning the compiler happened not to emit. It is licensed by the host-side
    // idsFitUint32 gate that must hold before the reservoir flag is ever set (IndexIVFRaBitQ.cpp), which
    // checks the largest id the index has actually stored -- not a count, and not an assumption about how
    // the caller labels its vectors. The check cannot live here: this function's `ntotal` parameter is only
    // the current block's baseSize, so the index-wide id range is not visible from the kernel.
    auto accept = [&](float d, int64_t candId)
    {
        if (!isSelected(candId))
        {
            return;
        }
        buf[res.count].dist = d;
        buf[res.count].id = static_cast<uint32_t>(candId);
        res.count++;
        if (res.count == bufCap)
        {
            // Compacts in place back down to ~k_ and refreshes thresh; the scan then carries on writing at
            // buf[res.count].
            RefillReservoir(res, cmp);
        }
    };

    int64_t idx = 0;
    const int64_t burstSize = ntotal / BURST_LEN;
    for (int64_t i = 0; i < burstSize; ++i)
    {
        // Burst-level prune. While the reservoir has never filled, thresh is the accept-all sentinel and
        // this cannot fire — mirroring the heap path, where a not-yet-full heap also cannot prune.
        if (res.everFull && !cmp(res.thresh, vmdists[i * 2]))
        {
            idx += BURST_LEN;
            continue;
        }
        // Bound is the BURST_LEN literal, and there is no `idx < ntotal` guard, because within a full burst
        // it cannot fire: burstSize is a floor division, so the last index this loop reaches is
        // burstSize * BURST_LEN - 1 <= ntotal - 1. The ragged tail of ntotal % BURST_LEN candidates is what
        // the while loop below exists for. Dropping the guard matters more than it looks -- it was the only
        // thing in the condition the compiler could not prove loop-invariant, and with a literal trip count
        // this becomes an unrollable, vectorizable loop over a contiguous float array.
        for (int64_t j = 0; j < BURST_LEN; ++j, ++idx)
        {
            // The id is only needed when the candidate is actually accepted, which at k=5000 is ~9% of
            // those scanned (21.5k accepts / 235k seen). Loading it up front streamed 8B per candidate
            // through a loop whose real working set is the 4B distance — ~1.9MB of dead traffic per query.
            if (!res.everFull || cmp(res.thresh, indists[idx]))
            {
                accept(indists[idx], *(id + idx));
            }
        }
    }
    while (idx < ntotal)
    {
        if (!res.everFull || cmp(res.thresh, indists[idx]))
        {
            accept(indists[idx], *(id + idx));
        }
        ++idx;
    }
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::DrainReservoir(int64_t qidx, KernelTensor<float> &outdistsTensor,
                                                KernelTensor<int64_t> &outlabelsTensor, RabitqReservoir &res,
                                                std::vector<uint64_t> &scratch, C &&cmp)
{
    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    // Final compaction, and the one that makes the whole loose-threshold scheme exact. The scan accepts
    // against a cutoff that is only refreshed at refill boundaries, so between refills it admits candidates
    // the true k-th best would have rejected; res.count on arrival here is therefore k_ + (however many the
    // stale cutoff let through since the last refill), in arbitrary buffer order. Taking the first k would
    // return some of those extras instead of the real top-k. RefillReservoir both selects the true best
    // k and puts them in [0, k). The monotonicity of the cutoff is what makes this sufficient: a stale
    // threshold only ever admits extras, never rejects a genuine top-k member, so nothing is missing from
    // the buffer by the time we select over it.
    RefillReservoir(res, cmp);
    RabitqCand *data = res.buf.data();
    const int64_t n = std::min(res.count, k_);

    // scratch is sized to exactly k_ when the packed sort is on, and n <= k_, so `>= n` is the right test
    // here — unlike Reorder, which always sorts the full k_ and can demand an exact match.
    if (static_cast<int64_t>(scratch.size()) >= n)
    {
        // Packed introsort, same representation as Reorder: [63:32] order key, [31:0] source slot. Sorting
        // plain uint64s needs no comparator call per comparison, and the sort key sits in the high bits so
        // the compare is a single integer compare on the whole word. RabitqCand is also 8B, so the swap
        // width is the same either way; what remains is the comparator the struct sort would call on every
        // comparison, plus the fact that the distance comes back out of the key in the scatter below, so
        // res.buf is never re-read.
        const bool descending = !cmp(1.0f, 0.0f);
        for (int64_t i = 0; i < n; ++i)
        {
            scratch[i] =
                (static_cast<uint64_t>(PackOrderKey(data[i].dist, descending)) << 32) | static_cast<uint32_t>(i);
        }
        std::sort(scratch.begin(), scratch.begin() + n);
        // Unlike Reorder, the source (res.buf) and the destination tensors are distinct buffers, so there is
        // no aliasing to work around — both fields can be written in a single pass.
        for (int64_t i = 0; i < n; ++i)
        {
            const uint64_t entry = scratch[i];
            outdists[i] = UnpackOrderKey(static_cast<uint32_t>(entry >> 32), descending);
            outlabel[i] = data[static_cast<uint32_t>(entry & 0xffffffffu)].id;
        }
    }
    else
    {
        // Scratch unavailable (packed sort disabled, or k_ outside its range). Sort the structs directly;
        // still a sequential scan, unlike the heap path's sift-down walk.
        std::sort(data, data + n, [&cmp](const RabitqCand &a, const RabitqCand &b) { return cmp(b.dist, a.dist); });
        for (int64_t i = 0; i < n; ++i)
        {
            outdists[i] = data[i].dist;
            outlabel[i] = data[i].id;
        }
    }
    // Fewer than k survivors (small index): InitTopkHeap already filled the output with the sentinel
    // distance and the 0xfff... invalid label, and the two-die merge skips label==-1, so the untouched tail
    // is correct as-is.
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::Reorder(int64_t qidx, KernelTensor<float> &outdistsTensor,
                                         KernelTensor<int64_t> &outlabelsTensor, std::vector<uint64_t> &scratch,
                                         C &&cmp)
{
    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    if (static_cast<int64_t>(scratch.size()) != k_)
    {
        // Scratch unavailable (k_ outside the packed-sort range, or allocation skipped) — drain the heap
        // in place. O(k log k) sift-downs over a k*12B working set; for the small-k case that set is
        // L1-resident so the packed sort would not pay for itself anyway.
        for (int64_t i = k_ - 1; i >= 1; --i)
        {
            std::swap(outdists[0], outdists[i]);
            std::swap(outlabel[0], outlabel[i]);
            UpdateHeap(outdists, outlabel, i, 0, cmp);
        }
        return;
    }

    // Packed introsort. Heapsort's sift-down walks a root-to-leaf path on each of the k-1 drains, so it
    // random-walks a k*12B working set (20KB dists + 40KB labels at k=5000) that overflows L1D — every
    // level is a likely miss and the hardware prefetcher cannot help a tree descent. std::sort's partition
    // scans are sequential, so the same O(k log k) comparison count runs at a far lower cost per visit.
    //
    // The two parallel arrays (SoA) cannot be fed to std::sort directly, and sorting an index array would
    // reintroduce the indirection we are trying to remove. So pack each entry into one uint64:
    //   [63:32] order key — the distance as an order-preserving uint32
    //   [31:0]  source slot, for gathering the matching label afterwards
    // Sorting plain uint64s needs no comparator and touches 8B/element contiguously.
    //
    // Ascending uint64 order must equal the heapsort output order. cmp is funcGreater for asc_!=0 (max-heap,
    // drains ascending) and funcLess otherwise (min-heap, drains descending); probing it with a known pair
    // keeps the direction tied to the comparator actually in use rather than re-deriving it from asc_.
    const bool descending = !cmp(1.0f, 0.0f);
    for (int64_t i = 0; i < k_; ++i)
    {
        scratch[i] = (static_cast<uint64_t>(PackOrderKey(outdists[i], descending)) << 32) | static_cast<uint32_t>(i);
    }

    std::sort(scratch.begin(), scratch.end());

    // Scatter back. The key is a bijection on the float's bit pattern, so each distance is reconstructed
    // from the key itself and outdists is never read here — writing outdists[i] cannot clobber a value a
    // later iteration still needs. Labels do need the original array, so stage them into scratch[i] (whose
    // packed entry is fully consumed by then) and write outlabel only in the second pass.
    for (int64_t i = 0; i < k_; ++i)
    {
        const uint64_t entry = scratch[i];
        const uint32_t slot = static_cast<uint32_t>(entry & 0xffffffffu);
        outdists[i] = UnpackOrderKey(static_cast<uint32_t>(entry >> 32), descending);
        scratch[i] = static_cast<uint64_t>(outlabel[slot]);
    }
    for (int64_t i = 0; i < k_; ++i)
    {
        outlabel[i] = static_cast<int64_t>(scratch[i]);
    }
}

REGISTER_CPU_KERNEL(TOPK_IVF_RABITQ_FP32, TopkIvfRabitqfP32CpuKernel);
}  // namespace aicpu
