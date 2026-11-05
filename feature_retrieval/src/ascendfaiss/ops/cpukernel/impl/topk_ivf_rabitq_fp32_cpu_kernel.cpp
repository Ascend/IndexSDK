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
#include <iostream>
#include <limits>
#include <string>

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
                                            KernelTensor<float> &outdists, KernelTensor<int64_t> &outlabels, C &&cmp,
                                            Pred &&isSelected)
{
    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt)
    {
        int64_t thisBlockNum = *(blocknums.GetSubTensorDim0(qidx));
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
            ComputeQuery(qidx, slot, indists, vmdists, ids, size, outdists, outlabels, cmp, isSelected);
        }
        Reorder(qidx, outdists, outlabels, cmp);
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

    const bool negate = selNegate_ != 0;
    switch (selMode_)
    {
        case RABITQ_ID_FILTER_NONE:
            RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels, cmp,
                       [](int64_t) { return true; });
            break;
        case RABITQ_ID_FILTER_RANGE:
        {
            const int64_t lo = selAux0_;
            const int64_t hi = selAux1_;
            if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [lo, hi](int64_t id) { return id < lo || id >= hi; });
            }
            else
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [lo, hi](int64_t id) { return id >= lo && id < hi; });
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
                           cmp, [negate](int64_t) { return negate; });
            }
            else if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [sorted, n](int64_t id) { return !std::binary_search(sorted, sorted + n, id); });
            }
            else
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [sorted, n](int64_t id) { return std::binary_search(sorted, sorted + n, id); });
            }
            break;
        }
        case RABITQ_ID_FILTER_BITMAP:
        {
            if (selBitmap_ == nullptr)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [negate](int64_t) { return negate; });
            }
            else
            {
                RunBitmapQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists,
                                 outlabels, cmp);
            }
            break;
        }
        case RABITQ_ID_FILTER_ROARING:
        {
            const roaring_bitmap_t *roaringView = static_cast<const roaring_bitmap_t *>(selRoaringView_);
            if (roaringView == nullptr)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp, [negate](int64_t) { return negate; });
            }
            else if (negate)
            {
                RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels,
                           cmp,
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
                           cmp,
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
            RunQueries(tcnt, tid, indists, vmdists, ids, size, blocknums, opflag, blockSlots, outdists, outlabels, cmp,
                       [](int64_t) { return false; });
            break;
    }
}

template <typename C>
void TopkIvfRabitqfP32CpuKernel::RunBitmapQueries(size_t tcnt, size_t tid, KernelTensor<float> &indists,
                                                  KernelTensor<float> &vmdists, KernelTensor<int64_t> &ids,
                                                  KernelTensor<uint32_t> &size, KernelTensor<int64_t> &blocknums,
                                                  KernelTensor<uint16_t> &opflag, KernelTensor<int64_t> &blockSlots,
                                                  KernelTensor<float> &outdists, KernelTensor<int64_t> &outlabels,
                                                  C &&cmp)
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
        Reorder(qidx, outdists, outlabels, cmp);
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
                                              KernelTensor<int64_t> &outlabelsTensor, C &&cmp, Pred &&isSelected)
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
void TopkIvfRabitqfP32CpuKernel::Reorder(int64_t qidx, KernelTensor<float> &outdistsTensor,
                                         KernelTensor<int64_t> &outlabelsTensor, C &&cmp)
{
    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    for (int64_t i = k_ - 1; i >= 1; --i)
    {
        std::swap(outdists[0], outdists[i]);
        std::swap(outlabel[0], outlabel[i]);
        UpdateHeap(outdists, outlabel, i, 0, cmp);
    }
}

REGISTER_CPU_KERNEL(TOPK_IVF_RABITQ_FP32, TopkIvfRabitqfP32CpuKernel);
}  // namespace aicpu
