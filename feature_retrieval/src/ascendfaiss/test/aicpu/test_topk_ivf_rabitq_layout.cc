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

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "cpu_nodedef_builder.h"
#undef private
#undef protected

#include "topk_ivf_rabitq_fp32_cpu_kernel.h"
#include "utils.h"

using namespace std;
using namespace aicpu;
using namespace aicpu::unittest;

#define CREATE_RABITQ_LAYOUT_NODEDEF(shapes, dataTypes, dataPtrs, idx)           \
    auto nodeDef##idx = NodeDefBuilder::CreateNodeDef();                         \
    NodeDefBuilder(nodeDef##idx.get(), "TopkIvfRabitqFp32", "TopkIvfRabitqFp32") \
        .Input({"indists", dataTypes[0], shapes[0], dataPtrs[0]})                \
        .Input({"vmdists", dataTypes[1], shapes[1], dataPtrs[1]})                \
        .Input({"ids", dataTypes[2], shapes[2], dataPtrs[2]})                    \
        .Input({"size", dataTypes[3], shapes[3], dataPtrs[3]})                   \
        .Input({"blocknums", dataTypes[4], shapes[4], dataPtrs[4]})              \
        .Input({"opflag", dataTypes[5], shapes[5], dataPtrs[5]})                 \
        .Input({"attr", dataTypes[6], shapes[6], dataPtrs[6]})                   \
        .Input({"blockSlots", dataTypes[7], shapes[7], dataPtrs[7]})             \
        .Output({"outdists", dataTypes[8], shapes[8], dataPtrs[8]})              \
        .Output({"outlabels", dataTypes[9], shapes[9], dataPtrs[9]})

namespace
{
const int64_t CORE_NUM = 8;
const int64_t BURST_LEN = 64;
const int64_t BLOCK_SIZE = 2048;
const int64_t FLAG_STRIDE = 32;

// Mirror of BuildBlockSlotPlan in IndexIVFRaBitQ.cpp. Deliberately a reimplementation rather than a call into
// the host: what the kernel contract rests on is the property (the mapping is a bijection, and the kernel reads
// back the same numbering the host wrote), not one particular host function. Sharing the implementation would
// make the GROUPED-equals-LEGACY assertion below pass even if both sides drifted together.
void BuildPlan(const vector<int64_t> &blockNumPerQ, int64_t maxBlocksPerQ, int64_t group, vector<int64_t> &slots)
{
    const int64_t batch = static_cast<int64_t>(blockNumPerQ.size());
    const int64_t window = min<int64_t>(batch, group > 0 ? group : 1);
    vector<int64_t> emitted(static_cast<size_t>(batch), 0);
    int64_t slot = 0;
    for (int64_t winStart = 0; winStart < batch; winStart += window)
    {
        const int64_t winEnd = min(batch, winStart + window);
        bool progressed = true;
        while (progressed)
        {
            progressed = false;
            for (int64_t q = winStart; q < winEnd; ++q)
            {
                const size_t qu = static_cast<size_t>(q);
                if (emitted[qu] >= blockNumPerQ[qu])
                {
                    continue;
                }
                slots[static_cast<size_t>(q * maxBlocksPerQ + emitted[qu])] = slot++;
                emitted[qu]++;
                progressed = true;
            }
        }
    }
}

// One query's logical blocks, independent of where they land in the grid. The whole point of the layout change
// is that these contents are identical under both layouts and only their grid positions move, so the test
// generates them once and stages them twice.
struct LogicalData
{
    vector<vector<float>> blocks;
    vector<vector<int64_t>> labels;
};

// The test's stand-in for fillDisOpInputDataByBlock: it is the second of the three places that have to agree on
// the slot numbering, and staging through it is what makes a kernel-side addressing bug observable at all.
void StageGrid(const vector<LogicalData> &logical, const vector<int64_t> &blockNumPerQ, int64_t maxBlocksPerQ,
               const vector<int64_t> &slots, int64_t asc, int64_t vcMaxLen, vector<float> &indists,
               vector<float> &vmdists, vector<int64_t> &ids, vector<uint32_t> &sizes, vector<uint16_t> &opflag,
               vector<vector<int64_t>> &labelStore)
{
    const int64_t nq = static_cast<int64_t>(blockNumPerQ.size());
    for (int64_t q = 0; q < nq; ++q)
    {
        for (int64_t i = 0; i < blockNumPerQ[static_cast<size_t>(q)]; ++i)
        {
            const int64_t slot = slots[static_cast<size_t>(q * maxBlocksPerQ + i)];
            const vector<float> &blk = logical[static_cast<size_t>(q)].blocks[static_cast<size_t>(i)];
            copy(blk.begin(), blk.end(), indists.begin() + static_cast<size_t>(slot * BLOCK_SIZE));

            // Burst extrema, the way distance_ivf_rabitq_l2_fp32 produces them: L2 (asc) stores the burst
            // minimum, IP (desc) the maximum. Getting this backwards would make the burst-level prune in
            // ComputeQuery skip bursts that still hold candidates, and the two layouts would then disagree for
            // a reason that has nothing to do with addressing.
            const int64_t burstNum = (BLOCK_SIZE + BURST_LEN - 1) / BURST_LEN;
            for (int64_t b = 0; b < burstNum; ++b)
            {
                const int64_t start = b * BURST_LEN;
                const int64_t end = min(start + BURST_LEN, BLOCK_SIZE);
                float m = blk[static_cast<size_t>(start)];
                for (int64_t j = start + 1; j < end; ++j)
                {
                    const float v = blk[static_cast<size_t>(j)];
                    m = (asc != 0) ? min(m, v) : max(m, v);
                }
                vmdists[static_cast<size_t>(slot * vcMaxLen + b * 2)] = m;
            }

            labelStore[static_cast<size_t>(slot)] = logical[static_cast<size_t>(q)].labels[static_cast<size_t>(i)];
            ids[static_cast<size_t>(slot)] = reinterpret_cast<int64_t>(labelStore[static_cast<size_t>(slot)].data());
            sizes[static_cast<size_t>(slot)] = static_cast<uint32_t>(BLOCK_SIZE);
            opflag[static_cast<size_t>(slot * FLAG_STRIDE)] = 1;
        }
    }
}

struct RunResult
{
    uint32_t status;
    vector<float> outdists;
    vector<int64_t> outlabels;
};

// Run the kernel once under a given layout. slotOverride, when non-empty, replaces the plan AFTER the grid has
// been staged from the correct one -- that is what turns an injected slot into a genuine host/kernel
// disagreement rather than just differently-arranged data. declaredDim1Extra of -1 declares the row width as 0,
// which is what resetL2TopkOp actually emits (the op is built per searchBatchSizes entry at init time, before
// the probe result that determines maxBlocksPerQ exists); >= 0 declares maxBlocksPerQ plus that much.
RunResult RunLayout(const vector<LogicalData> &logical, const vector<int64_t> &blockNumPerQ, int64_t k, int64_t asc,
                    int64_t layoutMode, int64_t group, const vector<int64_t> &slotOverride, int64_t declaredDim1Extra)
{
    const int64_t nq = static_cast<int64_t>(blockNumPerQ.size());
    const int64_t totalBlock = accumulate(blockNumPerQ.begin(), blockNumPerQ.end(), static_cast<int64_t>(0));
    const int64_t maxBlocksPerQ = *max_element(blockNumPerQ.begin(), blockNumPerQ.end());
    const int64_t allBlock = (totalBlock + CORE_NUM - 1) / CORE_NUM * CORE_NUM;
    const int64_t vcMaxLen = (BLOCK_SIZE + BURST_LEN - 1) / BURST_LEN * 2;
    // The stride the buffer is laid out at and the width declared on the tensor are kept separate on purpose.
    // The mismatch case declares a wider row than the kernel derives, and the allocation has to cover the
    // declared width so a failure to reject shows up as a wrong status rather than as a read past the buffer.
    const int64_t stride = maxBlocksPerQ + (declaredDim1Extra > 0 ? declaredDim1Extra : 0);
    const int64_t shapeDim1 = (declaredDim1Extra < 0) ? 0 : stride;

    vector<int64_t> slots(static_cast<size_t>(nq * stride), -1);
    // group=1 is the query-major order, i.e. exactly the prefix sum SlotOf derives for LEGACY. Staging LEGACY
    // through it keeps both layouts reading the same logical block from the same grid position.
    BuildPlan(blockNumPerQ, stride, layoutMode == RABITQ_LAYOUT_LEGACY ? 1 : group, slots);

    vector<float> indists(static_cast<size_t>(allBlock * BLOCK_SIZE), 0.0f);
    vector<float> vmdists(static_cast<size_t>(allBlock * vcMaxLen), 0.0f);
    vector<int64_t> ids(static_cast<size_t>(allBlock), 0);
    vector<uint32_t> sizes(static_cast<size_t>(allBlock), 0);
    vector<uint16_t> opflag(static_cast<size_t>(allBlock * FLAG_STRIDE), 0);
    vector<vector<int64_t>> labelStore(static_cast<size_t>(allBlock));

    StageGrid(logical, blockNumPerQ, stride, slots, asc, vcMaxLen, indists, vmdists, ids, sizes, opflag, labelStore);

    if (!slotOverride.empty())
    {
        slots = slotOverride;
    }

    vector<int64_t> blocknums = blockNumPerQ;
    vector<int64_t> attrs(static_cast<size_t>(TOPK_IVF_RABITQ_ATTR_IDX_COUNT), 0);
    attrs[TOPK_IVF_RABITQ_ATTR_ASC_IDX] = asc;
    attrs[TOPK_IVF_RABITQ_ATTR_K_IDX] = k;
    attrs[TOPK_IVF_RABITQ_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[TOPK_IVF_RABITQ_ATTR_QUERY_NUM_IDX] = nq;
    attrs[TOPK_IVF_RABITQ_ATTR_CORE_NUM_IDX] = CORE_NUM;
    attrs[TOPK_IVF_RABITQ_ATTR_SEL_MODE_IDX] = static_cast<int64_t>(RABITQ_ID_FILTER_NONE);
    attrs[TOPK_IVF_RABITQ_ATTR_LAYOUT_MODE_IDX] = layoutMode;

    RunResult res;
    res.status = KERNEL_STATUS_PARAM_INVALID;
    res.outdists.assign(static_cast<size_t>(nq * k), 0.0f);
    res.outlabels.assign(static_cast<size_t>(nq * k), 0);

    vector<DataType> dataTypes = {DT_FLOAT,  DT_FLOAT, DT_INT64, DT_UINT32, DT_INT64,
                                  DT_UINT16, DT_INT64, DT_INT64, DT_FLOAT,  DT_INT64};
    vector<vector<int64_t>> shapes = {{allBlock, BLOCK_SIZE},
                                      {allBlock, vcMaxLen},
                                      {allBlock, 1},
                                      {allBlock, 1},
                                      {nq},
                                      {allBlock, FLAG_STRIDE},
                                      {static_cast<int64_t>(TOPK_IVF_RABITQ_ATTR_IDX_COUNT)},
                                      {nq, shapeDim1},
                                      {nq, k},
                                      {nq, k}};
    vector<void *> dataPtrs = {(void *)indists.data(),      (void *)vmdists.data(),   (void *)ids.data(),
                               (void *)sizes.data(),        (void *)blocknums.data(), (void *)opflag.data(),
                               (void *)attrs.data(),        (void *)slots.data(),     (void *)res.outdists.data(),
                               (void *)res.outlabels.data()};

    CREATE_RABITQ_LAYOUT_NODEDEF(shapes, dataTypes, dataPtrs, 0);
    CpuKernelContext ctx(DEVICE);
    if (ctx.Init(nodeDef0.get()) != 0)
    {
        return res;
    }
    TopkIvfRabitqfP32CpuKernel kernel;
    res.status = kernel.Compute(ctx);
    return res;
}

// Deterministic, self-contained generator. Avoids <random>, whose distribution output is not specified to be
// identical across standard libraries -- the equivalence assertions need both runs to see the same bytes, and
// pinning the generator here is cheaper than reasoning about that.
float NextValue(uint32_t &state)
{
    state = state * 1103515245u + 12345u;
    return static_cast<float>((state >> 8) & 0xffffu) / 65536.0f;
}

vector<LogicalData> MakeBatch(const vector<int64_t> &blockNumPerQ, uint32_t seed)
{
    vector<LogicalData> out;
    for (size_t q = 0; q < blockNumPerQ.size(); ++q)
    {
        uint32_t state = seed + static_cast<uint32_t>(q) * 7919u;
        const int64_t labelBase = static_cast<int64_t>(q) * 1000000;
        LogicalData d;
        for (int64_t i = 0; i < blockNumPerQ[q]; ++i)
        {
            vector<float> blk(static_cast<size_t>(BLOCK_SIZE));
            vector<int64_t> lab(static_cast<size_t>(BLOCK_SIZE));
            for (int64_t j = 0; j < BLOCK_SIZE; ++j)
            {
                blk[static_cast<size_t>(j)] = NextValue(state);
                lab[static_cast<size_t>(j)] = labelBase + i * BLOCK_SIZE + j;
            }
            d.blocks.push_back(blk);
            d.labels.push_back(lab);
        }
        out.push_back(d);
    }
    return out;
}

// Ragged on purpose. Equal blockNum per query is the one case where a row-stride or emit-order off-by-one can
// still come out looking right.
vector<int64_t> RaggedBatch()
{
    const int64_t raw[] = {3, 1, 7, 4, 4, 2, 9, 1};
    return vector<int64_t>(raw, raw + sizeof(raw) / sizeof(raw[0]));
}
}  // namespace

class TopkIvfRabitqLayoutTest : public testing::Test
{
};

// Invariant segment (1): the plan is a bijection from (query, i-th block) onto [0, totalBlock).
TEST_F(TopkIvfRabitqLayoutTest, SlotPlanIsBijection)
{
    const vector<int64_t> blockNumPerQ = RaggedBatch();
    const int64_t totalBlock = accumulate(blockNumPerQ.begin(), blockNumPerQ.end(), static_cast<int64_t>(0));
    const int64_t maxBlocksPerQ = *max_element(blockNumPerQ.begin(), blockNumPerQ.end());
    const int64_t groups[] = {1, 2, 3, 6, 8, 16};

    for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); ++g)
    {
        const int64_t group = groups[g];
        vector<int64_t> slots(blockNumPerQ.size() * static_cast<size_t>(maxBlocksPerQ), -1);
        BuildPlan(blockNumPerQ, maxBlocksPerQ, group, slots);

        set<int64_t> used;
        for (size_t q = 0; q < blockNumPerQ.size(); ++q)
        {
            for (int64_t i = 0; i < blockNumPerQ[q]; ++i)
            {
                const int64_t slot = slots[static_cast<size_t>(static_cast<int64_t>(q) * maxBlocksPerQ + i)];
                EXPECT_GE(slot, 0) << "group=" << group << " q=" << q << " i=" << i;
                EXPECT_LT(slot, totalBlock) << "group=" << group << " q=" << q << " i=" << i;
                EXPECT_TRUE(used.insert(slot).second) << "duplicate slot " << slot << " group=" << group;
            }
            // Columns past this query's blockNum must stay untouched. SlotOf is reached for blkidx < blocknums
            // only, so a stale value here would mean a query that shrank between requests still reaches a slot
            // belonging to the previous one.
            for (int64_t i = blockNumPerQ[q]; i < maxBlocksPerQ; ++i)
            {
                EXPECT_EQ(slots[static_cast<size_t>(static_cast<int64_t>(q) * maxBlocksPerQ + i)],
                          static_cast<int64_t>(-1));
            }
        }
        EXPECT_EQ(static_cast<int64_t>(used.size()), totalBlock) << "group=" << group << " not onto";
    }
}

// Invariant segments (2) and (3) together: GROUPED and LEGACY must produce identical TopK. The two runs share
// the logical data and differ only in where it sits in the grid, so any disagreement between how the plan
// numbers slots, how the grid is staged, and how the kernel reads it back lands here.
TEST_F(TopkIvfRabitqLayoutTest, GroupedMatchesLegacy)
{
    const vector<int64_t> blockNumPerQ = RaggedBatch();
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 1234u);
    const int64_t k = 64;
    const int64_t ascModes[] = {1, 0};
    const int64_t groups[] = {2, 3, 6};

    for (size_t a = 0; a < sizeof(ascModes) / sizeof(ascModes[0]); ++a)
    {
        const int64_t asc = ascModes[a];
        RunResult legacy = RunLayout(logical, blockNumPerQ, k, asc, RABITQ_LAYOUT_LEGACY, 1, vector<int64_t>(), 0);
        ASSERT_EQ(legacy.status, static_cast<uint32_t>(KERNEL_STATUS_OK)) << "asc=" << asc;

        for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); ++g)
        {
            const int64_t group = groups[g];
            RunResult grouped =
                RunLayout(logical, blockNumPerQ, k, asc, RABITQ_LAYOUT_GROUPED, group, vector<int64_t>(), 0);
            ASSERT_EQ(grouped.status, static_cast<uint32_t>(KERNEL_STATUS_OK)) << "asc=" << asc << " group=" << group;
            EXPECT_EQ(grouped.outdists, legacy.outdists) << "asc=" << asc << " group=" << group;
            EXPECT_EQ(grouped.outlabels, legacy.outlabels) << "asc=" << asc << " group=" << group;
        }
    }
}

// A positive but out-of-range slot must be rejected. Without ValidateBlockSlots this returns OK and the output
// quietly contains candidates read from past the end of the staged grid.
TEST_F(TopkIvfRabitqLayoutTest, RejectsOutOfRangeSlot)
{
    const int64_t raw[] = {3, 1, 7, 4};
    const vector<int64_t> blockNumPerQ(raw, raw + 4);
    const int64_t totalBlock = accumulate(blockNumPerQ.begin(), blockNumPerQ.end(), static_cast<int64_t>(0));
    const int64_t maxBlocksPerQ = *max_element(blockNumPerQ.begin(), blockNumPerQ.end());
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 99u);

    vector<int64_t> bad(blockNumPerQ.size() * static_cast<size_t>(maxBlocksPerQ), -1);
    BuildPlan(blockNumPerQ, maxBlocksPerQ, 3, bad);
    bad[0] = totalBlock;  // one past the last staged block

    RunResult r = RunLayout(logical, blockNumPerQ, 16, 1, RABITQ_LAYOUT_GROUPED, 3, bad, 0);
    EXPECT_EQ(r.status, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID));
}

// A duplicate slot must be rejected. This is the case a pure range check cannot catch: both values are legal
// indices, but one block gets scanned twice and another never at all, and the only symptom is recall.
TEST_F(TopkIvfRabitqLayoutTest, RejectsDuplicateSlot)
{
    const int64_t raw[] = {3, 1, 7, 4};
    const vector<int64_t> blockNumPerQ(raw, raw + 4);
    const int64_t maxBlocksPerQ = *max_element(blockNumPerQ.begin(), blockNumPerQ.end());
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 98u);

    vector<int64_t> bad(blockNumPerQ.size() * static_cast<size_t>(maxBlocksPerQ), -1);
    BuildPlan(blockNumPerQ, maxBlocksPerQ, 3, bad);
    bad[1] = bad[0];  // query 0's second block aliases its first

    RunResult r = RunLayout(logical, blockNumPerQ, 16, 1, RABITQ_LAYOUT_GROUPED, 3, bad, 0);
    EXPECT_EQ(r.status, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID));
}

// A declared row width that disagrees with what blocknums implies must be rejected rather than silently
// overwritten: it means the host sized the device buffer to one stride while the kernel is about to read it
// with another, so every row past the first lands at the wrong offset.
TEST_F(TopkIvfRabitqLayoutTest, RejectsMismatchedDeclaredRowWidth)
{
    const int64_t raw[] = {3, 1, 7, 4};
    const vector<int64_t> blockNumPerQ(raw, raw + 4);
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 97u);

    RunResult r = RunLayout(logical, blockNumPerQ, 16, 1, RABITQ_LAYOUT_GROUPED, 3, vector<int64_t>(), 1);
    EXPECT_EQ(r.status, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID));
}

// The complement of the case above, and the one the production host actually hits: a row width declared 0 is
// the legal "framework left it dynamic" value and must be accepted, with the kernel resolving the stride from
// blocknums. Worth pinning separately, because tightening the mismatch check into an unconditional equality
// would reject every real request while still passing every other test here.
TEST_F(TopkIvfRabitqLayoutTest, AcceptsDynamicDeclaredRowWidth)
{
    const vector<int64_t> blockNumPerQ = RaggedBatch();
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 95u);
    const int64_t k = 64;

    RunResult dynamicWidth = RunLayout(logical, blockNumPerQ, k, 1, RABITQ_LAYOUT_GROUPED, 3, vector<int64_t>(), -1);
    ASSERT_EQ(dynamicWidth.status, static_cast<uint32_t>(KERNEL_STATUS_OK));

    RunResult legacy = RunLayout(logical, blockNumPerQ, k, 1, RABITQ_LAYOUT_LEGACY, 1, vector<int64_t>(), 0);
    ASSERT_EQ(legacy.status, static_cast<uint32_t>(KERNEL_STATUS_OK));
    EXPECT_EQ(dynamicWidth.outdists, legacy.outdists);
    EXPECT_EQ(dynamicWidth.outlabels, legacy.outlabels);
}

// LEGACY derives slots from the prefix sum and never consults the table, so a corrupted one must not change
// its result. Pins the claim that both the indirection and its validation are inert on the historical path.
TEST_F(TopkIvfRabitqLayoutTest, LegacyIgnoresSlotTable)
{
    const int64_t raw[] = {3, 1, 7, 4};
    const vector<int64_t> blockNumPerQ(raw, raw + 4);
    const int64_t maxBlocksPerQ = *max_element(blockNumPerQ.begin(), blockNumPerQ.end());
    const vector<LogicalData> logical = MakeBatch(blockNumPerQ, 96u);

    RunResult clean = RunLayout(logical, blockNumPerQ, 16, 1, RABITQ_LAYOUT_LEGACY, 1, vector<int64_t>(), 0);
    ASSERT_EQ(clean.status, static_cast<uint32_t>(KERNEL_STATUS_OK));

    vector<int64_t> garbage(blockNumPerQ.size() * static_cast<size_t>(maxBlocksPerQ), 123456);
    RunResult corrupted = RunLayout(logical, blockNumPerQ, 16, 1, RABITQ_LAYOUT_LEGACY, 1, garbage, 0);
    ASSERT_EQ(corrupted.status, static_cast<uint32_t>(KERNEL_STATUS_OK));
    EXPECT_EQ(corrupted.outdists, clean.outdists);
    EXPECT_EQ(corrupted.outlabels, clean.outlabels);
}
