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

#ifndef AICPU_TOPK_IVF_FP32_CPU_KERNEL_H
#define AICPU_TOPK_IVF_FP32_CPU_KERNEL_H

#include <cstdint>
#include <vector>

#include "cpu_kernel.h"
#include "kernel_shared_def.h"
#include "kernel_tensor.h"

namespace aicpu
{
// One accepted candidate held in the reservoir. nth_element moves whole entries with sequential copies;
// the alternative — nth_element over an index array into the two parallel output arrays — would
// reintroduce exactly the random indirection the reservoir exists to remove.
//
// id is deliberately uint32_t and not the int64_t the ids input tensor carries. Two reasons, and the
// second is the one that matters:
//   - Size. {float; int64_t} is 16B on LP64: 4B of it is pure padding forced by the int64's 8B
//     alignment. {float; uint32_t} is 8B with no padding at all, so the buffer at k=5000 goes from
//     110KB to 55KB and a sequential walk over it touches half as many cache lines.
//   - Cost shape. The saving is in BYTES MOVED PER ELEMENT, not in how many elements get moved.
//     nth_element still makes its ~2n visits; each one is just half as wide.
//
// The contract is id <= UINT32_MAX, and it is NOT expressed in the type system: the daemon's idx_t is
// uint64_t and addVectors/add_with_ids accept the full range without constraining a label by the vector
// count. Dense ordinals (vearch pushes vid = indexed_count_ + i) satisfy it with ~10x headroom at the 4e8
// scale this is deployed at, but nothing in the ingest path enforces that, so the bound is checked against
// the largest id actually stored rather than assumed from the deployment.
//
// The check is enforced ON THE HOST, where the reservoir flag is set (IndexIVFRaBitQ.cpp, search for
// idsFitUint32, which gates on maxStoredId_), and it has to be: this kernel is handed one block at a time
// and sees only that block's baseSize as `ntotal`, never the index-wide id range, so it cannot check
// itself. Past the bound the host simply does not request the reservoir and the k-heap path runs with
// full-width ids.
//
// UNSIGNED is load-bearing, not stylistic. Widening back to the int64_t outlabel is a zero-extend, so
// the worst-case id 2^32-1 stays positive and can never collide with the -1 (0xffff...ffff) invalid-label
// sentinel that InitTopkHeap writes and the two-die merge skips on. An int32_t would sign-extend
// 0xffffffff straight onto that sentinel and make a real result disappear in the merge.
struct RabitqCand
{
    float dist;
    uint32_t id;
};

// Per-query reservoir state. ComputeQuery is called once per block but the reservoir spans all blocks of a
// query, so DoCompute owns it and passes it by reference.
struct RabitqReservoir
{
    std::vector<RabitqCand> buf;  // capacity k_ + refillChunk_; entries [0, count) are live
    int64_t count = 0;
    // Coarse top-k cutoff: the k-th best distance as of the last refill. Both the burst-level and the
    // candidate-level pruning tests read this instead of a heap root. Starts at the accept-all sentinel
    // (+/-inf by sort direction) and only tightens once k candidates have been accepted.
    float thresh = 0.0f;
    bool everFull = false;  // has a refill ever produced a real (non-sentinel) threshold
    bool active = false;    // false => fall back to the in-place k-heap path
};

class TopkIvfRabitqfP32CpuKernel : public CpuKernel
{
    struct Inputs
    {
        Tensor *indists = nullptr;
        Tensor *vmdists = nullptr;
        Tensor *ids = nullptr;
        Tensor *size = nullptr;
        Tensor *blocknums = nullptr;
        Tensor *opflag = nullptr;
        Tensor *attr = nullptr;
        Tensor *blockSlots = nullptr;
    };

    struct Outputs
    {
        Tensor *outdists = nullptr;
        Tensor *outlabels = nullptr;
    };

   public:
    TopkIvfRabitqfP32CpuKernel() = default;

    ~TopkIvfRabitqfP32CpuKernel() override = default;

    TopkIvfRabitqfP32CpuKernel(const TopkIvfRabitqfP32CpuKernel &) = delete;
    TopkIvfRabitqfP32CpuKernel &operator=(const TopkIvfRabitqfP32CpuKernel &) = delete;
    TopkIvfRabitqfP32CpuKernel(TopkIvfRabitqfP32CpuKernel &&) = delete;
    TopkIvfRabitqfP32CpuKernel &operator=(TopkIvfRabitqfP32CpuKernel &&) = delete;

    uint32_t Compute(CpuKernelContext &ctx) override;

   private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);

    // Resolves the dynamic dims the op desc leaves at 0, and fails if blockSlots' declared shape disagrees
    // with what blocknums implies. Returns a status rather than void because that disagreement has to abort
    // the call: the shape it would otherwise install is the stride every later read of the table uses.
    uint32_t UpdateInOutShape(Inputs &inputs, Outputs &outputs) const;

    // Check that the staged slot table really is a bijection from (query, i-th block of that query) onto
    // [0, totalBlock). Only meaningful under a scattering layout; LEGACY derives slots from the prefix sum
    // and never reads the table, so it returns OK immediately.
    //
    // Why this is a whole-table pass before any output is written, rather than a bound test inside SlotOf:
    // the invariant being checked is not per-slot ("is this index in range") but global ("is the mapping
    // injective and total"). An in-range duplicate — two queries handed the same slot, or one query handed
    // the same slot twice — is exactly the failure that loses a block's candidates while double-counting
    // another's, and no amount of per-slot range checking can see it. Running it up front also means a bad
    // plan is rejected with PARAM_INVALID instead of being discovered partway through a heap that has
    // already been written to.
    uint32_t ValidateBlockSlots(const Inputs &inputs) const;

    void InitTopkHeap(Outputs &outputs) const;

    template <typename C>
    void DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename C, typename Pred>
    void RunQueries(size_t tcnt, size_t tid, KernelTensor<float> &indists, KernelTensor<float> &vmdists,
                    KernelTensor<int64_t> &ids, KernelTensor<uint32_t> &size, KernelTensor<int64_t> &blocknums,
                    KernelTensor<uint16_t> &opflag, KernelTensor<int64_t> &blockSlots, KernelTensor<float> &outdists,
                    KernelTensor<int64_t> &outlabels, std::vector<uint64_t> &scratch, RabitqReservoir &res, C &&cmp,
                    Pred &&isSelected);

    template <typename C>
    void RunBitmapQueries(size_t tcnt, size_t tid, KernelTensor<float> &indists, KernelTensor<float> &vmdists,
                          KernelTensor<int64_t> &ids, KernelTensor<uint32_t> &size, KernelTensor<int64_t> &blocknums,
                          KernelTensor<uint16_t> &opflag, KernelTensor<int64_t> &blockSlots,
                          KernelTensor<float> &outdists, KernelTensor<int64_t> &outlabels,
                          std::vector<uint64_t> &scratch, C &&cmp);

    template <typename C, typename Pred>
    void ComputeQuery(int64_t qidx, int64_t bidx, KernelTensor<float> &indistsTensor,
                      KernelTensor<float> &vmdistsTensor, KernelTensor<int64_t> &idsTensor,
                      KernelTensor<uint32_t> &sizeTensor, KernelTensor<float> &outdistsTensor,
                      KernelTensor<int64_t> &outlabelsTensor, RabitqReservoir &res, C &&cmp, Pred &&isSelected);

    // Reservoir path (env IVFRABITQ_TOPK_RESERVOIR): scan candidates into a flat buffer, compacting it
    // down to the best k with nth_element whenever it fills, instead of sifting a k-heap per accepted
    // candidate. See ComputeQueryReservoir for the cost argument.
    template <typename C, typename Pred>
    void ComputeQueryReservoir(const float *indists, const float *vmdists, const int64_t *id, int64_t ntotal,
                               RabitqReservoir &res, C &&cmp, Pred &&isSelected);

    // Compact the reservoir down to the best k with nth_element and refresh res.thresh to the new k-th
    // best. Called when the buffer fills, and again at drain time, where it is what makes the loose-cutoff
    // scheme exact -- see DrainReservoir. A no-op while fewer than k candidates have been seen, since the
    // cutoff must stay accept-all until the top-k is actually full.
    template <typename C>
    void RefillReservoir(RabitqReservoir &res, C &&cmp);

    // Sort the reservoir's best k and write them to the output tensors, replacing Reorder on this path.
    // Shares `scratch` (and the packed-uint64 representation) with Reorder; only one of the two runs per
    // query, so one buffer serves both.
    template <typename C>
    void DrainReservoir(int64_t qidx, KernelTensor<float> &outdistsTensor, KernelTensor<int64_t> &outlabelsTensor,
                        RabitqReservoir &res, std::vector<uint64_t> &scratch, C &&cmp);

    template <typename C>
    void Reorder(int64_t qidx, KernelTensor<float> &outdistsTensor, KernelTensor<int64_t> &outlabelsTensor,
                 std::vector<uint64_t> &scratch, C &&cmp);

    int64_t SlotOf(int64_t qidx, int64_t blkidx, KernelTensor<int64_t> &blockSlots) const;

   private:
    int64_t nq_ = 0;
    int64_t handleBatch_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t burstLen_ = 0;
    int64_t core_ = 0;
    int64_t layoutMode_ = RABITQ_LAYOUT_LEGACY;
    int64_t maxBlocksPerQ_ = 0;

    RabitqIdFilterMode selMode_ = RABITQ_ID_FILTER_NONE;
    int64_t selNegate_ = 0;
    int64_t selAux0_ = 0;
    int64_t selAux1_ = 0;
    const int64_t *selSorted_ = nullptr;
    const uint8_t *selBitmap_ = nullptr;
    const char *selRoaringBuf_ = nullptr;
    const void *selRoaringView_ = nullptr;

    // Reservoir path toggle (host env IVFRABITQ_TOPK_RESERVOIR, default ON; set to 0 to fall back to the
    // historical k-heap). Arrives in a high bit of the QUERY_NUM attr slot rather than being read from the
    // environment here: AICPU kernels execute in a separate device-side environment that does not inherit
    // the host process's environ, so std::getenv in this file would always return null. Riding a spare bit
    // of an already-unused attr slot also keeps the om signature and attr count fixed.
    //
    // The MEMBER initialiser stays false even though the env defaults on, and the asymmetry is deliberate:
    // this is what the kernel assumes when no host told it otherwise, and the only host that tells it
    // nothing is one too old to know the bit exists. Defaulting the member on would make that skew silently
    // select the new path from a host that never requested it.
    bool reservoirEnabled_ = false;

    // Reorder's packed-introsort toggle (host env IVFRABITQ_TOPK_PACKED_SORT, default ON; set to 0 to fall
    // back to the in-place heap drain). Same attr-bit transport, and the same false initialiser, as
    // reservoirEnabled_: "no host told us" must mean the historical path for every one of these toggles, and
    // that has to hold in the member initialiser itself rather than resting on CheckInputShapes always
    // getting to overwrite it. Default-ON is expressed solely by the host setting the flag bit.
    //
    // Exists so both this and the reservoir can be A/B'd inside ONE deployment: comparing across redeploys
    // has a noise floor of the same order as the effect being measured.
    bool packedSortEnabled_ = false;

    // Reservoir chunk size B: how many candidates are accepted past k before the buffer is compacted back
    // down. Host env IVFRABITQ_TOPK_REFILL_CHUNK, transported as log2(B) in QUERY_NUM bits 36-39 (see
    // kernel_shared_def.h for why a log2 in spare bits rather than a new attr slot). Initialised to the
    // compiled-in default here so that the field is sane even on the paths that never parse the attrs, and
    // re-derived in CheckInputShapes on every call -- it is NOT a latched flag, it is a value, and leaving a
    // stale one behind would silently size the buffer for a different B than the one being measured.
    //
    // Read once per DoCompute, where it sizes res.buf, and once more for bufCap. Deliberately not used
    // inside the per-candidate loop: that loop's bound is the BURST_LEN literal precisely so the compiler
    // can see a constant trip count, and a member read in there would undo it.
    int64_t refillChunk_ = TOPK_IVF_RABITQ_REFILL_DEFAULT;

    std::vector<int64_t> blockOffset_;
};
}  // namespace aicpu
#endif
