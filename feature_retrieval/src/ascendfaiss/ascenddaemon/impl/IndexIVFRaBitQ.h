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

#ifndef ASCEND_INDEXIVFRABITQ_INCLUDED
#define ASCEND_INDEXIVFRABITQ_INCLUDED
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>

#include "ascenddaemon/impl/IndexIVF.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendStackMemory.h"
#include "ascenddaemon/utils/DeviceMemArena.h"
#include "ascenddaemon/utils/SegmentedDeviceVector.h"
#include "common/RabitqIdFilter.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend
{
namespace
{
const int IVF_RABITQ_BURST_LEN = 64;
constexpr uint8_t IVF_RABITQ_BURST_LEN_LOW = 32;
constexpr uint8_t IVF_RABITQ_BURST_LEN_HIGH = 64;
constexpr uint8_t IVF_RABITQ_BURST_BLOCK_RATIO = 2;
constexpr uint8_t IVF_RABITQ_OPTIMIZE_BATCH_THRES = 48;
}  // namespace

class IndexIVFRaBitQ : public IndexIVF
{
   public:
    template <typename T>
    using SegmentedVector = SegmentedDeviceVector<T>;

    // One inverted list's device storage. The four vectors share the same slot numbering, so slot s of
    // codes/indexL1/indexL2/ids all describe the same vector; listLen is the number of slots in use.
    struct ListStorage
    {
        // blockSize is the per-segment logical slot count; growthStep is the tail allocation granularity.
        // Geometry is fixed here at construction (see SegmentedDeviceVector).
        ListStorage(size_t blockSize, size_t growthStep, size_t codeElementsPerSlot,
                    std::shared_ptr<DeviceMemArena> arena)
            : codes(blockSize, growthStep, codeElementsPerSlot, arena),
              indexL1(blockSize, growthStep, 1, arena),
              indexL2(blockSize, growthStep, 1, arena),
              ids(blockSize, growthStep, 1, arena)
        {
        }

        SegmentedVector<uint8_t> codes;
        SegmentedVector<float> indexL1;
        SegmentedVector<float> indexL2;
        SegmentedVector<idx_t> ids;
        size_t listLen{0};
    };

    // An immutable view of every inverted list. Readers hold one of these for the whole duration of a
    // search; the writer never mutates a published snapshot, it clones the one list it touches and
    // publishes a new snapshot. Lists are held by shared_ptr so an unmodified list is shared between
    // snapshots rather than copied, and ListStorage's defaulted copy shares the segments themselves,
    // so cloning a list copies four small segment-pointer vectors, not device memory.
    struct GlobalSnapshot
    {
        std::vector<std::shared_ptr<const ListStorage>> lists;
        uint64_t publishVer{0};
        size_t total{0};
        // Label ceiling belongs to this immutable version, just like total.
        idx_t maxStoredId{0};
    };

    // Keeps the snapshot it acquires alive for its lifetime by publishing it to the single reader
    // hazard slot, which the writer consults before freeing a retired snapshot. There is one hazard
    // slot and one readerContext, so the guard also serialises readers against each other: a second
    // reader would otherwise overwrite the first one's hazard and let the writer free segments the
    // first is still scanning, besides sharing the not-MT-safe AscendStackMemory.
    class SnapshotGuard
    {
       public:
        explicit SnapshotGuard(const IndexIVFRaBitQ &index);
        ~SnapshotGuard();
        SnapshotGuard(const SnapshotGuard &) = delete;
        SnapshotGuard &operator=(const SnapshotGuard &) = delete;
        const GlobalSnapshot &get() const;
        // Finish before returning a status to the caller; the destructor is only an exception-path fallback.
        aclError finish();

       private:
        const IndexIVFRaBitQ &index_;
        std::lock_guard<std::mutex> readerLock_;
        const GlobalSnapshot *snapshot_{nullptr};
        bool finished_{false};
    };

    // Per-role streams and scratch memory. The resources held by Index are a process-wide singleton
    // (see AscendResourcesProxy) and AscendStackMemory is documented as not MT safe, so the reader and
    // the writer each need their own rather than sharing the default stream and memory manager.
    struct ExecutionContext
    {
        explicit ExecutionContext(size_t memorySize);
        ~ExecutionContext();
        ExecutionContext(const ExecutionContext &) = delete;
        ExecutionContext &operator=(const ExecutionContext &) = delete;

        aclrtStream computeStream{nullptr};
        aclrtStream aicpuStream{nullptr};
        AscendStackMemory memory;
    };

   public:
    IndexIVFRaBitQ(int numList, int dim, int nprobes, const std::string &metricType, int64_t resourceSize = -1,
                   int quantBits = 1, size_t retiredMemoryHighWatermark = 0);

    ~IndexIVFRaBitQ() override;

    IndexIVFRaBitQ(const IndexIVFRaBitQ &) = delete;
    IndexIVFRaBitQ &operator=(const IndexIVFRaBitQ &) = delete;
    IndexIVFRaBitQ(IndexIVFRaBitQ &&) = delete;
    IndexIVFRaBitQ &operator=(IndexIVFRaBitQ &&) = delete;

    APP_ERROR reset() override;

    APP_ERROR updateCoarseCenterImpl(std::vector<float> &centerData);

    APP_ERROR addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices);

    APP_ERROR removeIds(size_t numVecs, const idx_t *indices);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k, AscendTensor<float16_t, DIMS_2> &outDistance,
                         AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
                          AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);

    size_t getListLength(int listId) const;
    std::unique_ptr<DeviceVector<float>> centroidsOnDevice;
    std::unique_ptr<DeviceVector<float>> OrthogonalMatrixOnDevice;  // 旋转矩阵
    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        if (faiss::ascend::SocUtils::GetInstance().IsAscend910B())
        {
            burstLen = IVF_RABITQ_BURST_LEN_HIGH;
        }
        else
        {
            burstLen = (nq > IVF_RABITQ_OPTIMIZE_BATCH_THRES) ? IVF_RABITQ_BURST_LEN_LOW : IVF_RABITQ_BURST_LEN_HIGH;
        }
        return utils::divUp(blockSize, burstLen) * IVF_RABITQ_BURST_BLOCK_RATIO;
    }
    APP_ERROR searchImpl(int n, const float *x, int k, float *distances, idx_t *labels, const float *srcIndexes,
                         const RabitqIdFilterHost *idFilter = nullptr, int searchNprobe = 0);

    size_t getCodeSize() const;

    size_t getCodePartBytes() const { return static_cast<size_t>(dims) * static_cast<size_t>(quantBits_) / 8; }

    APP_ERROR getListIds(int listId, std::vector<int64_t> &ids) const;

    APP_ERROR getListRawCodes(int listId, std::vector<uint8_t> &rawCodes) const;

    APP_ERROR getCentroids(std::vector<float> &centroids) const;

    APP_ERROR addEncodedVectors(int listId, size_t numVecs, const uint8_t *encodedCodes, const idx_t *indices);

   protected:
    void uploadLUTMatrix();
    APP_ERROR addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);
    APP_ERROR resetCenterRotateL2Op();
    void runCenterRotateL2Op(AscendTensor<float, DIMS_2> &centroid, AscendTensor<int32_t, DIMS_1> &vectorSize,
                             AscendTensor<float, DIMS_2> &matrix, AscendTensor<float, DIMS_2> &rotateCentroid,
                             AscendTensor<float, DIMS_1> &centroidl2, aclrtStream stream);
    APP_ERROR resetCenterLUTOp();
    void runCenterLUTOp(AscendTensor<float, DIMS_2> &centroid, AscendTensor<float, DIMS_2> &matrix,
                        AscendTensor<float, DIMS_2> &centroidlut, aclrtStream stream);
    APP_ERROR resetIndexRotateL2Op();
    void runIndexRotateL2Op(AscendTensor<float, DIMS_2> &index, AscendTensor<int32_t, DIMS_1> &vectorSize,
                            AscendTensor<float, DIMS_2> &matrix, AscendTensor<float, DIMS_2> &rotateIndex,
                            AscendTensor<float, DIMS_1> &indexl2, aclrtStream stream);
    APP_ERROR resetIndexCodeAndPreComputeOp();
    void runIndexCodeAndPreComputeOp(AscendTensor<int32_t, DIMS_1> &vectorSize, AscendTensor<float, DIMS_2> &index,
                                     AscendTensor<float, DIMS_1> &indexl2, AscendTensor<float, DIMS_2> &centroid,
                                     AscendTensor<float, DIMS_1> &centroidl2, AscendTensor<uint8_t, DIMS_2> &indexCodes,
                                     AscendTensor<float, DIMS_1> &l2Result, AscendTensor<float, DIMS_1> &l1Result,
                                     aclrtStream stream);
    APP_ERROR resetQueryRotateL2Op();
    void runQueryRotateL2Op(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<int32_t, DIMS_1> &vectorSize,
                            AscendTensor<float, DIMS_2> &matrix, AscendTensor<float, DIMS_2> &rotateQueries,
                            AscendTensor<float, DIMS_1> &queryl2, aclrtStream stream);
    APP_ERROR resetQueryLUTOp();
    void runQueryLUTOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &matrix,
                       AscendTensor<float, DIMS_2> &querieslut, aclrtStream stream);
    APP_ERROR resetL1TopkOp();
    void runL1TopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                     AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream);
    APP_ERROR resetL1DistOp();
    void runL1DistOp();
    APP_ERROR searchImplL1(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &rotateQueries,
                           AscendTensor<float, DIMS_1> &queryL2, AscendTensor<float, DIMS_2> &queriesLut,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, int probeK);
    void runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroidsDev,
                     AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                     AscendTensor<uint16_t, DIMS_2> &opFlag, aclrtStream stream);
    bool l2DisOpReset(std::unique_ptr<AscendOperator> &op, int64_t batch);
    APP_ERROR resetL2DistOp();
    void resizeDistResult(size_t iterNum, size_t coreNum, size_t ivfRabitqBlockSize);
    APP_ERROR searchImplL2(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_1> &queryL2,
                           AscendTensor<float, DIMS_2> &queriesLut,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, int k, float *distances, idx_t *labels,
                           const RabitqIdFilterHost *idFilter = nullptr);
    APP_ERROR searchWithBatch(int n, const float *x, int k, float *distances, idx_t *labels, const float *srcIndexes,
                              const RabitqIdFilterHost *idFilter = nullptr, int searchNprobe = 0);
    void refine(int n, const float *x, int k, float *distances, idx_t *labels, float *topkdist, idx_t *topklabel,
                const float *srcIndexes);
    void runL2DistOp(AscendTensor<float, DIMS_1> &queryL2Vec, AscendTensor<float, DIMS_2> &subQuerylut,
                     AscendTensor<float, DIMS_2, size_t> &centroidslut,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subQueryid,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subCentroidsid,
                     AscendTensor<float, DIMS_1, size_t> &subCentroidsl2,
                     AscendTensor<uint8_t, DIMS_2, size_t> &codeVec, AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize,
                     AscendTensor<float, DIMS_1, size_t> &subIndexl2, AscendTensor<float, DIMS_1, size_t> &subIndexl1,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl2Offset,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl1Offset,
                     AscendTensor<int64_t, DIMS_1, size_t> &subIds, AscendTensor<int64_t, DIMS_1> &distFilterAttrs,
                     AscendTensor<float, DIMS_2, size_t> &subDis, AscendTensor<float, DIMS_2, size_t> &subVcMaxDis,
                     AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag, aclrtStream stream);
    APP_ERROR resetL2TopkOp();
    void runL2TopkOp(int batch, AscendTensor<float, DIMS_2, size_t> &distResult,
                     AscendTensor<float, DIMS_2, size_t> &vmdistResult, AscendTensor<int64_t, DIMS_2, size_t> &ids,
                     AscendTensor<uint32_t, DIMS_2, size_t> &sizes, AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQ,
                     AscendTensor<uint16_t, DIMS_2, size_t> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<int64_t, DIMS_2, size_t> &blockSlots, AscendTensor<float, DIMS_2, size_t> &outdists,
                     AscendTensor<uint64_t, DIMS_2, size_t> &outlabel, aclrtStream stream);
    APP_ERROR fillDisOpInputData(
        int k, size_t batch, size_t totalBlockNum, size_t coreNum, AscendTensor<uint64_t, DIMS_2, size_t> &offset,
        AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset, AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
        AscendTensor<uint32_t, DIMS_2, size_t> &queryid, AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
        AscendTensor<float, DIMS_2, size_t> &centroidsl2, AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
        AscendTensor<int64_t, DIMS_2, size_t> &ids, AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQ,
        AscendTensor<int64_t, DIMS_2, size_t> &blockSlots, AscendTensor<int64_t, DIMS_1> &attrs,
        AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost,
        const std::vector<int64_t> &blockNumPlan, size_t maxBlocksPerQ,
        aicpu::RabitqIdFilterMode selMode = aicpu::RABITQ_ID_FILTER_NONE, int64_t selPtr = 0, int64_t selAux0 = 0,
        int64_t selAux1 = 0, int64_t selNegate = 0);
    APP_ERROR fillL2TopkOpInputData(int k, size_t batch, size_t coreNum, AscendTensor<int64_t, DIMS_1> &attrs,
                                    aicpu::RabitqIdFilterMode selMode = aicpu::RABITQ_ID_FILTER_NONE,
                                    int64_t selPtr = 0, int64_t selAux0 = 0, int64_t selAux1 = 0,
                                    int64_t selNegate = 0);
    APP_ERROR fillL1TopkOpInputData(AscendTensor<int64_t, DIMS_1> &attrsInput, int probeK);
    // Retries the AI CPU count query if the constructor ran before a device was set on this thread.
    void EnsureTopkGroupSizeResolved();
    void fillDisOpInputDataByBlock(size_t batch, size_t coreNum, AscendTensor<uint32_t, DIMS_2, size_t> &queryidHostVec,
                                   AscendTensor<uint32_t, DIMS_2, size_t> &centroidsidHostVec,
                                   AscendTensor<float, DIMS_2, size_t> &centroidsl2HostVec,
                                   AscendTensor<uint32_t, DIMS_2, size_t> &baseSizeHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &offsetHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &indexl2OffsetHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &indexl1OffsetHostVec,
                                   AscendTensor<int64_t, DIMS_2, size_t> &idsHostVec,
                                   AscendTensor<int64_t, DIMS_1, size_t> &blockNumPerQHostVec,
                                   AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                   AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost, const int64_t *slotPlan,
                                   size_t maxBlocksPerQ);
    void callL2DistanceOp(
        size_t batch, size_t totalBlockNum, size_t coreNum, size_t vcMaxLen, AscendTensor<float, DIMS_1> &queryL2Vec,
        AscendTensor<float, DIMS_2> &queryLutVec, AscendTensor<float, DIMS_2, size_t> &centroidsLutVec,
        AscendTensor<uint32_t, DIMS_2, size_t> &queryid, AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
        AscendTensor<float, DIMS_2, size_t> &centroidsl2, AscendTensor<uint64_t, DIMS_2, size_t> &offset,
        AscendTensor<uint32_t, DIMS_2, size_t> &baseSize, AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
        AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset, AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
        AscendTensor<float, DIMS_2, size_t> &disVec, AscendTensor<float, DIMS_2, size_t> &vcMaxDisVec,
        AscendTensor<uint8_t, DIMS_2, size_t> &codeVec, AscendTensor<float, DIMS_1, size_t> &Indexl2,
        AscendTensor<float, DIMS_1, size_t> &Indexl1, AscendTensor<int64_t, DIMS_2, size_t> &ids,
        AscendTensor<int64_t, DIMS_1> &distFilterAttrs, aclrtStream &stream);
    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost) const;
    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k, float *distances,
                         idx_t *labels) const;
    size_t getTotalBlockNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                            float *distances, idx_t *labels) const;
    size_t getBlockNumPerQ(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           std::vector<int64_t> &blockNumPerQ) const;

    APP_ERROR EnsureFilterPayloadOnDevice(const RabitqIdFilterHost *idFilter, int64_t &selPtr);
    APP_ERROR EnsurePrefixFilterPayloadOnDevice(const RabitqIdFilterHost *idFilter, int64_t &selPtr);

    // Mark the cached filter payload stale without freeing it. Safe to call from a writer.
    void InvalidateCachedFilterPayload();

    // Free the cached payload outright. Only safe where no reader can be in flight, i.e. under
    // exclusive indexLock.
    void ClearCachedFilterPayload();

    size_t removeIds(const ascend::IDSelector &sel) override;
    void initListStorage();

    // Build the snapshot every list starts out empty in. Used by the constructor and by reset().
    std::unique_ptr<GlobalSnapshot> makeEmptySnapshot() const;

    // Structural copy: the new snapshot shares every list with `base` until a list is replaced.
    std::unique_ptr<GlobalSnapshot> cloneSnapshot(const GlobalSnapshot &base) const;

    // Install `snapshot` as the one readers will pick up from now on and retire the previous one.
    // Must be called with writerLock held.
    void publishSnapshot(std::unique_ptr<GlobalSnapshot> snapshot);

    // Free every retired snapshot no reader is parked on. Must be called with writerLock held.
    void reclaimRetiredSnapshots();

    // Device bytes that would be freed by dropping `snapshot`, i.e. only the segments it is the last
    // owner of.
    size_t retainedBytes(const GlobalSnapshot &snapshot) const;

    // Drain both reader streams with one retry. Called under readerLock, or exclusive indexLock in reset.
    aclError drainReaderStreams() const;

    // Refuse a write when readers have parked long enough on old snapshots that the segments they pin
    // exceed the watermark. Must be called with writerLock held.
    APP_ERROR checkRetiredMemory();

    // Grow all four vectors of `list` so slots [0, newLen) are addressable.
    void ensureListCapacity(ListStorage &list, size_t newLen, aclrtStream stream);

    // Rebuild `newList` (a fresh copy of `oldList`) with `deletedSlots` removed, by moving the
    // surviving tail slots into the holes they leave behind and shrinking to the new length.
    // `deletedSlots` need not be sorted. Every segment written into is copy-on-write cloned first, so
    // `oldList` and the snapshot holding it are left untouched.
    APP_ERROR compactList(const ListStorage &oldList, ListStorage &newList, std::vector<size_t> &deletedSlots,
                          aclrtStream stream);

    // Remove every id in `ids`, rebuilding each affected list in a fresh snapshot.
    APP_ERROR removeIdsInternal(const std::unordered_set<idx_t> &ids, size_t *removed);

   protected:
    // Declared first so it is destroyed after the list storage that carves segments out of it.
    std::shared_ptr<DeviceMemArena> listArena_;
    MetricType metric;                                             // metric type
    std::unique_ptr<DeviceVector<float>> originCentroidsOnDevice;  // 原始聚类中心
    std::unique_ptr<DeviceVector<float>> LUTMatrixOnDevice;        // LUT 计算常值矩阵
    std::unique_ptr<DeviceVector<float>> CentroidLUTOnDevice;      // 聚类中心 LUT
    std::unique_ptr<DeviceVector<float>> CentroidL2OnDevice;       // 聚类中心 L2
    std::unique_ptr<AscendOperator> ivfRaBitQRotateL2Ops;
    std::unique_ptr<AscendOperator> ivfCenterLUTOps;
    std::unique_ptr<AscendOperator> ivfRaBitQIndexRotateL2Ops;
    std::unique_ptr<AscendOperator> ivfRaBitQIndexCodeAndPreComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> ivfRaBitQL2DistOps;

    // The snapshot readers pick up. Owned by activeSnapshot; raw so readers can load it atomically.
    std::atomic<const GlobalSnapshot *> currentSnapshot{nullptr};
    // Single-reader hazard slot: the snapshot the in-flight search is using, or nullptr. Only the one
    // reader thread stores here, and only the writer reads it, which is what lets reclamation be a
    // plain pointer comparison instead of a refcount.
    mutable std::atomic<const GlobalSnapshot *> readerHazard{nullptr};
    // A failed drain leaves the hazard pinned. No new guard may overwrite it until recovery succeeds.
    mutable std::atomic<aclError> readerDrainError{ACL_SUCCESS};
    // Set by searchImpl so the helpers it calls can reach the snapshot without threading it through
    // every signature.
    static thread_local const GlobalSnapshot *searchSnapshot;

    struct RetiredSnapshot
    {
        std::unique_ptr<GlobalSnapshot> snapshot;
        size_t bytes{0};
    };

    std::unique_ptr<GlobalSnapshot> activeSnapshot;
    std::vector<RetiredSnapshot> retiredSnapshots;
    size_t retiredBytes{0};
    size_t retiredMemoryHighWatermark{0};
    // Excludes writers from reset()/updateCoarseCenterImpl, which rebuild state no snapshot covers.
    mutable std::shared_mutex indexLock;
    // Serialises writers against each other: clone-mutate-publish is not atomic.
    std::mutex writerLock;
    // Serialises readers against each other: there is one hazard slot and one readerContext, so two
    // concurrent searches would clobber each other. Taken by SnapshotGuard.
    mutable std::mutex readerLock;
    std::unique_ptr<ExecutionContext> readerContext;
    std::unique_ptr<ExecutionContext> writerContext;

    std::map<int, std::unique_ptr<::ascend::AscendOperator>> queryRotateL2Ops;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> queryLUTOps;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkFp32;
    std::map<int, std::unique_ptr<AscendOperator>> topkL2Fp32;
    std::map<int, std::unique_ptr<AscendOperator>> l1DistFp32Ops;
    int blockSize;

    std::unique_ptr<DeviceVector<float>> distResultOnDevice;
    size_t distResultSpace;
    int64_t topkLayoutMode_;
    int64_t topkGroupSize_;
    // False while topkGroupSize_ still holds the compile-time fallback because the device could not be
    // queried at construction; the first search retries. See EnsureTopkGroupSizeResolved.
    bool topkGroupSizeResolved_ = false;

    // Bumped by writers to invalidate the filter payload cache. Writers must not touch the cache
    // itself: a reader may already have handed its device address to an async-dispatched operator,
    // so freeing the buffer here would be a use-after-free on the device side.
    std::atomic<uint64_t> filterCacheEpoch{0};

    // Only ever touched by readers, which readerLock serializes.
    std::unique_ptr<DeviceVector<uint8_t>> cachedFilterPayload;
    const void *cachedFilterSrc = nullptr;
    size_t cachedFilterBytes = 0;
    int64_t cachedFilterMode = 0;
    int64_t cachedFilterNegate = 0;
    int64_t cachedFilterAux0 = 0;
    int64_t cachedFilterAux1 = 0;
    uint64_t cachedFilterGeneration = 0;
    uint64_t cachedFilterEpoch = 0;
    std::unique_ptr<DeviceVector<uint8_t>> cachedPrefixFilterPayload;
    const void *cachedPrefixFilterSrc = nullptr;
    size_t cachedPrefixFilterBytes = 0;
    uint64_t cachedPrefixFilterGeneration = 0;
    uint64_t cachedPrefixFilterEpoch = 0;

    int quantBits_ = 1;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFRABITQ_INCLUDED
