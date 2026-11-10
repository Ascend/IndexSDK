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

#ifndef ASCEND_INDEX_IVFRABITQ_IMPL_INCLUDED
#define ASCEND_INDEX_IVFRABITQ_IMPL_INCLUDED

#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ascend/AscendIndexIVFRaBitQ.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascend/impl/LocationTable.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"
#include "ascenddaemon/impl/IndexIVFRaBitQ.h"
#include "common/RabitqIdFilter.h"

namespace faiss
{
namespace ascend
{
struct RandomGenerator
{
    std::mt19937 mt;

    /// random positive integer
    int rand_int();

    /// random int64_t
    int64_t rand_int64();

    /// generate random integer between 0 and max-1
    int rand_int(int max);

    /// between 0 and 1
    float rand_float();

    double rand_double();

    explicit RandomGenerator(int64_t seed = 1234);
};

class AscendIndexIVFRaBitQImpl : public AscendIndexIVFImpl
{
   public:
    // Construct an empty index
    AscendIndexIVFRaBitQImpl(AscendIndexIVFRaBitQ *intf, int dims, int nlist,
                             faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
                             AscendIndexIVFRaBitQConfig config = AscendIndexIVFRaBitQConfig());

    ~AscendIndexIVFRaBitQImpl() override;

    void train(idx_t n, const float *x, bool clearNpuData = true);
    size_t getAddElementSize() const override;

    AscendIndexIVFRaBitQImpl(const AscendIndexIVFRaBitQImpl &) = delete;
    AscendIndexIVFRaBitQImpl &operator=(const AscendIndexIVFRaBitQImpl &) = delete;
    AscendIndexIVFRaBitQImpl(AscendIndexIVFRaBitQImpl &&) = delete;
    AscendIndexIVFRaBitQImpl &operator=(AscendIndexIVFRaBitQImpl &&) = delete;
    // Takes a read lock instead of the base class' write lock: the daemon serialises writers against
    // each other itself, so a delete no longer has to exclude in-flight searches.
    void deleteConcurrent(int n, const idx_t *ids);
    void deleteImpl(int n, const idx_t *ids);
    // Resolves the selector against the LocationTable and routes through deleteImpl. The base class
    // version hands the selector to the daemon, which would compact without reporting the slots tail
    // backfill relocated, leaving the host mapping stale.
    size_t removeImpl(const IDSelector &sel) override;
    // slots carries the LocationTable slot of each id in deleteIds, so the daemon deletes by exact
    // position instead of searching. movedOut receives the ids its tail backfill relocated.
    void deleteFromIVFRaBitQ(
        int deviceId, const std::vector<idx_t> &deleteIds, const std::vector<int> &listIds,
        const std::vector<size_t> &slots, std::vector<::ascend::IndexIVFRaBitQ::MovedSlot> &movedOut,
        const ::ascend::IndexIVFRaBitQ::MoveValidator &validateMoves = ::ascend::IndexIVFRaBitQ::MoveValidator{});
    std::vector<idx_t> update(idx_t n, const float *x, const idx_t *ids);
    // Read-locked counterpart of the base class' add_with_ids, with local duplicate-ID checks.
    void addConcurrent(int n, const float *x, const idx_t *ids);
    void addPaged(int n, const float *x, const idx_t *ids) override;
    size_t getAddPagedSize(int n) const override;
    void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const override;
    void searchWithSelector(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const IDSelector *sel,
                            int searchNprobe = 0) const;
    void invalidateFilterCache() const;

    // Copy data from a CPU IndexIVFRaBitQ
    void copyFrom(const faiss::IndexIVFRaBitQ *index);
    // Copy data to a CPU IndexIVFRaBitQ
    void copyTo(faiss::IndexIVFRaBitQ *index) const;

    void loadRefineSources(const float *x, idx_t n);

    // 零拷贝方式挂载精排原文：只保存指针，不做深拷贝。
    // 契约：IDs 完整覆盖 [0, ntotal)，原文第 ID 行对应该向量。
    // 调用方保证 x 的 n 行数据在挂载期间有效；挂载期间禁止 add/update/delete。
    // 与 loadRefineSources（深拷贝）及 add（addPaged 追加 srcIndexes）互斥使用。
    void attachRefineSources(const float *x, idx_t n);
    bool refineSourcesAttached() const { return attachedSrc_ != nullptr; }

   protected:
    void threadUnsafeReset() override;
    void checkRefineIds();
    const float *getRefineSources() const;

    void mergeSearchResultSingleQuery(idx_t qIdx, size_t devices, std::vector<std::vector<float>> &dist,
                                      std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k,
                                      size_t eachdeviceK, float *distances, idx_t *labels,
                                      std::function<bool(float, float)> &compFunc) const;
    // merge topk results from all devices used in search process
    void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
                           std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
                           idx_t *labels) const override;
    void indexSearch(IndexParam<float, float, ascend_idx_t> &param) const;
    void indexSearch(IndexParam<float, float, ascend_idx_t> &param, const ::ascend::RabitqIdFilterHost *idFilter,
                     int searchNprobe = 0) const;
    void checkParams() const;
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Called from AscendIndex for add/add_with_ids
    void addL1(int n, const float *x, std::unique_ptr<idx_t[]> &assign);
    void addImpl(int n, const float *x, const idx_t *ids) override;
    // One batch append per device instead of one per list: the daemon clones and publishes a snapshot
    // once for the whole fan-out. The caller preallocates baseSlots to reqs.size(); it receives each
    // list's append start slot.
    void indexIVFRaBitQAddBatch(int deviceId, const std::vector<::ascend::IndexIVFRaBitQ::ListAddRequest> &reqs,
                                std::vector<size_t> &baseSlots);
    void indexIVFRaBitQAddEncodedBatch(int deviceId,
                                       const std::vector<::ascend::IndexIVFRaBitQ::ListEncodedAddRequest> &reqs,
                                       std::vector<size_t> &baseSlots);
    inline ::ascend::IndexIVFRaBitQ *getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(), "deviceId is out of range, deviceId=%d.",
                               deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFRaBitQ *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }
    void updateCoarseCenter(std::vector<float> &centerData);
    struct StagedDeviceAdd
    {
        std::vector<float> codes;
        std::vector<ascend_idx_t> ids;
        bool published = false;
    };
    struct StagedListAdd
    {
        explicit StagedListAdd(const std::vector<int> &counts) : devices(counts.size()), occupancy(counts) {}
        std::vector<StagedDeviceAdd> devices;
        std::vector<int> occupancy;
    };
    using AddStaging = std::unordered_map<int, StagedListAdd>;
    void stageAdd(int n, const float *x, const idx_t *ids, AddStaging &staging);
    void copyVectorToDevice(int n, AddStaging &staging);
    void initFlatAtFp32();
    void randomOrthogonalGivens(int n, std::vector<float> &orthogonalMatrix);
    void uploadorthogonalMatrix(std::vector<float> &orthogonalMatrix);

    AscendIndexIVFRaBitQ *intf_;
    std::vector<float> srcIndexes;
    const float *attachedSrc_ = nullptr;  // attach 模式下外部提供的精排原文（零拷贝，不持有所有权）
    idx_t attachedSrcRows_ = 0;
    std::vector<float> centroidsData;
    std::vector<float> orthogonalMatrix;
    std::unique_ptr<::ascend::IndexIVFFlat> assignIndex;  // 复用ivfflat一阶段检索能力加速add过程和npu聚类

   private:
    struct FilterCacheKey
    {
        const IDSelector *selPtr = nullptr;
        const void *payload = nullptr;
        size_t n = 0;
        int64_t negate = 0;
        uint64_t contentHash = 0;
        enum class Kind
        {
            Object = 0,
            Array,
            Bitmap,
            Roaring
        };
        Kind kind = Kind::Object;
    };

    void searchImplFiltered(int n, const float *x, int k, float *distances, idx_t *labels,
                            const ::ascend::RabitqIdFilterHost *idFilter, int searchNprobe = 0) const;

    const ::ascend::RabitqIdFilterHost *getCachedFilter(const IDSelector *sel) const;

    static FilterCacheKey MakeFilterCacheKey(const IDSelector *sel);
    static bool IsFilterCacheHit(const FilterCacheKey &cached, const FilterCacheKey &key);
    static bool IsFilterSlotEmpty(const FilterCacheKey &key);
    int FindFilterCacheHit(const FilterCacheKey &key) const;
    int AllocFilterCacheSlot() const;
    void InvalidateFilterCache() const;

    AscendIndexIVFRaBitQConfig ivfrabitqConfig;

    static constexpr int kFilterCacheSlots = 2;
    struct FilterCacheSlot
    {
        FilterCacheKey key;
        ::ascend::RabitqIdFilterHost filter;
    };

    mutable std::mutex filterCacheMutex;
    mutable FilterCacheSlot filterSlots[kFilterCacheSlots];
    mutable int filterSlotLru = 0;
    mutable uint64_t cachedFilterGeneration = 0;

    // Single source of truth for where an id lives. Only writers touch it, but copyFrom and
    // threadUnsafeReset reach it without going through writerMutex, so it keeps its own.
    std::unique_ptr<LocationTable> locationTable;
    std::mutex locationMutex;

    // Serialises writers against each other now that they only hold a read lock on mtx.
    std::mutex writerMutex;

    // Separate pools so a write fanning out over the devices cannot be starved by, or queue behind, a
    // concurrent search fanning out over the same devices.
    std::shared_ptr<AscendThreadPool> readPool;
    std::shared_ptr<AscendThreadPool> writePool;

    // Source ownership is a mode, not a test of vector size: an empty copied source still forbids mutation.
    bool copiedSources_ = false;
    bool refineSourcesReady_ = true;
};
}  // namespace ascend
}  // namespace faiss
#endif
