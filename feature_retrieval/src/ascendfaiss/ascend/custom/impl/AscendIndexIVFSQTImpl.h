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


#ifndef ASCEND_INDEX_IVFSQT_IMPL_INCLUDED
#define ASCEND_INDEX_IVFSQT_IMPL_INCLUDED
#include <limits>
#include <omp.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexFlat.h>

#include "ascend/custom/AscendIndexIVFSQT.h"
#include "ascend/custom/impl/AscendIndexIVFSQCImpl.h"
#include "index_custom/IndexIVFSQTIPAicpu.h"
#include "ascend/custom/AscendClustering.h"

struct fp16;

namespace faiss {
namespace ascend {
class AscendIndexFlatATInt8;

struct AddNpuClusInputParam {
    int startListId { 0 };
    std::vector<uint8_t> &quantCodes;
    size_t offset { 0 };
    const size_t numBuck { 0 };
    size_t curBatchLoop { 0 };
    bool upToNumThres { false };
};

struct AddNpuClusOutputParam {
    int &batchLen;
    std::vector<size_t> &listLenBucks;
    size_t &batchSubNlist;
    AscendClustering &npuClus;
};

struct AssignParam {
    size_t listId { 0 };
    size_t listLen { 0 };
    size_t oriListLen { 0 };
    int threadNum { 0 };
    bool upToNumThres { false };
    float *baseData { nullptr };
    std::vector<int> &oribaseIds;
    std::vector<uint8_t> &quantCodes;
    faiss::IndexFlat &subCpuQuantizer;
    std::vector<faiss::idx_t> &label;
    std::vector<int> &values;
};

class AscendIndexIVFSQTImpl : public AscendIndexIVFSQCImpl {
public:
    // Construct an index from CPU IndexIVFSQ
    AscendIndexIVFSQTImpl(AscendIndexIVFSQ *intf, const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());

    // Construct an empty index
    AscendIndexIVFSQTImpl(AscendIndexIVFSQ *intf, int dimIn, int dimOut, int nlist,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());

    virtual ~AscendIndexIVFSQTImpl();

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQTImpl(const AscendIndexIVFSQTImpl &) = delete;

    AscendIndexIVFSQTImpl &operator = (const AscendIndexIVFSQTImpl &) = delete;

    void copyTo(faiss::IndexIVFScalarQuantizer *index);

    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    void train(idx_t n, const float *x);

    void update(bool cleanData = true);
    
    void setNumProbes(int nprobes) override;

    void updateTParams(int l2Probe, int l3SegmentNum);

    void setLowerBound(int lowerBound);
    int getLowerBound() const;

    void setMergeThres(int mergeThres);
    int getMergeThres() const;

    void setMemoryLimit(float memoryLimit);

    void setAddTotal(size_t addTotal);

    void setPreciseMemControl(bool preciseMemControl);

    void reset();

    size_t remove_ids(const faiss::IDSelector &sel);

    float getQMin() const;

    float getQMax() const;

    uint32_t getListLength(int listId) const override;

    void getListCodesAndIds(int listId, std::vector<uint8_t> &codes, std::vector<ascend_idx_t> &ids) const override;

    void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);

    void setSortMode(int mode);

    void setUseCpuUpdate(int numThreads);

protected:
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    void searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const override;

    void resizeSearchPipeline(int k) const;

    template <typename T, typename D>
    void insertFragmentation(int n, const T *data, int stride, std::vector<std::shared_ptr<std::vector<D>>> &dest);

    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const idx_t *ids) override;

    void addWithPageCodes(int devIdx);

    void calbaseInfo();

    void updateDeviceSubCenter();

    void TransDeviceSubCentPage(int n, const float *x, bool lastFlag);

    void updateBaseOffsetMask(int devIdx);

    void compressProc(int n, const float *x, const idx_t *ids, uint8_t *codesIn,
        std::vector<idx_t> &resultSearchId);

    // split SubData from sq.trained to load from a index
    void splitSubData();

    void appendTrained() override;

    void splitTrained() override;

    void copyCodes(const faiss::IndexIVFScalarQuantizer *index) override;

    // reset Internal for subnlist
    void resetInternal();

    void initTrainedValue();
    // functions for pipelining search
    void initSearchPipeline();
    void searchPipelinePrepareSingleDevice(int idx, int n, int k, const float *x, size_t offset) const;
    void searchPipelineFinishSingleDevice(int idx, int n, int k, float *distances, idx_t *labels,
        size_t offset) const;

    size_t getSendDeviceSize(int n);
    void initDeviceNpuType(); // for variable deviceNpuType

    void trainHostBaseSubCluster(std::vector<int> &listsPerDevice, int deviceId);

    size_t getListLen(size_t curListId, bool upToNumThres, size_t oriListLen, const std::vector<int> &oribaseIds);

    void trainCpuQuantizer(size_t listLen, int subnlist, float *baseData, int thisThread,
        faiss::IndexFlat &subCpuQuantizer);

    void cpuQuantizerAssignWithLargeListLen(AssignParam &param, int &subnlist);

    void cpuQuantizerAssign(AssignParam &param, int &subnlist);

    void reassignLabels(AssignParam &param, const std::vector<faiss::idx_t> &toPreserve,
        const std::vector<faiss::idx_t> &toMerge);

    void trainHostBaseSubClusterFor310(int numThreads);

    void updateForCpu(int numThreads, bool cleanData = true);

    void trainSubClusterMultiDevices();

    void fetchDeviceTmpData();

    void deviceBaseAssignCoarse();

    void deviceBaseAssignSub(int num);

    bool isToDevice(int id);

    void getFuzzyList(size_t n, const float *x, std::vector<idx_t> &resultSearchId) override;

    void getFuzzyList(size_t n, const float *x, std::vector<idx_t> &resultSearchId, int cpuAvail);

    void trainQuantizer(idx_t n, const float *x, bool clearNpuData = true) override;

    void fillingTrainedData(std::vector<uint16_t> &trainedFp16, uint16_t *vmin, uint16_t *vdiff, size_t length);

    void addNpuClus(AddNpuClusInputParam &inputParam, AddNpuClusOutputParam &outputParam);

    void addNpuClusByListLen(AddNpuClusInputParam &inputParam, size_t curBuck,
        size_t oriListLen, std::vector<int> &oribaseIds, AddNpuClusOutputParam &outputParam);

    void fillingAllLabels(size_t listId, size_t oriListLen, bool upToNumThres,
        std::vector<int> &oribaseIds);

    inline ::ascend::IndexIVFSQTIPAicpu* getActualIndex (int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFSQTIPAicpu *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    AscendIndexIVFSQTConfig ivfsqtConfig;

    std::vector<std::vector<int>> allLabels;

    std::vector<float> adaptiveSubCenters;

    std::unordered_map<int, std::vector<int>> baseNums;
    // Device pageSize is 32768, need a param to save page count
    size_t pageCount = 1;

    std::vector<int> deviceAddList; // idx: pageId, value: nlist
    std::vector<int> deviceAddNum; // idx: pageId, value: data num

    // for pipelining search (we use pointer because search has a const interface)
    std::unique_ptr<std::vector<uint16_t>> searchPipelineQuery = nullptr;
    std::unique_ptr<std::vector<uint16_t>> searchPipelineQueryPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineQueries = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineQueriesPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDist = nullptr;
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDistPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalf = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalfPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineLabel = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineLabelPrev = nullptr;

    // Add Popped
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDistPopped = nullptr;
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDistPrevPopped = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalfPopped = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalfPrevPopped = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineIndexPopped = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineIndexPrevPopped = nullptr;

    // quantize dimIn codes internal
    faiss::ScalarQuantizer sqIn;

    std::vector<std::vector<std::vector<int>>> subBucketNum;
    std::vector<std::vector<std::vector<int>>> subBucketOffset;
    std::vector<std::vector<std::vector<int>>> devSubBucketOffset;

    std::vector<std::shared_ptr<std::vector<uint8_t>>> globalInCodesPointer;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> globalOutCodesPointer;

    std::vector<std::shared_ptr<std::vector<ascend_idx_t>>> globalIdsPointer;
    std::vector<std::vector<int>> bucketDetails;

    typedef std::pair<float, ascend_idx_t> HostDistIndPair;

    std::vector<std::vector<int>> assignCountsDevice;
    std::vector<uint8_t> globalInCodesDevice;
    std::vector<int> idDictDevice;
    std::vector<int> idDictHost;
    std::vector<int> idDictglobalInCodesDevice;

    const size_t numHostThreshold = MAX_N;

    int lowerBound = 32;

    int mergeThres = 5;

    int subcenterNum = 64;

    int l2NProbe = 48; // L2 nprobe

    int l3SegmentNum = 96; // L3 ops Segment num

    float memoryLimit = 32; // Host Memory Limit, GigaByte

    size_t addTotal = 100000000; // Num of Base Vec expected to be added

    bool preciseMemControl = false; // Whether to precisely limit Host Memory

    bool isUpdated = false;

    bool isCpuUpdated = false;

    int defaultnumThreads = omp_get_max_threads();

    float qMin = std::numeric_limits<float>::max();

    float qMax = std::numeric_limits<float>::min();

    std::vector<std::unique_ptr<AscendIndexFlatAT>> npuQuantizerLists;

    uint32_t deviceNpuType = -1; // 0 reps 310, 1 reps 310P

private:
    void checkSearchParams(int nprobe, int l2Probe, int l3SegmentNum) const;

    void threadUnsafeReset() override;

    void checkParams() const;

    void sortDistDesc(size_t n, std::vector<idx_t> &resultSearchId, const idx_t *searchId,
                      const float *distances) const;
    void subBucketHandle(int listId, int deviceCnt);

    void resetPointer();

    size_t hostNtotal = 0;

    size_t deviceNtotal = 0;

    size_t fuzzyTotalDevice = 0;

    size_t fuzzyTotalHost = 0;

    size_t replaceCurId = 0;

    size_t deviceDataNums = 0;

    size_t strideHost = 3;

    size_t strideDevice = 2;

    bool resetAfterUpdate = false;

    int ivfFuzzyTopkMode = 0;

    mutable std::mutex searchMtx;
    mutable std::mutex copyToMtx;
};
} // ascend
} // faiss
#endif
