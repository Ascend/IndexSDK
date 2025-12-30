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


#include "ascend/custom/impl/AscendIndexIVFSQTImpl.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <omp.h>
#include <numeric>
#include <sys/time.h>

#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/sorting.h>
#include <faiss/utils/utils.h>

#include "ascend/utils/AscendIVFAddInfo.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "common/utils/CommonUtils.h"
#include "index_custom/IndexIVFSQTIPAicpu.h"

namespace faiss {
namespace ascend {
// Default params in case of nullptr index
const int DEFAULT_DIM_IN = 256;
const int DEFAULT_DIM_OUT = 64;
const int DEFAULT_NLIST = 8192;
// For copy index we append information to sq.trained, but standard sq trained size is 2
const int SQ_VAL_SIZE = 2;
const float PADDING_NUM = 100;
const int KMEANS_ITER = 16;
const int KMEANS_MIN_POINTS_PER_CENTROID = 10;
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;
const size_t DEVICE_PAGE_SIZE = 8192;
const size_t SEARCH_PAGE_SIZE = 2048;
const int K_BUFFER_RATIO = 2;
const int K_HEAP_RATIO = 2;
const size_t GLOBAL_CODES_PAGE_SIZE = 5000000;
const int MAX_CLUS_POINTS = 16384;
const size_t GB = 1024 * 1024 * 1024;
const float MEM_LIMIT_RATIO = 0.95;
const float EPSILON = 1e-6;
const int INT8_LOWER_BOUND = -128;
const int INT8_UPPER_BOUND = 127;
const int UINT8_UPPER_BOUND = 255;
const int CPU_AVAIL = 20;
const idx_t MAX_TRAIN_NUM = 7000000;
const std::set<int> L3_SEG_SIZES = { 24,  36,  48,  60,  72,  84,  96,  120, 144, 156, 168,
                                     192, 216, 240, 360, 480, 600, 720, 840, 960, 1020 };
const std::set<int> NPROBE_NUMS = { 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 };

// 0表示默认原有近似排序模式，1表示精确排序模式，精确排序模式提升精度但会降低部分性能
const std::set<int> SORT_MODES = { 0, 1 };
// The value range of dimIn
const std::vector<int> DIMS_IN = { 256 };

// The value range of dimOut
const std::vector<int> DIMS_OUT = { 32, 64, 128 };

// The value range of nlist
const std::vector<int> NLISTS = { 1024, 2048, 4096, 8192, 16384, 32768 };

const int MAX_OMP_THREAD_NUM = 96;

AscendIndexIVFSQTImpl::AscendIndexIVFSQTImpl(AscendIndexIVFSQ *intf, const faiss::IndexIVFScalarQuantizer *index,
    AscendIndexIVFSQTConfig config)
    : AscendIndexIVFSQCImpl(intf, ((index == nullptr) ? DEFAULT_DIM_IN : index->d),
    ((index == nullptr) ? DEFAULT_DIM_OUT : index->sq.d), ((index == nullptr) ? DEFAULT_NLIST : index->nlist), false,
    ((index == nullptr) ? ScalarQuantizer::QuantizerType::QT_8bit : index->sq.qtype), MetricType::METRIC_INNER_PRODUCT,
    config),
      ivfsqtConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");

    checkParams();
    this->fuzzyType = TYPE_T;
    sqIn = faiss::ScalarQuantizer(dimIn, ScalarQuantizer::QuantizerType::QT_8bit);
    initTrainedValue();
    initSearchPipeline();
    copyFrom(index);
    initDeviceNpuType();
}

AscendIndexIVFSQTImpl::AscendIndexIVFSQTImpl(AscendIndexIVFSQ *intf, int dimIn, int dimOut, int nlist,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric, AscendIndexIVFSQTConfig config)
    : AscendIndexIVFSQCImpl(intf, dimIn, dimOut, nlist, false, qtype, metric, config), ivfsqtConfig(config)
{
    checkParams();
    // Some params and check can get from parent classes
    initIndexes();
    // Init subcenter params, Process clone an empty index.
    initTrainedValue();
    initSearchPipeline();
    initDeviceNpuType();
    this->fuzzyType = TYPE_T;

    if (this->ivfsqtConfig.useKmeansPP) {
        if (deviceNpuType != 0) {
            faiss::ascend::AscendIndexFlatATInt8Config npuInt8Conf(ivfsqtConfig.deviceList, ivfsqtConfig.resourceSize);
            pQuantizerImpl->npuInt8Quantizer =
                std::make_shared<AscendIndexFlatATInt8>(this->intf_->d, this->nlist, npuInt8Conf);
        }
    }

    sqIn = faiss::ScalarQuantizer(dimIn, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
}

void AscendIndexIVFSQTImpl::checkParams() const
{
    // only support IP
    FAISS_THROW_IF_NOT_FMT(this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type: %d", this->intf_->metric_type);
    // only support SQ8
    FAISS_THROW_IF_NOT_FMT(sq.qtype == faiss::ScalarQuantizer::QT_8bit,
        "Unsupported qtype: %d", sq.qtype);
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS_IN.begin(), DIMS_IN.end(), this->dimIn) != DIMS_IN.end(),
        "Unsupported dimIn: %d", this->dimIn);
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS_OUT.begin(), DIMS_OUT.end(), this->dimOut) != DIMS_OUT.end(),
        "Unsupported dimOut: %d", this->dimOut);
    FAISS_THROW_IF_NOT_FMT(dimIn % dimOut == 0,
        "The dimIn(%d) must be an integer multiple of the dimOut(%d)", this->dimIn, this->dimOut);
    FAISS_THROW_IF_NOT_FMT(dimIn > dimOut,
        "The dimIn(%d) must be greater than the dimOut(%d)", this->dimIn, this->dimOut);
    FAISS_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(),
        "Unsupported nlists: %d", this->nlist);
    FAISS_THROW_IF_NOT_FMT(NPROBE_NUMS.find(nprobe) != NPROBE_NUMS.end(),
        "nprobe %d is invalid, please check it", nprobe);
}

AscendIndexIVFSQTImpl::~AscendIndexIVFSQTImpl() {}

void AscendIndexIVFSQTImpl::trainQuantizer(faiss::idx_t n, const float *x, bool clearNpuData)
{
    APP_LOG_INFO("AscendIndexIVFSQT start to trainQuantizer with %ld vector(s).\n", n);
    if (n == 0) {
        // nothing to do
        return;
    }

    if (pQuantizerImpl->cpuQuantizer->is_trained && (pQuantizerImpl->cpuQuantizer->ntotal == nlist)) {
        if (this->intf_->verbose) {
            printf("IVFSQT quantizer does not need training.\n");
        }

        return;
    }

    if (this->intf_->verbose) {
        printf("Training IVFSQT quantizer on %ld vectors in %dD to %d cluster\n", n, this->intf_->d, nlist);
    }

    pQuantizerImpl->cpuQuantizer->reset();
    if (this->ivfConfig.useKmeansPP) {
        std::vector<float> tmpCentroids(nlist * this->intf_->d);
        if (pQuantizerImpl->npuClus->GetNTotal() == 0) {
            pQuantizerImpl->npuClus->Add(n, x);
        }

        // AscendClustering does NOT perform sampling on data
        pQuantizerImpl->npuClus->DistributedTrain(this->ivfConfig.cp.niter, tmpCentroids.data(),
                                                  indexConfig.deviceList, clearNpuData);
        pQuantizerImpl->cpuQuantizer->add(nlist, tmpCentroids.data());
        if (deviceNpuType == 0) {
            pQuantizerImpl->npuQuantizer->reset();
            pQuantizerImpl->npuQuantizer->add(nlist, tmpCentroids.data());
        } else {
            pQuantizerImpl->npuInt8Quantizer->reset();
            pQuantizerImpl->npuInt8Quantizer->add(nlist, tmpCentroids.data());
        }
    } else {
        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));
    }
    pQuantizerImpl->cpuQuantizer->is_trained = true;

    FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
    APP_LOG_INFO("AscendIndexIVF trainQuantizer finished.\n");
}

void AscendIndexIVFSQTImpl::initSearchPipeline()
{
    searchPipelineQuery = std::unique_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>());
    searchPipelineQueryPrev = std::unique_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>());
    searchPipelineQueries =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineQueriesPrev =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineDist = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
    searchPipelineDistPrev = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
    searchPipelineDistHalf =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineDistHalfPrev =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineLabel =
        std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
    searchPipelineLabelPrev =
        std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
    // Add DistancePopped and IndexPopped
    searchPipelineDistPopped = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
    searchPipelineDistPrevPopped =
        std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
    searchPipelineDistHalfPopped =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineDistHalfPrevPopped =
        std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
    searchPipelineIndexPopped =
        std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
    searchPipelineIndexPrevPopped =
        std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
}

void AscendIndexIVFSQTImpl::searchPipelinePrepareSingleDevice(int idx, int n, int k, const float *x,
    size_t offset) const
{
    APP_LOG_INFO("AscendIndexIVFSQT start to searchPipelinePrepare: n=%d, k=%d.\n", n, k);

    // convert query data from float to fp16, device use fp16 data to search
    searchPipelineQueries->at(idx).resize(n * this->intf_->d);

    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());

    int cpuAvg = CPU_AVAIL / deviceCnt;
    FAISS_THROW_IF_NOT_FMT(cpuAvg != 0, "deviceCnt %d must <= %d, please check it", deviceCnt, CPU_AVAIL);
    std::shared_ptr<AscendThreadPool> poolPrepare = std::make_shared<AscendThreadPool>(cpuAvg);

    int fragN = (n + cpuAvg - 1) / cpuAvg;

    auto prepareFunctor = [&](int nIdx) {
        ::ascend::CommonUtils::attachToCpu(nIdx);
        int start = std::min(fragN * nIdx, n);
        int end = std::min(fragN * (nIdx + 1), n);

        transform(x + (offset + start) * this->intf_->d, x + (offset + end) * this->intf_->d,
            searchPipelineQueries->at(idx).begin() + start * this->intf_->d,
            [](float temp) { return fp16(temp).data; });
    };

    std::vector<std::future<void>> prepareFunctorRet;
    for (size_t nIdx = 0; nIdx < (size_t)cpuAvg; nIdx++) {
        prepareFunctorRet.emplace_back(poolPrepare->Enqueue(prepareFunctor, nIdx));
    }
    int prepareWait = 0;
    try {
        for_each(prepareFunctorRet.begin(), prepareFunctorRet.end(), [&](std::future<void> &ret) {
            prepareWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        for_each(prepareFunctorRet.begin() + prepareWait, prepareFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for prepare functor failed %s", e.what());
    }

    searchPipelineDist->at(idx).resize(n * k);
    searchPipelineDistHalf->at(idx).resize(n * k);
    searchPipelineLabel->at(idx).resize(n * k);

    searchPipelineDistPopped->at(idx).resize(n * k);
    searchPipelineDistHalfPopped->at(idx).resize(n * k);
    searchPipelineIndexPopped->at(idx).resize(n * k);
    APP_LOG_INFO("AscendIndexIVFSQT searchPipelinePrepare operation finished.\n");
}

void AscendIndexIVFSQTImpl::searchPipelineFinishSingleDevice(int idx, int n, int k, float *distances,
    idx_t *labels, size_t offset) const
{
    APP_LOG_INFO("AscendIndexIVFSQT start to searchPipelineFinish: n=%d, k=%d.\n", n, k);
    int kBuffer = k / K_BUFFER_RATIO;
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());

    int cpuAvg = CPU_AVAIL / deviceCnt;
    FAISS_THROW_IF_NOT_FMT(cpuAvg != 0, "deviceCnt %d must <= %d, please check it", deviceCnt, CPU_AVAIL);
    std::shared_ptr<AscendThreadPool> poolFinish = std::make_shared<AscendThreadPool>(cpuAvg);

    int fragN = (n + cpuAvg - 1) / cpuAvg;

    auto finishFunctor = [&](int nIdx) {
        ::ascend::CommonUtils::attachToCpu(nIdx);
        int start = std::min(fragN * nIdx, n);
        int end = std::min(fragN * (nIdx + 1), n);
        // convert result data from fp16 to float
        transform(searchPipelineDistHalfPrev->at(idx).begin() + start * k,
            searchPipelineDistHalfPrev->at(idx).begin() + end * k, searchPipelineDistPrev->at(idx).begin() + start * k,
            [](uint16_t temp) { return (float)fp16(temp); });
        transform(searchPipelineDistHalfPrevPopped->at(idx).begin() + start * k,
            searchPipelineDistHalfPrevPopped->at(idx).begin() + end * k,
            searchPipelineDistPrevPopped->at(idx).begin() + start * k, [](uint16_t temp) { return (float)fp16(temp); });

        // post process with popped: reorder topK results and merge the topk results from devices
        for (int i = start; i < end; i++) {
            std::vector<HostDistIndPair> p(k + kBuffer);
            for (int j = 0; j < k; j++) {
                p[j].first = searchPipelineDistPrev->at(idx).at(j + i * k);
                p[j].second = searchPipelineLabelPrev->at(idx).at(j + i * k);
            }
            for (int j = 0; j < kBuffer; j++) {
                p[j + k].first = searchPipelineDistPrevPopped->at(idx).at(j + i * kBuffer);
                p[j + k].second = searchPipelineIndexPrevPopped->at(idx).at(j + i * kBuffer);
            }

            std::sort(p.begin(), p.end(), std::greater<HostDistIndPair>());

            int flag = 0;
            const ascend_idx_t INVALID_IDX = std::numeric_limits<ascend_idx_t>::max();
            std::unordered_set<ascend_idx_t> labelSet;
            size_t offsetFinal = offset * (size_t)k + (size_t)i * (size_t)k;
            for (size_t j = 0; j < size_t(k + kBuffer); j++) {
                if (flag >= k) {
                    break;
                }
    
                if (labelSet.find(p[j].second) != labelSet.end()) {
                    continue;
                }
                labelSet.insert(p[j].second);

                if (p[j].second == INVALID_IDX) {
                    break;
                }
                *(distances + offsetFinal + flag) = p[j].first;
                *(labels + offsetFinal + flag) = p[j].second;
                flag += 1;
            }
            for (; flag < k; flag++) {
                *(labels + offsetFinal + flag) = INVALID_IDX;
            }
        }
    };

    std::vector<std::future<void>> finishFunctorRet;
    for (size_t nIdx = 0; nIdx < (size_t)cpuAvg; nIdx++) {
        finishFunctorRet.emplace_back(poolFinish->Enqueue(finishFunctor, nIdx));
    }
    int finishWait = 0;
    try {
        for_each(finishFunctorRet.begin(), finishFunctorRet.end(), [&](std::future<void> &ret) {
            finishWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        for_each(finishFunctorRet.begin() + finishWait, finishFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for finish functor failed %s", e.what());
    }

    searchPipelineQueriesPrev->at(idx).clear();
    searchPipelineDistPrev->at(idx).clear();
    searchPipelineDistHalfPrev->at(idx).clear();
    searchPipelineLabelPrev->at(idx).clear();

    searchPipelineDistPrevPopped->at(idx).clear();
    searchPipelineDistHalfPrevPopped->at(idx).clear();
    searchPipelineIndexPrevPopped->at(idx).clear();
    APP_LOG_INFO("AscendIndexIVFSQT searchPipelineFinish operation finished.\n");
}

void AscendIndexIVFSQTImpl::resizeSearchPipeline(int k) const
{
    size_t deviceCnt = indexConfig.deviceList.size();
    searchPipelineQueries->resize(deviceCnt, std::vector<uint16_t>(SEARCH_PAGE_SIZE * this->intf_->d, 0));
    searchPipelineDist->resize(deviceCnt, std::vector<float>(SEARCH_PAGE_SIZE * k, 0));
    searchPipelineDistHalf->resize(deviceCnt, std::vector<uint16_t>(SEARCH_PAGE_SIZE * k, 0));
    searchPipelineLabel->resize(deviceCnt, std::vector<ascend_idx_t>(SEARCH_PAGE_SIZE * k, 0));
    searchPipelineDistPopped->resize(deviceCnt, std::vector<float>(SEARCH_PAGE_SIZE * k, 0));
    searchPipelineDistHalfPopped->resize(deviceCnt, std::vector<uint16_t>(SEARCH_PAGE_SIZE * k, 0));
    searchPipelineIndexPopped->resize(deviceCnt, std::vector<ascend_idx_t>(SEARCH_PAGE_SIZE * k, 0));

    searchPipelineQueriesPrev->resize(deviceCnt);
    searchPipelineDistPrev->resize(deviceCnt);
    searchPipelineDistHalfPrev->resize(deviceCnt);
    searchPipelineLabelPrev->resize(deviceCnt);
    searchPipelineDistPrevPopped->resize(deviceCnt);
    searchPipelineDistHalfPrevPopped->resize(deviceCnt);
    searchPipelineIndexPrevPopped->resize(deviceCnt);
}

void AscendIndexIVFSQTImpl::searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const
{
    // 该接口修改成员变量，不支持并发；但考虑到这些成员变量只有本接口使用，因此可以和其他读接口并发。
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(searchMtx);

    APP_LOG_INFO("AscendIndexIVFSQT start to searchPaged: n=%d, k=%d.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(isUpdated == true, "Please do update before search!");

    resizeSearchPipeline(k);
    auto searchFunctorSingleDevice = [&](int idx, size_t n1, const float *x1, float *distances1,
        idx_t *labels1) {
        size_t tileSize = SEARCH_PAGE_SIZE;
        size_t prevNum = 0;

        auto searchFunctor = [&](size_t num, uint16_t *queryPtr, uint16_t *distHalfPtr, ascend_idx_t *labelPtr,
            uint16_t *distHalfPoppedPtr, ascend_idx_t *indexPoppedPtr) {
            int deviceId = indexConfig.deviceList[idx];
            auto pIndex = getActualIndex(deviceId);
            // When dividend k is 1, set kBufferRatio and kHeapRatio is 1 to ensure that quotient is not 0
            if (k == 1 || ivfFuzzyTopkMode == 1) {
                pIndex->setRatio(1, 1);
            } else {
                pIndex->setRatio(K_BUFFER_RATIO, K_HEAP_RATIO);
            }

            pIndex->setPoppedDistAndIndex(distHalfPoppedPtr, indexPoppedPtr);
            auto ret = pIndex->search(num, queryPtr, k, distHalfPtr, labelPtr);
            FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ivfsqt search failed, ret: %d", ret);
        };

        std::shared_ptr<AscendThreadPool> poolSingleDevice = std::make_shared<AscendThreadPool>(1);
        std::vector<std::future<void>> functorRets;
        for (size_t i = 0; i < (size_t)n1; i += tileSize) {
            size_t curNum = std::min(tileSize, (size_t)(n1)-i);

            // prepare data for current page
            searchPipelinePrepareSingleDevice(idx, curNum, k, x1, i);

            if (i > 0) {
                // wait for previous page to finish on device
                for (auto &ret : functorRets) {
                    ret.get();
                }
                functorRets.clear();
            }
            functorRets.emplace_back(poolSingleDevice->Enqueue(searchFunctor, curNum,
                searchPipelineQueries->at(idx).data(), searchPipelineDistHalf->at(idx).data(),
                searchPipelineLabel->at(idx).data(), searchPipelineDistHalfPopped->at(idx).data(),
                searchPipelineIndexPopped->at(idx).data()));

            // run post process for previous page
            if (i > 0) {
                searchPipelineFinishSingleDevice(idx, prevNum, k, distances1, labels1, i - tileSize);
            }

            // move data to prepare the next page
            searchPipelineQueriesPrev->at(idx) = std::move(searchPipelineQueries->at(idx));
            searchPipelineDistPrev->at(idx) = std::move(searchPipelineDist->at(idx));
            searchPipelineDistHalfPrev->at(idx) = std::move(searchPipelineDistHalf->at(idx));
            searchPipelineLabelPrev->at(idx) = std::move(searchPipelineLabel->at(idx));

            // Add Popped
            searchPipelineDistPrevPopped->at(idx) = std::move(searchPipelineDistPopped->at(idx));
            searchPipelineDistHalfPrevPopped->at(idx) = std::move(searchPipelineDistHalfPopped->at(idx));
            searchPipelineIndexPrevPopped->at(idx) = std::move(searchPipelineIndexPopped->at(idx));

            prevNum = curNum;
        }
        for (auto &ret : functorRets) {
            ret.get();
        }
        size_t lastOffset = (n1 - 1) / tileSize * tileSize;
        searchPipelineFinishSingleDevice(idx, prevNum, k, distances1, labels1, lastOffset);
    };

    if (indexConfig.deviceList.size() == 1) {
        return searchFunctorSingleDevice(0, n, x, distances, labels);
    }

    std::vector<std::future<void>> functorRets;
    size_t deviceCnt = indexConfig.deviceList.size();
    size_t numSingleDevice = ((size_t)n + deviceCnt - 1) / deviceCnt;
    for (size_t idx = 0; idx < deviceCnt; idx++) {
        size_t startSingleDevice = std::min(idx * numSingleDevice, (size_t)n);
        size_t endSingleDevice = std::min((idx + 1) * numSingleDevice, (size_t)n);
        if (startSingleDevice >= endSingleDevice) {
            continue;
        }
        functorRets.emplace_back(pool->Enqueue(searchFunctorSingleDevice, idx, endSingleDevice - startSingleDevice,
            x + startSingleDevice * dimIn, distances + startSingleDevice * k, labels + startSingleDevice * k));
    }
    for (auto &ret : functorRets) {
        ret.get();
    }
    APP_LOG_INFO("AscendIndexIVFSQT searchPaged operation finished.\n");
}

void AscendIndexIVFSQTImpl::updateDeviceSubCenter()
{
    APP_LOG_INFO("AscendIndexIVFSQT updateDeviceSubCenter operation started.\n");
    if (!this->intf_->is_trained) {
        return;
    }
    resetInternal();
    size_t total = adaptiveSubCenters.size() / static_cast<size_t>(dimIn);
    TransDeviceSubCentPage(total, adaptiveSubCenters.data(), true);
    APP_LOG_INFO("AscendIndexIVFSQT updateDeviceSubCenter operation finished.\n");
}

void AscendIndexIVFSQTImpl::initTrainedValue()
{
    APP_LOG_INFO("AscendIndexIVFSQT initTrainedValue operation started.\n");
    AscendIndexIVFSQCImpl::initTrainedValue();
    adaptiveSubCenters = std::vector<float>(nlist * subcenterNum * dimIn, PADDING_NUM);
    baseNums.clear();
    baseNums.reserve(nlist);
    // init bucketDetails
    bucketDetails.resize(nlist);
    assignCountsDevice.resize(nlist);
    for (int listId = 0; listId < nlist; listId++) {
        std::vector<int> tmpNums(subcenterNum, 0);
        baseNums.emplace(listId, tmpNums);
    }
    APP_LOG_INFO("AscendIndexIVFSQT initTrainedValue operation finished.\n");
}

void AscendIndexIVFSQTImpl::TransDeviceSubCentPage(int n, const float *x, bool)
{
    APP_LOG_INFO("AscendIndexIVFSQT TransDeviceSubCentPage operation started.\n");
    std::vector<uint16_t> subCentroidsFp16(n * dimIn);

    std::shared_ptr<AscendThreadPool> pool;
    pool = std::make_shared<AscendThreadPool>(CPU_AVAIL);
    int fragN = (n + CPU_AVAIL - 1) / CPU_AVAIL;
    auto transFunctor = [&](int idx) {
        int actualSize = std::min(fragN, (int)n - idx * fragN);
        int start = idx * fragN * dimIn;
        if (actualSize > 0) {
            transform(x + start, x + std::min(n * dimIn, start + actualSize * dimIn), begin(subCentroidsFp16) + start,
                [](float temp) { return fp16(temp).data; });
        }
    };

    std::vector<std::future<void>> transFunctorRet;
    for (size_t idx = 0; idx < (size_t)CPU_AVAIL; idx++) {
        transFunctorRet.emplace_back(pool->Enqueue(transFunctor, idx));
    }

    int transWait = 0;
    try {
        for_each(transFunctorRet.begin(), transFunctorRet.end(), [&](std::future<void> &ret) {
            transWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        for_each(transFunctorRet.begin() + transWait, transFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for cent trans functor failed %s", e.what());
    }

    std::shared_ptr<AscendThreadPool> poolTrans;
    poolTrans = std::make_shared<AscendThreadPool>(indexConfig.deviceList.size());
    auto transSubFunctor = [&](int i) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->updateSubCentroidsData(n, subCentroidsFp16.data());
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "updateSubCentroidsData failed:%d", ret);
    };

    std::vector<std::future<void>> transSubFunctorRet;
    for (size_t idx = 0; idx < indexConfig.deviceList.size(); idx++) {
        transSubFunctorRet.emplace_back(pool->Enqueue(transSubFunctor, idx));
    }

    int transSubWait = 0;
    try {
        for_each(transSubFunctorRet.begin(), transSubFunctorRet.end(), [&](std::future<void> &ret) {
            transSubWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        for_each(transSubFunctorRet.begin() + transSubWait, transSubFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for cent trans functor failed %s", e.what());
    }
    APP_LOG_INFO("AscendIndexIVFSQT TransDeviceSubCentPage operation finished.\n");
}

/**
 * append SubData to sq.trained to implement Copy Ascend Index to CPU
 */
void AscendIndexIVFSQTImpl::appendTrained()
{
    APP_LOG_INFO("AscendIndexIVFSQT appendTrained operation started.\n");
    std::vector<float> addTmp(deviceAddList.size());
    transform(begin(deviceAddList), end(deviceAddList), begin(addTmp),
        [](int x) { return static_cast<float>(x); });
    sq.trained.insert(sq.trained.end(), addTmp.begin(), addTmp.end());
    transform(begin(deviceAddNum), end(deviceAddNum), begin(addTmp),
        [](int x) { return static_cast<float>(x); });
    sq.trained.insert(sq.trained.end(), addTmp.begin(), addTmp.end());
    sq.trained.push_back(deviceAddNum.size());

    sq.trained.insert(sq.trained.end(), adaptiveSubCenters.begin(), adaptiveSubCenters.end());
    for (int i = 0; i < nlist; ++i) {
        sq.trained.insert(sq.trained.end(), baseNums[i].begin(), baseNums[i].end());
    }
    // append sqIn data to clone index
    sqIn.trained.resize(dimIn * SQ_VAL_SIZE);
    sq.trained.insert(sq.trained.end(), sqIn.trained.begin(), sqIn.trained.end());
    AscendIndexIVFSQCImpl::appendTrained();
    APP_LOG_INFO("AscendIndexIVFSQT appendTrained operation finished.\n");
}

/**
 * Copy Ascend Index SQT to CPU is diff from SQ index, need to get codes from each page
 */
void AscendIndexIVFSQTImpl::copyTo(faiss::IndexIVFScalarQuantizer *index)
{
    // 修改成员变量，因此需要单独加锁
    auto copyToLock = ::ascend::AscendMultiThreadManager::GetWriteLock(copyToMtx);
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    std::vector<int> tmpDeviceList = indexConfig.deviceList;
    tmpDeviceList = std::vector<int> { tmpDeviceList[0] };
    APP_LOG_INFO("AscendIndexIVFSQT copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    FAISS_THROW_IF_NOT_FMT(baseNums.size() == (size_t)nlist, "baseNums size: %zu, nlist: %d", baseNums.size(), nlist);
    FAISS_THROW_IF_NOT_MSG(isUpdated == true, "not updated yet!");

    appendTrained();
    AscendIndexIVFImpl::copyTo(index);
    index->by_residual = false;
    index->sq = sq;
    index->code_size = sq.code_size;

    // recover sq status
    sq.trained.resize(SQ_VAL_SIZE * sq.d);
    pageCount = (pageCount - 1) / indexConfig.deviceList.size() + 1;

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);
    // GetCodes with page
    if (!this->intf_->is_trained || this->intf_->ntotal <= 0) {
        APP_LOG_INFO("AscendIndexIVFSQT copyTo operation finished ahead.\n");
        return;
    }
    for (size_t i = 0; i < tmpDeviceList.size(); i++) {
        int deviceId = tmpDeviceList[i];
        auto pIndex = getActualIndex(deviceId);
        for (int pageId = 0; pageId < static_cast<int>(deviceAddList.size()); pageId++) {
            size_t listSize = pIndex->getPageLength(pageId);
            if (listSize > 0) {
                size_t dataSize = static_cast<size_t>(pIndex->getDim()) * listSize;
                std::vector<uint8_t> codes(dataSize);
                int ret = pIndex->getPageVectorsReshaped(pageId, codes.data());
                FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "getPageVectorsReshaped error, ret=%d", ret);

                std::vector<::ascend::idx_t> ids(listSize);
                ret = pIndex->getPageIndices(pageId, ids.data());
                FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "getPageIndices error, ret=%d", ret);

                std::vector<idx_t> indice(listSize);
                std::transform(std::begin(ids), std::end(ids), std::begin(indice),
                               [](ascend_idx_t x) { return idx_t(x); });
                ivf->add_entries(deviceAddList[pageId], listSize, indice.data(), codes.data());
            }
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT copyTo operation finished.\n");
}

void AscendIndexIVFSQTImpl::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(!isUpdated, "this index had updated, cannot copy from other index");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVFSQCImpl::copyFrom(index);
    checkParams();
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "index must be trained");
    FAISS_THROW_IF_NOT_MSG(sq.code_size == static_cast<size_t>(dimOut), "invalid sq!");

    FAISS_THROW_IF_NOT_MSG(devSubBucketOffset.size() == static_cast<size_t>(nlist),
        "invalid data of devSubBucketOffset.");
    FAISS_THROW_IF_NOT_MSG(subBucketNum.size() == static_cast<size_t>(nlist), "invalid data of subBucketNum.");

    for (size_t listId = 0; listId < static_cast<size_t>(nlist); listId++) {
        FAISS_THROW_IF_NOT_MSG(!devSubBucketOffset[listId].empty(),
            "invalid data of devSubBucketOffset.");
        FAISS_THROW_IF_NOT_MSG(!subBucketNum[listId].empty(), "invalid data of subBucketNum.");
    }

    for (size_t idx = 0; idx < indexConfig.deviceList.size(); ++idx) {
        updateBaseOffsetMask(idx);
    }
    isUpdated = true;
    APP_LOG_INFO("AscendIndexIVFSQT copyFrom operation finished.\n");
}


void AscendIndexIVFSQTImpl::train(idx_t n, const float *x)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    FAISS_THROW_IF_NOT_MSG(x, "Invalid parameter, x can not be nullptr. \n");
    FAISS_THROW_IF_NOT_FMT(n >= static_cast<idx_t>(nlist) && n <= MAX_TRAIN_NUM,
        "Invalid n %ld, shuold be in range [%ld, %ld]\n", n, static_cast<idx_t>(nlist), MAX_TRAIN_NUM);
    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }
    if (isUpdated) {
        FAISS_THROW_IF_NOT_MSG(isUpdated == false, "train after Update is not supported");
        return;
    }

    std::thread sqInTrain([&]() {
        for (size_t i = 0; i < size_t(n * dimIn); i++) {
            if (*(x + i) < qMin) {
                qMin = *(x + i);
            }
            if (*(x + i) > qMax) {
                qMax = *(x + i);
            }
        }
        if (this->ivfsqtConfig.useKmeansPP) {
            if (deviceNpuType != 0) {
                pQuantizerImpl->npuInt8Quantizer->sendMinMax(qMin, qMax);
            }
        }
        // train quantizer to quantize code with dimIn internal
        sqIn.train(n, x);
    });

    APP_LOG_INFO("AscendIndexIVFSQT start to train with %ld vector(s).\n", n);

    try {
        AscendIndexIVFSQCImpl::train(n, x);
    } catch (std::exception &e) {
        sqInTrain.join();
        ASCEND_THROW_FMT("wait for train functor failed %s", e.what());
    }
    sqInTrain.join();
    APP_LOG_INFO("AscendIndexIVFSQT train operation finished.\n");
}

/**
 * AscendIndexIVFSQT resetInternal, clear codes but keep total nums.
 */
void AscendIndexIVFSQTImpl::resetInternal()
{
    APP_LOG_INFO("AscendIndexIVFSQT resetInternal operation started.\n");
    // keep code counts
    int baseTotal = this->intf_->ntotal;
    size_t baseFuzzy = this->fuzzyTotal;
    threadUnsafeReset();
    this->intf_->ntotal = baseTotal;
    this->fuzzyTotal = baseFuzzy;
    APP_LOG_INFO("AscendIndexIVFSQT resetInternal operation finished.\n");
}

void AscendIndexIVFSQTImpl::splitSubData()
{
    APP_LOG_INFO("AscendIndexIVFSQT splitSubData operation started.\n");
    // split sqIn trained value from sq.trained
    sqIn.trained.resize(0);
    size_t sqInSize = static_cast<size_t>(dimIn) * static_cast<size_t>(SQ_VAL_SIZE);
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() >= sqInSize,
        "sq.trained.size should not be less than sqInSize");
    size_t sqInTag = sq.trained.size() - sqInSize;
    sqIn.trained.insert(sqIn.trained.end(), sq.trained.begin() + sqInTag,
        sq.trained.begin() + sqInTag + sqInSize);
    sq.trained.resize(sqInTag);

    // when splitSubData, subData is in the last subDataSize of sq.trained
    size_t subDataSize = static_cast<size_t>(nlist) * static_cast<size_t>(subcenterNum) * static_cast<size_t>(dimIn) +
        static_cast<size_t>(nlist) * static_cast<size_t>(subcenterNum);
    // subDataTag has 2 meanings: 1: subData start id; 2: after split subData, counts of data remaining
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() >= subDataSize,
        "sq.trained.size should not be less than subDataSize");
    size_t subDataTag = sq.trained.size() - subDataSize;
    FAISS_THROW_IF_NOT_MSG(subDataTag >= static_cast<size_t>(SQ_VAL_SIZE * dimOut),
        "subDataTag should not be less than SQ_VAL_SIZE * dimOut");

    // SubData has 2 parts: nlist * subcenterNum * dimIn subcenter, nlist * subcenterNum subNum
    size_t subCenterSize = static_cast<size_t>(nlist) * static_cast<size_t>(subcenterNum) * static_cast<size_t>(dimIn);
    adaptiveSubCenters.resize(0);
    adaptiveSubCenters.insert(adaptiveSubCenters.end(), sq.trained.begin() + subDataTag,
        sq.trained.begin() + subDataTag + subCenterSize);

    baseNums.clear();
    baseNums.reserve(nlist);

    // subNum start index: subDataTag + subCenterSize
    size_t startId = subDataTag + subCenterSize;
    for (int listId = 0; listId < nlist; listId++) {
        std::vector<std::vector<int>> tmpNums(1, std::vector<int>(subcenterNum, 0));
        std::vector<std::vector<int>> tmpOffsets(1, std::vector<int>(subcenterNum, 0));
        for (int j = 0; j < subcenterNum; j++) {
            int labelSize = (int)sq.trained[static_cast<int>(startId) + listId * subcenterNum + j];
            if (labelSize == 0) {
                continue;
            }
            tmpNums[0][j] = labelSize;
        }
        baseNums.emplace(listId, tmpNums[0]);
        subBucketHandle(listId, 1);
    }
    // After split subData, resize sq.trained to origin format
    sq.trained.resize(subDataTag);

    const size_t pageCountDataSize = 1;
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() >= pageCountDataSize,
        "sq.trained.size should not be less than addPageCount");
    size_t pageCountTag = sq.trained.size() - pageCountDataSize;

    size_t pageNum = static_cast<size_t>(sq.trained[pageCountTag]);
    FAISS_THROW_IF_NOT_MSG(pageCountTag >= static_cast<size_t>(SQ_VAL_SIZE * dimOut) +
        static_cast<size_t>(SQ_VAL_SIZE * pageNum), "subDataTag should not be less than SQ_VAL_SIZE * dimOut");
    sq.trained.resize(pageCountTag);
    size_t addListTag = sq.trained.size() - pageNum;

    deviceAddNum.resize(pageNum);
    transform(sq.trained.begin() + addListTag, sq.trained.begin() + addListTag + pageNum,
        begin(deviceAddNum), [](float x) { return static_cast<int>(x); });
    sq.trained.resize(addListTag);
    size_t addNumTag = sq.trained.size() - pageNum;

    deviceAddList.resize(pageNum);
    transform(sq.trained.begin() + addNumTag, sq.trained.begin() + addNumTag + pageNum,
        begin(deviceAddList), [](float x) { return static_cast<int>(x); });
    sq.trained.resize(addNumTag);

    updateDeviceSubCenter();
    APP_LOG_INFO("AscendIndexIVFSQT splitSubData operation finished.\n");
}
void AscendIndexIVFSQTImpl::subBucketHandle(int listId, int deviceCnt)
{
    std::vector<std::vector<int>> subBucketNumItem(deviceCnt, std::vector<int>(subcenterNum, 0));
    std::vector<std::vector<int>> subBucketOffsetItem(deviceCnt, std::vector<int>(subcenterNum, 0));
    std::vector<std::vector<int>> devSubBucketOffsetItem(deviceCnt, std::vector<int>(subcenterNum, 0));
    subBucketNum.emplace_back(subBucketNumItem);
    subBucketOffset.emplace_back(subBucketOffsetItem);
    devSubBucketOffset.emplace_back(devSubBucketOffsetItem);

    int idx = 0;
    for (int j = 0; j < subcenterNum; ++j) {
        for (int k = 0; k < baseNums[listId][j]; ++k) {
            subBucketNum[listId][idx][j]++;
            idx = (idx + 1) % deviceCnt;
        }
    }

    // calculate offset in each sub bucket
        for (int j = 0; j < subcenterNum; ++j) {
            for (int k = 1; k < deviceCnt; ++k) {
                subBucketOffset[listId][k][j] = subBucketOffset[listId][k - 1][j] + subBucketNum[listId][k - 1][j];
            }
        }

        // calculate offset in each device
        for (int idx = 0; idx < deviceCnt; ++idx) {
            for (int j = 1; j < subcenterNum; ++j) {
                devSubBucketOffset[listId][idx][j] =
                devSubBucketOffset[listId][idx][j - 1] + subBucketNum[listId][idx][j - 1];
            }
        }
}

void AscendIndexIVFSQTImpl::splitTrained()
{
    APP_LOG_INFO("AscendIndexIVFSQT splitTrained operation started.\n");
    FAISS_THROW_IF_NOT_MSG(sq.d == static_cast<size_t>(dimOut), "Invalid parameter of sq\n");
    AscendIndexIVFSQCImpl::splitTrained();
    splitSubData();
    APP_LOG_INFO("AscendIndexIVFSQT splitTrained operation finished.\n");
}

/**
 * Copy Index SQT From CPU to Ascend need to copy codes from each page, and arrange to each sublist
 */
void AscendIndexIVFSQTImpl::copyCodes(const faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQT copyCodes operation started.\n");
    const InvertedLists *ivf = index->invlists;
    FAISS_THROW_IF_NOT_MSG(ivf, "Invalid parameter, index->invlists can not be nullptr. \n");
    auto arrayIvf = dynamic_cast<ArrayInvertedLists *>(index->invlists);
    FAISS_THROW_IF_NOT_MSG(arrayIvf != nullptr, "invlists of index invalid.");
    FAISS_THROW_IF_NOT_MSG(arrayIvf->codes.size() == arrayIvf->nlist, "codes size error.");
    FAISS_THROW_IF_NOT_MSG(arrayIvf->ids.size() == arrayIvf->nlist, "ids size error.");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(this->nlist) == arrayIvf->nlist,
                           "nlist of index->invlists must equal to this->nlist");
    size_t nlist = ivf->nlist;
    FAISS_THROW_IF_NOT_FMT(deviceAddList.size() == deviceAddNum.size(),
        "deviceAddList size: %zu, deviceAddNum size: %zu", deviceAddList.size(), deviceAddNum.size());

    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);

        std::vector<int> listSendOffset(nlist, 0);
        for (size_t pageId = 0; pageId < deviceAddList.size(); pageId++) {
            int listId = deviceAddList[pageId];
            FAISS_THROW_IF_NOT_MSG(listId >= 0 && (static_cast<size_t>(listId) < arrayIvf->ids.size()),
                "invalid listId");
            size_t listSize = ivf->list_size(listId);
            if (listSize == 0 || deviceAddNum[pageId] == 0) {
                continue;
            }

            FAISS_THROW_IF_NOT_MSG((deviceAddNum[pageId] >= 0) &&
                (deviceAddNum[pageId] <= static_cast<int>(DEVICE_PAGE_SIZE)), "invalid deviceAddNum");
            FAISS_THROW_IF_NOT_MSG(listSize >= static_cast<size_t>(deviceAddNum[pageId]), "invalid listSize");
            FAISS_THROW_IF_NOT_MSG(listSize * static_cast<size_t>(dimOut) == arrayIvf->codes[listId].size(),
                "invalid codes size");

            FAISS_THROW_IF_NOT_MSG(ivf->get_codes(listId) != nullptr, "invalid codes");
            FAISS_THROW_IF_NOT_MSG(ivf->get_ids(listId) != nullptr, "invalid ids");

            FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(listSendOffset[listId]) +
                static_cast<size_t>(deviceAddNum[pageId]) <= arrayIvf->ids[listId].size(),
                "invalid ids size");

            uint8_t *listCodes = const_cast<uint8_t *>(ivf->get_codes(listId)) + listSendOffset[listId] * dimOut;
            idx_t *tmpIds = const_cast<idx_t *>(ivf->get_ids(listId)) + listSendOffset[listId];

            // trandform idx_t to ascend_idx_t
            std::vector<ascend_idx_t> listIds(tmpIds, tmpIds + deviceAddNum[pageId]);
            transform(tmpIds, tmpIds + deviceAddNum[pageId], begin(listIds),
                [](idx_t x) { return ascend_idx_t(x); });
            auto ret = pIndex->addPageVectors(deviceAddNum[pageId], listCodes, listIds.data());
            FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "addPageVectors failed:%d", ret);
            listSendOffset[listId] += deviceAddNum[pageId];
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT copyCodes operation finished.\n");
}

void AscendIndexIVFSQTImpl::update(bool cleanData)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT update operation started.\n");
    FAISS_THROW_IF_NOT_MSG(!resetAfterUpdate, "update after update and reset is not supported");
    if (deviceNpuType == 0 || isCpuUpdated) {
        updateForCpu(defaultnumThreads, cleanData);
        return;
    }

    if (isUpdated == true) {
        FAISS_THROW_IF_NOT_MSG(adaptiveSubCenters.size() == size_t(nlist * subcenterNum * dimIn),
            "Cannot update twice");
        return;
    }
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained == true, "not train yet!");

    // train sub cluster
    trainSubClusterMultiDevices();

    // send back base from device
    fetchDeviceTmpData();

    if (cleanData) {
        idDictHost.clear();
        subBucketNum.clear();
        subBucketOffset.clear();
        devSubBucketOffset.clear();
        globalInCodesDevice.clear();
        globalInCodesDevice.shrink_to_fit();
        if (this->ivfsqtConfig.useKmeansPP) {
            pQuantizerImpl->npuQuantizer->clearAscendTensor();
            pQuantizerImpl->npuInt8Quantizer->reset();
            pQuantizerImpl->npuInt8Quantizer->clearAscendTensor();
        }
    }

    updateDeviceSubCenter();
    calbaseInfo();

    std::shared_ptr<AscendThreadPool> pool;

    pool = std::make_shared<AscendThreadPool>(indexConfig.deviceList.size());

    auto updateFunctor = [&](int idx) {
        ::ascend::CommonUtils::attachToCpu(idx);
        addWithPageCodes(idx);
        updateBaseOffsetMask(idx);
    };

    std::vector<std::future<void>> updateFunctorRet;
    for (size_t idx = 0; idx < indexConfig.deviceList.size(); idx++) {
        updateFunctorRet.emplace_back(pool->Enqueue(updateFunctor, idx));
    }

    int updateWait = 0;
    try {
        for_each(updateFunctorRet.begin(), updateFunctorRet.end(), [&](std::future<void> &ret) {
            updateWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        for_each(updateFunctorRet.begin() + updateWait, updateFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for cent update functor failed %s", e.what());
    }

    if (cleanData) {
        resetPointer();
        APP_LOG_INFO("AscendIndexIVFSQT data has been cleared out.\n");
    }
    isUpdated = true;
    APP_LOG_INFO("AscendIndexIVFSQT update operation finished.\n");
}

void AscendIndexIVFSQTImpl::resetPointer()
{
    for (size_t i = 0; i < globalIdsPointer.size(); i++) {
        globalIdsPointer[i].reset();
    }
    for (size_t i = 0; i < globalInCodesPointer.size(); i++) {
        globalInCodesPointer[i].reset();
    }
    for (size_t i = 0; i < globalOutCodesPointer.size(); i++) {
        globalOutCodesPointer[i].reset();
    }
}

void AscendIndexIVFSQTImpl::updateForCpu(int numThreads, bool cleanData)
{
    APP_LOG_INFO("AscendIndexIVFSQT update operation started.\n");
    if (isUpdated == true) {
        FAISS_THROW_IF_NOT_MSG(adaptiveSubCenters.size() == size_t(nlist * subcenterNum * dimIn),
            "Cannot update twice");
        return;
    }
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained == true, "not train yet!");

    // train sub cluster
    trainHostBaseSubClusterFor310(numThreads);

    // send back base from device
    fetchDeviceTmpData();

    if (cleanData) {
        devSubBucketOffset.clear();
        subBucketNum.clear();
        idDictHost.clear();
        subBucketOffset.clear();
        globalInCodesDevice.clear();
        globalInCodesDevice.shrink_to_fit();
        if (this->ivfsqtConfig.useKmeansPP) {
            pQuantizerImpl->npuQuantizer->clearAscendTensor();
        }
    }

    updateDeviceSubCenter();
    calbaseInfo();

    for (size_t idx = 0; idx < indexConfig.deviceList.size(); ++idx) {
        addWithPageCodes(idx);
        updateBaseOffsetMask(idx);
    }
    if (cleanData) {
        resetPointer();
        APP_LOG_INFO("AscendIndexIVFSQT data has been cleared out.\n");
    }
    isUpdated = true;
    APP_LOG_INFO("AscendIndexIVFSQT update operation finished.\n");
}

void AscendIndexIVFSQTImpl::trainCpuQuantizer(size_t listLen, int subnlist, float *baseData, int thisThread,
    faiss::IndexFlat &subCpuQuantizer)
{
    faiss::ClusteringParameters cp;
    cp.niter = KMEANS_ITER;
    cp.min_points_per_centroid = KMEANS_MIN_POINTS_PER_CENTROID;
    cp.max_points_per_centroid = static_cast<int>(listLen);

    faiss::Clustering clus(dimIn, subnlist, cp);
    clus.verbose = false;
    clus.train(MAX_CLUS_POINTS, baseData + thisThread * MAX_CLUS_POINTS * dimIn, subCpuQuantizer);
}

void AscendIndexIVFSQTImpl::cpuQuantizerAssignWithLargeListLen(AssignParam &param, int &subnlist)
{
    std::vector<int> pickedBaseIds;
    for (size_t j = 0; j < param.oriListLen; j++) {
        if (param.upToNumThres && isToDevice(param.oribaseIds[j])) {
            continue;
        }
        pickedBaseIds.push_back(param.oribaseIds[j]);
    }
    size_t sampleStride = pickedBaseIds.size() / MAX_CLUS_POINTS;
    size_t j = 0;
    for (size_t cnt = 0; cnt < MAX_CLUS_POINTS; cnt++) {
        size_t hostId =
            static_cast<size_t>((static_cast<size_t>(pickedBaseIds[j]) >= numHostThreshold) ?
                idDictHost[pickedBaseIds[j]] : pickedBaseIds[j]);
        size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
        size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;

        std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
            globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
            param.quantCodes.data() + (param.threadNum * MAX_CLUS_POINTS + cnt) * dimIn);
        j += sampleStride;
    }

    sqIn.decode(param.quantCodes.data() + param.threadNum * MAX_CLUS_POINTS * dimIn,
        param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, MAX_CLUS_POINTS);

    subnlist = ceil(static_cast<float>(param.listLen) / static_cast<float>(lowerBound));
    subnlist = (subnlist > subcenterNum) ? subcenterNum : subnlist;
    trainCpuQuantizer(param.listLen, subnlist, param.baseData, param.threadNum, param.subCpuQuantizer);

    size_t segNums = (param.listLen + MAX_CLUS_POINTS - 1) / MAX_CLUS_POINTS;

    for (size_t s = 0; s < segNums; s++) {
        size_t actualSegSize = 0;
        for (size_t j = s * MAX_CLUS_POINTS; j < std::min((s + 1) * MAX_CLUS_POINTS, param.listLen); j++) {
            size_t hostId = static_cast<size_t>((static_cast<size_t>(pickedBaseIds[j]) >= numHostThreshold) ?
                idDictHost[pickedBaseIds[j]] : pickedBaseIds[j]);
            size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
            size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;

            std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
                globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
                param.quantCodes.data() + (param.threadNum * MAX_CLUS_POINTS + j - s * MAX_CLUS_POINTS) * dimIn);
            actualSegSize += 1;
        }
        sqIn.decode(param.quantCodes.data() + param.threadNum * MAX_CLUS_POINTS * dimIn,
            param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, actualSegSize);
        std::vector<faiss::idx_t> labelSub(actualSegSize, 0);
        param.subCpuQuantizer.assign(
            actualSegSize, param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, labelSub.data());
        param.label.insert(param.label.end(), labelSub.begin(), labelSub.end());
        for (size_t j = 0; j < labelSub.size(); j++) {
            param.values[labelSub[j] + subcenterNum * param.threadNum]++;
        }
    }
}

void AscendIndexIVFSQTImpl::cpuQuantizerAssign(AssignParam &param, int &subnlist)
{
    if (param.listLen > MAX_CLUS_POINTS) {
        return cpuQuantizerAssignWithLargeListLen(param, subnlist);
    }

    // If need to train Subcenter, dequant globalInCodes to get train data.
    param.label.resize(param.listLen, 0);
    size_t realJ = 0;
    for (size_t j = 0; j < param.oriListLen; j++) {
        if (param.upToNumThres && isToDevice(param.oribaseIds[j])) {
            continue;
        }
        size_t hostId = static_cast<size_t>((static_cast<size_t>(param.oribaseIds[j]) >= numHostThreshold) ?
            idDictHost[param.oribaseIds[j]] : param.oribaseIds[j]);
        size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
        size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;
        std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
            globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
            param.quantCodes.data() + (param.threadNum * MAX_CLUS_POINTS + realJ) * dimIn);
        realJ += 1;
    }
    sqIn.decode(param.quantCodes.data() + param.threadNum * MAX_CLUS_POINTS * dimIn,
        param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, param.listLen);
    subnlist = static_cast<int>(ceil(float(param.listLen) / float(lowerBound)));
    subnlist = (subnlist > subcenterNum) ? subcenterNum : subnlist;
    faiss::ClusteringParameters cp;
    cp.niter = KMEANS_ITER;
    cp.min_points_per_centroid = KMEANS_MIN_POINTS_PER_CENTROID;
    cp.max_points_per_centroid = static_cast<int>(param.listLen);

    faiss::Clustering clus(dimIn, subnlist, cp);
    clus.verbose = false;
    clus.train(param.listLen, param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, param.subCpuQuantizer);
    param.subCpuQuantizer.assign(
        param.listLen, param.baseData + param.threadNum * MAX_CLUS_POINTS * dimIn, param.label.data());

    for (size_t j = 0; j < param.listLen; j++) {
        param.values[param.label[j] + subcenterNum * param.threadNum]++;
    }
}

size_t AscendIndexIVFSQTImpl::getListLen(size_t curListId, bool upToNumThres, size_t oriListLen,
    const std::vector<int> &oribaseIds)
{
    size_t listLen = oriListLen;
    for (size_t j = 0; j < oriListLen; j++) {
        if (upToNumThres && isToDevice(oribaseIds[j])) {
            assignCountsDevice[curListId].push_back(idDictglobalInCodesDevice[oribaseIds[j]]);
            listLen--;
            continue;
        }
    }
    return listLen;
}

void AscendIndexIVFSQTImpl::reassignLabels(AssignParam &param, const std::vector<faiss::idx_t> &toPreserve,
    const std::vector<faiss::idx_t> &toMerge)
{
    std::unique_ptr<faiss::IndexFlat> cpuQuantizerC = std::make_unique<faiss::IndexFlatL2>(dimIn);
    for (size_t j = 0; j < toPreserve.size(); j++) {
        cpuQuantizerC->add(1, param.subCpuQuantizer.get_xb() + toPreserve[j] * dimIn);
    }
    std::vector<float> reSubCenter(subcenterNum * dimIn, PADDING_NUM);
    std::vector<bool> relativeSeen(toPreserve.size(), false);
    for (size_t j = 0; j < toMerge.size(); j++) {
        std::unique_ptr<idx_t[]> subRelativeLabel = std::make_unique<idx_t[]>(1);

        cpuQuantizerC->assign(1, param.subCpuQuantizer.get_xb() + toMerge[j] * dimIn, subRelativeLabel.get());
        auto subLabel = toPreserve[subRelativeLabel[0]];
        int labelK = 0;
        for (size_t k = 0; k < param.oriListLen; k++) {
            if (param.upToNumThres && isToDevice(param.oribaseIds[k])) {
                continue;
            }
            param.label[labelK] = (param.label[labelK] == toMerge[j]) ? subLabel : param.label[labelK];
            labelK++;
        }

        int mSize = param.values[toMerge[j] + subcenterNum * param.threadNum];
        int pSize = param.values[subLabel + subcenterNum * param.threadNum];
        for (int d = 0; d < dimIn; d++) {
            reSubCenter[subLabel * dimIn + d] = (*(param.subCpuQuantizer.get_xb() + subLabel * dimIn + d) * pSize +
                *(param.subCpuQuantizer.get_xb() + toMerge[j] * dimIn + d) * mSize) / (mSize + pSize);
        }
        relativeSeen[subLabel] = true;
    }
    for (size_t j = 0; j < toPreserve.size(); j++) {
        if (relativeSeen[toPreserve[j]] == true) {
            continue;
        }
        for (int d = 0; d < dimIn; d++) {
            reSubCenter[toPreserve[j] * dimIn + d] = *(param.subCpuQuantizer.get_xb() + toPreserve[j] * dimIn + d);
        }
    }
    int labelJ = 0;
    for (size_t j = 0; j < param.oriListLen; j++) {
        if (param.upToNumThres && isToDevice(param.oribaseIds[j])) {
            continue;
        }
        allLabels[param.listId * subcenterNum + param.label[labelJ]].push_back(param.oribaseIds[j]);
        labelJ += 1;
    }
    std::copy(reSubCenter.begin(), reSubCenter.end(), adaptiveSubCenters.data() + param.listId * subcenterNum * dimIn);
}

void AscendIndexIVFSQTImpl::trainHostBaseSubClusterFor310(int numThreads)
{
    APP_LOG_INFO("AscendIndexIVFSQT trainHostBaseSubCluster operation started.\n");
    allLabels.clear();
    allLabels.resize(nlist * subcenterNum);
    replaceCurId -= strideHost;
    bool upToNumThres = ((size_t)this->intf_->ntotal > numHostThreshold) ? true : false;

    int curNumThreads = omp_get_max_threads();
    FAISS_THROW_IF_NOT_MSG(numThreads > 0, "thread num is 0");

    if (numThreads > std::min(MAX_OMP_THREAD_NUM, curNumThreads)) {
        APP_LOG_ERROR("AscendIndexIVFSQT trainHostBaseSubCluster omp max threads exceeds the limit, use the default");
        numThreads = std::min(MAX_OMP_THREAD_NUM, curNumThreads);
    }

    std::vector<int> values(subcenterNum * numThreads, 0);
    std::vector<uint8_t> quantCodes(MAX_CLUS_POINTS * dimIn * numThreads, 0);
    std::unique_ptr<float[]> baseData = std::make_unique<float[]>(MAX_CLUS_POINTS * dimIn * numThreads);

#pragma omp parallel for num_threads(numThreads)
    for (auto i = 0; i < nlist; i++) {
        int thisThread = omp_get_thread_num();
        std::vector<int> &oribaseIds = bucketDetails[i];
        size_t orilistLen = oribaseIds.size();
        size_t listLen = getListLen(i, upToNumThres, orilistLen, oribaseIds);
        if (listLen == 0) {
            if (assignCountsDevice[i].size() > 0) {
                std::copy(pQuantizerImpl->cpuQuantizer->get_xb() + i * dimIn,
                    pQuantizerImpl->cpuQuantizer->get_xb() + (i + 1) * dimIn,
                    adaptiveSubCenters.data() + i * subcenterNum * dimIn);
            }
            continue;
        }
        if (listLen <= (size_t)lowerBound) {
            std::copy(pQuantizerImpl->cpuQuantizer->get_xb() + i * dimIn,
                pQuantizerImpl->cpuQuantizer->get_xb() + (i + 1) * dimIn,
                adaptiveSubCenters.data() + i * subcenterNum * dimIn);
            fillingAllLabels(i, orilistLen, upToNumThres, oribaseIds);
            continue;
        }

        if (listLen >= DEVICE_PAGE_SIZE * subcenterNum) {
            continue;
        }

        std::vector<faiss::idx_t> label;
        label.reserve(listLen);
        std::unique_ptr<faiss::IndexFlat> subCpuQuantizer(new faiss::IndexFlatL2(dimIn));
        int subnlist = subcenterNum;
        AssignParam param { static_cast<size_t>(i), listLen, orilistLen, thisThread, upToNumThres, baseData.get(),
            oribaseIds, quantCodes, *subCpuQuantizer, label, values };
        cpuQuantizerAssign(param, subnlist);

        // Merge small buckets
        std::vector<faiss::idx_t> toMerge;
        std::vector<faiss::idx_t> toPreserve;
        for (int j = 0; j < subnlist; j++) {
            (values[j + subcenterNum * thisThread] <= mergeThres) ? toMerge.push_back(j) : toPreserve.push_back(j);
        }

        if (toMerge.size() == 0) {
            std::copy(subCpuQuantizer->get_xb(),
                subCpuQuantizer->get_xb() + subCpuQuantizer->codes.size() / sizeof(float),
                adaptiveSubCenters.data() + i * subcenterNum * dimIn);
            int labelJ = 0;
            for (size_t j = 0; j < orilistLen; j++) {
                if (upToNumThres && isToDevice(oribaseIds[j])) {
                    continue;
                }
                allLabels[i * subcenterNum + label[labelJ]].push_back(oribaseIds[j]);
                labelJ += 1;
            }
            continue;
        }
        if (toMerge.size() == (size_t)subnlist) {
            std::copy(pQuantizerImpl->cpuQuantizer->get_xb() + i * dimIn,
                pQuantizerImpl->cpuQuantizer->get_xb() + (i + 1) * dimIn,
                adaptiveSubCenters.data() + i * subcenterNum * dimIn);
            int labelJ = 0;
            for (size_t j = 0; j < orilistLen; j++) {
                if (upToNumThres && isToDevice(oribaseIds[j])) {
                    continue;
                }
                allLabels[i * subcenterNum].push_back(oribaseIds[j]);
                labelJ += 1;
            }
            continue;
        }

        // Reassign labels
        reassignLabels(param, toPreserve, toMerge);
    }

    APP_LOG_INFO("AscendIndexIVFSQT trainHostBaseSubCluster operation finished.\n");
}

void AscendIndexIVFSQTImpl::updateBaseOffsetMask(int devIdx)
{
    APP_LOG_INFO("AscendIndexIVFSQT updateBaseOffsetMask operation started.\n");
    if (!this->intf_->is_trained) {
        return;
    }

    int deviceId = indexConfig.deviceList[devIdx];
    auto pIndex = getActualIndex(deviceId);
    devIdx = 0;

    for (int listId = 0; listId < nlist; listId++) {
        std::vector<int> baseOffsetI = devSubBucketOffset[listId][devIdx];
        std::vector<int> baseNumI = subBucketNum[listId][devIdx];
        pIndex->receiveBaseOffsetNum(baseOffsetI.size(), listId,  baseOffsetI.data(), baseNumI.data());
        if (listId == nlist - 1) {
            pIndex->getBaseMaskSeg();
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT updateBaseOffsetMask operation finished.\n");
}

size_t AscendIndexIVFSQTImpl::remove_ids(const faiss::IDSelector &sel)
{
    std::string msg = std::string("remove_ids() not implemented for this type of index, check member: ") +
        std::to_string(sel.is_member(0));
    FAISS_THROW_MSG(msg);
    return 0;
}

float AscendIndexIVFSQTImpl::getQMin() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return qMin;
}

float AscendIndexIVFSQTImpl::getQMax() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return qMax;
}

std::shared_ptr<::ascend::Index> AscendIndexIVFSQTImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFSQT  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    using namespace ::ascend;
    std::shared_ptr<::ascend::Index> index = std::make_shared<IndexIVFSQTIPAicpu>(nlist, dimIn, dimOut,
                                                                                  nprobe, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to create index ivfsqt:%d", ret);
    APP_LOG_INFO("AscendIndexIVFSQT createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFSQTImpl::checkSearchParams(int nprobe, int l2Probe, int l3SegmentNum) const
{
    FAISS_THROW_IF_NOT_FMT(NPROBE_NUMS.find(nprobe) != NPROBE_NUMS.end(), "Please Check nprobe[%d] is legal", nprobe);
    FAISS_THROW_IF_NOT_FMT(L3_SEG_SIZES.find(l3SegmentNum) != L3_SEG_SIZES.end(),
        "Please Check l3SegmentNum[%d] is legal", l3SegmentNum);

    FAISS_THROW_IF_NOT_FMT(l2Probe >= nprobe,
        "Please Check l2Probe[%d] is larger than or equal to nprobe[%d]", l2Probe, nprobe);
    FAISS_THROW_IF_NOT_FMT(l2Probe <= l3SegmentNum,
        "Please Check l2Probe[%d] is less than or equal to l3SegmentNum[%d]", l2Probe, l3SegmentNum);
    FAISS_THROW_IF_NOT_FMT(l2Probe <= nprobe * subcenterNum,
        "Please Check l2Probe[%d] is less than or equal to nprobe * subcenterNum [%d]", l2Probe, nprobe * subcenterNum);
}

void AscendIndexIVFSQTImpl::setSearchParams(int nprobe, int l2Probe, int l3SegmentNum)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT setSearchParams operation started.\n");

    checkSearchParams(nprobe, l2Probe, l3SegmentNum);

    for (const auto &index : indexes) {
        auto actualIndex = getActualIndex(index.first);
        actualIndex->setSearchParams(nprobe, l2Probe, l3SegmentNum);
    }

    this->nprobe = nprobe;
    this->l2NProbe = l2Probe;
    this->l3SegmentNum = l3SegmentNum;
    APP_LOG_INFO("AscendIndexIVFSQT setSearchParams operation finished.\n");
}

void AscendIndexIVFSQTImpl::setSortMode(int mode)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT setSortMode operation started.\n");
    FAISS_THROW_IF_NOT_FMT(SORT_MODES.find(mode) != SORT_MODES.end(),
        "Please Check mode[%d] is legal, mode must be 0 or 1", mode);
    ivfFuzzyTopkMode = mode;
    for (const auto &index : indexes) {
        auto actualIndex = getActualIndex(index.first);
        actualIndex->setSortMode(mode);
    }
    APP_LOG_INFO("AscendIndexIVFSQT setSortMode operation finished.\n");
}

void AscendIndexIVFSQTImpl::setUseCpuUpdate(int numThreads)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT setUseCpuUpdate operation started.\n");
    isCpuUpdated = true;
    defaultnumThreads = numThreads;
    APP_LOG_INFO("AscendIndexIVFSQT setUseCpuUpdate operation finished.\n");
}

void AscendIndexIVFSQTImpl::updateTParams(int l2Probe, int l3Segment)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT updateTParams operation started.\n");
    
    checkSearchParams(nprobe, l2Probe, l3Segment);

    l2NProbe = l2Probe;
    l3SegmentNum = l3Segment;
    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        pIndex->updateTParams(l2Probe, l3SegmentNum);
    }
    APP_LOG_INFO("AscendIndexIVFSQT updateTParams operation finished.\n");
}

template <typename T, typename D>
void AscendIndexIVFSQTImpl::insertFragmentation(int n, const T *data, int stride,
    std::vector<std::shared_ptr<std::vector<D>>> &dest)
{
    for (size_t i = 0; i < (size_t)n; i++) {
        size_t destCurSize = dest.size();
        if ((static_cast<size_t>(this->intf_->ntotal) + i) >= GLOBAL_CODES_PAGE_SIZE * destCurSize) {
            std::vector<D> initVec;
            initVec.reserve(GLOBAL_CODES_PAGE_SIZE * stride);
            dest.push_back(std::move(std::make_shared<std::vector<D>>(std::move(initVec))));
            dest[destCurSize]->insert(dest[destCurSize]->end(), data + i * stride, data + (i + 1) * stride);
        } else {
            dest[destCurSize - 1]->insert(dest[destCurSize - 1]->end(), data + i * stride, data + (i + 1) * stride);
        }
    }
}

void AscendIndexIVFSQTImpl::getFuzzyList(size_t n, const float *x, std::vector<idx_t> &resultSearchId,
    int CPU_AVAIL)
{
    std::shared_ptr<AscendThreadPool> poolFuzzy;
    poolFuzzy = std::make_shared<AscendThreadPool>(CPU_AVAIL);

    APP_LOG_INFO("AscendIndexIVFSQFuzzy getFuzzyList operation started.\n");
    std::unique_ptr<idx_t[]> searchId = std::make_unique<idx_t[]>(n * fuzzyK);
    std::unique_ptr<float[]> distances = std::make_unique<float[]>(n * fuzzyK);

    std::unique_ptr<int8_t[]> xInt8 = std::make_unique<int8_t[]>(n * static_cast<size_t>(dimIn));

    int fragN = (static_cast<int>(n) + CPU_AVAIL - 1) / CPU_AVAIL;

    auto searchFunctor = [&](int idx) {
        ::ascend::CommonUtils::attachToCpu(idx);
        int actualSize = std::min(fragN, int(n) - idx * fragN);
        if (actualSize > 0) {
            size_t start = (size_t)idx * (size_t)fragN;
            if (this->ivfsqtConfig.useKmeansPP && this->fuzzyType != TYPE_FAST) {
                float qMin_ = this->qMin;
                float qMax_ = this->qMax;

                for (size_t i = 0; i < size_t(actualSize * dimIn); i++) {
                    if (*(x + i + start * (size_t)dimIn) < qMin_) {
                        *(xInt8.get() + i + start * (size_t)dimIn) = static_cast<int8_t>(INT8_LOWER_BOUND);
                    } else if (*(x + i + start * (size_t)dimIn) > qMax_) {
                        *(xInt8.get() + i + start * (size_t)dimIn) = static_cast<int8_t>(INT8_UPPER_BOUND);
                    } else {
                        // 1e-6 is a minimum value to prevent division by 0
                        *(xInt8.get() + i + start * (size_t)dimIn) = static_cast<int8_t>(
                            ((*(x + start * (size_t)dimIn + i) - qMin_) / (qMax_ - qMin_ + 1e-6) * UINT8_UPPER_BOUND) +
                            INT8_LOWER_BOUND);
                    }
                }
            }
        }
    };

    std::vector<std::future<void>> searchFunctorRet;
    for (size_t idx = 0; idx < (size_t)CPU_AVAIL; idx++) {
        searchFunctorRet.emplace_back(poolFuzzy->Enqueue(searchFunctor, idx));
    }
    int updateWait = 0;
    try {
        for_each(searchFunctorRet.begin(), searchFunctorRet.end(), [&](std::future<void> &ret) {
            updateWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        for_each(searchFunctorRet.begin() + updateWait, searchFunctorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for cent update functor failed %s", e.what());
    }

    pQuantizerImpl->npuInt8Quantizer->searchInt8(n, xInt8.get(), static_cast<idx_t>(fuzzyK),
        distances.get(), searchId.get());

    // sort distance from small to big and remain with threshold
    sortDistDesc(n, resultSearchId, searchId.get(), distances.get());
    APP_LOG_INFO("AscendIndexIVFSQFuzzy getFuzzyList operation finished.\n");
}


void AscendIndexIVFSQTImpl::getFuzzyList(size_t n, const float *x, std::vector<idx_t> &resultSearchId)
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy getFuzzyList operation started.\n");

    std::unique_ptr<idx_t[]> searchId = std::make_unique<idx_t[]>(n * fuzzyK);

    std::unique_ptr<float[]> distances = std::make_unique<float[]>(n * fuzzyK);

    if (this->ivfsqtConfig.useKmeansPP && this->fuzzyType != TYPE_FAST) {
        float qMin_ = getQMin();
        float qMax_ = getQMax();

        std::unique_ptr<int8_t[]> xInt8 = std::make_unique<int8_t[]>(n * static_cast<size_t>(dimIn));
        for (size_t i = 0; i < n * (size_t)dimIn; i++) {
            if (*(x + i) < qMin_) {
                *(xInt8.get() + i) = static_cast<int8_t>(INT8_LOWER_BOUND);
            } else if (*(x + i) > qMax_) {
                *(xInt8.get() + i) = static_cast<int8_t>(INT8_UPPER_BOUND);
            } else {
                *(xInt8.get() + i) =
                    static_cast<int8_t>(((*(x + i) - qMin_) / (qMax_ - qMin_) * UINT8_UPPER_BOUND) + INT8_LOWER_BOUND);
            }
        }
        if (deviceNpuType == 0) {
            pQuantizerImpl->npuQuantizer->search(n, x, fuzzyK, distances.get(), searchId.get());
        } else {
            pQuantizerImpl->npuInt8Quantizer->searchInt8(n, xInt8.get(), fuzzyK, distances.get(), searchId.get());
        }
    } else {
        pQuantizerImpl->cpuQuantizer->search(n, x, fuzzyK, distances.get(), searchId.get());
    }
    // sort distance from small to big and remain with threshold
    sortDistDesc(n, resultSearchId, searchId.get(), distances.get());

    APP_LOG_INFO("AscendIndexIVFSQFuzzy getFuzzyList operation finished.\n");
}
void AscendIndexIVFSQTImpl::sortDistDesc(size_t n, std::vector<idx_t> &resultSearchId,
                                         const idx_t *searchId, const float *distances) const
{
    size_t remainNumber = static_cast<size_t>(n * threshold);
    size_t sortCount = n * (fuzzyK - 1);
    FAISS_THROW_IF_NOT_MSG(sortCount >= remainNumber, "FuzzyK and threshold mismatch");

    std::unique_ptr<float[]> sortDis = std::make_unique<float[]>(sortCount);
    std::unique_ptr<idx_t[]> sortId = std::make_unique<idx_t[]>(sortCount);

#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < fuzzyK - 1; k++) {
            if (fuzzyType != TYPE_FAST) {
                sortDis[n * k + i] = distances[i * fuzzyK + k + 1] / (distances[i * fuzzyK] + EPSILON);
            } else {
                sortDis[n * k + i] = distances[i * fuzzyK] / (distances[i * fuzzyK + k + 1] + EPSILON);
            }
            sortId[n * k + i] = searchId[i * fuzzyK + k + 1];
        }
    }

    std::vector<idx_t> idx(sortCount);
    fvec_argsort(sortCount, sortDis.get(), (size_t *)idx.data());

#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < n; i++) {
        resultSearchId[i] = searchId[i * fuzzyK];
    }

#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < remainNumber; i++) {
        resultSearchId[n + idx[i]] = sortId[idx[i]];
    }
}

void AscendIndexIVFSQTImpl::compressProc(int n, const float *x, const idx_t *ids, uint8_t *codesIn,
    std::vector<idx_t> &resultSearchId)
{
    std::string compressError = "";
    std::thread compressThreaed([&]() {
        std::unique_ptr<uint8_t[]> codesOut = std::make_unique<uint8_t[]>(static_cast<size_t>(n) * sq.code_size);
        std::unique_ptr<float[]> compressX = std::make_unique<float[]>(static_cast<size_t>(n) *
            static_cast<size_t>(dimOut));

        idDictDevice.resize(this->intf_->ntotal + n);
        idDictHost.resize(this->intf_->ntotal + n);
        idDictglobalInCodesDevice.resize(this->intf_->ntotal + n);

        sqIn.compute_codes(x, codesIn, n);
        insertFragmentation(n, ids, 1, globalIdsPointer);
        compress(n, x, compressX.get());
        normalize(n, compressX.get());
        sq.compute_codes(compressX.get(), codesOut.get(), n);
        insertFragmentation(n, codesOut.get(), sq.code_size, globalOutCodesPointer);
    });

    try {
        // get bucket Results of each value
        if ((this->ivfsqtConfig.useKmeansPP && this->fuzzyType != TYPE_FAST) && ((static_cast<size_t>(n) >
            (this->indexConfig.deviceList.size() - 1) * (this->indexConfig.deviceList.size() - 1))) &&
            (deviceNpuType != 0)) {
            getFuzzyList(n, x, resultSearchId, CPU_AVAIL);
        } else {
            getFuzzyList(n, x, resultSearchId);
        }
    } catch (std::exception &e) {
        compressThreaed.join();
        ASCEND_THROW_FMT("wait for add functor failed %s", e.what());
    }
    compressThreaed.join();
    if (!compressError.empty()) {
        ASCEND_THROW_FMT("wait for compress functor failed %s", compressError.c_str());
    }
}

void AscendIndexIVFSQTImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    // v_i = (code_i + 0.5) / 255 * vdiff_i + vmin_i (reconstruct code to float)
    // save origin data in uint8_t
    APP_LOG_INFO("AscendIndexIVFSQT addImpl operation started: n=%d.\n", n);
    if (isUpdated) {
        FAISS_THROW_IF_NOT_MSG(isUpdated == false, "Add after Update is not supported");
        return;
    }
    FAISS_THROW_IF_NOT_MSG(!resetAfterUpdate, "Add after update and reset is not supported");
    size_t hostI = 0;
    size_t deviceI = 0;
    std::unique_ptr<uint8_t[]> codesIn = std::make_unique<uint8_t[]>(static_cast<size_t>(n) * sqIn.code_size);
    // 3. compute the sq codes
    std::vector<idx_t> resultSearchId(static_cast<size_t>(n) * fuzzyK, -1);
    compressProc(n, x, ids, codesIn.get(), resultSearchId);

    size_t hostStopSign = 0;
    if ((size_t)this->intf_->ntotal < numHostThreshold) {
        hostStopSign = std::min((size_t)n, numHostThreshold - this->intf_->ntotal);
    }

    for (size_t i = 0; i < hostStopSign; i++) {
        idx_t listIdHost = resultSearchId[i];
        FAISS_THROW_IF_NOT(listIdHost >= 0 && listIdHost < this->nlist);
        bucketDetails[listIdHost].push_back(this->intf_->ntotal + i);
        hostI += 1;
        fuzzyTotal += 1;
        for (size_t fuzzy = 1; fuzzy < fuzzyK; fuzzy++) {
            listIdHost = resultSearchId[static_cast<size_t>(n) * fuzzy + i];
            if (listIdHost != -1) {
                fuzzyTotal += 1;
                FAISS_THROW_IF_NOT(listIdHost >= 0 && listIdHost < this->nlist);
                bucketDetails[listIdHost].push_back(this->intf_->ntotal + i);
            }
        }
    }

    if (hostStopSign > 0) {
        insertFragmentation(hostStopSign, codesIn.get(), sqIn.code_size, globalInCodesPointer);
    }

    std::vector<uint8_t> codesVec;
    codesVec.reserve((n - hostStopSign) * dimIn);
    for (size_t i = hostStopSign; i < (size_t)n; i++) {
        if ((static_cast<size_t>(this->intf_->ntotal) + i) % strideDevice == 0 && replaceCurId < numHostThreshold) {
            idx_t listIdHost = resultSearchId[i];
            FAISS_THROW_IF_NOT(listIdHost >= 0 && listIdHost < this->nlist);
            bucketDetails[listIdHost].push_back(this->intf_->ntotal + i);
            size_t replacePageId = replaceCurId / GLOBAL_CODES_PAGE_SIZE;
            size_t pageOffset = replaceCurId % GLOBAL_CODES_PAGE_SIZE;

            codesVec.insert(codesVec.end(), globalInCodesPointer[replacePageId]->data() + pageOffset * dimIn,
                globalInCodesPointer[replacePageId]->data() + (pageOffset + 1) * dimIn);

            std::copy(codesIn.get() + i * dimIn, codesIn.get() + (i + 1) * dimIn,
                globalInCodesPointer[replacePageId]->data() + pageOffset * sqIn.code_size);
            fuzzyTotal += 1;
            for (size_t fuzzy = 1; fuzzy < fuzzyK; fuzzy++) {
                listIdHost = resultSearchId[static_cast<size_t>(n) * fuzzy + i];
                if (listIdHost != -1) {
                    fuzzyTotal += 1;
                    FAISS_THROW_IF_NOT(listIdHost >= 0 && listIdHost < this->nlist);
                    bucketDetails[listIdHost].push_back(this->intf_->ntotal + i);
                }
            }
            idDictDevice[this->deviceNtotal + deviceI] = static_cast<int>(replaceCurId);
            idDictHost[static_cast<size_t>(this->intf_->ntotal) + i] = static_cast<int>(replaceCurId);
            idDictglobalInCodesDevice[replaceCurId] = static_cast<int>(this->deviceNtotal + deviceI);
            replaceCurId += strideHost;
            deviceI += 1;
        } else {
            idx_t listDevice = resultSearchId[i];
            FAISS_THROW_IF_NOT(listDevice >= 0 && listDevice < this->nlist);
            codesVec.insert(codesVec.end(), codesIn.get() + i * dimIn, codesIn.get() + (i + 1) * dimIn);
            assignCountsDevice[listDevice].push_back(this->deviceNtotal + deviceI);
            idDictDevice[this->deviceNtotal + deviceI] = this->intf_->ntotal + static_cast<int>(i);
            fuzzyTotal += 1;
            for (size_t fuzzy = 1; fuzzy < fuzzyK; fuzzy++) {
                listDevice = resultSearchId[static_cast<size_t>(n) * fuzzy + i];
                if (listDevice != -1) {
                    fuzzyTotal += 1;
                    assignCountsDevice[listDevice].push_back(this->deviceNtotal + deviceI);
                }
            }
            deviceI += 1;
        }
    }
    int deviceId = indexConfig.deviceList[0];
    auto pIndex = getActualIndex(deviceId);
    if (codesVec.size() > 0) {
        size_t tileSize = getSendDeviceSize(deviceI);
        for (size_t i = 0; i < deviceI; i += tileSize) {
            int curNum = std::min(tileSize, deviceI - i);
            auto ret = pIndex->addTmpDeviceData(curNum, dimIn, codesVec.data() + i * dimIn);
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,  "addTmpDeviceData failed, ret: %d", ret);
        }
    }
    this->hostNtotal += hostI;
    this->deviceNtotal += deviceI;
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexIVFSQT addImpl operation finished.\n");
}

inline bool AscendIndexIVFSQTImpl::isToDevice(int id)
{
    return ((size_t)id <= replaceCurId) && ((size_t)id % strideHost) == 0;
}

void AscendIndexIVFSQTImpl::trainSubClusterMultiDevices()
{
    replaceCurId -= strideHost;
    size_t deviceCnt = this->ivfsqtConfig.deviceList.size();
    allLabels.clear();
    allLabels.resize(nlist * subcenterNum);

    int listsPerDeviceLen = (nlist + static_cast<int>(deviceCnt) - 1) / static_cast<int>(deviceCnt);

    auto updateFunctor = [&](int deviceId) {
        ::ascend::CommonUtils::attachToCpu(deviceId);
        int start = listsPerDeviceLen * deviceId;
        int end = std::min(nlist, listsPerDeviceLen * (deviceId + 1));
        std::vector<int> listsPerDevice = { start, end };
        trainHostBaseSubCluster(listsPerDevice, deviceId);
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, updateFunctor);
}

void AscendIndexIVFSQTImpl::fillingTrainedData(std::vector<uint16_t> &trainedFp16, uint16_t *vmin, uint16_t *vdiff,
    size_t length)
{
    switch (sq.qtype) {
        case faiss::ScalarQuantizer::QT_8bit:
            transform(begin(sqIn.trained), end(sqIn.trained), begin(trainedFp16),
                [](float trainedVal) { return fp16(trainedVal).data; });
            break;
        case faiss::ScalarQuantizer::QT_8bit_uniform:
            for (size_t i = 0; i < length; i++) {
                vmin[i] = fp16(sqIn.trained[0]).data; // 均值vmin存储在元素0
                vdiff[i] = fp16(sqIn.trained[1]).data; // 均值vdiff存储在元素1
            }
            break;
        default:
            FAISS_THROW_FMT("not supportted qtype(%d).", sqIn.qtype);
            break;
    }
}

void AscendIndexIVFSQTImpl::addNpuClusByListLen(AddNpuClusInputParam &inputParam, size_t curBuck,
    size_t orilistLen, std::vector<int> &oribaseIds, AddNpuClusOutputParam &outputParam)
{
    size_t listLen = outputParam.listLenBucks[curBuck];
    auto &quantCodes = inputParam.quantCodes;
    if (listLen > MAX_CLUS_POINTS) {
        std::vector<int> pickedBaseIds;
        for (size_t j = 0; j < orilistLen; j++) {
            if (inputParam.upToNumThres && isToDevice(oribaseIds[j])) {
                continue;
            }
            pickedBaseIds.push_back(oribaseIds[j]);
        }
        size_t sampleStride = pickedBaseIds.size() / MAX_CLUS_POINTS;
        for (size_t cnt = 0, j = 0; cnt < MAX_CLUS_POINTS; cnt++) {
            size_t hostId = (static_cast<size_t>(pickedBaseIds[j]) >= numHostThreshold) ?
                static_cast<size_t>(idDictHost[pickedBaseIds[j]]) : static_cast<size_t>(pickedBaseIds[j]);
            size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
            size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;

            std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
                      globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
                      quantCodes.data() + cnt * dimIn + inputParam.offset * curBuck);
            j += sampleStride;
        }

        int subnlist = ceil(static_cast<float>(listLen) / static_cast<float>(lowerBound));
        subnlist = (subnlist > subcenterNum) ? subcenterNum : subnlist;
        outputParam.batchSubNlist += static_cast<size_t>(subnlist);

        outputParam.npuClus.SubClusAddInt8(MAX_CLUS_POINTS, quantCodes.data() + inputParam.offset * curBuck,
            curBuck, subnlist);

        outputParam.batchLen += MAX_CLUS_POINTS;
        return;
    }

    size_t realJ = 0;
    for (size_t j = 0; j < orilistLen; j++) {
        if (inputParam.upToNumThres && isToDevice(oribaseIds[j])) {
            continue;
        }
        size_t hostId = (static_cast<size_t>(oribaseIds[j]) >= numHostThreshold) ?
            static_cast<size_t>(idDictHost[oribaseIds[j]]) : static_cast<size_t>(oribaseIds[j]);
        size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
        size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;
        std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
            globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
            quantCodes.data() + realJ * dimIn + inputParam.offset * curBuck);
        realJ += 1;
    }
    int subnlist = static_cast<int>(ceil(float(listLen) / float(lowerBound)));
    subnlist = (subnlist > subcenterNum) ? subcenterNum : subnlist;
    outputParam.batchSubNlist += static_cast<size_t>(subnlist);

    outputParam.npuClus.SubClusAddInt8(listLen, quantCodes.data() + inputParam.offset * curBuck, curBuck, subnlist);

    outputParam.batchLen += static_cast<int>(listLen);
}

void AscendIndexIVFSQTImpl::fillingAllLabels(size_t listId, size_t oriListLen, bool upToNumThres,
    std::vector<int> &oribaseIds)
{
    for (size_t j = 0; j < oriListLen; j++) {
        if (upToNumThres && isToDevice(oribaseIds[j])) {
            continue;
        }
        allLabels[listId * subcenterNum].push_back(oribaseIds[j]);
    }
}

void AscendIndexIVFSQTImpl::addNpuClus(AddNpuClusInputParam &inputParam, AddNpuClusOutputParam &outputParam)
{
    for (size_t i = 0; i < inputParam.numBuck; i++) {
        size_t listId = inputParam.curBatchLoop * static_cast<size_t>(inputParam.numBuck) + i +
            static_cast<size_t>(inputParam.startListId);
        if (listId >= static_cast<size_t>(nlist)) {
            break;
        }
        std::vector<int> &oribaseIds = bucketDetails[listId];
        size_t oriListLen = oribaseIds.size();
        size_t listLen = oriListLen;
        for (size_t j = 0; j < oriListLen; j++) {
            if (inputParam.upToNumThres && isToDevice(oribaseIds[j])) {
                assignCountsDevice[listId].push_back(idDictglobalInCodesDevice[oribaseIds[j]]);
                listLen--;
                continue;
            }
        }
        outputParam.listLenBucks[i] = listLen;
        if (listLen == 0) {
            if (assignCountsDevice[listId].size() > 0) {
                std::copy(pQuantizerImpl->cpuQuantizer->get_xb() + listId * dimIn,
                    pQuantizerImpl->cpuQuantizer->get_xb() + (listId + 1) * dimIn,
                    adaptiveSubCenters.data() + listId * subcenterNum * dimIn);
            }
            continue;
        }
        if (listLen < static_cast<size_t>(lowerBound)) {
            std::copy(pQuantizerImpl->cpuQuantizer->get_xb() + listId * dimIn,
                pQuantizerImpl->cpuQuantizer->get_xb() + (listId + 1) * dimIn,
                adaptiveSubCenters.data() + listId * subcenterNum * dimIn);
            fillingAllLabels(listId, oriListLen, inputParam.upToNumThres, oribaseIds);
            continue;
        }
        if (listLen >= DEVICE_PAGE_SIZE * static_cast<size_t>(subcenterNum)) {
            continue;
        }

        addNpuClusByListLen(inputParam, i, oriListLen, oribaseIds, outputParam);
    }
}

void AscendIndexIVFSQTImpl::trainHostBaseSubCluster(std::vector<int> &listsPerDevice, int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFSQT trainHostBaseSubCluster operation started.\n");

    const int NUMBUCK = 64;
    int startListId = listsPerDevice[0];
    int endListId = listsPerDevice[1];
    int batches = (endListId - startListId + NUMBUCK - 1) / NUMBUCK;

    bool upToNumThres = ((size_t)this->intf_->ntotal > numHostThreshold) ? true : false;

    std::vector<uint8_t> quantCodes(MAX_CLUS_POINTS * dimIn * NUMBUCK, 0);

    size_t offset = (size_t)MAX_CLUS_POINTS * (size_t)dimIn;

    std::unique_ptr<float[]> baseData = std::make_unique<float[]>(MAX_CLUS_POINTS * dimIn);

    // add move trained here, save vmin and vdiff, so dimIn * 2
    std::vector<uint16_t> trainedFp16(dimIn * 2);
    uint16_t *vmin = trainedFp16.data();
    uint16_t *vdiff = trainedFp16.data() + dimIn;

    fillingTrainedData(trainedFp16, vmin, vdiff, dimIn);

    for (size_t b = 0; b < size_t(batches); b++) {
        AscendClusteringConfig npuClusConf({ ivfsqtConfig.deviceList[deviceId] }, ivfsqtConfig.resourceSize);
        AscendClustering npuClus(dimIn, NUMBUCK, faiss::METRIC_L2, npuClusConf); // 这里的中心个数暂时不重要

        npuClus.UpdateVdm(vmin, vdiff);

        size_t batchSubNlist = 0;
        std::vector<float> tmpStackCentroids;
        int batchLen = 0;
        std::vector<size_t> listLenBucks(NUMBUCK, 0);

        AddNpuClusInputParam inputParam { startListId, quantCodes, offset, static_cast<size_t>(NUMBUCK),
            b, upToNumThres };
        AddNpuClusOutputParam outputParam { batchLen, listLenBucks, batchSubNlist, npuClus };
        addNpuClus(inputParam, outputParam);

        if (batchLen == 0) {
            continue;
        }

        tmpStackCentroids.resize(batchSubNlist * dimIn, 0);
        std::vector<uint16_t> deviceLabel(batchLen, 0);

        npuClus.SubClusExecInt8(KMEANS_ITER, deviceLabel.data(), tmpStackCentroids.data(), batchSubNlist, batchLen);

        size_t curCentPos = 0;
        int deviceIdx_ = 0;

        for (auto i = 0; i < NUMBUCK; i++) {
            size_t listId = b * (size_t)NUMBUCK + (size_t)i + (size_t)startListId;
            if (listId >= size_t(nlist)) {
                break;
            }

            std::vector<int> &oribaseIds = bucketDetails[listId];
            size_t orilistLen = oribaseIds.size();
            size_t listLen = listLenBucks[i];

            bool listLenInvalid = (listLen < static_cast<size_t>(lowerBound)) ||
                (listLen >= DEVICE_PAGE_SIZE * static_cast<size_t>(subcenterNum));
            if (listLenInvalid) {
                continue;
            }
            int curCentNum = npuClus.GetSubBucketNum(i);

            std::copy(tmpStackCentroids.data() + curCentPos, tmpStackCentroids.data() + curCentPos + curCentNum * dimIn,
                adaptiveSubCenters.data() + listId * subcenterNum * dimIn);
            if (listLen > MAX_CLUS_POINTS) {
                std::vector<faiss::idx_t> label;
                std::unique_ptr<faiss::IndexFlat> subCpuQuantizer(new faiss::IndexFlatL2(dimIn));
                subCpuQuantizer->add(curCentNum, tmpStackCentroids.data() + curCentPos);

                int segNums = (static_cast<int>(listLen) + MAX_CLUS_POINTS - 1) / MAX_CLUS_POINTS;
                size_t t = 0;
                for (size_t s = 0; s < (size_t)segNums; s++) {
                    size_t actualSegSize = 0;
                    while (actualSegSize < MAX_CLUS_POINTS && t < orilistLen) {
                        if (upToNumThres && isToDevice(oribaseIds[t])) {
                            t += 1;
                            continue;
                        }
                        size_t hostId = ((size_t)oribaseIds[t] >= numHostThreshold) ?
                            (size_t)idDictHost[oribaseIds[t]] : (size_t)oribaseIds[t];
                        size_t pageId = hostId / GLOBAL_CODES_PAGE_SIZE;
                        size_t pageOffset = hostId % GLOBAL_CODES_PAGE_SIZE;

                        std::copy(globalInCodesPointer[pageId]->data() + pageOffset * dimIn,
                            globalInCodesPointer[pageId]->data() + (pageOffset + 1) * dimIn,
                            quantCodes.data() + i * offset + actualSegSize * dimIn);
                        actualSegSize += 1;
                        t += 1;
                    }
                    sqIn.decode(quantCodes.data() + i * offset, baseData.get(), actualSegSize);
                    std::vector<faiss::idx_t> labelSub(actualSegSize, 0);
                    subCpuQuantizer->assign(actualSegSize, baseData.get(), labelSub.data());
                    label.insert(label.end(), labelSub.begin(), labelSub.end());
                }
                int labelJ = 0;
                for (size_t j = 0; j < orilistLen; j++) {
                    if (upToNumThres && isToDevice(oribaseIds[j])) {
                        continue;
                    }
                    allLabels[listId * subcenterNum + label[labelJ]].push_back(oribaseIds[j]);
                    labelJ += 1;
                }
                curCentPos += (size_t)curCentNum * (size_t)dimIn;
                deviceIdx_ += MAX_CLUS_POINTS;
                continue;
            }
            int labelJ = 0;
            for (size_t j = 0; j < orilistLen; j++) {
                if (upToNumThres && isToDevice(oribaseIds[j])) {
                    continue;
                }
                allLabels[listId * subcenterNum + deviceLabel[labelJ + deviceIdx_]].push_back(oribaseIds[j]);
                labelJ += 1;
            }
            curCentPos += static_cast<size_t>(curCentNum) * static_cast<size_t>(dimIn);
            deviceIdx_ += static_cast<int>(listLen);
        }
    }
}

void AscendIndexIVFSQTImpl::fetchDeviceTmpData()
{
    APP_LOG_INFO("AscendIndexIVFSQT fetchDeviceTmpData operation started.\n");
    int deviceId = indexConfig.deviceList[0];
    auto pIndex = getActualIndex(deviceId);
    uint32_t pageNums = static_cast<uint32_t>(pIndex->getTmpPageNums());
    globalInCodesDevice.clear();
    deviceBaseAssignCoarse();

    for (uint32_t pageId = 0; pageId < pageNums; pageId++) {
        using namespace ::ascend;
        DeviceVector<unsigned char> &tmpData = pIndex->getPageTmpData(pageId);
        size_t num = tmpData.size();
        globalInCodesDevice.resize(num);
        auto ret = aclrtMemcpy(globalInCodesDevice.data(), num, tmpData.data(), num, ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy data error %d", ret);
        num /= static_cast<size_t>(dimIn);

        deviceBaseAssignSub(num);
        globalInCodesDevice.clear();
        deviceDataNums += num;
    }

    APP_LOG_INFO("AscendIndexIVFSQT fetchDeviceTmpData operation finished.\n");
}

void AscendIndexIVFSQTImpl::deviceBaseAssignCoarse()
{
    APP_LOG_INFO("AscendIndexIVFSQT deviceBaseAssignCoarse operation started.\n");
#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (int listId = 0; listId < nlist; listId++) {
        auto deviceBaseI = assignCountsDevice[listId];
        int listLen = deviceBaseI.size();
        if (listLen != 0) {
            std::vector<int> deviceBaseIRemap(listLen, 0);
            for (int i = 0; i < listLen; i++) {
                deviceBaseIRemap[i] = idDictDevice[deviceBaseI[i]];
            }
            bucketDetails[listId].insert(bucketDetails[listId].end(), deviceBaseIRemap.begin(), deviceBaseIRemap.end());
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT deviceBaseAssignCoarse operation finished.\n");
}

void AscendIndexIVFSQTImpl::deviceBaseAssignSub(int num)
{
    APP_LOG_INFO("AscendIndexIVFSQT deviceBaseAssignSub operation started.\n");
#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (int listId = 0; listId < nlist; listId++) {
        auto deviceBaseI = assignCountsDevice[listId];
        int listLen = deviceBaseI.size();
        if (listLen == 0) {
            continue;
        }
        for (size_t j = 0; j < (size_t)listLen; j++) {
            if ((size_t)deviceBaseI[j] < (size_t)num + deviceDataNums && (size_t)deviceBaseI[j] >= deviceDataNums) {
                idx_t deviceBaseIRemap = idDictDevice[deviceBaseI[j]];
                std::unique_ptr<float[]> baseData = std::make_unique<float[]>(1 * dimIn);

                sqIn.decode(globalInCodesDevice.data() + (ascend_idx_t(deviceBaseI[j]) - deviceDataNums) * dimIn,
                    baseData.get(), 1);
                std::vector<faiss::idx_t> label(1, 0);
                std::unique_ptr<faiss::IndexFlat> subCpuQuantizer(new faiss::IndexFlatL2(dimIn));
                subCpuQuantizer->add(subcenterNum, adaptiveSubCenters.data() + listId * subcenterNum * dimIn);
                subCpuQuantizer->assign(1, baseData.get(), label.data());
                allLabels[listId * subcenterNum + label[0]].push_back(deviceBaseIRemap);
            }
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT deviceBaseAssignSub operation finished.\n");
}

void AscendIndexIVFSQTImpl::calbaseInfo()
{
    APP_LOG_INFO("AscendIndexIVFSQT calbaseInfo operation started.\n");
    baseNums.clear();
    baseNums.reserve(nlist);

    // calculate the num of sub bucket
    for (int listId = 0; listId < nlist; ++listId) {
        std::vector<int> tmpNums(subcenterNum, 0);

        size_t listLen = bucketDetails[listId].size();
        if (listLen == 0) {
            baseNums.emplace(listId, tmpNums);
            continue;
        }

        for (int j = 0; j < subcenterNum; j++) {
            std::vector<faiss::idx_t> labels(allLabels[listId * subcenterNum + j].begin(),
                allLabels[listId * subcenterNum + j].end());
            if (labels.size() == 0 || labels.size() > DEVICE_PAGE_SIZE) {
                continue;
            }
            tmpNums[j] = static_cast<int>(labels.size());
        }
        baseNums.emplace(listId, tmpNums);
    }

    // calculate the num and offset of each device
    for (int listId = 0; listId < nlist; ++listId) {
        subBucketHandle(listId, 1);
    }
    APP_LOG_INFO("AscendIndexIVFSQT calbaseInfo operation finished.\n");
}

void AscendIndexIVFSQTImpl::addWithPageCodes(int devIdx)
{
    APP_LOG_INFO("AscendIndexIVFSQT addWithPageCodes operation started.\n");
    int deviceId = indexConfig.deviceList[devIdx];
    auto pIndex = getActualIndex(deviceId);
    size_t originLen = 0;
    std::vector<uint8_t> codes;
    std::vector<ascend_idx_t> ids;

    int devIdxBak = devIdx;
    devIdx = 0;
    // add Based on subcenter
    int listRecord = 0;
    for (size_t listId = 0; listId < (size_t)nlist; listId++) {
        for (int j = 0; j < subcenterNum; j++) {
            size_t bucketSize = static_cast<size_t>(subBucketNum[listId][devIdx][j]);
            size_t bucketOffset = static_cast<size_t>(subBucketOffset[listId][devIdx][j]);
            std::vector<faiss::idx_t> labels(allLabels[listId * subcenterNum + j].begin(),
                allLabels[listId * subcenterNum + j].end());

            if (bucketSize == 0) {
                continue;
            }
            listRecord = static_cast<int>(listId);
            // update page count
            if (originLen + bucketSize > DEVICE_PAGE_SIZE) {
                auto ret = pIndex->addPageVectors(originLen, codes.data(), ids.data());
                FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "addPageVectors failed:%d", ret);
                if (devIdxBak == 0) {
                    deviceAddList.push_back(listRecord);
                    deviceAddNum.push_back(originLen);
                }
                originLen = 0;
                pageCount += 1;
                codes.resize(0);
                ids.resize(0);
            }
            originLen += bucketSize;
            for (size_t k = bucketOffset; k < (bucketOffset + bucketSize); ++k) {
                auto label = labels[k];
                size_t pageId = static_cast<size_t>(label) / GLOBAL_CODES_PAGE_SIZE;
                size_t offset = static_cast<size_t>(label) % GLOBAL_CODES_PAGE_SIZE;
                codes.insert(codes.end(), globalOutCodesPointer[pageId]->data() + offset * sq.code_size,
                    globalOutCodesPointer[pageId]->data() + offset *sq.code_size +
                    sq.code_size);
                ids.push_back((*globalIdsPointer[pageId])[offset]);
            }
        }
    }
    if (originLen > 0) {
        auto ret = pIndex->addPageVectors(originLen, codes.data(), ids.data());
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "addPageVectors failed:%d", ret);
        if (devIdxBak == 0) {
            deviceAddList.push_back(listRecord);
            deviceAddNum.push_back(originLen);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQT addWithPageCodes operation started.\n");
}

uint32_t AscendIndexIVFSQTImpl::getListLength(int listId) const
{
    VALUE_UNUSED(listId);
    FAISS_THROW_MSG("getListLength not implemented for this type of index");
    return 0;
}

void AscendIndexIVFSQTImpl::getListCodesAndIds(int listId, std::vector<uint8_t> &codes,
    std::vector<ascend_idx_t> &ids) const
{
    VALUE_UNUSED(listId);
    VALUE_UNUSED(codes);
    VALUE_UNUSED(ids);
    FAISS_THROW_MSG("getListCodesAndIds not implemented for this type of index");
}

size_t AscendIndexIVFSQTImpl::getSendDeviceSize(int n)
{
    APP_LOG_INFO("AscendIndexIVFSQT getSendDeviceSize operation started.\n");
    size_t maxNumVecsForPageSize = static_cast<size_t>(ADD_PAGE_SIZE / dimIn);
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);
    APP_LOG_INFO("AscendIndexIVFSQT getSendDeviceSize operation finished.\n");
    return std::min((size_t)n, maxNumVecsForPageSize);
}

void AscendIndexIVFSQTImpl::initDeviceNpuType()
{
    APP_LOG_INFO("AscendIndexIVFSQT initDeviceNpuType operation started.\n");
    deviceNpuType = faiss::ascend::SocUtils::GetInstance().IsAscend310() ? 0 : 1;
    APP_LOG_INFO("AscendIndexIVFSQT initDeviceNpuType operation finish.\n");
}

void AscendIndexIVFSQTImpl::setLowerBound(int lowerBoundIn)
{
    VALUE_UNUSED(lowerBoundIn);
    APP_LOG_ERROR("setLowerBound is not implemented, and will be deprecated.");
}

int AscendIndexIVFSQTImpl::getLowerBound() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return lowerBound;
}

void AscendIndexIVFSQTImpl::setMergeThres(int mergeThresIn)
{
    VALUE_UNUSED(mergeThresIn);
    APP_LOG_ERROR("setMergeThres is not implemented, and will be deprecated.");
}

int AscendIndexIVFSQTImpl::getMergeThres() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return mergeThres;
}

void AscendIndexIVFSQTImpl::setMemoryLimit(float memoryLimitIn)
{
    VALUE_UNUSED(memoryLimitIn);
    APP_LOG_ERROR("setMemoryLimit is not implemented, and will be deprecated.");
}

void AscendIndexIVFSQTImpl::setAddTotal(size_t addTotalIn)
{
    VALUE_UNUSED(addTotalIn);
    APP_LOG_ERROR("setAddTotal is not implemented, and will be deprecated.");
}

void AscendIndexIVFSQTImpl::setPreciseMemControl(bool preciseMemControlIn)
{
    VALUE_UNUSED(preciseMemControlIn);
    APP_LOG_ERROR("setPreciseMemControl is not implemented, and will be deprecated.");
}

void AscendIndexIVFSQTImpl::threadUnsafeReset()
{
    APP_LOG_INFO("AscendIndexIVFSQT reset operation started.\n");
    AscendIndexIVFImpl::threadUnsafeReset();
    if (isUpdated) {
        resetAfterUpdate = true;
    }
    isUpdated = false;
    APP_LOG_INFO("AscendIndexIVFSQT reset operation finished.\n");
}

void AscendIndexIVFSQTImpl::reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    threadUnsafeReset();
}

void AscendIndexIVFSQTImpl::setNumProbes(int nprobes)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQT setNumProbes operation started.\n");

    checkSearchParams(nprobes, l2NProbe, l3SegmentNum);

    this->nprobe = nprobes;
    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        pIndex->setNumProbes(nprobe);
    }
    APP_LOG_INFO("AscendIndexIVFSQT setNumProbes operation finished.\n");
}
} // ascend
} // faiss
