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


#include "ascend/impl/AscendIndexIVFImpl.h"

#include <atomic>
#include <algorithm>

#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>

#include "AuxIndexStructures.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/utils/fp16.h"

namespace faiss {
namespace ascend {
constexpr int DEFAULT_NPROBE = 64;
const int MAX_NITER = 65536;
// divide 5 represents 0.2billion
const idx_t MAX_NUM_RESERVE = MAX_N / 5;

// implementation of AscendIndexIVF
AscendIndexIVFImpl::AscendIndexIVFImpl(AscendIndex *intf, int dims, faiss::MetricType metric, int listNum,
    AscendIndexIVFConfig config)
    : AscendIndexImpl(dims, metric, config, intf), nlist(listNum), nprobe(DEFAULT_NPROBE), ivfConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
        this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    checkIVFParams();

    ivfConfig.cp.verbose = this->intf_->verbose;

    pQuantizerImpl = std::make_shared<AscendIndexQuantizerImpl>();
    pQuantizerImpl->cpuQuantizer = std::make_shared<IndexFlatL2>(dims);
}

AscendIndexIVFImpl::~AscendIndexIVFImpl() {}

void AscendIndexIVFImpl::initFlatAT()
{
    if (this->ivfConfig.useKmeansPP) {
        // Now npuQuantizer use only in add stage
        faiss::ascend::AscendIndexFlatATConfig npuConf(ivfConfig.deviceList, ivfConfig.resourceSize);
        pQuantizerImpl->npuQuantizer = std::make_shared<AscendIndexFlatAT>(this->intf_->d, this->nlist, npuConf);
        AscendClusteringConfig npuClusConf({ ivfConfig.deviceList[0] }, ivfConfig.resourceSize);
        pQuantizerImpl->npuClus =
            std::make_shared<AscendClustering>(this->intf_->d, this->nlist, this->intf_->metric_type, npuClusConf);
    }
}

void AscendIndexIVFImpl::checkIVFParams()
{
    FAISS_THROW_IF_NOT_FMT(ivfConfig.cp.niter > 0 && ivfConfig.cp.niter <= MAX_NITER,
        "ClusteringParameters niter must be in range (0, %d] ", MAX_NITER);

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.nredo > 0, "ClusteringParameters nredo must > 0");

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.min_points_per_centroid > 0,
        "ClusteringParameters min_points_per_centroid must > 0");

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.max_points_per_centroid > 0,
        "ClusteringParameters max_points_per_centroid must > 0");
}

// Copy what we need from the CPU equivalent
void AscendIndexIVFImpl::copyFrom(const faiss::IndexIVF *index)
{
    APP_LOG_INFO("AscendIndexIVF copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Index is nullptr.");
    
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
        index->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(index->nprobe > 0 && index->nprobe <= index->nlist,
        "Invalid number of nprobe %zu", index->nprobe);
    FAISS_THROW_IF_NOT_FMT(index->ntotal >= 0 && index->ntotal < MAX_N, "Ntotal must be >= 0 and < %ld", MAX_N);
    
    FAISS_THROW_IF_NOT_MSG(this->intf_->d == index->d, "Invalid dim");
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == index->metric_type, "Invalid metric_type");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(nlist) == index->nlist, "Invalid nlist");
    nprobe = index->nprobe;

    pQuantizerImpl = std::make_shared<AscendIndexQuantizerImpl>();
    pQuantizerImpl->cpuQuantizer = std::make_shared<IndexFlatL2>(this->intf_->d);

    if (!index->is_trained) {
        this->intf_->is_trained = false;
        this->intf_->ntotal = 0;
        return;
    }

    this->intf_->is_trained = true;
    this->intf_->ntotal = index->ntotal;

    auto flat = dynamic_cast<faiss::IndexFlat *>(index->quantizer);
    FAISS_THROW_IF_NOT_MSG(flat, "Only IndexFlat is supported for the coarse quantizer "
        "for copying from an IndexIVF into a AscendIndexIVF");
    
    FAISS_THROW_IF_NOT_FMT(flat->ntotal >= 0 && flat->ntotal < MAX_N,
        "Quantizer ntotal must be >= 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(flat->codes.size() / sizeof(float) ==
        static_cast<size_t>(flat->ntotal) * static_cast<size_t>(this->intf_->d),
        "The size of quantizer codes Should be equal to quantizer_ntotal * dim, expected=%zu, actual=%zu",
        flat->codes.size() / sizeof(float), static_cast<size_t>(flat->ntotal) * static_cast<size_t>(this->intf_->d));

    pQuantizerImpl->cpuQuantizer->add(flat->ntotal, flat->get_xb());
    APP_LOG_INFO("AscendIndexIVF copyFrom operation finished.\n");
}

// / Copy what we have to the CPU equivalent
void AscendIndexIVFImpl::copyTo(faiss::IndexIVF *index) const
{
    APP_LOG_INFO("AscendIndexIVF copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Index is nullptr.");
    if (dynamic_cast<faiss::ArrayInvertedLists *>(index->invlists)) {
        index->reset();
    }
    index->ntotal = this->intf_->ntotal;
    index->d = this->intf_->d;
    index->metric_type = this->intf_->metric_type;
    index->is_trained = this->intf_->is_trained;
    index->nlist = nlist;
    index->nprobe = nprobe;

    faiss::IndexFlat *q = new faiss::IndexFlatL2(this->intf_->d);

    *q = *(pQuantizerImpl->cpuQuantizer);

    if (index->own_fields) {
        delete index->quantizer;
        index->quantizer = nullptr;
    }

    index->quantizer = q;
    index->quantizer_trains_alone = 0;
    index->own_fields = true;
    index->cp = this->ivfConfig.cp;
    index->direct_map.clear();
    APP_LOG_INFO("AscendIndexIVF copyTo operation finished.\n");
}

void AscendIndexIVFImpl::threadUnsafeReset()
{
    APP_LOG_INFO("AscendIndexIVF reset operation started.\n");
    auto resetFunctor = [&](int i) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->reset();
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to reset index, device id: %d, result is %d\n", deviceId, ret);
    };

    // parallel reset to every device
    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, resetFunctor);

    initDeviceAddNumMap();
    this->intf_->ntotal = 0;
    APP_LOG_INFO("AscendIndexIVF reset operation finished.\n");
}

void AscendIndexIVFImpl::reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    threadUnsafeReset();
}

// Sets the number of list probes per query
void AscendIndexIVFImpl::setNumProbes(int nprobes)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVF setNumProbes operation started.\n");
    FAISS_THROW_IF_NOT_MSG(nprobes > 0, "nprobe must be greater than 0");
    FAISS_THROW_IF_NOT_MSG(nprobes <= this->nlist, "nprobe must be less than or equal to nlist");
    this->nprobe = nprobes;
    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        pIndex->setNumProbes(nprobe);
    }
    APP_LOG_INFO("AscendIndexIVF setNumProbes operation finished.\n");
}

void AscendIndexIVFImpl::trainQuantizer(faiss::idx_t n, const float *x, bool clearNpuData)
{
    APP_LOG_INFO("AscendIndexIVF start to trainQuantizer with %ld vector(s).\n", n);
    if (n == 0) {
        // nothing to do
        return;
    }

    if (pQuantizerImpl->cpuQuantizer->is_trained && (pQuantizerImpl->cpuQuantizer->ntotal == nlist)) {
        if (this->intf_->verbose) {
            printf("IVF quantizer does not need training.\n");
        }

        return;
    }

    if (this->intf_->verbose) {
        printf("Training IVF quantizer on %ld vectors in %dD to %d cluster\n", n, this->intf_->d, nlist);
    }

    pQuantizerImpl->cpuQuantizer->reset();
    if (this->ivfConfig.useKmeansPP) {
        std::vector<float> tmpCentroids(nlist * this->intf_->d);
        if (pQuantizerImpl->npuClus->GetNTotal() == 0) {
            pQuantizerImpl->npuClus->Add(n, x);
        }
        if (this->intf_->verbose) {
            printf("Ascend cluster start training %zu vectors\n", pQuantizerImpl->npuClus->GetNTotal());
        }
        // AscendClustering does NOT perform sampling on data
        pQuantizerImpl->npuClus->Train(this->ivfConfig.cp.niter, tmpCentroids.data(), clearNpuData);

        pQuantizerImpl->cpuQuantizer->add(nlist, tmpCentroids.data());
        pQuantizerImpl->npuQuantizer->reset();
        pQuantizerImpl->npuQuantizer->add(nlist, tmpCentroids.data());
    } else {
        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));
    }
    pQuantizerImpl->cpuQuantizer->is_trained = true;

    FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.ntotal must be nlist");
    APP_LOG_INFO("AscendIndexIVF trainQuantizer finished.\n");
}

uint32_t AscendIndexIVFImpl::getListLength(int listId) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVF getListLength operation started.\n");
    FAISS_THROW_IF_NOT((listId >= 0) && (listId < nlist));

    uint32_t len = 0;
    for (auto &index : indexes) {
        // get the list length from devices
        uint32_t tmpLen = 0;
        auto pIndex = getActualIndex(index.first);
        tmpLen = pIndex->getListLength(listId);
        len += tmpLen;
    }
    APP_LOG_INFO("AscendIndexIVF getListLength operation finished.\n");
    return len;
}

void AscendIndexIVFImpl::getListCodesAndIds(int listId, std::vector<uint8_t> &codes,
    std::vector<ascend_idx_t> &ids) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVF getListCodesAndIds operation started.\n");
    FAISS_THROW_IF_NOT((listId >= 0) && (listId < nlist));
    codes.clear();
    ids.clear();

    // use for(deviceList) rather than for(auto& index : indexMap),
    // to ensure merged codes and ids in sequence
    for (size_t j = 0; j < indexConfig.deviceList.size(); j++) {
        int deviceId = indexConfig.deviceList[j];
        std::vector<uint8_t> tmpCodes;
        std::vector<ascend_idx_t> tmpIds;
        indexIVFGetListCodes(deviceId, listId, tmpCodes, tmpIds);
        std::copy(tmpCodes.begin(), tmpCodes.end(), back_inserter(codes));
        std::copy(tmpIds.begin(), tmpIds.end(), back_inserter(ids));
    }
    APP_LOG_INFO("AscendIndexIVF getListCodesAndIds operation finished.\n");
}

void AscendIndexIVFImpl::reserveMemory(size_t numVecs)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVF reserveMemory operation started.\n");
    FAISS_THROW_IF_NOT_FMT((numVecs > 0) && (numVecs <= MAX_N), "numVecs must be > 0 and <= %ld", MAX_N);
    size_t deviceNum = indexConfig.deviceList.size();
    size_t numPerDev = (numVecs + deviceNum - 1) / deviceNum;
    uint32_t numReserve = static_cast<uint32_t>(numPerDev);

    FAISS_THROW_IF_NOT_FMT((numReserve > 0) && (numReserve <= MAX_NUM_RESERVE), "numReserve must be > 0 and <= %ld",
        MAX_NUM_RESERVE);
    auto reserveFunctor = [&](int idx) {
        auto pIndex = getActualIndex(indexConfig.deviceList[idx]);
        auto ret = pIndex->reserveMemory(numReserve);
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to reserveMemory index,device is %d,result: %d\n",
                               indexConfig.deviceList[idx], ret);
    };

    // parallel reserveMemory to every device
    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, reserveFunctor);
    APP_LOG_INFO("AscendIndexIVF reserveMemory operation finished.\n");
}

size_t AscendIndexIVFImpl::reclaimMemory()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVF reclaimMemory operation started.\n");
    std::atomic<size_t> tmpReclaimed(0);

    auto reclaimFunctor = [&](int idx) {
        uint32_t tmp = 0;
        auto pIndex = getActualIndex(indexConfig.deviceList[idx]);
        tmp = pIndex->reclaimMemory();
        tmpReclaimed += tmp;
    };

    // parallel reclaimMemory to every device
    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, reclaimFunctor);
    APP_LOG_INFO("AscendIndexIVF reclaimMemory operation finished.\n");
    return tmpReclaimed.load();
}

void AscendIndexIVFImpl::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexIVF searchPostProcess operation started.\n");
    mergeSearchResult(devices, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexIVF searchPostProcess operation finished.\n");
}

void AscendIndexIVFImpl::indexIVFFastGetListCodes(int deviceId, int nlist, InvertedLists *ivf) const
{
    auto pIndex = getActualIndex(deviceId);
    using namespace ::ascend;
    for (int i = 0; i < nlist; i++) {
        if (pIndex->getListLength(i) == 0) {
            continue;
        }
        std::vector<uint8_t> codes;
        auto appRet = pIndex->getListVectorsReshaped(i, codes);
        FAISS_THROW_IF_NOT_FMT(appRet == APP_ERR_OK, "failed to get vector shaped, ret: %d", appRet);
        DeviceVector<::ascend::Index::idx_t> &idsVec = pIndex->getListIndices(i);
        std::vector<faiss::idx_t> ids(idsVec.size());
        auto ret = aclrtMemcpy(ids.data(), ids.size() * sizeof(::ascend::Index::idx_t),
                               idsVec.data(), idsVec.size() * sizeof(faiss::idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to memcpy ids, ret: %d", ret);
        ivf->add_entries(i, ids.size(), ids.data(), codes.data());
    }
}

void AscendIndexIVFImpl::indexIVFGetListCodes(int deviceId, int listId, std::vector<uint8_t> &codes,
                                              std::vector<ascend_idx_t> &ids) const
{
    auto pIndex = getActualIndex(deviceId);
    using namespace ::ascend;
    size_t listNum = pIndex->getListLength(listId);
    if (listNum == 0) {
        return;
    }
    if (pIndex->listVectorsNeedReshaped()) {
        auto ret = pIndex->getListVectorsReshaped(listId, codes);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to get vector shaped, ret: %d", ret);
    } else {
        DeviceVector<unsigned char> &codesVec = pIndex->getListVectors(listId);
        codes.resize(codesVec.size());
        auto ret = aclrtMemcpy(codes.data(), codes.size() * sizeof(uint8_t),
                               codesVec.data(), codesVec.size() * sizeof(unsigned char), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to memcpy codes, ret: %d", ret);
    }
    DeviceVector<::ascend::Index::idx_t> &idsVec = pIndex->getListIndices(listId);
    ids.resize(idsVec.size());
    auto ret = aclrtMemcpy(ids.data(), ids.size() * sizeof(ascend_idx_t),
                           idsVec.data(), idsVec.size() * sizeof(::ascend::Index::idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to memcpy ids, ret: %d", ret);
}

void AscendIndexIVFImpl::initDeviceAddNumMap()
{
    APP_LOG_INFO("AscendIndexIVF initDeviceAddNumMap operation started.\n");
    deviceAddNumMap.clear();
    deviceAddNumMap.resize(this->nlist);
    for (int i = 0; i < this->nlist; i++) {
        deviceAddNumMap[i] = std::vector<int>(indexConfig.deviceList.size(), 0);
    }
    APP_LOG_INFO("AscendIndexIVF initDeviceAddNumMap operation finished.\n");
}

void AscendIndexIVFImpl::updateDeviceCoarseCenter()
{
    APP_LOG_INFO("AscendIndexIVF updateDeviceCoarseCenter operation started.\n");
    // coarse centroids have been stored on host in fp16 format
    std::vector<uint16_t> corseCentroidsFp16(pQuantizerImpl->cpuQuantizer->codes.size() / sizeof(float));
    transform(pQuantizerImpl->cpuQuantizer->get_xb(), pQuantizerImpl->cpuQuantizer->get_xb() +
        pQuantizerImpl->cpuQuantizer->codes.size() / sizeof(float), begin(corseCentroidsFp16),
        [](float temp) { return fp16(temp).data; });

    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        ::ascend::AscendTensor<float16_t, ::ascend::DIMS_2> coarseCent(corseCentroidsFp16.data(),
                                                                       { this->nlist, this->intf_->d });
        pIndex->updateCoarseCentroidsData(coarseCent);
    }
    APP_LOG_INFO("AscendIndexIVF updateDeviceCoarseCenter operation finished.\n");
}

size_t AscendIndexIVFImpl::removeImpl(const IDSelector &sel)
{
    APP_LOG_INFO("AscendIndexIVF removeImpl operation started.\n");
    size_t deviceCnt = indexConfig.deviceList.size();
    uint32_t removeCnt = 0;

    // remove vector from device
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        size_t removeSize = rangeSel->set.size();
        FAISS_THROW_IF_NOT_FMT(removeSize <= static_cast<size_t>(this->intf_->ntotal),
            "the size of removed codes should be in range [0, %ld], actual=%zu.", this->intf_->ntotal, removeSize);
        std::vector<ascend_idx_t> removeBatch(removeSize);
        transform(begin(rangeSel->set), end(rangeSel->set), begin(removeBatch),
            [](idx_t temp) { return (ascend_idx_t)temp; });

#pragma omp parallel for reduction(+ : removeCnt) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
        for (size_t i = 0; i < deviceCnt; i++) {
            int deviceId = indexConfig.deviceList[i];
            auto pIndex = getActualIndex(deviceId);
            ::ascend::IDSelectorBatch batch(removeBatch.size(), removeBatch.data());
            removeCnt = pIndex->removeIds(batch);
        }
    } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
#pragma omp parallel for reduction(+ : removeCnt) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
        for (size_t i = 0; i < deviceCnt; i++) {
            int deviceId = indexConfig.deviceList[i];
            auto pIndex = getActualIndex(deviceId);
            ::ascend::IDSelectorRange range(rangeSel->imin, rangeSel->imax);
            removeCnt = pIndex->removeIds(range);
        }
    }

    // update vector nums of deviceAddNumMap
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        for (int listId = 0; listId < nlist; listId++) {
            uint32_t len = 0;
            auto pIndex = getActualIndex(deviceId);
            len = pIndex->getListLength(listId);
            deviceAddNumMap[listId][i] = len;
        }
    }
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal >= removeCnt,
        "removeCnt should be in range [0, %ld], actual=%d.", this->intf_->ntotal, removeCnt);
    this->intf_->ntotal -= static_cast<faiss::idx_t>(removeCnt);
    APP_LOG_INFO("AscendIndexIVF removeImpl operation finished.\n");
    return (size_t)removeCnt;
}
} // namespace ascend
} // namespace faiss
