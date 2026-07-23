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

#include "AscendIndexIVFPQImpl.h"

#include <faiss/Clustering.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <securec.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>

#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/custom/AscendClustering.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/DevVecMemStrategyIntf.h"

namespace faiss
{
namespace ascend
{

namespace
{
int computeTrainTotalSize(int nlist, int n, int trainSamplesPerList, int maxTrainSamples)
{
    // Use 64-bit multiply to avoid overflow when nlist * trainSamplesPerList exceeds INT_MAX
    // (e.g. nlist=524288 and trainSamplesPerList=256).
    const int64_t byLists = static_cast<int64_t>(nlist) * static_cast<int64_t>(trainSamplesPerList);
    const int64_t capped = std::min(byLists, static_cast<int64_t>(n));
    return static_cast<int>(std::min(capped, static_cast<int64_t>(maxTrainSamples)));
}

uint64_t getActualRngSeed(int seed)
{
    return (seed >= 0) ? static_cast<uint64_t>(seed)
                       : static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

void sampleTrainData(const float* x, int n, int dim, int totalSize, int seed, std::vector<float>& trainData)
{
    const size_t dimSz = static_cast<size_t>(dim);
    const size_t totalSizeSz = static_cast<size_t>(totalSize);
    trainData.resize(totalSizeSz * dimSz);
    const size_t numBytes = totalSizeSz * dimSz * sizeof(float);

    if (totalSize >= n)
    {
        auto ret = ::ascend::memcpySChunked(trainData.data(), numBytes, x, numBytes);
        FAISS_THROW_IF_NOT_FMT(ret == EOK, "trainData memcpy_s failed %d", ret);
        return;
    }

    std::vector<size_t> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(static_cast<uint32_t>(getActualRngSeed(seed)));
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < totalSize; i++)
    {
        const size_t srcIdx = indices[static_cast<size_t>(i)];
        const float* src = x + srcIdx * dimSz;
        float* dst = trainData.data() + static_cast<size_t>(i) * dimSz;
        auto ret = memcpy_s(dst, dimSz * sizeof(float), src, dimSz * sizeof(float));
        FAISS_THROW_IF_NOT_FMT(ret == EOK, "trainData sample memcpy_s failed %d", ret);
    }
    APP_LOG_INFO("IVFPQ sampleTrainData: shuffled %d vectors from input n=%d\n", totalSize, n);
}

void buildSampleIndices(int n, int totalSize, int seed, std::vector<size_t>& sampleIndices)
{
    sampleIndices.resize(static_cast<size_t>(totalSize));
    if (totalSize >= n)
    {
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        return;
    }

    std::vector<size_t> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(static_cast<uint32_t>(getActualRngSeed(seed)));
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < totalSize; i++)
    {
        sampleIndices[static_cast<size_t>(i)] = indices[static_cast<size_t>(i)];
    }
}

long long elapsedMs(const std::chrono::high_resolution_clock::time_point& start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
        .count();
}
}  // namespace

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 128;
// Default nlist in case of nullptr index
const size_t DEFAULT_NLIST = 1024;
// Default msub in case of nullptr index
const size_t DEFAULT_MSUB = 4;
// Default nbit in case of nullptr index
const size_t DEFAULT_NBIT = 8;

// The value range of dim
const std::vector<int> DIMS = {128};

// The value range of nlist
const std::vector<int> NLISTS = {1024, 2048, 4096, 8192, 16384, 262144, 524288};

// The value range of msub
const std::vector<int> MSUBS = {2, 4, 8, 16, 32};

// The value range of nbit
const std::vector<int> NBITS = {8};

const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 640;
const size_t ADD_PAGE_SIZE = (UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE);
const size_t UNIT_VEC_SIZE = 5120;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

AscendIndexIVFPQImpl::AscendIndexIVFPQImpl(AscendIndexIVFPQ* intf, int dims, int nlist, int msubs, int nbits,
                                           faiss::MetricType metric, AscendIndexIVFPQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config), intf_(intf), msubs(msubs), nbits(nbits)
{
    checkParams();
    initCoarseClustering();
    initIndexes();
    initDeviceAddNumMap();
    centroidsData.resize(nlist * dims);
    initProductQuantizer();
    this->intf_->is_trained = false;
}

void AscendIndexIVFPQImpl::initCoarseClustering()
{
    if (!ivfConfig.useKmeansPP)
    {
        return;
    }
    const int64_t clusteringMaxMem = 0x100000000LL;
    int64_t clusResource = ivfConfig.resourceSize;
    if (clusResource > clusteringMaxMem)
    {
        clusResource = clusteringMaxMem;
    }
    AscendClusteringConfig npuClusConf({ivfConfig.deviceList[0]}, clusResource);
    pQuantizerImpl->npuClus =
        std::make_shared<AscendClustering>(this->intf_->d, this->nlist, this->intf_->metric_type, npuClusConf);
}

AscendIndexIVFPQImpl::~AscendIndexIVFPQImpl() {}

void AscendIndexIVFPQImpl::copyFromCentroids(const faiss::IndexIVFPQ* index)
{
    // copy centroids from index
    APP_LOG_INFO("Uploading centroids to devices...\n");

    std::vector<float> centroids_buffer(nlist * intf_->d);
    index->quantizer->reconstruct_n(0, nlist, centroids_buffer.data());

    updateCoarseCenter(centroids_buffer);
}

void AscendIndexIVFPQImpl::copyFromCodebook(const faiss::IndexIVFPQ* index)
{
    // copy codebook from index
    APP_LOG_INFO("Uploading PQ codebook to devices...\n");

    this->pq.M = index->pq.M;
    this->pq.nbits = index->pq.nbits;
    this->pq.dim = index->d;
    this->pq.ksub = 1 << index->pq.nbits;
    this->pq.dsub = index->d / index->pq.M;

    const float* codeBook_data = index->pq.centroids.data();
    const size_t codebook_size = this->pq.M * this->pq.ksub * this->pq.dsub;
    this->pq.codeBook.assign(codeBook_data, codeBook_data + codebook_size);
    FAISS_THROW_IF_NOT_FMT(pq.M > 0, "invalid msubs: %zu", pq.M);
    for (int deviceId : indexConfig.deviceList)
    {
        auto pIndex = getActualIndex(deviceId);
        if (!pIndex) continue;
        if (pIndex->codeBookOnDevice->size() != codebook_size)
        {
            pIndex->codeBookOnDevice->resize(codebook_size);
        }

        auto ret = aclrtMemcpy(pIndex->codeBookOnDevice->data(), codebook_size * sizeof(float),
                               index->pq.centroids.data(), codebook_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to upload PQ codebook to device %d: %d", deviceId, ret);
    }
    size_t code_size = pq.M;
    FAISS_THROW_IF_NOT_FMT(index->code_size == code_size, "Code size mismatch: CPU index has %zu, expected %zu",
                           index->code_size, code_size);
}

void AscendIndexIVFPQImpl::copyFromPQCodes(const faiss::IndexIVFPQ* index)
{
    // copy pqcode from index
    APP_LOG_INFO("Uploading inverted lists data to devices...\n");

    deviceAddNumMap.clear();
    deviceAddNumMap.resize(index->nlist);
    for (size_t i = 0; i < index->nlist; i++)
    {
        deviceAddNumMap[i].resize(indexConfig.deviceList.size(), 0);
    }

    const faiss::InvertedLists* invlists = index->invlists;
    FAISS_THROW_IF_NOT_MSG(invlists != nullptr, "Source index has no inverted lists");

    size_t deviceCount = indexConfig.deviceList.size();
    size_t totalVectors = index->ntotal;

    std::vector<std::vector<std::pair<size_t, size_t>>> deviceAssignments = assignListsToDevices(invlists, deviceCount);

    uploadToDevicesParallel(deviceAssignments, invlists);

    // Build ID to listId and deviceId mapping for delete support
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        idToListMap.clear();
        idToDeviceMap.clear();
        listInfos.clear();
        listInfos.resize(nlist);

        for (size_t devIdx = 0; devIdx < deviceCount; devIdx++)
        {
            int deviceId = indexConfig.deviceList[devIdx];
            for (const auto& [listNo, listSize] : deviceAssignments[devIdx])
            {
                if (listSize == 0) continue;

                const faiss::idx_t* srcIds = invlists->get_ids(listNo);
                for (size_t i = 0; i < listSize; i++)
                {
                    idx_t id = srcIds[i];
                    idToListMap[id] = listNo;
                    idToDeviceMap[id] = deviceId;
                    listInfos[listNo].idSet.insert(id);
                }
            }
        }
    }

    this->intf_->ntotal = index->ntotal;
    this->intf_->is_trained = index->is_trained;

    size_t totalUploaded = 0;
    for (size_t listNo = 0; listNo < static_cast<size_t>(nlist); listNo++)
    {
        for (size_t devIdx = 0; devIdx < deviceCount; devIdx++)
        {
            totalUploaded += static_cast<size_t>(deviceAddNumMap[listNo][devIdx]);
        }
    }

    FAISS_THROW_IF_NOT_FMT(totalUploaded == totalVectors, "Copied vector count mismatch. Expected: %zu, Actual: %zu",
                           totalVectors, totalUploaded);

    APP_LOG_INFO(
        "AscendIndexIVFPQ copyFrom operation finished. "
        "Successfully copied %zu vectors from source index.\n",
        totalVectors);
}

std::vector<std::vector<std::pair<size_t, size_t>>> AscendIndexIVFPQImpl::assignListsToDevices(
    const faiss::InvertedLists* invlists, size_t deviceCount)
{
    std::vector<std::vector<std::pair<size_t, size_t>>> deviceAssignments(deviceCount);
    for (size_t listNo = 0; listNo < static_cast<size_t>(nlist); listNo++)
    {
        size_t listSize = invlists->list_size(listNo);
        if (listSize == 0) continue;

        size_t selectedDevice = 0;
        size_t minCount = static_cast<size_t>(deviceAddNumMap[listNo][0]);
        for (size_t devIdx = 1; devIdx < deviceCount; devIdx++)
        {
            if (static_cast<size_t>(deviceAddNumMap[listNo][devIdx]) < minCount)
            {
                minCount = static_cast<size_t>(deviceAddNumMap[listNo][devIdx]);
                selectedDevice = devIdx;
            }
        }
        deviceAssignments[selectedDevice].emplace_back(listNo, listSize);
        deviceAddNumMap[listNo][selectedDevice] += static_cast<int>(listSize);
    }

    return deviceAssignments;
}

void AscendIndexIVFPQImpl::uploadToDevicesParallel(
    const std::vector<std::vector<std::pair<size_t, size_t>>>& deviceAssignments, const faiss::InvertedLists* invlists)
{
    auto uploadFunctor = [&](int devIdx)
    {
        int deviceId = indexConfig.deviceList[devIdx];
        auto pIndex = getActualIndex(deviceId);
        if (!pIndex)
        {
            return;
        }

        size_t totalForDevice = 0;
        const auto& assignments = deviceAssignments[devIdx];

        for (const auto& [listNo, listSize] : assignments)
        {
            if (listSize == 0) continue;

            const uint8_t* srcCodes = invlists->get_codes(listNo);
            const faiss::idx_t* srcIds = invlists->get_ids(listNo);
            IndexParam<uint8_t, float, ascend_idx_t> param(deviceId, listSize, 0, 0);
            param.listId = listNo;
            param.query = srcCodes;
            param.label = const_cast<ascend_idx_t*>(reinterpret_cast<const ascend_idx_t*>(srcIds));
            indexIVFPQAdd(param);
            totalForDevice += listSize;

            APP_LOG_DEBUG("Successfully uploaded list %zu (%zu vectors) to device %d\n", listNo, listSize, deviceId);
        }

        APP_LOG_INFO("Device %d: uploaded %zu vectors\n", deviceId, totalForDevice);
    };

    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, uploadFunctor);
}

void AscendIndexIVFPQImpl::copyFrom(const faiss::IndexIVFPQ* index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFPQ copyFrom operation started.\n");

    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->is_trained, "Source index is not trained");

    for (int deviceId : indexConfig.deviceList)
    {
        auto pIndex = getActualIndex(deviceId);
        if (pIndex)
        {
            pIndex->reset();
        }
    }

    this->intf_->metric_type = index->metric_type;
    this->intf_->is_trained = index->is_trained;
    this->intf_->ntotal = index->ntotal;
    this->ivfConfig.cp = index->cp;
    this->intf_->d = index->d;
    nlist = index->nlist;
    nprobe = index->nprobe;

    copyFromCentroids(index);

    copyFromCodebook(index);

    copyFromPQCodes(index);

    APP_LOG_INFO("AscendIndexIVFPQ copyFrom operation finished.\n");
}

void AscendIndexIVFPQImpl::copyToPQCodes(faiss::IndexIVFPQ* index) const
{
    // copy pqcode to index
    index->code_size = pq.M;

    InvertedLists* ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->intf_->is_trained)
    {
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++)
        {
            int deviceId = indexConfig.deviceList[i];
            indexIVFFastGetListCodes(deviceId, nlist, ivf);
        }
    }
    index->ntotal = 0;
    for (int i = 0; i < nlist; i++)
    {
        index->ntotal += static_cast<idx_t>(ivf->list_size(i));
    }
}

void AscendIndexIVFPQImpl::copyTo(faiss::IndexIVFPQ* index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVFPQ copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index is not trained");
    if (index->ntotal > 0)
    {
        index->reset();
    }
    index->d = this->intf_->d;
    index->metric_type = this->intf_->metric_type;
    index->is_trained = this->intf_->is_trained;
    index->nlist = nlist;
    index->nprobe = nprobe;
    index->by_residual = false;
    index->cp = this->ivfConfig.cp;
    index->pq.M = pq.M;
    index->pq.nbits = pq.nbits;
    index->pq.dsub = pq.dim / pq.M;
    index->pq.ksub = pq.ksub;
    faiss::IndexFlat* quantizer = nullptr;
    if (this->intf_->metric_type == faiss::METRIC_INNER_PRODUCT)
    {
        quantizer = new faiss::IndexFlatIP(this->intf_->d);
    }
    else
    {
        quantizer = new faiss::IndexFlatL2(this->intf_->d);
    }
    if (!indexConfig.deviceList.empty())
    {
        std::vector<float> centroids(intf_->d * nlist);
        auto pIndex = getActualIndex(indexConfig.deviceList[0]);
        auto ret = aclrtMemcpy(centroids.data(), intf_->d * nlist * sizeof(float), pIndex->centroidsOnDevice->data(),
                               pIndex->centroidsOnDevice->size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        quantizer->add(nlist, centroids.data());
    }
    index->quantizer = quantizer;
    index->own_fields = true;
    size_t codebook_size = pq.M * pq.ksub * pq.dsub;
    index->pq.centroids.resize(codebook_size);
    if (indexConfig.deviceList.size() > 0)
    {
        auto pIndex = getActualIndex(indexConfig.deviceList[0]);
        FAISS_THROW_IF_NOT_MSG(pIndex, "invalid device index");
        auto ret =
            aclrtMemcpy(index->pq.centroids.data(), codebook_size * sizeof(float), pIndex->codeBookOnDevice->data(),
                        codebook_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
    }

    copyToPQCodes(index);
    APP_LOG_INFO("AscendIndexIVFPQ copyTo operation finished.\n");
}

void AscendIndexIVFPQImpl::checkParams() const
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
                               this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");

    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(), "Unsupported dims: %d\n",
                           this->intf_->d);
    FAISS_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(),
                           "Unsupported nlists: %d\n", this->nlist);
    FAISS_THROW_IF_NOT_FMT(
        std::find(MSUBS.begin(), MSUBS.end(), this->msubs) != MSUBS.end() && this->intf_->d % this->msubs == 0,
        "Unsupported msubs: %d\n", this->msubs);
    FAISS_THROW_IF_NOT_FMT(std::find(NBITS.begin(), NBITS.end(), this->nbits) != NBITS.end(), "Unsupported nbits: %d\n",
                           this->nbits);
}

void AscendIndexIVFPQImpl::initProductQuantizer()
{
    APP_LOG_INFO("AscendIndexIVFPQImpl initProductQuantizer operation started\n");
    pq.nlist = static_cast<uint32_t>(nlist);
    pq.dim = static_cast<uint32_t>(this->intf_->d);
    pq.nbits = static_cast<uint32_t>(nbits);
    pq.M = static_cast<uint32_t>(msubs);
    pq.ksub = 1 << nbits;
    pq.dsub = static_cast<uint32_t>(this->intf_->d / msubs);
    pq.codeBook.resize(pq.M * pq.ksub * pq.dsub);
    APP_LOG_INFO("AscendIndexIVFPQImpl initProductQuantizer operation finished\n");
}

std::vector<idx_t> AscendIndexIVFPQImpl::update(idx_t n, const float* x, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "vector list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(ids != nullptr, "vector ID list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(n > 0, "update vector number must be greater than 0!");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(n) * this->intf_->d == std::distance(x, x + n * this->intf_->d),
                           "vector list size is not match!");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(n) == std::distance(ids, ids + n), "vector ID list size is not match!");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "AscendIndexIVFPQ is not trained!");
    APP_LOG_INFO("AscendIndexIVFPQImpl update operation started: n=%ld.\n", n);
    std::lock_guard<std::mutex> lock(mapMutex);

    std::vector<idx_t> noExistIds;
    std::vector<idx_t> existIds;
    std::vector<float> existVectors;
    idx_t noExistNum = 0;
    idx_t existNum = 0;

    for (idx_t i = 0; i < n; ++i)
    {
        idx_t id = ids[i];
        if (idToListMap.find(id) == idToListMap.end())
        {
            noExistIds.push_back(id);
            noExistNum++;
            continue;
        }
        idx_t listId = idToListMap[id];
        if (listInfos[listId].idSet.find(id) == listInfos[listId].idSet.end())
        {
            noExistIds.push_back(id);
            noExistNum++;
            continue;
        }
        existIds.push_back(id);
        const float* vector = x + i * this->intf_->d;
        existVectors.insert(existVectors.end(), vector, vector + this->intf_->d);
        existNum++;
    }

    if (!noExistIds.empty())
    {
        APP_LOG_WARNING("The following %d IDs do not exist: \n", noExistNum);
        for (idx_t i = 0; i < noExistNum; i++)
        {
            APP_LOG_WARNING("ID: %ld\n", noExistIds[i]);
        }
        APP_LOG_WARNING("Updating other vectors with ids\n");
    }
    if (existNum > 0)
    {
        deleteImpl(existNum, existIds.data());
        addImpl(existNum, existVectors.data(), existIds.data());
    }
    APP_LOG_INFO("AscendIndexIVFPQ update operation finished.\n");

    return noExistIds;
}

void AscendIndexIVFPQImpl::addL1(int n, const float* x, std::vector<int64_t>& assign)
{
    auto l1_start = std::chrono::high_resolution_clock::now();
    if (ivfConfig.useKmeansPP)
    {
        const size_t deviceCnt = indexConfig.deviceList.size();
        FAISS_THROW_IF_NOT_MSG(deviceCnt > 0, "device list is empty");
        auto assignOnDevice = [&](size_t deviceIdx, int count, size_t offset, std::vector<int64_t>& outAssign)
        {
            auto pIndex = getActualIndex(indexConfig.deviceList[deviceIdx]);
            FAISS_THROW_IF_NOT_MSG(pIndex != nullptr, "device is invalid");
            auto ret = pIndex->assignCentroid(this->nlist, this->intf_->d, count, centroidsOnHost,
                                              const_cast<float*>(x + offset * this->intf_->d), outAssign, true);
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "failed to assign centroids from device %d, ret: %d",
                                   indexConfig.deviceList[deviceIdx], ret);
        };

        if (deviceCnt == 1)
        {
            // Avoid thread-pool overhead on the common single-device path.
            const auto deviceStart = std::chrono::high_resolution_clock::now();
            assignOnDevice(0, n, 0, assign);
            if (this->intf_->verbose)
            {
                APP_LOG_INFO("IVFPQ addL1 device: id=%d, vectors=%d, elapsed=%lld ms\n", indexConfig.deviceList[0], n,
                             elapsedMs(deviceStart));
            }
        }
        else
        {
            std::vector<long long> deviceElapsed(deviceCnt, 0);
            std::vector<int> deviceCounts(deviceCnt, 0);
            auto assignFunctor = [&](size_t deviceIdx)
            {
                const size_t base = static_cast<size_t>(n) / deviceCnt;
                const size_t remainder = static_cast<size_t>(n) % deviceCnt;
                const size_t count = base + (deviceIdx < remainder ? 1 : 0);
                const size_t offset = deviceIdx * base + std::min(deviceIdx, remainder);
                deviceCounts[deviceIdx] = static_cast<int>(count);
                if (count == 0)
                {
                    return;
                }

                const auto deviceStart = std::chrono::high_resolution_clock::now();
                std::vector<int64_t> deviceAssign(count);
                assignOnDevice(deviceIdx, static_cast<int>(count), offset, deviceAssign);
                std::copy(deviceAssign.begin(), deviceAssign.end(), assign.data() + offset);
                deviceElapsed[deviceIdx] = elapsedMs(deviceStart);
            };
            CALL_PARALLEL_FUNCTOR(deviceCnt, pool, assignFunctor);
            if (this->intf_->verbose)
            {
                for (size_t i = 0; i < deviceCnt; ++i)
                {
                    APP_LOG_INFO("IVFPQ addL1 device: id=%d, vectors=%d, elapsed=%lld ms\n", indexConfig.deviceList[i],
                                 deviceCounts[i], deviceElapsed[i]);
                }
            }
        }
    }
    else
    {
        pQuantizerImpl->cpuQuantizer->assign(n, x, assign.data());
    }
    if (this->intf_->verbose)
    {
        APP_LOG_INFO("IVFPQ addL1: assigned %d vectors, elapsed=%lld ms\n", n, elapsedMs(l1_start));
    }
}

void AscendIndexIVFPQImpl::addL2(int n, const float* x, std::vector<uint8_t>& pqCodes)
{
    const auto encodeStart = std::chrono::high_resolution_clock::now();
    const size_t codeSize = pq.M;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        const float* vector = x + static_cast<size_t>(i) * intf_->d;
        uint8_t* code = pqCodes.data() + static_cast<size_t>(i) * codeSize;
        encodeSingleVectorPQ(vector, code);
    }
    if (this->intf_->verbose)
    {
        const long long elapsed = elapsedMs(encodeStart);
        const double throughput = elapsed > 0 ? 1000.0 * n / elapsed : 0.0;
        APP_LOG_INFO("IVFPQ addL2: encoded %d vectors, elapsed=%lld ms, throughput=%.2f vectors/s\n", n, elapsed,
                     throughput);
    }
}

void AscendIndexIVFPQImpl::addImpl(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFPQImpl addImpl operation started: n=%d.\n", n);
    this->intf_->metric_type = faiss::METRIC_L2;
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<int64_t> assign(n);
    addL1(n, x, assign);
    size_t code_size = pq.M;
    std::vector<uint8_t> pqCodes(n * static_cast<int>(code_size));
    addL2(n, x, pqCodes);

    std::vector<idx_t> idList;
    std::vector<idx_t> listIdList;
    idList.reserve(n);
    listIdList.reserve(n);

    const auto bucketStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);

        if (ids != nullptr)
        {
            idList.push_back(ids[i]);
            listIdList.push_back(listId);
        }

        auto it = assignCounts.find(listId);
        if (it != assignCounts.end())
        {
            int deviceIdx = it->second.addDeviceIdx;
            deviceAddNumMap[listId][deviceIdx]++;
            if (ids != nullptr)
            {
                idToDeviceMap[ids[i]] = indexConfig.deviceList[deviceIdx];
            }
            it->second.Add(pqCodes.data() + i * pq.M, ids ? (ids + i) : nullptr);
            continue;
        }
        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++)
        {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx])
            {
                devIdx = j;
            }
        }
        deviceAddNumMap[listId][devIdx]++;
        if (ids != nullptr)
        {
            idToDeviceMap[ids[i]] = indexConfig.deviceList[devIdx];
        }
        assignCounts.emplace(listId, AscendIVFAddInfo(devIdx, deviceCnt, code_size));
        assignCounts.at(listId).Add(pqCodes.data() + i * code_size, ids ? (ids + i) : nullptr);
    }
    // update idList for delete
    if (ids != nullptr)
    {
        updateIdMapping(idList, listIdList);
    }
    if (this->intf_->verbose)
    {
        APP_LOG_INFO("IVFPQ add bucket: grouped %d vectors into %zu lists, elapsed=%lld ms\n", n, assignCounts.size(),
                     elapsedMs(bucketStart));
    }
}

void AscendIndexIVFPQImpl::copyVectorToDevice(int n)
{
    const auto uploadStart = std::chrono::high_resolution_clock::now();
    size_t deviceCnt = indexConfig.deviceList.size();
    auto addFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        for (auto& centroid : assignCounts)
        {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0)
            {
                continue;
            }
            uint8_t* pqCodePtr = nullptr;
            ascend_idx_t* idPtr = nullptr;

            centroid.second.GetCodeAndIdPtr(idx, &pqCodePtr, &idPtr);
            IndexParam<uint8_t, float, ascend_idx_t> param(deviceId, num, 0, 0);
            param.listId = listId;
            param.query = pqCodePtr;
            param.label = idPtr;
            indexIVFPQAdd(param);
            deviceAddNumMap[listId][idx] += num;
        }
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    this->intf_->ntotal += n;
    if (this->intf_->verbose)
    {
        const long long elapsed = elapsedMs(uploadStart);
        const double throughput = elapsed > 0 ? 1000.0 * n / elapsed : 0.0;
        APP_LOG_INFO(
            "IVFPQ add upload: copied %d vectors to %zu device(s), elapsed=%lld ms, "
            "throughput=%.2f vectors/s\n",
            n, deviceCnt, elapsed, throughput);
    }
    APP_LOG_INFO("AscendIndexIVFPQ addImpl operation finished.\n");
}

size_t AscendIndexIVFPQImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

void AscendIndexIVFPQImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    const auto addStart = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("AscendIndexIVFPQImpl addPaged operation started.\n");
    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    size_t addPageSize = ADD_PAGE_SIZE;
    if (totalSize > addPageSize || static_cast<size_t>(n) > ADD_VEC_SIZE * 10)
    {
        size_t tileSize = getAddPagedSize(n);
        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize)
        {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose)
            {
                APP_LOG_INFO("AscendIndexIVFPQImpl::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr);
        }
    }
    else
    {
        if (this->intf_->verbose)
        {
            APP_LOG_INFO("AscendIndexIVFPQImpl::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    copyVectorToDevice(n);
    std::unordered_map<int, AscendIVFAddInfo>().swap(assignCounts);  // 释放host侧占用的内存
    // Free assign/train workspaces left by addL1::assignCentroid on every device.
    for (int deviceId : indexConfig.deviceList)
    {
        auto pIndex = getActualIndex(deviceId);
        if (pIndex)
        {
            pIndex->resetTrainSession();
        }
    }
    if (this->intf_->verbose)
    {
        const long long elapsed = elapsedMs(addStart);
        const double throughput = elapsed > 0 ? 1000.0 * n / elapsed : 0.0;
        APP_LOG_INFO("IVFPQ add total: added %d vectors, elapsed=%lld ms, throughput=%.2f vectors/s\n", n, elapsed,
                     throughput);
    }
    APP_LOG_INFO("AscendIndexIVFPQImpl addPaged operation finished.\n");
}

void AscendIndexIVFPQImpl::indexIVFPQAdd(IndexParam<uint8_t, float, ascend_idx_t>& param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;

    const uint8_t* pqCodes = param.query;
    const ascend_idx_t* ids = param.label;

    auto ret = pIndex->addPQCodes(param.listId, param.n, pqCodes, static_cast<const ::ascend::Index::idx_t*>(ids));

    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to add to ivf PQ, ret: %d", ret);
}

void AscendIndexIVFPQImpl::encodeSingleVectorPQ(const float* vector, uint8_t* pqCode) const
{
    for (size_t m = 0; m < pq.M; m++)
    {
        const float* subVector = vector + m * this->pq.dsub;
        pqCode[m] = findCentroidInSubQuantizer(m, subVector);
    }
}

uint8_t AscendIndexIVFPQImpl::findCentroidInSubQuantizer(size_t subqIdx, const float* subVector) const
{
    if (subVector == nullptr || subqIdx >= this->pq.M)
    {
        return 0;
    }

    float minDist = std::numeric_limits<float>::max();
    uint8_t findCentroid = 0;
    const float* subCodebook = this->pq.codeBook.data() + subqIdx * this->pq.ksub * this->pq.dsub;
    for (size_t k = 0; k < this->pq.ksub; k++)
    {
        const float* centroid = subCodebook + k * this->pq.dsub;
        float dist = calDistance(subVector, centroid, this->pq.dsub);
        if (dist < minDist)
        {
            minDist = dist;
            findCentroid = static_cast<uint8_t>(k);
        }
    }

    return findCentroid;
}

float AscendIndexIVFPQImpl::calDistance(const float* a, const float* b, size_t dim) const
{
    float dist = 0.0;
    for (size_t i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

std::shared_ptr<::ascend::Index> AscendIndexIVFPQImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFPQ createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::IndexIVF> index =
        std::make_shared<::ascend::IndexIVFPQ>(nlist, this->intf_->d, msubs, nbits, nprobe, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf PQ, result is %d", ret);

    APP_LOG_INFO("AscendIndexIVFPQ createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFPQImpl::extractAllSubspaces(int nSampled, const std::vector<size_t>& sampleIndices, const float* x,
                                               std::vector<std::vector<float>>& subspaceData)
{
    const idx_t dim = static_cast<idx_t>(this->pq.dim);
    const idx_t dsub = static_cast<idx_t>(this->pq.dsub);
    subspaceData.resize(this->pq.M);
    for (size_t m = 0; m < this->pq.M; m++)
    {
        subspaceData[m].resize(static_cast<size_t>(nSampled) * static_cast<size_t>(dsub));
    }

#pragma omp parallel for if (nSampled > 100)
    for (int i = 0; i < nSampled; i++)
    {
        const idx_t srcRow = static_cast<idx_t>(sampleIndices[static_cast<size_t>(i)]);
        const float* row = x + srcRow * dim;
        for (size_t m = 0; m < this->pq.M; m++)
        {
            const float* sub_vec = row + static_cast<idx_t>(m) * dsub;
            float* dst = subspaceData[m].data() + static_cast<idx_t>(i) * dsub;
            std::copy(sub_vec, sub_vec + dsub, dst);
        }
    }
}

void AscendIndexIVFPQImpl::trainPQCodeBook(idx_t n, const float* x)
{
    auto pq_start = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("Training PQ codebook with %ld vectors\n", n);

    FAISS_THROW_IF_NOT_MSG(n >= static_cast<idx_t>(this->pq.M * this->pq.ksub), "Insufficient training data");
    FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "Coarse quantizer not trained");

    const int pqTrainSize = computeTrainTotalSize(static_cast<int>(this->pq.ksub), static_cast<int>(n),
                                                  ivfConfig.trainSamplesPerList, ivfConfig.maxTrainSamples);
    std::vector<size_t> sampleIndices;
    if (ivfConfig.useKmeansPP)
    {
        buildSampleIndices(static_cast<int>(n), pqTrainSize, this->ivfConfig.cp.seed, sampleIndices);
        APP_LOG_INFO("IVFPQ trainPQCodeBook: pqTrainSize=%d (from n=%ld, ksub=%zu)\n", pqTrainSize,
                     static_cast<long>(n), this->pq.ksub);
    }
    else
    {
        sampleIndices.resize(static_cast<size_t>(n));
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        APP_LOG_INFO("IVFPQ trainPQCodeBook: CPU path using full n=%ld vectors\n", static_cast<long>(n));
    }
    const int trainCount = ivfConfig.useKmeansPP ? pqTrainSize : static_cast<int>(n);
    const int pqNiter = ivfConfig.pqNiter >= 0 ? ivfConfig.pqNiter : this->ivfConfig.cp.niter;

    auto extract_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> subspaceData;
    extractAllSubspaces(trainCount, sampleIndices, x, subspaceData);
    APP_LOG_INFO("IVFPQ trainPQCodeBook: extracted %zu subspaces, nSampled=%d, elapsed=%lld ms\n", this->pq.M,
                 trainCount, static_cast<long long>(elapsedMs(extract_start)));

    const size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<std::mutex> deviceTrainMutex(deviceCnt);
    auto trainFunctor = [&](size_t m)
    {
        const size_t devIdx = m % deviceCnt;
        const int deviceId = indexConfig.deviceList[devIdx];
        std::lock_guard<std::mutex> lock(deviceTrainMutex[devIdx]);
        trainSubQuantizer(m, trainCount, subspaceData[m], deviceId, pqNiter);
    };

    if (ivfConfig.useKmeansPP && deviceCnt > 1 && this->pq.M > 1)
    {
        std::vector<std::future<void>> futures;
        for (size_t m = 0; m < this->pq.M; m++)
        {
            futures.emplace_back(GetPool()->Enqueue(trainFunctor, m));
        }
        for (auto& future : futures)
        {
            future.get();
        }
    }
    else
    {
        for (size_t m = 0; m < this->pq.M; m++)
        {
            trainFunctor(m);
        }
    }

    if (!indexConfig.deviceList.empty())
    {
        for (int deviceId : indexConfig.deviceList)
        {
            getActualIndex(deviceId)->resetTrainSession();
        }
    }

    APP_LOG_INFO("IVFPQ trainPQCodeBook finished, elapsed=%lld ms\n", static_cast<long long>(elapsedMs(pq_start)));
}

void AscendIndexIVFPQImpl::trainSubQuantizer(size_t m, int nSampled, const std::vector<float>& subspace_data,
                                             int deviceId, int pqNiter)
{
    FAISS_THROW_IF_NOT_MSG(nSampled >= static_cast<int>(this->pq.ksub),
                           "Insufficient data for sub-quantizer clustering");

    if (ivfConfig.useKmeansPP)
    {
        try
        {
            APP_LOG_INFO("IVFPQ PQ sub-quantizer %zu: NPU K-Means, n_data=%d, dsub=%d, ksub=%d, device=%d\n", m,
                         nSampled, this->pq.dsub, this->pq.ksub, deviceId);
            auto train_start = std::chrono::high_resolution_clock::now();
            std::vector<float> localCentroids;
            indexTrainImpl(nSampled, subspace_data.data(), this->intf_->d / this->pq.M, this->pq.ksub, deviceId,
                           localCentroids, true, pqNiter);
            APP_LOG_INFO("IVFPQ PQ sub-quantizer %zu: NPU training finished, train=%lld ms\n", m,
                         static_cast<long long>(elapsedMs(train_start)));
            savePQCodeBook(m, localCentroids);
        }
        catch (std::exception& e)
        {
            FAISS_THROW_FMT("IVFPQ NPU training failed for sub-quantizer %zu: %s", m, e.what());
        }
    }
    else
    {
        APP_LOG_INFO("IVFPQ PQ sub-quantizer %zu: CPU Clustering, n_data=%d, dsub=%d, ksub=%d\n", m, nSampled,
                     this->pq.dsub, this->pq.ksub);
        faiss::ClusteringParameters cp;
        cp.niter = pqNiter;
        cp.spherical = true;
        cp.nredo = 1;
        cp.verbose = this->intf_->verbose;

        faiss::Clustering clus(this->pq.dsub, this->pq.ksub, cp);
        faiss::IndexFlatL2 index(this->pq.dsub);

        clus.train(nSampled, subspace_data.data(), index);

        savePQCodeBook(m, clus.centroids);
    }

    APP_LOG_DEBUG("Sub-quantizer %zu: trained with %d vectors\n", m, nSampled);
}

void AscendIndexIVFPQImpl::savePQCodeBook(size_t m, const std::vector<float>& centroids)
{
    FAISS_THROW_IF_NOT_FMT(centroids.size() == this->pq.ksub * this->pq.dsub,
                           "centroids size error: expect %zu, actual %zu\n", this->pq.ksub * this->pq.dsub,
                           centroids.size());
    FAISS_THROW_IF_NOT_FMT(m < this->pq.M, "sub-quantizer index out of range: %zu", m);
    float* dst = this->pq.codeBook.data() + m * this->pq.ksub * this->pq.dsub;
    std::copy(centroids.begin(), centroids.end(), dst);
}

void AscendIndexIVFPQImpl::updatePQCodeBook()
{
    auto upload_start = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("Updating PQ codebook to device...\n");

    const size_t codebookElems = this->pq.codeBook.size();
    const size_t codebookBytes = codebookElems * sizeof(float);
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());

    for (int deviceId : indexConfig.deviceList)
    {
        auto pIndex = getActualIndex(deviceId);
        float* device_ptr = pIndex->codeBookOnDevice->data();

        auto ret =
            aclrtMemcpy(device_ptr, codebookBytes, this->pq.codeBook.data(), codebookBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
    }

    APP_LOG_INFO("IVFPQ updatePQCodeBook: bulk H2D %zu bytes to %d device(s), elapsed=%lld ms\n", codebookBytes,
                 deviceCnt, static_cast<long long>(elapsedMs(upload_start)));
}

void AscendIndexIVFPQImpl::updateCoarseCenter(std::vector<float>& centerData)
{
    std::vector<float> centroidsSqrSum(nlist, 0.0f);
    for (int i = 0; i < nlist; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < intf_->d; j++)
        {
            float val = centerData[i * intf_->d + j];
            sum += val * val;
        }
        centroidsSqrSum[i] = sum;
    }
    centroidsOnHost = centerData;
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());
    for (int i = 0; i < deviceCnt; i++)
    {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = aclrtMemcpy(pIndex->centroidsOnDevice->data(), pIndex->centroidsOnDevice->size() * sizeof(float),
                               centerData.data(), intf_->d * nlist * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy centroids error %d", ret);

        ret = aclrtMemcpy(pIndex->centroidsSqrSumOnDevice->data(),
                          pIndex->centroidsSqrSumOnDevice->size() * sizeof(float), centroidsSqrSum.data(),
                          nlist * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy centroidsSqrSum error %d", ret);
    }
}

void AscendIndexIVFPQImpl::train(idx_t n, const float* x)
{
    auto train_total_start = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("AscendIndexIVFPQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained)
    {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT)
    {
        APP_LOG_INFO("METRIC_INNER_PRODUCT must set spherical to true in cpu train case\n");
        this->ivfConfig.cp.spherical = true;
    }

    auto coarse_start = std::chrono::high_resolution_clock::now();
    if (!ivfConfig.useKmeansPP)
    {
        APP_LOG_INFO("IVFPQ Coarse quantizer: CPU Clustering, n=%ld, d=%d, nlist=%d\n", static_cast<long>(n),
                     this->intf_->d, this->nlist);
        this->ivfConfig.cp.niter = 25;        // iter nums
        this->ivfConfig.cp.spherical = true;  // spherical clus flag
        this->ivfConfig.cp.nredo = 1;
        this->ivfConfig.cp.verbose = this->intf_->verbose;

        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer, "cpuQuantizer is not init.");
        clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));

        updateCoarseCenter(clus.centroids);
        centroidsData = clus.centroids;
    }
    else
    {
        const int totalSize = computeTrainTotalSize(this->nlist, static_cast<int>(n), ivfConfig.trainSamplesPerList,
                                                    ivfConfig.maxTrainSamples);
        const bool useDistributed = ivfConfig.useDistributedCoarse && indexConfig.deviceList.size() > 1;
        APP_LOG_INFO(
            "IVFPQ Coarse quantizer: AscendClustering, n=%ld, sampled=%d, d=%d, nlist=%d, device=%d, distributed=%d\n",
            static_cast<long>(n), totalSize, this->intf_->d, this->nlist, indexConfig.deviceList[0],
            static_cast<int>(useDistributed));

        std::vector<float> trainData;
        sampleTrainData(x, static_cast<int>(n), this->intf_->d, totalSize, this->ivfConfig.cp.seed, trainData);

        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->npuClus, "npuClus is not init.");
        pQuantizerImpl->npuClus->verbose = this->intf_->verbose;
        std::vector<float> tmpCentroids(static_cast<size_t>(nlist) * static_cast<size_t>(this->intf_->d));

        if (useDistributed)
        {
            // Distributed path clusters in fp32 across all devices via
            // IndexIVFPQ::runKMeans (same assignment backend as single-device
            // TrainFp32). It shares the fp32 code buffer with TrainFp32 and
            // performs no internal sampling, so the host-side sample above
            // still applies.
            if (pQuantizerImpl->npuClus->GetNTotal() == 0)
            {
                pQuantizerImpl->npuClus->AddFp32(totalSize, trainData.data());
            }
            if (this->intf_->verbose)
            {
                APP_LOG_INFO("Ascend cluster start distributed training %zu vectors on %zu devices\n",
                             pQuantizerImpl->npuClus->GetNTotal(), indexConfig.deviceList.size());
            }
            pQuantizerImpl->npuClus->DistributedTrainFp32(this->ivfConfig.cp.niter, tmpCentroids.data(),
                                                          indexConfig.deviceList, true);
        }
        else
        {
            if (pQuantizerImpl->npuClus->GetNTotal() == 0)
            {
                pQuantizerImpl->npuClus->AddFp32(totalSize, trainData.data());
            }
            if (this->intf_->verbose)
            {
                APP_LOG_INFO("Ascend cluster start training %zu vectors\n", pQuantizerImpl->npuClus->GetNTotal());
            }
            pQuantizerImpl->npuClus->TrainFp32(this->ivfConfig.cp.niter, tmpCentroids.data(), true);
        }

        centroidsData = tmpCentroids;
        updateCoarseCenter(tmpCentroids);
        centroidsOnHost = tmpCentroids;

        pQuantizerImpl->cpuQuantizer->reset();
        pQuantizerImpl->cpuQuantizer->add(nlist, tmpCentroids.data());
        pQuantizerImpl->cpuQuantizer->is_trained = true;
    }
    APP_LOG_INFO("IVFPQ train coarse quantizer finished, elapsed=%lld ms\n",
                 static_cast<long long>(elapsedMs(coarse_start)));

    trainPQCodeBook(n, x);

    auto upload_start = std::chrono::high_resolution_clock::now();
    updatePQCodeBook();
    APP_LOG_INFO("IVFPQ train codebook upload finished, elapsed=%lld ms\n",
                 static_cast<long long>(elapsedMs(upload_start)));

    this->intf_->is_trained = true;
    APP_LOG_INFO("IVFPQ train total finished, elapsed=%lld ms\n", static_cast<long long>(elapsedMs(train_total_start)));
}

void AscendIndexIVFPQImpl::indexTrainImpl(int n, const float* x, int dim, int nlist, int deviceId,
                                          std::vector<float>& centroidsOut, bool dataAlreadySampled, int niterOverride)
{
    if (n <= 0 || x == nullptr)
    {
        FAISS_THROW_MSG("train data invalid");
    }

    if (indexConfig.deviceList.empty())
    {
        FAISS_THROW_MSG("NPU device invalid");
    }

    auto pIndex = getActualIndex(deviceId);
    if (!pIndex)
    {
        FAISS_THROW_FMT("device %d Index not reset", deviceId);
    }

    const size_t data_bytes = static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(float);
    APP_LOG_INFO("IVFPQ NPU clustering indexTrainImpl: device=%d, n=%d, dim=%d, nlist=%d, dataBytes=%zu\n", deviceId, n,
                 dim, nlist, data_bytes);

    auto train_start = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("IVFPQ NPU clustering: IndexIVFPQ::trainImpl starting on device %d\n", deviceId);

    const int niter = niterOverride >= 0 ? niterOverride : this->ivfConfig.cp.niter;
    auto ret = pIndex->trainImpl(n, x, dim, nlist, niter, this->ivfConfig.cp.seed, dataAlreadySampled);
    FAISS_THROW_IF_NOT_MSG(ret == ::ascend::APP_ERR_OK, "IndexIVFPQ::trainImpl failed");
    centroidsOut.resize(static_cast<size_t>(dim) * static_cast<size_t>(nlist));
    size_t centroids_bytes = static_cast<size_t>(nlist) * static_cast<size_t>(dim) * sizeof(float);
    ret = aclrtMemcpy(centroidsOut.data(), centroids_bytes, pIndex->clusteringOnDevice->data(), centroids_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "update centroids result to Host failed: %d", ret);

    APP_LOG_INFO("IVFPQ NPU clustering done: copied %zu centroid bytes, elapsed=%lld ms\n", centroids_bytes,
                 elapsedMs(train_start));
}

void AscendIndexIVFPQImpl::indexSearch(IndexParam<float, float, ascend_idx_t>& param) const
{
    auto pIndex = getActualIndex(param.deviceId);
    auto ret = pIndex->searchImpl(param.n, param.query, param.k, param.distance, param.label);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to search index,deviceId is %d, result is: %d\n",
                           param.deviceId, ret);
}

void AscendIndexIVFPQImpl::searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        IndexParam<float, float, ascend_idx_t> param(deviceId, n, this->intf_->d, k);
        param.query = x;
        param.distance = dist[idx].data();
        param.label = label[idx].data();
        indexSearch(param);
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
}

void AscendIndexIVFPQImpl::deleteImpl(int n, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFPQImpl deleteImpl operation started: n=%d.\n", n);

    // Group by (deviceId, listId) pair
    std::map<std::pair<int, idx_t>, std::vector<idx_t>> deviceListIdMap;

    for (int i = 0; i < n; i++)
    {
        idx_t id = ids[i];
        int deviceId = findDeviceId(id);
        idx_t listId = findListId(id);

        if (deviceId >= 0 && listId >= 0 && listId < this->nlist)
        {
            deviceListIdMap[{deviceId, listId}].push_back(id);
        }
        else
        {
            APP_LOG_WARNING("Could not find valid mapping for ID %ld, skipping\n", id);
        }
    }

    for (auto& entry : deviceListIdMap)
    {
        int deviceId = entry.first.first;
        idx_t listId = entry.first.second;
        auto& deleteIds = entry.second;

        if (deleteIds.empty())
        {
            continue;
        }

        std::vector<ascend_idx_t> ascendIds(deleteIds.begin(), deleteIds.end());

        IndexParam<void, void, ascend_idx_t> param(deviceId, deleteIds.size(), 0, 0);
        param.listId = listId;
        param.label = ascendIds.data();
        deleteFromIVFPQ(param);

        removeIdMapping(deleteIds);
    }

    this->intf_->ntotal -= n;
    APP_LOG_INFO("AscendIndexIVFPQImpl deleteImpl operation finished.\n");
}

void AscendIndexIVFPQImpl::deleteFromIVFPQ(IndexParam<void, void, ascend_idx_t>& param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;

    const ascend_idx_t* ids = param.label;

    auto ret = pIndex->deletePQCodes(param.listId, param.n, static_cast<const ::ascend::Index::idx_t*>(ids));

    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to delete from ivf PQ, ret: %d", ret);
}

idx_t AscendIndexIVFPQImpl::findListId(idx_t id)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    auto it = idToListMap.find(id);
    if (it != idToListMap.end())
    {
        return it->second;
    }

    APP_LOG_WARNING("ID %ld not found in host mapping, attempting to find in device data\n", id);

    idx_t fallbackListId = id % this->nlist;
    APP_LOG_WARNING("ID %ld not found, using fallback listId: %ld\n", id, fallbackListId);
    return fallbackListId;
}

int AscendIndexIVFPQImpl::findDeviceId(idx_t id)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    auto it = idToDeviceMap.find(id);
    if (it != idToDeviceMap.end())
    {
        return it->second;
    }

    APP_LOG_WARNING("ID %ld not found in device mapping, attempting to find in device data\n", id);

    size_t deviceCnt = indexConfig.deviceList.size();
    int fallbackDeviceId = indexConfig.deviceList[id % deviceCnt];
    APP_LOG_WARNING("ID %ld not found, using fallback deviceId: %d\n", id, fallbackDeviceId);
    return fallbackDeviceId;
}

void AscendIndexIVFPQImpl::updateIdMapping(const std::vector<idx_t>& ids, const std::vector<idx_t>& listIds)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    FAISS_THROW_IF_NOT(ids.size() == listIds.size());

    for (size_t i = 0; i < ids.size(); i++)
    {
        idx_t id = ids[i];
        idx_t listId = listIds[i];

        idToListMap[id] = listId;

        if (static_cast<idx_t>(listInfos.size()) <= listId)
        {
            listInfos.resize(listId + 1);
        }
        listInfos[listId].idSet.insert(id);
    }
    APP_LOG_DEBUG("Updated batch mapping for %zu IDs\n", ids.size());
}

void AscendIndexIVFPQImpl::removeIdMapping(const std::vector<idx_t>& ids)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    for (idx_t id : ids)
    {
        auto it = idToListMap.find(id);
        if (it != idToListMap.end())
        {
            idx_t listId = it->second;

            if (listId < static_cast<idx_t>(listInfos.size()))
            {
                listInfos[listId].idSet.erase(id);
            }
            idToListMap.erase(it);
        }
        idToDeviceMap.erase(id);
    }
    APP_LOG_DEBUG("Removed batch mapping for %zu IDs\n", ids.size());
}

size_t AscendIndexIVFPQImpl::getAddElementSize() const { return static_cast<size_t>(intf_->d) * sizeof(float); }
}  // namespace ascend
}  // namespace faiss
