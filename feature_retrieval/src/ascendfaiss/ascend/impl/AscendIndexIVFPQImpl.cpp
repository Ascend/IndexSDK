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
#include <cfloat>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/impl/ProductQuantizer.h>
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/custom/AscendClustering.h"

namespace faiss {
namespace ascend {

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
const std::vector<int> NLISTS = {1024, 2048, 4096, 8192, 16384};

// The value range of msub
const std::vector<int> MSUBS = {4};

// The value range of nbit
const std::vector<int> NBITS = {8};

AscendIndexIVFPQImpl::AscendIndexIVFPQImpl(AscendIndexIVFPQ* intf, int dims, int nlist, int msubs, int nbits,
                                           faiss::MetricType metric, AscendIndexIVFPQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config),
      intf_(intf),
      msubs(msubs),
      nbits(nbits)
{
    checkParams();
    initIndexes();
    initDeviceAddNumMap();
    centroidsData.resize(nlist * dims);
    initProductQuantizer();
    this->intf_->is_trained = false;
}

AscendIndexIVFPQImpl::~AscendIndexIVFPQImpl()
{
}

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

    this->pq.codeBook.clear();
    this->pq.codeBook.resize(this->pq.M);

    const float* codeBook_data = index->pq.centroids.data();

    for (size_t m = 0; m < this->pq.M; m++) {
        this->pq.codeBook[m].resize(this->pq.ksub);
        for (size_t k = 0; k < this->pq.ksub; k++) {
            this->pq.codeBook[m][k].resize(this->pq.dsub);
            size_t offset = (m * this->pq.ksub + k) * this->pq.dsub;
            for (size_t d = 0; d < this->pq.dsub; d++) {
                this->pq.codeBook[m][k][d] = codeBook_data[offset + d];
            }
        }
    }
    FAISS_THROW_IF_NOT_FMT(pq.M > 0, "invalid msubs: %zu", pq.M);
    size_t codebook_size = pq.M * pq.ksub * (static_cast<size_t>(intf_->d) / pq.M);
    for (int deviceId : indexConfig.deviceList) {
        auto pIndex = getActualIndex(deviceId);
        if (!pIndex) continue;
        if (pIndex->codeBookOnDevice->size() != codebook_size) {
            pIndex->codeBookOnDevice->resize(codebook_size);
        }

        auto ret = aclrtMemcpy(pIndex->codeBookOnDevice->data(), codebook_size * sizeof(float),
                               index->pq.centroids.data(), codebook_size * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE);

        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                               "Failed to upload PQ codebook to device %d: %d",
                               deviceId, ret);
    }
    size_t code_size = pq.M;
    FAISS_THROW_IF_NOT_FMT(index->code_size == code_size,
                           "Code size mismatch: CPU index has %zu, expected %zu",
                           index->code_size, code_size);
}

void AscendIndexIVFPQImpl::copyFromPQCodes(const faiss::IndexIVFPQ* index)
{
    // copy pqcode from index
    APP_LOG_INFO("Uploading inverted lists data to devices...\n");

    deviceAddNumMap.clear();
    deviceAddNumMap.resize(index->nlist);
    for (size_t i = 0; i < index->nlist; i++) {
        deviceAddNumMap[i].resize(indexConfig.deviceList.size(), 0);
    }

    const faiss::InvertedLists* invlists = index->invlists;
    FAISS_THROW_IF_NOT_MSG(invlists != nullptr, "Source index has no inverted lists");

    size_t deviceCount = indexConfig.deviceList.size();
    size_t totalVectors = index->ntotal;

    std::vector<std::vector<std::pair<size_t, size_t>>> deviceAssignments =
            assignListsToDevices(invlists, deviceCount);

    uploadToDevicesParallel(deviceAssignments, invlists);

    this->intf_->ntotal = index->ntotal;
    this->intf_->is_trained = index->is_trained;

    size_t totalUploaded = 0;
    for (size_t listNo = 0; listNo < static_cast<size_t>(nlist); listNo++) {
        for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
            totalUploaded += static_cast<size_t>(deviceAddNumMap[listNo][devIdx]);
        }
    }

    FAISS_THROW_IF_NOT_FMT(totalUploaded == totalVectors,
                           "Copied vector count mismatch. Expected: %zu, Actual: %zu",
                           totalVectors, totalUploaded);

    APP_LOG_INFO("AscendIndexIVFPQ copyFrom operation finished. "
                 "Successfully copied %zu vectors from source index.\n",
                 totalVectors);
}

std::vector<std::vector<std::pair<size_t, size_t>>> AscendIndexIVFPQImpl::assignListsToDevices(
    const faiss::InvertedLists* invlists, size_t deviceCount)
{
    std::vector<std::vector<std::pair<size_t, size_t>>> deviceAssignments(deviceCount);
    for (size_t listNo = 0; listNo < static_cast<size_t>(nlist); listNo++) {
        size_t listSize = invlists->list_size(listNo);
        if (listSize == 0)
            continue;

        size_t selectedDevice = 0;
        size_t minCount = static_cast<size_t>(deviceAddNumMap[listNo][0]);
        for (size_t devIdx = 1; devIdx < deviceCount; devIdx++) {
            if (static_cast<size_t>(deviceAddNumMap[listNo][devIdx]) < minCount) {
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
    auto uploadFunctor = [&](int devIdx) {
        int deviceId = indexConfig.deviceList[devIdx];
        auto pIndex = getActualIndex(deviceId);
        if (!pIndex) {
            return;
        }

        size_t totalForDevice = 0;
        const auto& assignments = deviceAssignments[devIdx];

        for (const auto& [listNo, listSize] : assignments) {
            if (listSize == 0)
                continue;

            const uint8_t* srcCodes = invlists->get_codes(listNo);
            const faiss::idx_t* srcIds = invlists->get_ids(listNo);
            IndexParam<uint8_t, float, ascend_idx_t> param(deviceId, listSize, 0, 0);
            param.listId = listNo;
            param.query = srcCodes;
            param.label = const_cast<ascend_idx_t*>(
                    reinterpret_cast<const ascend_idx_t*>(srcIds));
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

    for (int deviceId : indexConfig.deviceList) {
        auto pIndex = getActualIndex(deviceId);
        if (pIndex) {
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

void AscendIndexIVFPQImpl::copyToPQCodes(faiss::IndexIVFPQ *index) const
{
    // copy pqcode to index
    index->code_size = pq.M;

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->intf_->is_trained) {
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
            int deviceId = indexConfig.deviceList[i];
            indexIVFFastGetListCodes(deviceId, nlist, ivf);
        }
    }
    index->ntotal = 0;
    for (int i = 0; i < nlist; i++) {
        index->ntotal += static_cast<idx_t>(ivf->list_size(i));
    }
}

void AscendIndexIVFPQImpl::copyTo(faiss::IndexIVFPQ *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVFPQ copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index is not trained");
    if (index->ntotal > 0) {
        index->reset();
    }
    index->d = this->intf_->d;
    index->metric_type = this->intf_->metric_type;
    index->is_trained = this->intf_->is_trained;
    index->nlist = nlist;
    index->nprobe = nprobe;
    index->cp = this->ivfConfig.cp;
    index->pq.M = pq.M;
    index->pq.nbits = pq.nbits;
    index->pq.dsub = pq.dim / pq.M;
    index->pq.ksub = pq.ksub;
    faiss::IndexFlat* quantizer = nullptr;
    if (this->intf_->metric_type == faiss::METRIC_INNER_PRODUCT) {
        quantizer = new faiss::IndexFlatIP(this->intf_->d);
    } else {
        quantizer = new faiss::IndexFlatL2(this->intf_->d);
    }
    if (!indexConfig.deviceList.empty()) {
        std::vector<float> centroids(intf_->d * nlist);
        auto pIndex = getActualIndex(indexConfig.deviceList[0]);
        auto ret = aclrtMemcpy(centroids.data(), intf_->d * nlist * sizeof(float),
                               pIndex->centroidsOnDevice->data(), pIndex->centroidsOnDevice->size() * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        quantizer->add(nlist, centroids.data());
    }
    index->quantizer = quantizer;
    index->own_fields = true;
    size_t codebook_size = pq.M * pq.ksub * pq.dsub;
    index->pq.centroids.resize(codebook_size);
    if (indexConfig.deviceList.size() > 0) {
        auto pIndex = getActualIndex(indexConfig.deviceList[0]);
        FAISS_THROW_IF_NOT_MSG(pIndex, "invalid device index");
        auto ret = aclrtMemcpy(index->pq.centroids.data(), codebook_size * sizeof(float),
                               pIndex->codeBookOnDevice->data(), codebook_size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                               "aclrtMemcpy error %d", ret);
    }

    copyToPQCodes(index);
    APP_LOG_INFO("AscendIndexIVFPQ copyTo operation finished.\n");
}

void AscendIndexIVFPQImpl::checkParams() const
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
                           this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");

    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(),
                           "Unsupported dims: %d\n", this->intf_->d);
    FAISS_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(),
                           "Unsupported nlists: %d\n", this->nlist);
    FAISS_THROW_IF_NOT_FMT(
        std::find(MSUBS.begin(), MSUBS.end(), this->msubs) != MSUBS.end() && this->intf_->d % this->msubs == 0,
        "Unsupported msubs: %d\n", this->msubs);
    FAISS_THROW_IF_NOT_FMT(std::find(NBITS.begin(), NBITS.end(), this->nbits) != NBITS.end(),
                           "Unsupported nbits: %d\n", this->nbits);
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
    pq.codeBook.resize(pq.M, std::vector<std::vector<float>>(pq.ksub, std::vector<float>(pq.dsub, 0.0f)));
    APP_LOG_INFO("AscendIndexIVFPQImpl initProductQuantizer operation finished\n");
}

std::vector<idx_t> AscendIndexIVFPQImpl::update(idx_t n, const float* x, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "vector list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(ids != nullptr, "vector ID list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(n > 0, "update vector number must be greater than 0!");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(n) * this->intf_->d == std::distance(x, x + n * this->intf_->d),
                           "vector list size is not match!");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(n) == std::distance(ids, ids + n),
                           "vector ID list size is not match!");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "AscendIndexIVFPQ is not trained!");
    APP_LOG_INFO("AscendIndexIVFPQImpl update operation started: n=%ld.\n", n);
    std::lock_guard<std::mutex> lock(mapMutex);

    std::vector<idx_t> noExistIds;
    std::vector<idx_t> existIds;
    std::vector<float> existVectors;
    idx_t noExistNum = 0;
    idx_t existNum = 0;

    for (idx_t i = 0; i < n; ++i) {
        idx_t id = ids[i];
        if (idToListMap.find(id) == idToListMap.end()) {
            noExistIds.push_back(id);
            noExistNum++;
            continue;
        }
        idx_t listId = idToListMap[id];
        if (listInfos[listId].idSet.find(id) == listInfos[listId].idSet.end()) {
            noExistIds.push_back(id);
            noExistNum++;
            continue;
        }
        existIds.push_back(id);
        const float* vector = x + i * this->intf_->d;
        existVectors.insert(existVectors.end(), vector, vector + this->intf_->d);
        existNum++;
    }

    if (!noExistIds.empty()) {
        APP_LOG_WARNING("The following %d IDs do not exist: \n", noExistNum);
        for (idx_t i = 0; i < noExistNum; i++) {
            APP_LOG_WARNING("ID: %ld\n", noExistIds[i]);
        }
        APP_LOG_WARNING("Updating other vectors with ids\n");
    }
    if (existNum > 0) {
        deleteImpl(existNum, existIds.data());
        addImpl(existNum, existVectors.data(), existIds.data());
    }
    APP_LOG_INFO("AscendIndexIVFPQ update operation finished.\n");
    
    return noExistIds;
}

void AscendIndexIVFPQImpl::addL1(int n, const float* x, std::unique_ptr<idx_t[]>& assign)
{
    pQuantizerImpl->cpuQuantizer->assign(n, x, assign.get());
}

void AscendIndexIVFPQImpl::addImpl(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFPQImpl addImpl operation started: n=%d.\n", n);
    this->intf_->metric_type = faiss::METRIC_L2;
    size_t deviceCnt = indexConfig.deviceList.size();
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);
    addL1(n, x, assign);  // waiting for npu clustering

    size_t code_size = pq.M;
    std::vector<uint8_t> pq_codes(n * static_cast<int>(code_size));

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const float* vector = x + i * intf_->d;
        std::vector<uint8_t> singlePQCode = encodeSingleVectorPQ(vector);
        std::copy(singlePQCode.begin(), singlePQCode.end(), pq_codes.begin() + i * code_size);
    }

    std::vector<idx_t> idList;
    std::vector<idx_t> listIdList;
    idList.reserve(n);
    listIdList.reserve(n);

    std::unordered_map<int, AscendIVFAddInfo> assignCounts;
    for (int i = 0; i < n; i++) {
        idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);

        idList.push_back(ids[i]);
        listIdList.push_back(listId);

        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(pq_codes.data() + i * pq.M, ids + i);
            continue;
        }
        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }
        assignCounts.emplace(listId, AscendIVFAddInfo(devIdx, deviceCnt, code_size));
        assignCounts.at(listId).Add(pq_codes.data() + i * code_size, ids + i);
    }
    // update idList for delete
    updateIdMapping(idList, listIdList);

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        for (auto& centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0) {
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
    APP_LOG_INFO("AscendIndexIVFPQ addImpl operation finished.\n");
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

std::vector<uint8_t> AscendIndexIVFPQImpl::encodeSingleVectorPQ(const float* vector)
{
    size_t code_size = pq.M;
    std::vector<uint8_t> pq_code(code_size);

    for (size_t m = 0; m < code_size; m++) {
        const float* sub_vector = vector + m * this->pq.dsub;

        uint8_t centroid_idx = findCentroidInSubQuantizer(m, sub_vector);
        pq_code[m] = centroid_idx;
    }
    return pq_code;
}

uint8_t AscendIndexIVFPQImpl::findCentroidInSubQuantizer(size_t subq_idx, const float* sub_vector)
{
    if (sub_vector == nullptr || subq_idx >= this->pq.codeBook.size()) {
        return 0;
    }

    float min_dist = std::numeric_limits<float>::max();
    uint8_t find_centroid = 0;

    std::vector<float> distances(this->pq.ksub);

    for (size_t k = 0; k < this->pq.ksub; k++) {
        const float* centroid = this->pq.codeBook[subq_idx][k].data();
        float dist = calDistance(sub_vector, centroid, this->pq.dsub);
        if (dist < min_dist) {
            min_dist = dist;
            find_centroid = static_cast<uint8_t>(k);
        }
    }

    return find_centroid;
}

float AscendIndexIVFPQImpl::calDistance(const float* a, const float* b, size_t dim)
{
    float dist = 0.0;
    for (size_t i = 0; i < dim; i++) {
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

void AscendIndexIVFPQImpl::trainPQCodeBook(idx_t n, const float* x)
{
    APP_LOG_INFO("Training PQ codebook with %ld vectors\n", n);

    FAISS_THROW_IF_NOT_MSG(n >= static_cast<idx_t>(this->pq.M * this->pq.ksub), "Insufficient training data");
    FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "Coarse quantizer not trained");

    for (size_t m = 0; m < this->pq.M; m++) {
        trainSubQuantizer(m, n, x);
    }

    APP_LOG_INFO("PQ codebook training finished\n");
}

void AscendIndexIVFPQImpl::trainSubQuantizer(size_t m, idx_t n, const float* x)
{
    std::vector<float> subspace_data(n * static_cast<idx_t>(this->pq.dsub));

    for (idx_t i = 0; i < n; i++) {
        const float* sub_vec = x + i * static_cast<idx_t>(this->pq.dim) + m * this->pq.dsub;
        std::copy(sub_vec, sub_vec + this->pq.dsub, subspace_data.data() + i * this->pq.dsub);
    }

    idx_t n_data = n;
    FAISS_THROW_IF_NOT_MSG(n_data >= static_cast<idx_t>(this->pq.ksub),
                           "Insufficient data for sub-quantizer clustering");

    if (ivfConfig.useKmeansPP) {
        try {
            indexTrainImpl(n_data, subspace_data.data(), this->intf_->d / this->pq.M, this->pq.ksub);
        } catch (std::exception& e) {
            APP_LOG_ERROR("NPU training failed for sub-quantizer %zu: %s\n", m, e.what());
        }
    } else {
        faiss::ClusteringParameters cp;
        cp.niter = 25;
        cp.spherical = true;
        cp.nredo = 1;
        cp.verbose = this->intf_->verbose;

        faiss::Clustering clus(this->pq.dsub, this->pq.ksub, cp);
        faiss::IndexFlatL2 index(this->pq.dsub);

        clus.train(n_data, subspace_data.data(), index);

        savePQCodeBook(m, clus.centroids);
    }

    APP_LOG_DEBUG("Sub-quantizer %zu: trained with %ld vectors\n", m, n_data);
}

void AscendIndexIVFPQImpl::savePQCodeBook(size_t m, const std::vector<float>& centroids)
{
    FAISS_THROW_IF_NOT_FMT(centroids.size() == this->pq.ksub * this->pq.dsub,
                           "centroids size error: expect %zu, actual %zu\n",
                           this->pq.ksub * this->pq.dsub, centroids.size());
    if (this->pq.codeBook.size() <= m) {
        this->pq.codeBook.resize(m + 1);
    }

    this->pq.codeBook[m].resize(pq.ksub);

    for (size_t k = 0; k < this->pq.ksub; k++) {
        this->pq.codeBook[m][k].resize(this->pq.dsub);
        const float* src = centroids.data() + k * this->pq.dsub;
        std::copy(src, src + this->pq.dsub, this->pq.codeBook[m][k].begin());
    }
}

void AscendIndexIVFPQImpl::updatePQCodeBook()
{
    APP_LOG_INFO("Updating PQ codebook to device...\n");

    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());

    for (int deviceId : indexConfig.deviceList) {
        auto pIndex = getActualIndex(deviceId);

        float* device_ptr = pIndex->codeBookOnDevice->data();

        for (size_t m = 0; m < this->pq.M; m++) {
            for (size_t k = 0; k < this->pq.ksub; k++) {
                const auto& center = this->pq.codeBook[m][k];
                size_t bytes = this->pq.dsub * sizeof(float);

                auto ret = aclrtMemcpy(device_ptr + m * this->pq.ksub * this->pq.dsub + k * this->pq.dsub, bytes,
                                       center.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
                FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
            }
        }
    }

    APP_LOG_INFO("PQ codebook updated to %d device(s)\n", deviceCnt);
}

void AscendIndexIVFPQImpl::updateCoarseCenter(std::vector<float>& centerData)
{
    std::vector<float> centroidsSqrSum(nlist, 0.0f);
    for (int i = 0; i < nlist; i++) {
        float sum = 0.0f;
        for (int j = 0; j < intf_->d; j++) {
            float val = centerData[i * intf_->d + j];
            sum += val * val;
        }
        centroidsSqrSum[i] = sum;
    }
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());
    for (int i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = aclrtMemcpy(pIndex->centroidsOnDevice->data(), pIndex->centroidsOnDevice->size() * sizeof(float),
                               centerData.data(), intf_->d * nlist * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy centroids error %d", ret);

        ret = aclrtMemcpy(pIndex->centroidsSqrSumOnDevice->data(),
                          pIndex->centroidsSqrSumOnDevice->size() * sizeof(float),
                          centroidsSqrSum.data(),
                          nlist * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy centroidsSqrSum error %d", ret);
    }
}

void AscendIndexIVFPQImpl::train(idx_t n, const float* x)
{
    APP_LOG_INFO("AscendIndexIVFPQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT) {
        APP_LOG_INFO("METRIC_INNER_PRODUCT must set spherical to true in cpu train case\n");
        this->ivfConfig.cp.spherical = true;
    }
    if (!ivfConfig.useKmeansPP) {
        this->ivfConfig.cp.niter = 25; // iter nums
        this->ivfConfig.cp.spherical = true; // spherical clus flag
        this->ivfConfig.cp.nredo = 1;
        this->ivfConfig.cp.verbose = this->intf_->verbose;

        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer, "cpuQuantizer is not init.");
        clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));

        updateCoarseCenter(clus.centroids);
        centroidsData = clus.centroids;
    } else {
        indexTrainImpl(n, x, this->intf_->d, this->nlist);
        updateCoarseCenter(centroidsData);
    }
    trainPQCodeBook(n, x);
    updatePQCodeBook();

    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFPQ train operation finished.\n");
}

void AscendIndexIVFPQImpl::indexTrainImpl(int n, const float* x, int dim, int nlist)
{
    if (n <= 0 || x == nullptr) {
        APP_LOG_ERROR("train data invalid");
    }

    if (indexConfig.deviceList.empty()) {
        APP_LOG_ERROR("NPU device invalid");
    }

    int device_id = indexConfig.deviceList[0];

    auto pIndex = getActualIndex(device_id);
    if (!pIndex) {
        FAISS_THROW_FMT("device %d Index not reset", device_id);
    }

    auto ret = pIndex->trainImpl(n, x, dim, nlist);
    FAISS_THROW_IF_NOT_MSG(ret == ::ascend::APP_ERR_OK, "IndexIVFPQ::trainImpl failed");
    centroidsData.resize(dim * nlist);
    size_t centroids_bytes = nlist * dim * sizeof(float);
    ret = aclrtMemcpy(centroidsData.data(), centroids_bytes,
                      pIndex->clusteringOnDevice->data(), centroids_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                           "update centroids result to Host failed: %d", ret);
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

    auto searchFunctor = [&](int idx) {
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
    size_t deviceCnt = indexConfig.deviceList.size();
    std::unordered_map<int, std::vector<idx_t>> idMap;

    for (int i = 0; i < n; i++) {
        idx_t id = ids[i];
        idx_t listId = findListId(id);
        if (listId >= 0 && listId < this->nlist) {
            idMap[listId].push_back(id);
        } else {
            APP_LOG_WARNING("Could not find valid listId for ID %ld, skipping\n", id);
        }
    }

    for (auto& centroid : idMap) {
        int listId = centroid.first;
        auto& deleteIds = centroid.second;

        if (deleteIds.empty()) {
            continue;
        }

        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }

        std::vector<ascend_idx_t> ascendIds(deleteIds.begin(), deleteIds.end());

        IndexParam<void, void, ascend_idx_t> param(indexConfig.deviceList[devIdx], deleteIds.size(), 0, 0);
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
    if (it != idToListMap.end()) {
        return it->second;
    }

    APP_LOG_WARNING("ID %ld not found in host mapping, attempting to find in device data\n", id);

    idx_t fallbackListId = id % this->nlist;
    APP_LOG_WARNING("ID %ld not found, using fallback listId: %ld\n", id, fallbackListId);
    return fallbackListId;
}

void AscendIndexIVFPQImpl::updateIdMapping(const std::vector<idx_t>& ids, const std::vector<idx_t>& listIds)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    FAISS_THROW_IF_NOT(ids.size() == listIds.size());

    for (size_t i = 0; i < ids.size(); i++) {
        idx_t id = ids[i];
        idx_t listId = listIds[i];

        idToListMap[id] = listId;

        if (static_cast<idx_t>(listInfos.size()) <= listId) {
            listInfos.resize(listId + 1);
        }
        listInfos[listId].idSet.insert(id);
    }
    APP_LOG_DEBUG("Updated batch mapping for %zu IDs\n", ids.size());
}

void AscendIndexIVFPQImpl::removeIdMapping(const std::vector<idx_t>& ids)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    for (idx_t id : ids) {
        auto it = idToListMap.find(id);
        if (it != idToListMap.end()) {
            idx_t listId = it->second;

            if (listId < static_cast<idx_t>(listInfos.size())) {
                listInfos[listId].idSet.erase(id);
            }
            idToListMap.erase(it);
        }
    }
    APP_LOG_DEBUG("Removed batch mapping for %zu IDs\n", ids.size());
}

size_t AscendIndexIVFPQImpl::getAddElementSize() const
{
    return static_cast<size_t>(intf_->d) * sizeof(float);
}
}  // namespace ascend
}  // namespace faiss
