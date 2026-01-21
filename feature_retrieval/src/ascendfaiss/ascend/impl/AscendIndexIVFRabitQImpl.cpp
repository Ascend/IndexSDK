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

#include "AscendIndexIVFRabitQImpl.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <faiss/utils/distances.h>
#include "ascend/AscendIndexQuantizerImpl.h"

namespace faiss {
namespace ascend {

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 128;
// Default nlist in case of nullptr index
const size_t DEFAULT_NLIST = 1024;

// The value range of dim
const std::vector<int> DIMS = { 128 };

// The value range of nlist
const std::vector<int> NLISTS = { 1024, 2048, 4096, 8192, 16384, 32768 };
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 640;
const size_t ADD_PAGE_SIZE = (UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE);
const size_t UNIT_VEC_SIZE = 5120;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

extern "C" {
/* declare LAPACK functions */
int sgeqrf_(long* m, long* n, float* a, long* lda, float* tau, float* work, long* lwork, long* info);

int sorgqr_(long* m, long* n, long* k, float* a, long* lda, float* tau, float* work, long* lwork, long* info);
}

RandomGenerator::RandomGenerator(int64_t seed) : mt(static_cast<unsigned int>(seed)) {}

int RandomGenerator::rand_int()
{
    return mt() & 0x7fffffff;
}

int64_t RandomGenerator::rand_int64()
{
    return static_cast<int64_t>(rand_int()) | static_cast<int64_t>(rand_int()) << 31;
}

int RandomGenerator::rand_int(int max)
{
    return mt() % max;
}

float RandomGenerator::rand_float()
{
    return mt() / static_cast<float>(mt.max());
}

double RandomGenerator::rand_double()
{
    return mt() / static_cast<double>(mt.max());
}

void matrix_qr(int m, int n, float* a)
{
    FAISS_THROW_IF_NOT(m >= n);
    long mi = m;
    long ni = n;
    long ki = mi < ni ? mi : ni;
    std::vector<float> tau(ki);
    long lwork = -1;
    long info;
    float work_size;

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), &work_size, &lwork, &info);
    lwork = size_t(work_size);
    std::vector<float> work(lwork);

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), work.data(), &lwork, &info);

    sorgqr_(&mi, &ni, &ki, a, &mi, tau.data(), work.data(), &lwork, &info);
}

void float_randn(float* x, size_t n, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0(seed);
    int a0 = rng0.rand_int();
    int b0 = rng0.rand_int();

#pragma omp parallel for
    for (int64_t j = 0; j < nblock; j++) {
        RandomGenerator rng(a0 + j * b0);

        double a = 0;
        double b = 0;
        double s = 0;
        int state = 0; // generate two number per "do-while" loop

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++) {
            // Marsaglia's method (see Knuth)
            if (state == 1) {
                x[i] = a * sqrt(-2.0 * log(s) / s);
                state = 1 - state;
                continue;
            }
            do {
                a = 2.0 * rng.rand_double() - 1;
                b = 2.0 * rng.rand_double() - 1;
                s = a * a + b * b;
            } while (s >= 1.0);
            x[i] = a * sqrt(-2.0 * log(s) / s);
            state = 1 - state;
        }
    }
}

void orthonormalinit(std::vector<float>& A, int seed, int d_in, int d_out)
{
    if (d_out <= d_in) {
        A.resize(d_out * d_in);
        float* q = A.data();
        float_randn(q, d_out * d_in, seed);
        matrix_qr(d_in, d_out, q);
    } else {
        // use tight-frame transformation
        A.resize(d_out * d_out);
        float* q = A.data();
        float_randn(q, d_out * d_out, seed);
        matrix_qr(d_out, d_out, q);
        // remove columns
        int i;
        int j;
        for (i = 0; i < d_out; i++) {
            for (j = 0; j < d_in; j++) {
                q[i * d_in + j] = q[i * d_out + j];
            }
        }
        A.resize(d_in * d_out);
    }
}

AscendIndexIVFRabitQImpl::AscendIndexIVFRabitQImpl(AscendIndexIVFRabitQ *intf, int dims, int nlist,
    faiss::MetricType metric, AscendIndexIVFRabitQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config), intf_(intf), ivfrabitqConfig(config)
{
    checkParams();
    initIndexes();

    initDeviceAddNumMap();
    centroidsData.resize(nlist);
    this->intf_->is_trained = false;
    initFlatAtFp32();
    orthogonalMatrix.resize(dims * dims, 0);
}

AscendIndexIVFRabitQImpl::~AscendIndexIVFRabitQImpl() {}

void AscendIndexIVFRabitQImpl::initFlatAtFp32()
{
    APP_LOG_INFO("AscendIndexIVFRabitQImpl initFlatAtFp32 started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
    assignIndex = CREATE_UNIQUE_PTR(::ascend::IndexIVFFlat, nlist, intf_->d, 1, -1);
    assignIndex->init();
    if (this->ivfConfig.useKmeansPP) {
        AscendClusteringConfig npuClusConf({ ivfConfig.deviceList[0] }, ivfConfig.resourceSize);
        pQuantizerImpl->npuClus =
            std::make_shared<AscendClustering>(this->intf_->d, this->nlist, this->intf_->metric_type, npuClusConf);
    }
    APP_LOG_INFO("AscendIndexIVFRabitQImpl initFlatAtFp32 finished.\n");
}
 
void AscendIndexIVFRabitQImpl::checkParams() const
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2, "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(),
                           "Unsupported nlists %d", this->nlist);
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_MSG(ivfConfig.deviceList.size() != 0, "deviceList is empty");
}

void AscendIndexIVFRabitQImpl::addL1(int n, const float *x, std::unique_ptr<idx_t[]> &assign)
{
    // 使用npu能力加速add过程
    FAISS_THROW_IF_NOT_MSG(assignIndex != nullptr, "assignIndex is not init");
    FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
    ::ascend::AscendTensor<float, ::ascend::DIMS_2> codes(const_cast<float *>(x), {n, this->intf_->d});
    ::ascend::AscendTensor<idx_t, ::ascend::DIMS_2> indices(assign.get(), {n, 1});
    auto ret = assignIndex->assign(codes, indices);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "assign failed %d", ret);
}

void AscendIndexIVFRabitQImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexIVFRabitQImpl addImpl operation started: n=%d.\n", n);
    size_t deviceCnt = indexConfig.deviceList.size();
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);
    addL1(n, x, assign);
    size_t dim = static_cast<size_t>(intf_->d);
    for (size_t i = 0; i <  static_cast<size_t>(n); i++) {
        idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(const_cast<float *>(x) + i * dim, ids + i);
            deviceAddNumMap[listId][it->second.addDeviceIdx]++;
            continue;
        }
        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }
        deviceAddNumMap[listId][devIdx]++;
        assignCounts.emplace(listId, AscendIVFAddInfo(devIdx, deviceCnt, dim));
        assignCounts.at(listId).Add(const_cast<float *>(x) + i * dim, ids + i);
    }
}

void AscendIndexIVFRabitQImpl::copyVectorToDevice(int n)
{
    size_t deviceCnt = indexConfig.deviceList.size();
    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        for (auto &centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0) {
                continue;
            }
            float *codePtr = nullptr;
            ascend_idx_t *idPtr = nullptr;
            centroid.second.GetCodeAndIdPtr(idx, &codePtr, &idPtr);
            IndexParam<float, float, ascend_idx_t> param(deviceId, num, 0, 0);
            param.listId = listId;
            param.query = codePtr;
            param.label = idPtr;
            indexIVFRabitQAdd(param);
        }
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexIVFSQ addImpl operation finished.\n");
}

size_t AscendIndexIVFRabitQImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

void AscendIndexIVFRabitQImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFRabitQImpl addPaged operation started.\n");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "x cannot be nullptr");
    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    size_t addPageSize = ADD_PAGE_SIZE;
    if (totalSize > addPageSize || static_cast<size_t>(n) > ADD_VEC_SIZE * 10) {
        size_t tileSize = getAddPagedSize(n);
        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose) {
                printf("AscendIndexIVFRabitQImpl::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndexIVFRabitQImpl::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    copyVectorToDevice(n);
    this->srcIndexes.insert(this->srcIndexes.end(), x, x + n * this->intf_->d);
    std::unordered_map<int, AscendIVFAddInfo>().swap(assignCounts); // 释放host侧占用的内存
    APP_LOG_INFO("AscendIndexIVFRabitQImpl addPaged operation finished.\n");
}

void AscendIndexIVFRabitQImpl::indexIVFRabitQAdd(IndexParam<float, float, ascend_idx_t> &param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;
    const float *codes = param.query;
    const ascend_idx_t *ids = param.label;
    auto ret = pIndex->addVectors(param.listId, param.n, codes,
                                  static_cast<const ::ascend::Index::idx_t *>(ids));
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to add to ivf rabitq, ret: %d", ret);
}

std::shared_ptr<::ascend::Index> AscendIndexIVFRabitQImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFRabitQ  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::IndexIVF> index =
        std::make_shared<::ascend::IndexIVFRabitQ>(nlist, this->intf_->d, nprobe, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf rabitq, result is %d", ret);

    APP_LOG_INFO("AscendIndexIVFRabitQ createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFRabitQImpl::updateCoarseCenter(std::vector<float> &centerData)
{
    size_t deviceCnt = indexConfig.deviceList.size();
    auto updateCoarseCenterFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->updateCoarseCenterImpl(centerData);
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to update coarsecenter, deviceId is %d, result is: %d\n", deviceId, ret);
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, updateCoarseCenterFunctor);
}

void AscendIndexIVFRabitQImpl::randomOrthogonalGivens(int n, std::vector<float> &orthogonalMatrix)
{
    APP_LOG_INFO("AscendIndexIVFRabitQ  create randomOrthogonalMatrix started\n");
    if (ivfrabitqConfig.useRandomOrthogonalMatrix) {
        orthonormalinit(orthogonalMatrix, ivfrabitqConfig.matrixSeed, n, n);
    } else {
        // 使用单位矩阵
        for (int i = 0; i < n; ++i) {
            orthogonalMatrix[i * n + i] = 1.0;
        }
    }
}

void AscendIndexIVFRabitQImpl::uploadorthogonalMatrix(std::vector<float> &orthogonalMatrix)
{
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());
    for (int i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = aclrtMemcpy(pIndex->OrthogonalMatrixOnDevice->data(),
                               pIndex->OrthogonalMatrixOnDevice->size() * sizeof(float),
                               orthogonalMatrix.data(), intf_->d * intf_->d * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                               "Failed to aclrtMemcpy orthogonalMatrix, deviceId is %d, result is: %d\n",
                               deviceId, ret);
    }
}

void AscendIndexIVFRabitQImpl::train(idx_t n, const float *x, bool clearNpuData)
{
    APP_LOG_INFO("AscendIndexIVFRabitQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    
    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    randomOrthogonalGivens(this->intf_->d, this->orthogonalMatrix);
    uploadorthogonalMatrix(this->orthogonalMatrix);
    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT) {
        APP_LOG_INFO("METRIC_INNER_PRODUCT must set spherical to true in cpu train case\n");
        this->ivfConfig.cp.spherical = true;
    }
    if (ivfConfig.useKmeansPP) {
        pQuantizerImpl->npuClus->verbose = this->intf_->verbose;
        std::vector<float> tmpCentroids(nlist * this->intf_->d);
        if (pQuantizerImpl->npuClus->GetNTotal() == 0) {
            pQuantizerImpl->npuClus->AddFp32(n, x);
        }
        if (this->intf_->verbose) {
            printf("Ascend cluster start training %zu vectors\n", pQuantizerImpl->npuClus->GetNTotal());
        }
        pQuantizerImpl->npuClus->TrainFp32(this->ivfConfig.cp.niter, tmpCentroids.data(), clearNpuData);
        updateCoarseCenter(tmpCentroids);
        FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
        ::ascend::AscendTensor<float, ::ascend::DIMS_2> centroidsTrained(tmpCentroids.data(), {nlist, intf_->d});
        assignIndex->addVectorsAsCentroid(centroidsTrained);
    } else {
        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer, "cpuQuantizer is not init.");
        clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));
        updateCoarseCenter(clus.centroids);
        ::ascend::AscendTensor<float, ::ascend::DIMS_2> centroidsTrained(clus.centroids.data(), {nlist, intf_->d});
        assignIndex->addVectorsAsCentroid(centroidsTrained);
    }
    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFRabitQ train operation finished.\n");
}

void AscendIndexIVFRabitQImpl::mergeSearchResultSingleQuery(idx_t qIdx, size_t devices,
                                                            std::vector<std::vector<float>>& dist,
                                                            std::vector<std::vector<ascend_idx_t>>& label,
                                                            idx_t n, idx_t k, size_t eachdeviceK,
                                                            float* distances, idx_t* labels,
                                                            std::function<bool(float, float)> &compFunc) const
{
    idx_t num = 0;
    const idx_t offset = qIdx * k;
    const idx_t offsetDevice = qIdx * eachdeviceK;
    std::vector<int> posit(devices, 0);
    while (num < k) {
        size_t id = 0;
        while (posit[id] >= eachdeviceK) {
            id++;
        }
        float disMerged = dist[id][offsetDevice + posit[id]];
        ascend_idx_t labelMerged = label[id][offsetDevice + posit[id]];
        for (size_t j = id + 1; j < devices && posit[j] < eachdeviceK; j++) {
            idx_t pos = offsetDevice + posit[j];
            if (static_cast<idx_t>(label[j][pos]) != -1 && compFunc(dist[j][pos], disMerged)) {
                disMerged = dist[j][pos];
                labelMerged = label[j][pos];
                id = j;
            }
        }

        *(distances + offset + num) = disMerged;
        *(labels + offset + num) = (idx_t)labelMerged;
        posit[id]++;
        num++;
    }
}

void AscendIndexIVFRabitQImpl::mergeSearchResult(size_t devices, std::vector<std::vector<float>>& dist,
                                                 std::vector<std::vector<ascend_idx_t>>& label, idx_t n, idx_t k,
                                                 float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex mergeSearchResult operation started.\n");
    size_t eachdeviceK = dist[0].size() / n;
    FAISS_THROW_IF_NOT_FMT(devices * eachdeviceK >= k, "deviceNum %ld * %ld must be >= k %ld", devices, eachdeviceK, k);
    std::function<bool(float, float)> compFunc = GetCompFunc();

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 100) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < n; i++) {
        mergeSearchResultSingleQuery(i, devices, dist, label, n, k, eachdeviceK, distances, labels, compFunc);
    }
    APP_LOG_INFO("AscendIndex mergeSearchResult operation finished.\n");
}

void AscendIndexIVFRabitQImpl::indexSearch(IndexParam<float, float, ascend_idx_t> &param) const
{
    auto pIndex = getActualIndex(param.deviceId);
    const float* indexPtr = nullptr;
    if (ivfrabitqConfig.needRefine) {
        indexPtr = this->srcIndexes.data();
    }
    auto ret = pIndex->searchImpl(param.n, param.query, param.k, param.distance, param.label, indexPtr);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                           "Failed to search index,deviceId is %d, result is: %d\n", param.deviceId, ret);
}

void AscendIndexIVFRabitQImpl::searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    int finalk = k;
    if (ivfrabitqConfig.needRefine) {
        k = finalk * ivfrabitqConfig.refineAlpha;
    }
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
    searchPostProcess(deviceCnt, dist, label, n, finalk, distances, labels);
}

size_t AscendIndexIVFRabitQImpl::getAddElementSize() const
{
    return static_cast<size_t>(intf_->d) * sizeof(float);
}
} // ascend
} // faiss
