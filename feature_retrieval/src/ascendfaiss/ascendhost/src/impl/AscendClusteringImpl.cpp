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


#include "AscendClusteringImpl.h"
#include <omp.h>
#include <algorithm>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/Random.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"
#include "index_custom/IndexFlatATAicpu.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace faiss {
namespace ascend {
namespace {
const int64_t CLUSTERING_MAX_MEM = 0x100000000; // 0x100000000 mean 4096MB
const idx_t CLUSTERING_MAX_N = 7000001;
const int MAX_CPU_AVAILABLE = 20;
const int NUM_BUCK_PER_BATCH = 64;
const int SUBCENTER_NUM = 64;
const int LOWER_BOUND = 32;
const int MAX_CLUS_POINTS = 16384;
const int MAX_NCENTROIDS = 32768;
const int MAX_NITER = 65536;
const std::vector<int> DIMS = { 64, 128, 256, 384, 512, 1024, 2048 };
const int MAX_NLISTS = 32768;
const int64_t RAND_SEED = 1234;
const int CORR_COMPUTE_SIZE = 1024;

const int UINT8_UPPER_BOUND = 255;
const float16_t SQ_DECODE_PARAM = 0.5;
const int QUERY_ALIGN_CLUS = 32;
const float EPS = 1 / 1024.0;
}

AscendClusteringImpl::AscendClusteringImpl(int d, int k,
    MetricType metricType, AscendClusteringConfig config, AscendClustering *intf)
    : intf_(intf),
      metricType(metricType),
      config(config),
      ntotal(0),
      pool(std::make_shared<AscendThreadPool>(MAX_CPU_AVAILABLE)),
      isUpdateVDM(false),
      updateNtotal(0)
{
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), d) != DIMS.end(), "Invalid number of dimensions");
    FAISS_THROW_IF_NOT_FMT(k > 0 && k <= MAX_NLISTS, "number of centroids(%d) should be in range (0, %d]",
        k, MAX_NLISTS);
    FAISS_THROW_IF_NOT_MSG(metricType == MetricType::METRIC_L2 || metricType == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size() == 1, "Only 1 chip supported for Ascend Clustering.");
    FAISS_THROW_IF_NOT_MSG(config.resourceSize == -1 ||
                           (config.resourceSize >= 0 && config.resourceSize <= CLUSTERING_MAX_MEM),
                           "resources should be -1 or in range [0, 4096MB]!");
    init();
}

AscendClusteringImpl::~AscendClusteringImpl()
{
    int deviceId = config.deviceList[0];
    (void)aclrtResetDevice(deviceId);
}

// Do NOT support clustering on multi chips
// Do NOT support clustering at same time on 1 chip
void AscendClusteringImpl::init()
{
    int deviceId = config.deviceList[0];
    ASCEND_THROW_IF_NOT(deviceId >= 0);
    uint32_t devCount = SocUtils::GetInstance().GetDeviceCount();
    FAISS_THROW_IF_NOT_FMT(static_cast<uint32_t>(deviceId) < devCount,
                           "Device %d is invalid, total device %u", deviceId, devCount);
    ACL_REQUIRE_OK(aclrtSetDevice(deviceId));
    resources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
}

void AscendClusteringImpl::add(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendClustering add start: searchNum=%ld, d %zu, k %zu", n, intf_->d, intf_->k);

    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    ASCEND_THROW_IF_NOT_FMT((n > 0) && (n < CLUSTERING_MAX_N),
        "n must be > 0 and n < %ld", CLUSTERING_MAX_N);
    ASCEND_THROW_IF_NOT_FMT((n > 0) && (ntotal + static_cast<size_t>(n) < CLUSTERING_MAX_N),
        "n must be > 0 and ntotal+n < %ld", CLUSTERING_MAX_N);
    FAISS_THROW_IF_NOT_FMT(
        static_cast<size_t>(CLUSTERING_MAX_N) * static_cast<size_t>(intf_->d) < static_cast<size_t>(INT_MAX),
        "dim*n must be > 0 and < %zu", static_cast<size_t>(INT_MAX));

    codes.resize((ntotal + n) * intf_->d);

    // transform float to fp16 with multithreading
    int64_t fragN = static_cast<int64_t>(n + MAX_CPU_AVAILABLE - 1) / static_cast<int64_t>(MAX_CPU_AVAILABLE);
    auto transFunctor = [&](int64_t idx) {
        CommonUtils::attachToCpu(idx);
        int64_t actualSize = std::min(fragN, n - idx * fragN);
        size_t start = static_cast<size_t>(idx) * fragN * static_cast<uint32_t>(intf_->d);
        if (actualSize > 0) {
            std::transform(x + start, x + std::min(n * intf_->d, start + actualSize * intf_->d),
                codes.begin() + ntotal + start, [](float temp) { return fp16(temp).data; });
        }
    };
    std::vector<std::future<void>> transFunctorRet;
    for (int64_t i = 0; i < static_cast<int64_t>(MAX_CPU_AVAILABLE); i++) {
        transFunctorRet.emplace_back(pool->Enqueue(transFunctor, i));
    }

    int transWait = 0;
    try {
        for_each(transFunctorRet.begin(), transFunctorRet.end(), [&](std::future<void> &ret) {
            transWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        for_each(transFunctorRet.begin() + transWait,
            transFunctorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_MSG("wait for cent trans functor failed.");
    }

    ntotal += static_cast<size_t>(n);

    APP_LOG_INFO("AscendClustering add finished, ntotal %zu", ntotal);
}

void AscendClusteringImpl::addFp32(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendClustering addFp32 start: searchNum=%ld, d %zu, k %zu", n, intf_->d, intf_->k);

    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    ASCEND_THROW_IF_NOT_FMT((n > 0) && (n < CLUSTERING_MAX_N),
        "n must be > 0 and n < %ld", CLUSTERING_MAX_N);
    ASCEND_THROW_IF_NOT_FMT((n > 0) && (ntotal + static_cast<size_t>(n) < CLUSTERING_MAX_N),
        "n must be > 0 and ntotal+n < %ld", CLUSTERING_MAX_N);
    FAISS_THROW_IF_NOT_FMT(
        static_cast<size_t>(CLUSTERING_MAX_N) * static_cast<size_t>(intf_->d) < static_cast<size_t>(INT_MAX),
        "dim*n must be > 0 and < %zu", static_cast<size_t>(INT_MAX));
    codesFp32.resize((ntotal + n) * intf_->d);
    std::copy(x, x + n * static_cast<idx_t>(intf_->d), codesFp32.begin() + ntotal);
    ntotal += static_cast<size_t>(n);
    APP_LOG_INFO("AscendClustering addFp32 finished, ntotal %zu", ntotal);
}

void AscendClusteringImpl::distributedTrain(int niter, float *centroids,
                                            const std::vector<int> &deviceList, bool clearData)
{
    // Here k means num of centroids/nlist
    APP_LOG_INFO("AscendClustering distributedTrain start: ntotal=%zu, ncentroids=%d, dim=%d, niter=%d\n",
        ntotal, intf_->k, intf_->d, niter);
    ASCEND_THROW_IF_NOT_MSG(centroids, "distances can not be nullptr.");
    ASCEND_THROW_IF_NOT_FMT(niter > 0 && niter <= MAX_NITER,
        "Niter must be in range (0, %d], but given %d \n", MAX_NITER, niter);
    ASCEND_THROW_IF_NOT_FMT((ntotal > 0) && (ntotal < CLUSTERING_MAX_N),
        "ntotal %zu must be > 0 and < %ld", ntotal, CLUSTERING_MAX_N);
    ASCEND_THROW_IF_NOT_FMT(intf_->k > 0 && intf_->k <= MAX_NCENTROIDS,
        "Invalid number of ncentroids %zu, should be in range (0, %d]", intf_->k, MAX_NCENTROIDS);
    ASCEND_THROW_IF_NOT_FMT(ntotal >= static_cast<size_t>(intf_->k),
        "Number of training points (%zu) should be at least as large as number of clusters (%zu)",
        ntotal, static_cast<size_t>(intf_->k));

    ACL_REQUIRE_OK(aclrtSetDevice(deviceList[0]));
    resetKmUpdateCentroidsOp();

    std::vector<int> perm(ntotal, 0);
    RandPerm(perm.data(), ntotal, RAND_SEED);

    std::vector<float16_t> centrodataHost(intf_->k * intf_->d);
    for (size_t i = 0; i < intf_->k; i++) {
        size_t offset = i * intf_->d;
        auto ret = memcpy_s(centrodataHost.data() + offset, (centrodataHost.size() - offset) * sizeof(float16_t),
                            codes.data() + static_cast<size_t>(perm[i]) * intf_->d, intf_->d * sizeof(float16_t));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Failed to copy to centrodata (error %d)", (int)ret);
    }

    AscendTensor<float16_t, DIMS_2> codesDeviceAll({static_cast<int>(ntotal), static_cast<int>(intf_->d)});
    auto ret = aclrtMemcpy(codesDeviceAll.data(), codesDeviceAll.getSizeInBytes(),
                           codes.data(), codes.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy codes to device ret %d", ret);

    std::vector<std::unique_ptr<IndexFlatATAicpu>> npuFlatAts(deviceList.size());

    int deviceCount = static_cast<int>(deviceList.size());
    std::vector<uint64_t> assignHost(ntotal);
    std::atomic_int count(0);
    int fragN = (static_cast<int64_t>(ntotal) + deviceCount - 1) / deviceCount;
    auto trainFunctor = [&](int idx) {
        CommonUtils::attachToCpu(idx);
        int actualSize = std::min(fragN, static_cast<int>(ntotal - idx * fragN));
        int start = idx * fragN;
        if (actualSize <= 0) {
            return;
        }
        ACL_REQUIRE_OK(aclrtSetDevice(deviceList[idx]));

        auto resourcesDevice = CREATE_UNIQUE_PTR(AscendResourcesProxy);
        auto streamPtr = resourcesDevice->getDefaultStream();
        auto stream = streamPtr->GetStream();
        auto &mem = resourcesDevice->getMemoryManager();
        int d = static_cast<int>(intf_->d);
        int k = static_cast<int>(intf_->k);
        npuFlatAts[idx] = CREATE_UNIQUE_PTR(IndexFlatATAicpu, d, k, config.resourceSize);
        auto ret = npuFlatAts[idx]->init();
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to init NpuFlat, device id %d, ret %d", idx, ret);
        npuFlatAts[idx]->setResultCopyBack(false);

        AscendTensor<float16_t, DIMS_2> codesDevice(mem, {actualSize, d}, stream);
        ret = aclrtMemcpy(codesDevice.data(), codesDevice.getSizeInBytes(),
                          codes.data() + static_cast<size_t>(start) * static_cast<size_t>(d),
                          static_cast<size_t>(actualSize) * static_cast<size_t>(d) * sizeof(float16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy codes to device, device id %d, ret %d", idx, ret);

        AscendTensor<uint64_t, DIMS_1> assign(mem, { actualSize }, stream);
        AscendTensor<float16_t, DIMS_1> dis(mem, { actualSize }, stream);

        double t0 = 0.0;
        if (intf_->verbose) {
            t0 = utils::getMillisecs();
        }
        for (int i = 0; i < niter; i++) {
            AscendTensor<float16_t, DIMS_2> centrodata(mem, { k, d }, stream);
            ret = aclrtMemcpy(centrodata.data(), centrodata.getSizeInBytes(), centrodataHost.data(),
                              centrodataHost.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                "Failed to copy centrodata to device failed, id %d, ret %d", idx, ret);

            ret = npuFlatAts[idx]->addVectors(centrodata);
            ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to addVectors, device id %d, ret %d", idx, ret);
            ret = npuFlatAts[idx]->search(actualSize, codesDevice.data(), 1, dis.data(), assign.data());
            ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to search, device id %d, ret %d", idx, ret);

            ret = aclrtMemcpy(assignHost.data() + start, (ntotal - start) * sizeof(uint64_t), assign.data(),
                              static_cast<size_t>(actualSize) * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                "Failed to copy assign to host, device id %d, ret %d", idx, ret);
            count++;
            WAITING_FLAG_READY_TIMEOUT((count == deviceCount), TIMEOUT_MS);
            if (idx == 0) {
                AscendTensor<uint64_t, DIMS_1> assignDevice(mem, { static_cast<int>(ntotal) }, stream);
                ret = aclrtMemcpy(assignDevice.data(), assignDevice.getSizeInBytes(),
                                  assignHost.data(), assignHost.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                    "Failed to copy assignHost to device, device id %d, ret %d", idx, ret);

                AscendTensor<float16_t, DIMS_2> centrodataAll(mem, { k, d }, stream);
                this->runkmUpdateCentroidsCompute(codesDeviceAll, centrodataAll, assignDevice, stream);
                ret = synchronizeStream(stream);
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                    "update centroids compute synchronizeStream failed: %d", ret);

                ret = aclrtMemcpy(centrodataHost.data(), centrodataHost.size() * sizeof(float16_t),
                                  centrodataAll.data(), centrodataAll.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                    "Failed to copy centrodataAll to host, device id %d, ret %d", idx, ret);
                if (intf_->verbose) {
                    printf("Cluster Iteration %d (%.2f s)\n", i, (utils::getMillisecs() - t0) / 1000);
                }
                count = 0;
            }
            WAITING_FLAG_READY_TIMEOUT((count == 0), TIMEOUT_MS);

            npuFlatAts[idx]->reset();
        }
    };

    CALL_PARALLEL_FUNCTOR(static_cast<size_t>(deviceCount), pool, trainFunctor);

    std::transform(centrodataHost.begin(), centrodataHost.end(), centroids,
        [](float16_t temp) { return static_cast<float>(fp16(temp)); });

    // clear data
    if (clearData) {
        std::vector<float16_t>().swap(codes);
        ntotal = 0;
    }

    APP_LOG_INFO("AscendClustering distributedTrain finished.\n");
}

void AscendClusteringImpl::train(int niter, float *centroids, bool clearData)
{
    // Here k means num of centroids/nlist
    APP_LOG_INFO("AscendClustering train start: ntotal=%zu, ncentroids=%d, dim=%d, niter=%d\n",
        ntotal, intf_->k, intf_->d, niter);
    FAISS_THROW_IF_NOT_MSG(centroids, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(niter > 0 && niter <= MAX_NITER,
        "Niter must be in range (0, %d], but given %d \n", MAX_NITER, niter);
    FAISS_THROW_IF_NOT_FMT((ntotal > 0) && (ntotal < CLUSTERING_MAX_N),
        "ntotal %zu must be > 0 and < %ld", ntotal, CLUSTERING_MAX_N);
    FAISS_THROW_IF_NOT_FMT(intf_->k > 0 && intf_->k <= MAX_NCENTROIDS,
        "Invalid number of ncentroids, should be in range (0, %d]", MAX_NCENTROIDS);
    FAISS_THROW_IF_NOT_FMT(ntotal >= static_cast<size_t>(intf_->k),
        "Number of training points (%zu) should be at least as large as number of clusters (%zu)",
        ntotal, static_cast<size_t>(intf_->k));

    ACL_REQUIRE_OK(aclrtSetDevice(config.deviceList[0]));

    npuFlatAt = CREATE_UNIQUE_PTR(IndexFlatATAicpu, intf_->d, intf_->k, -1);
    FAISS_THROW_IF_NOT_MSG(npuFlatAt->init() == APP_ERR_OK, "Failed to init NpuFlat");
    npuFlatAt->setResultCopyBack(false);

    auto streamPtr = resources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources->getMemoryManager();

    resetKmUpdateCentroidsOp();

    AscendTensor<float16_t, DIMS_2> centrodata(mem, {static_cast<int>(intf_->k), static_cast<int>(intf_->d)}, stream);
    randomCentrodata(centrodata);

    AscendTensor<float16_t, DIMS_2> codesDevice(mem, {static_cast<int>(ntotal), static_cast<int>(intf_->d)}, stream);
    auto ret = aclrtMemcpy(codesDevice.data(), codesDevice.getSizeInBytes(),
                           codes.data(), codes.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy codes to device");

    AscendTensor<uint64_t, DIMS_1> assign(mem, { static_cast<int>(ntotal) }, stream);
    AscendTensor<float16_t, DIMS_1> dis(mem, { static_cast<int>(ntotal) }, stream);

    // Exec clustering
    double tSearchTotal = 0.0;
    double t0 = 0.0;
    double t0s = 0.0;
    if (intf_->verbose) {
        t0 = utils::getMillisecs();
    }
    for (int i = 0; i < niter; i++) {
        ret = npuFlatAt->addVectors(centrodata);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to addVectors, ret %d", ret);
        if (intf_->verbose) {
            t0s = utils::getMillisecs();
        }
        ret = npuFlatAt->search(ntotal, codesDevice.data(), 1, dis.data(), assign.data());
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to search, ret %d", ret);
        if (intf_->verbose) {
            tSearchTotal += utils::getMillisecs() - t0s;
            printf("Cluster Iteration %d (%.2f s, search %.2f s)\n",
                i, (utils::getMillisecs() - t0) / 1000, tSearchTotal / 1000);
        }

        runkmUpdateCentroidsCompute(codesDevice, centrodata, assign, stream);
        auto ret = synchronizeStream(stream);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream failed: %d", ret);

        ret = npuFlatAt->reset();
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to reset, ret %d", ret);
    }

    trainPostProcess(centrodata, centroids);

    // clear data
    if (clearData) {
        std::vector<float16_t>().swap(codes);
        ntotal = 0;
    }

    APP_LOG_INFO("AscendClustering train finished.\n");
}

void AscendClusteringImpl::computeCentroids(size_t d, // dim
                                            size_t k, // nlist
                                            size_t n, // ntotal
                                            const float* x,
                                            const int64_t* assign, // ntotal条向量所属桶
                                            float* centroids)
{
    auto ret = memset_s(centroids, d * k * sizeof(float), 0, d * k * sizeof(float));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "set centroids to 0 failed %d", ret);
    std::vector<int32_t> hassign(k);
#pragma omp parallel
    {
        int nt = omp_get_num_threads(); // 总线程数
        int rank = omp_get_thread_num(); // 获取当前的线程ID

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        for (size_t i = 0; i < n; i++) {
            ASCEND_THROW_IF_NOT_FMT(assign[i] >= 0, "assign[%zu] %jd < 0 is invalid", i, assign[i]);
            size_t ci = static_cast<size_t>(assign[i]);
            FAISS_THROW_IF_NOT_FMT(ci < k, "assign %zu >= nlist is invalid.", ci);
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi = x + i * d;
                hassign[ci] += 1;
                for (size_t j = 0; j < d; j++) {
                    c[j] += xi[j];
                }
            }
        }
    }
#pragma omp parallel for
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / static_cast<float>(hassign[ci]);
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

void AscendClusteringImpl::normalizeVecByHost(float *centrodataHost,
                                              AscendTensor<float, DIMS_2> &centrodataDev)
{
    fvec_renorm_L2(intf_->d, intf_->k, centrodataHost);
    auto ret = aclrtMemcpy(centrodataDev.data(), centrodataDev.getSizeInBytes(),
                           centrodataHost, intf_->d * intf_->k * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy centrodata to device");
}

void AscendClusteringImpl::checkParaFp32(int niter, float *centroids)
{
    FAISS_THROW_IF_NOT_MSG(centroids, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(niter > 0 && niter <= MAX_NITER,
        "Niter must be in range (0, %d], but given %d \n", MAX_NITER, niter);
    FAISS_THROW_IF_NOT_FMT((ntotal > 0) && (ntotal < CLUSTERING_MAX_N),
        "ntotal %zu must be > 0 and < %ld", ntotal, CLUSTERING_MAX_N);
    FAISS_THROW_IF_NOT_FMT(intf_->k > 0 && intf_->k <= MAX_NCENTROIDS,
        "Invalid number of ncentroids, should be in range (0, %d]", MAX_NCENTROIDS);
    FAISS_THROW_IF_NOT_FMT(ntotal >= static_cast<size_t>(intf_->k),
        "Number of training points (%zu) should be at least as large as number of clusters (%zu)",
        ntotal, static_cast<size_t>(intf_->k));
}

void AscendClusteringImpl::trainFp32(int niter, float *centroids, bool clearData)
{
    // Here k means num of centroids/nlist
    checkParaFp32(niter, centroids);
    APP_LOG_INFO("AscendClustering train start: ntotal=%zu, ncentroids=%d, dim=%d, niter=%d\n",
        ntotal, intf_->k, intf_->d, niter);
    ACL_REQUIRE_OK(aclrtSetDevice(config.deviceList[0]));
    npuFlatAtFp32 = CREATE_UNIQUE_PTR(IndexIVFFlat, intf_->k, intf_->d, 1, -1);
    auto ret = npuFlatAtFp32->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf flat, result is %d", ret);

    auto streamPtr = resources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources->getMemoryManager();

    AscendTensor<float, DIMS_2> centrodata(mem, {static_cast<int>(intf_->k), static_cast<int>(intf_->d)}, stream);
    AscendTensor<float, DIMS_2> centroidsHost(centroids, {static_cast<int>(intf_->k), static_cast<int>(intf_->d)});
    randomCentrodataFp32(centroidsHost);
    // 聚类中心进行归一化
    normalizeVecByHost(centroids, centrodata);
    AscendTensor<float, DIMS_2> codesTensor(codesFp32.data(), {static_cast<int>(ntotal), static_cast<int>(intf_->d)});
    AscendTensor<int64_t, DIMS_1> assign(mem, { static_cast<int>(ntotal) }, stream);
    std::vector<int64_t> indices(ntotal);
    AscendTensor<int64_t, DIMS_2> indicesHost{indices.data(), {static_cast<int>(ntotal), 1}};
    double tSearchTotal = 0.0;
    double t0 = 0.0;
    double t0s = 0.0;

    if (this->intf_->verbose) {
        t0 = utils::getMillisecs();
    }
    for (int i = 0; i < niter; i++) {
        ret = npuFlatAtFp32->addVectorsAsCentroid(centroidsHost);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to centrodata, ret %d", ret);
        if (this->intf_->verbose) {
            t0s = utils::getMillisecs();
        }
        ret = npuFlatAtFp32->assign(codesTensor, indicesHost);
        FAISS_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to exec search centroids!");
        if (this->intf_->verbose) {
            tSearchTotal += utils::getMillisecs() - t0s;
            printf("Ascend Cluster Iteration %d (%.2f s, search %.2f s)  \r",
                i, (utils::getMillisecs() - t0) / 1000, tSearchTotal / 1000); // 除以 1000 将单位转换为s
            fflush(stdout);
        }
        // aicpu速度不如cpu，直接使用cpu更新
        computeCentroids(intf_->d, intf_->k, ntotal, codesFp32.data(),  indicesHost.data(), centroids);
        normalizeVecByHost(centroids, centrodata);
    }
    if (clearData) {
        std::vector<float>().swap(codesFp32);
        ntotal = 0;
    }
    APP_LOG_INFO("AscendClustering train finished.\n");
}

size_t AscendClusteringImpl::getNTotal()
{
    return ntotal;
}

void AscendClusteringImpl::corrPreProcess(float *corr)
{
    APP_LOG_INFO("AscendClusteringImpl::computeCorr start");
    FAISS_THROW_IF_NOT_MSG(corr, "correlation result ptr can not be nullptr.");
    ASCEND_THROW_IF_NOT_FMT((ntotal >= intf_->k) && (ntotal < CLUSTERING_MAX_N),
        "ntotal must be >= %zu and < %ld", intf_->k, CLUSTERING_MAX_N);
    ACL_REQUIRE_OK(aclrtSetDevice(config.deviceList[0]));
    if (!corrComputeOp) {
        FAISS_THROW_IF_NOT_MSG(resetCorrComputeOp() == APP_ERR_OK, "Failed to reset corrcompute op");
    }
}

void AscendClusteringImpl::computeCorr(float *corr, bool clearData)
{
    corrPreProcess(corr);
    auto streamPtr = resources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources->getMemoryManager();
    int dims = static_cast<int>(intf_->d);
    size_t n = codes.size() / intf_->d;
    int dim2 = utils::divUp(static_cast<int>(n), CUBE_ALIGN);
    AscendTensor<float16_t, DIMS_4> codesShaped(mem, { 1, dim2, dims, CUBE_ALIGN }, stream);
    std::vector<float16_t> codesShapedVec(codesShaped.numElements());
    AscendTensor<float16_t, DIMS_4> codesShapedHost(codesShapedVec.data(), { 1, dim2, dims, CUBE_ALIGN });

    for (int j = 0; j < dim2; ++j) {
        float16_t *tmpDataSrc = codes.data() + static_cast<int64_t>(dims) * CUBE_ALIGN * j;
        float16_t *tmpDataDst = codesShapedHost[0][j].data();
        int hpadding = (j == (dim2 - 1)) ? ((j + 1) * CUBE_ALIGN - n) : 0;
        for (int h2 = 0; h2 < dims; ++h2) {
            int h1 = 0;
            for (; h1 < CUBE_ALIGN - hpadding; ++h1) {
                tmpDataDst[h2 * CUBE_ALIGN + h1] = tmpDataSrc[h1 * dims + h2];
            }
            for (; h1 < CUBE_ALIGN; ++h1) {
                tmpDataDst[h2 * CUBE_ALIGN + h1] = 0;
            }
        }
    }

    int ret = aclrtMemcpy(codesShaped.data(), codesShaped.getSizeInBytes(),
        codesShapedHost.data(), codesShapedHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy codesShaped to device");

    AscendTensor<uint64_t, DIMS_1> actualNum(mem, { CORE_NUM }, stream);
    actualNum[0] = dim2 * CUBE_ALIGN;

    AscendTensor<float16_t, DIMS_2> corrResult(mem, {dims, dims}, stream);
    AscendTensor<uint16_t, DIMS_2> flag(mem, { CORE_NUM, FLAG_SIZE }, stream);
    (void)flag.zero();
    runCorrComputeOp(codesShaped, actualNum, corrResult, flag, stream);

    ret = synchronizeStream(stream);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i\n", ret);

    // clear data
    if (clearData) {
        std::vector<float16_t>().swap(codes);
        ntotal = 0;
    }

    std::vector<float16_t> corrResultHost(corrResult.numElements());
    ret = aclrtMemcpy(corrResultHost.data(), corrResult.getSizeInBytes(), corrResult.data(),
        corrResult.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy corr back to host %i\n", ret);

    transform(corrResultHost.begin(), corrResultHost.end(),
        corr, [](float16_t temp) { return static_cast<float>(fp16(temp)); });
    APP_LOG_INFO("AscendClusteringImpl::computeCorr finish");
}

APP_ERROR AscendClusteringImpl::resetCorrComputeOp()
{
    auto corrComputeOpReset = [&](std::unique_ptr<AscendOperator> &op) {
        int dims = static_cast<int>(intf_->d);
        AscendOpDesc desc("CorrCompute");
        std::vector<int64_t> codeShape({ 1, CORR_COMPUTE_SIZE / CUBE_ALIGN, dims, CUBE_ALIGN });
        std::vector<int64_t> actualNumShape({ CORE_NUM });
        std::vector<int64_t> corrResultShape({ dims, dims });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, codeShape.size(), codeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, actualNumShape.size(), actualNumShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, corrResultShape.size(), corrResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    corrComputeOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(corrComputeOpReset(corrComputeOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}

void AscendClusteringImpl::runCorrComputeOp(const AscendTensor<float16_t, DIMS_4>& codesShaped,
    const AscendTensor<uint64_t, DIMS_1>& actualNum, AscendTensor<float16_t, DIMS_2>& corrResult,
    AscendTensor<uint16_t, DIMS_2>& flag, aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(corrComputeOp);

    std::shared_ptr<std::vector<const aclDataBuffer *>> corrOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    corrOpInput->emplace_back(aclCreateDataBuffer(codesShaped.data(), codesShaped.getSizeInBytes()));
    corrOpInput->emplace_back(aclCreateDataBuffer(actualNum.data(), actualNum.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> corrOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    corrOpOutput->emplace_back(aclCreateDataBuffer(corrResult.data(), corrResult.getSizeInBytes()));
    corrOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    corrComputeOp->exec(*corrOpInput, *corrOpOutput, stream);
}

void AscendClusteringImpl::subClusAddInt8(idx_t n, const uint8_t *x, int seq, int k)
{
    APP_LOG_INFO("subClusAddInt8 start: searchNum=%ld, topK=%ld.\n", n, k);

    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < CLUSTERING_MAX_N),
        "n must be > 0 and < %ld", CLUSTERING_MAX_N);

    if (seq == 0 || codesInt8.size() == 0) {
        numCodesEachBuck.resize(NUM_BUCK_PER_BATCH, 0);
        numCentroidsEachBuck.resize(NUM_BUCK_PER_BATCH, 0);
    }
    numCodesEachBuck[seq] = static_cast<uint16_t>(n);
    numCentroidsEachBuck[seq] = static_cast<uint16_t>(k);

    codesInt8.insert(codesInt8.end(), x, x + n * intf_->d);
    updateNtotal += static_cast<size_t>(n);

    APP_LOG_INFO("subClusAddInt8 finished.\n");
}

void AscendClusteringImpl::subClusExecInt8(
    int niter, uint16_t *labels, float *centroids, int batchSubNlist, size_t batchNTotal)
{
    checkParaInt8(niter, labels, centroids, batchSubNlist, batchNTotal);
    
    execClusteringInt8(niter, labels, centroids);
}


void AscendClusteringImpl::randomCentrodataInt8(const uint8_t *data, int n, int nlist,
    std::vector<float16_t> &centrodata)
{
    std::vector<int> perm(n, 0);
    RandPerm(perm.data(), n, RAND_SEED);

    std::vector<float> initCentroids(static_cast<idx_t>(nlist) * static_cast<idx_t>(intf_->d), 0);

    for (size_t i = 0; i < size_t(nlist); i++) {
        size_t c = static_cast<size_t>(perm[i]);
        for (size_t d = 0; d < static_cast<size_t>(intf_->d); d++) {
            initCentroids[i * static_cast<size_t>(intf_->d) + d] = static_cast<float>((static_cast<float>(
                data[c * intf_->d + d]) + SQ_DECODE_PARAM) / UINT8_UPPER_BOUND * vDiff[d] + vMin[d]);
        }
    }

    transform(begin(initCentroids), end(initCentroids), begin(centrodata),
        [](float temp) { return fp16(temp).data; });
}

void AscendClusteringImpl::execClusteringInt8(int niter, uint16_t *labels, float *centroids)
{
    size_t curCodePos = 0;
    size_t curCentPos = 0;

    codesInt8.resize(codesInt8.size() + CORE_NUM * QUERY_ALIGN_CLUS);
    std::vector<std::vector<float16_t>> centrodataInit(NUM_BUCK_PER_BATCH,
        std::vector<float16_t>(intf_->d * SUBCENTER_NUM, 0)); // 初始化为0代表k-mean不参与计算

    std::vector<size_t> curCodePosAccum(NUM_BUCK_PER_BATCH, 0); // 为多线程随机初始化准备一个curCodePos的累计数组
    for (size_t i = 1; i < NUM_BUCK_PER_BATCH; i++) {
        int prevNtotal = numCodesEachBuck[i - 1];
        curCodePosAccum[i] = curCodePosAccum[i - 1] + static_cast<size_t>(prevNtotal);
    }

#pragma omp parallel for num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < NUM_BUCK_PER_BATCH; i++) {
        int curNtotal = numCodesEachBuck[i];
        if (curNtotal == 0) {
            continue;
        }
        randomCentrodataInt8(codesInt8.data() + curCodePosAccum[i] * intf_->d, curNtotal,
            numCentroidsEachBuck[i], centrodataInit[i]);
    }

    npuFlatAicpu = CREATE_UNIQUE_PTR(IndexFlatATSubAicpu, intf_->d, SUBCENTER_NUM, config.resourceSize);
    int ret = npuFlatAicpu->init();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "DistanceFlatL2At op init failed");

    for (size_t i = 0; i < NUM_BUCK_PER_BATCH; i++) {
        int curNtotal = numCodesEachBuck[i];
        if (curNtotal == 0) {
            continue;
        }

        int curCent = numCentroidsEachBuck[i];
        trainInt8(centrodataInit[i], curCent, curNtotal, niter, SUBCENTER_NUM,
            codesInt8.data() + curCodePos * intf_->d, labels + curCodePos, centroids + curCentPos * intf_->d);

        curCodePos += static_cast<size_t>(curNtotal);
        curCentPos += static_cast<size_t>(curCent);
    }

    // clear data
    std::vector<uint8_t>().swap(codesInt8);
    updateNtotal = 0;
}

void AscendClusteringImpl::trainInt8(std::vector<float16_t> &centrodata,
    int ncentroids, int n, int niter, int reqCentroids, const uint8_t *data, uint16_t *labels, float *subCentroids)
{
    auto streamPtr = resources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources->getMemoryManager();
    int dims = static_cast<int>(intf_->d);
    AscendTensor<uint16_t, DIMS_1> labelSubClus(mem, {utils::roundUp(n, QUERY_ALIGN_CLUS)}, stream);
    AscendTensor<uint8_t, DIMS_2> codesDevice(mem, { n, dims }, stream);
    int ret = aclrtMemcpy(codesDevice.data(), n * dims * sizeof(uint8_t),
        data, n * dims * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy codes to device");

    int pages = utils::divUp(n, npuFlatAicpu->searchPage);
    AscendTensor<float16_t, DIMS_4> subCentroidsCores(mem, {pages, CORE_NUM, reqCentroids, dims}, stream);
    AscendTensor<int16_t, DIMS_3> hassignCores(mem, {pages, CORE_NUM, reqCentroids}, stream);

    AscendTensor<float16_t, DIMS_4> curSubcentroidsCores(subCentroidsCores.data(), {1, CORE_NUM, SUBCENTER_NUM, dims});
    AscendTensor<int16_t, DIMS_3> inputHassignCores(hassignCores.data(), {1, CORE_NUM, SUBCENTER_NUM});
    AscendTensor<int16_t, DIMS_1> outHassign(mem, {SUBCENTER_NUM, }, stream);
    AscendTensor<uint16_t, DIMS_1> inputActualSize(mem, {FLAG_SIZE, }, stream);
    std::vector<uint16_t> inputActualSizeHost(FLAG_SIZE, 0);
    inputActualSizeHost[0] = static_cast<uint16_t>(pages);
    inputActualSizeHost[1] = static_cast<uint16_t>(ncentroids);
    ret = aclrtMemcpy(inputActualSize.data(), inputActualSize.getSizeInBytes(),
        inputActualSizeHost.data(), inputActualSize.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy inputActualSize to device, ret: %i\n", ret);
 
    AscendTensor<uint16_t, DIMS_2> outOpFlag(mem, { CORE_NUM, FLAG_SIZE }, stream);
    AscendTensor<float16_t, DIMS_2> centrodataDevice(mem, {SUBCENTER_NUM, dims }, stream);

    ret = aclrtMemcpy(centrodataDevice.data(), centrodataDevice.getSizeInBytes(),
        centrodata.data(), centrodata.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy centrodata to device, ret: %i\n", ret);

    for (int iter = 0; iter < niter - 1; iter++) {
        ret = npuFlatAicpu->addVectors(reqCentroids, dims, centrodataDevice);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to addVectors, ret %d", ret);
        ret = npuFlatAicpu->search(
            n, codesDevice.data(), 1, ncentroids, labelSubClus.data(), subCentroidsCores, hassignCores, vDM);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to search, ret %d", ret);

        runSubCntCompute(curSubcentroidsCores, inputHassignCores, inputActualSize,
            centrodataDevice, outHassign, outOpFlag, stream);
        ret = synchronizeStream(stream);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream aicore stream failed: %i\n", ret);
 
        ret = npuFlatAicpu->reset();
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to reset, ret %d", ret);
    }

    ret = aclrtMemcpy(centrodata.data(), centrodataDevice.getSizeInBytes(),
        centrodataDevice.data(), centrodataDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy codesDevice to host");

    std::transform(centrodata.data(), centrodata.data() + ncentroids * dims, subCentroids,
        [](float16_t temp) { return static_cast<float>(fp16(temp)); });

    ret = aclrtMemcpy(labels, size_t(n) * sizeof(uint16_t), labelSubClus.data(),
        size_t(n) * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy labels back to host %i\n", ret);
}

void AscendClusteringImpl::checkParaInt8(
    int, uint16_t *labels, float *centroids, int batchSubNlist, size_t batchNTotal)
{
    FAISS_THROW_IF_NOT_MSG(labels, "labels ptr can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(centroids, "centroids result ptr can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(isUpdateVDM, "VDM not Update.");
    FAISS_THROW_IF_NOT_FMT(numCodesEachBuck.size() == (size_t)NUM_BUCK_PER_BATCH,
        "the size of numCodesEachBuck is error, expected=%zu, actual=%zu",
        (size_t)NUM_BUCK_PER_BATCH, numCodesEachBuck.size());
    FAISS_THROW_IF_NOT_FMT(numCentroidsEachBuck.size() == (size_t)NUM_BUCK_PER_BATCH,
        "the size of numCentroidsEachBuck is error, expected=%zu, actual=%zu",
        (size_t)NUM_BUCK_PER_BATCH, numCentroidsEachBuck.size());
    FAISS_THROW_IF_NOT_FMT(batchNTotal == getNumCodes(), "Invalid number of batchNTotal, expected=%zu, actual=%zu.\n",
        getNumCodes(), batchNTotal);
    FAISS_THROW_IF_NOT_FMT((size_t)batchSubNlist == getNumCentroids(),
        "Invalid number of batchNTotal, expected=%zu, actual=%zu.\n", getNumCentroids(), (size_t)batchSubNlist);
    FAISS_THROW_IF_NOT_FMT(batchNTotal <= updateNtotal,
        "batchNTotal should not be greater than ntotal, batchNTotal=%zu, ntotal=%zu.\n",
        batchNTotal, updateNtotal);

    // check value of numCentroidsEachBuck and numCodesEachBuck
    for (int i = 0; i < NUM_BUCK_PER_BATCH; ++i) {
        FAISS_THROW_IF_NOT_FMT(numCentroidsEachBuck[i] <= SUBCENTER_NUM,
            "numCentroidsEachBuck[%d] should be in range [0, %d], actual=%d.\n",
            i, SUBCENTER_NUM, static_cast<int>(numCentroidsEachBuck[i]));
        auto numCodes = numCodesEachBuck[i];
        FAISS_THROW_IF_NOT_FMT((numCodes >= LOWER_BOUND && numCodes <= MAX_CLUS_POINTS) || (numCodes == 0),
            "numCodesEachBuck[%d] should be 0 or in range [%d, %d], actual=%d.\n",
            i, LOWER_BOUND, MAX_CLUS_POINTS, static_cast<int>(numCodesEachBuck[i]));
    }
}

APP_ERROR AscendClusteringImpl::resetSubcentsAccumOp()
{
    int dims = static_cast<int>(intf_->d);
    AscendOpDesc desc("SubcentAccum");
    std::vector<int64_t> subcentsCoresShape({ 1, CORE_NUM, SUBCENTER_NUM, dims });
    std::vector<int64_t> hassignCoresShape({ 1, CORE_NUM, SUBCENTER_NUM });
    std::vector<int64_t> actualSizeShape({ FLAG_SIZE, });
    std::vector<int64_t> subcentsShape({ SUBCENTER_NUM, dims });
    std::vector<int64_t> hassignShape({ SUBCENTER_NUM, });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, subcentsCoresShape.size(), subcentsCoresShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT16, hassignCoresShape.size(), hassignCoresShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT16, actualSizeShape.size(), actualSizeShape.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, subcentsShape.size(), subcentsShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_INT16, hassignShape.size(), hassignShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    subCentsOp.reset();
    subCentsOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(subCentsOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}

void  AscendClusteringImpl::runSubCntCompute(AscendTensor<float16_t, DIMS_4> &subcentsCores,
                                             AscendTensor<int16_t, DIMS_3> &hassignCores,
                                             AscendTensor<uint16_t, DIMS_1> &actualSize,
                                             AscendTensor<float16_t, DIMS_2> &subcents,
                                             AscendTensor<int16_t, DIMS_1> &hassign,
                                             AscendTensor<uint16_t, DIMS_2> &flag,
                                             aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(subCentsOp);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(subcentsCores.data(), subcentsCores.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(hassignCores.data(), hassignCores.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(actualSize.data(), actualSize.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(subcents.data(), subcents.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(hassign.data(), hassign.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    subCentsOp->exec(*distOpInput, *distOpOutput, stream);
}

void AscendClusteringImpl::updateVDM(uint16_t *vmin, uint16_t *vdiff)
{
    FAISS_THROW_IF_NOT_MSG(vmin, "vmin ptr can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(vdiff, "vdiff ptr can not be nullptr.");
    const int dims = static_cast<int>(intf_->d);
    vMin.resize(dims, 0.0);
    vDiff.resize(dims, 0.0);

    std::transform(vmin, vmin + dims, vMin.data(), [](uint16_t temp) { return (float)fp16(temp); });
    std::transform(vdiff, vdiff + dims, vDiff.data(), [](uint16_t temp) { return (float)fp16(temp); });

    AscendTensor<float16_t, DIMS_2> dmTensor({ 2, dims });
    int ret = aclrtMemcpy(dmTensor[0].data(), dims * sizeof(float16_t),
        vmin, dims * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy vmin to dmTensor");
    ret = aclrtMemcpy(dmTensor[1].data(), dims * sizeof(float16_t),
        vdiff, dims * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy vdiff to dmTensor");
    vDM = std::move(dmTensor);

    FAISS_THROW_IF_NOT_MSG(resetSubcentsAccumOp() == APP_ERR_OK, "Failed to reset SubcentAccum op");
    isUpdateVDM = true;
}

size_t AscendClusteringImpl::getNumCodes() const
{
    size_t curCodePos = 0;
    for (size_t i = 0; i < NUM_BUCK_PER_BATCH; i++) {
        size_t curNtotal = static_cast<size_t>(numCodesEachBuck[i]);
        curCodePos += curNtotal;
    }
    return curCodePos;
}

size_t AscendClusteringImpl::getNumCentroids() const
{
    size_t curCentPos = 0;
    for (size_t i = 0; i < NUM_BUCK_PER_BATCH; i++) {
        size_t curCent = static_cast<size_t>(numCentroidsEachBuck[i]);
        curCentPos += curCent;
    }
    return curCentPos;
}

int AscendClusteringImpl::getSubBucketNum(int seq)
{
    FAISS_THROW_IF_NOT(seq >= 0 && static_cast<size_t>(seq) < numCentroidsEachBuck.size());
    return numCentroidsEachBuck[seq];
}

void AscendClusteringImpl::resetKmUpdateCentroidsOp()
{
    AscendOpDesc desc("KmUpdateCentroids");
    std::vector<int64_t> shape0 { static_cast<int64_t>(ntotal), static_cast<int64_t>(intf_->d) };
    std::vector<int64_t> shape1 { static_cast<int64_t>(ntotal) };
    std::vector<int64_t> shape2 { static_cast<int64_t>(intf_->k), static_cast<int64_t>(intf_->d) };

    desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, shape1.size(), shape1.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, shape2.size(), shape2.data(), ACL_FORMAT_ND);

    kmUpdateCentroidsComputeOp.reset();
    kmUpdateCentroidsComputeOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    FAISS_THROW_IF_NOT_MSG(kmUpdateCentroidsComputeOp->init(), "Failed to resetKmUpdateCentroidsOp");
}

void AscendClusteringImpl::runkmUpdateCentroidsCompute(AscendTensor<float16_t, DIMS_2> &codes,
                                                       AscendTensor<float16_t, DIMS_2> &centrodata,
                                                       AscendTensor<uint64_t, DIMS_1> &assign,
                                                       aclrtStream stream)
{
    std::shared_ptr<std::vector<const aclDataBuffer *>> kmUpdateCentroidsOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    kmUpdateCentroidsOpInput->emplace_back(aclCreateDataBuffer(codes.data(), codes.getSizeInBytes()));
    kmUpdateCentroidsOpInput->emplace_back(aclCreateDataBuffer(assign.data(), assign.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> kmUpdateCentroidsOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    kmUpdateCentroidsOpOutput->emplace_back(aclCreateDataBuffer(centrodata.data(), centrodata.getSizeInBytes()));

    kmUpdateCentroidsComputeOp->exec(*kmUpdateCentroidsOpInput, *kmUpdateCentroidsOpOutput, stream);
}

void AscendClusteringImpl::randomCentrodata(AscendTensor<float16_t, DIMS_2> &centrodata)
{
    std::vector<int> perm(ntotal, 0);
    RandPerm(perm.data(), ntotal, RAND_SEED);

    std::vector<float16_t> centrodataTmp(intf_->k * intf_->d);
    for (size_t i = 0; i < intf_->k; i++) {
        size_t offset = i * intf_->d;
        auto ret = memcpy_s(centrodataTmp.data() + offset, (centrodataTmp.size() - offset) * sizeof(float16_t),
                            codes.data() + perm[i] * intf_->d, intf_->d * sizeof(float16_t));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Failed to copy to centrodata (error %d)", (int)ret);
    }
    auto ret = aclrtMemcpy(centrodata.data(), centrodata.getSizeInBytes(),
                           centrodataTmp.data(), centrodataTmp.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy centrodata to device (error %d)", (int)ret);
}

uint64_t get_actual_rng_seed(const int seed)
{
    return (seed >= 0) ? seed : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                            .time_since_epoch()
                                            .count());
}

void AscendClusteringImpl::randomCentrodataFp32(AscendTensor<float, DIMS_2> &centrodata)
{
    std::vector<int> perm(ntotal, 0);
    const uint64_t actual_seed = get_actual_rng_seed(RAND_SEED);
    rand_perm(perm.data(), ntotal, actual_seed + 1);
    std::vector<float> centrodataTmp(intf_->k * intf_->d);
    for (size_t i = 0; i < intf_->k; i++) {
        size_t offset = i * intf_->d;
        auto ret = memcpy_s(centrodataTmp.data() + offset, (centrodataTmp.size() - offset) * sizeof(float),
                            codesFp32.data() + perm[i] * intf_->d, intf_->d * sizeof(float));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Failed to copy to centrodata (error %d)", ret);
    }
    auto ret = aclrtMemcpy(centrodata.data(), centrodata.getSizeInBytes(),
                           centrodataTmp.data(), centrodataTmp.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy centrodata to device (error %d)", ret);
}

void AscendClusteringImpl::trainPostProcess(AscendTensor<float16_t, DIMS_2> &centrodata, float *centroids)
{
    std::vector<float16_t> centroResult(intf_->k * intf_->d, 0);
    auto ret = aclrtMemcpy(centroResult.data(), centroResult.size() * sizeof(float16_t),
                           centrodata.data(), centrodata.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy centro result back to host (error %d)", (int)ret);
    std::transform(centroResult.begin(), centroResult.end(), centroids,
        [](float16_t temp) { return (float)fp16(temp); });
}
}
}