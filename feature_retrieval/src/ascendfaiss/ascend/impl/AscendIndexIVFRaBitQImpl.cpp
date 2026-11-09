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

#include "AscendIndexIVFRaBitQImpl.h"

#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

#include "ascend/AscendIDSelectorRoaring.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/impl/AscendRoaringFilter.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/MemDebug.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace faiss
{
namespace ascend
{

constexpr size_t MAX_TRAINNUM = 7000000;

// The value range of nlist
const std::vector<int> NLISTS = {1024, 2048, 4096, 8192, 10048, 16384, 32768};
// The value range of dims
const std::vector<int> DIMS = {64, 128, 256, 384, 512, 768, 1024};
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 640;
const size_t ADD_PAGE_SIZE = (UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE);
const size_t UNIT_VEC_SIZE = 5120;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

// Faiss 1.14 uses a larger per-vector layout for native multi-bit RaBitQ.
// Ascend keeps a compact [bit planes][L2][L1] payload and stores this trailer
// in the extra carrier bytes. Keep the original marker value so indexes saved
// by the earlier 2-bit implementation remain compatible.
constexpr uint64_t RABITQ_MULTIBIT_CARRIER_MAGIC = 0x4153435242513201ULL;
constexpr uint32_t RABITQ_MULTIBIT_CARRIER_VERSION = 1;
constexpr size_t RABITQ_MULTIBIT_CARRIER_TRAILER_SIZE = sizeof(uint64_t) + sizeof(uint32_t);

size_t getAscendRaBitQCodePartSize(int dims, int quantBits)
{
    return static_cast<size_t>(dims) * static_cast<size_t>(quantBits) / 8;
}

size_t getAscendRaBitQPayloadSize(int dims, int quantBits)
{
    return getAscendRaBitQCodePartSize(dims, quantBits) + 2 * sizeof(float);
}

size_t getFaissRaBitQCarrierSize(int dims, int quantBits)
{
    const size_t payloadSize = getAscendRaBitQPayloadSize(dims, quantBits);
    return quantBits > 1 ? payloadSize + RABITQ_MULTIBIT_CARRIER_TRAILER_SIZE : payloadSize;
}

void writeMultiBitCarrierTrailer(uint8_t* dst)
{
    std::memcpy(dst, &RABITQ_MULTIBIT_CARRIER_MAGIC, sizeof(RABITQ_MULTIBIT_CARRIER_MAGIC));
    std::memcpy(dst + sizeof(RABITQ_MULTIBIT_CARRIER_MAGIC), &RABITQ_MULTIBIT_CARRIER_VERSION,
                sizeof(RABITQ_MULTIBIT_CARRIER_VERSION));
}

bool isAscendMultiBitCarrier(const uint8_t* trailer)
{
    uint64_t magic = 0;
    uint32_t version = 0;
    std::memcpy(&magic, trailer, sizeof(magic));
    std::memcpy(&version, trailer + sizeof(magic), sizeof(version));
    return magic == RABITQ_MULTIBIT_CARRIER_MAGIC && version == RABITQ_MULTIBIT_CARRIER_VERSION;
}

extern "C"
{
    /* declare LAPACK functions */
    int sgeqrf_(long* m, long* n, float* a, long* lda, float* tau, float* work, long* lwork, long* info);

    int sorgqr_(long* m, long* n, long* k, float* a, long* lda, float* tau, float* work, long* lwork, long* info);
}

RandomGenerator::RandomGenerator(int64_t seed) : mt(static_cast<unsigned int>(seed)) {}

int RandomGenerator::rand_int() { return mt() & 0x7fffffff; }

int64_t RandomGenerator::rand_int64()
{
    return static_cast<int64_t>(rand_int()) | static_cast<int64_t>(rand_int()) << 31;
}

int RandomGenerator::rand_int(int max) { return mt() % max; }

float RandomGenerator::rand_float() { return mt() / static_cast<float>(mt.max()); }

double RandomGenerator::rand_double() { return mt() / static_cast<double>(mt.max()); }

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
    for (int64_t j = 0; j < static_cast<int64_t>(nblock); j++)
    {
        RandomGenerator rng(a0 + j * b0);

        double a = 0;
        double b = 0;
        double s = 0;
        int state = 0;  // generate two number per "do-while" loop

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++)
        {
            // Marsaglia's method (see Knuth)
            if (state == 1)
            {
                x[i] = a * sqrt(-2.0 * log(s) / s);
                state = 1 - state;
                continue;
            }
            do
            {
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
    if (d_out <= d_in)
    {
        A.resize(d_out * d_in);
        float* q = A.data();
        float_randn(q, d_out * d_in, seed);
        matrix_qr(d_in, d_out, q);
    }
    else
    {
        // use tight-frame transformation
        A.resize(d_out * d_out);
        float* q = A.data();
        float_randn(q, d_out * d_out, seed);
        matrix_qr(d_out, d_out, q);
        // remove columns
        int i;
        int j;
        for (i = 0; i < d_out; i++)
        {
            for (j = 0; j < d_in; j++)
            {
                q[i * d_in + j] = q[i * d_out + j];
            }
        }
        A.resize(d_in * d_out);
    }
}

AscendIndexIVFRaBitQImpl::AscendIndexIVFRaBitQImpl(AscendIndexIVFRaBitQ* intf, int dims, int nlist,
                                                   faiss::MetricType metric, AscendIndexIVFRaBitQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config), intf_(intf), ivfrabitqConfig(config)
{
    checkParams();
    if (config.niter > 0)
    {
        ivfConfig.cp.niter = config.niter;
    }
    // Reads and writes now run concurrently and fan out over the same devices, so each role gets its
    // own pool rather than queueing behind the other on the base class' single one. Like the base
    // class, only worth a pool when there is more than one device to fan out to.
    const size_t deviceCnt = indexConfig.deviceList.size();
    if (deviceCnt > 1)
    {
        readPool = std::make_shared<AscendThreadPool>(deviceCnt);
        writePool = std::make_shared<AscendThreadPool>(deviceCnt);
    }
    initIndexes();

    initDeviceAddNumMap();
    centroidsData.resize(nlist);
    this->intf_->is_trained = false;
    initFlatAtFp32();
    orthogonalMatrix.resize(dims * dims, 0);
}

AscendIndexIVFRaBitQImpl::~AscendIndexIVFRaBitQImpl() {}

void AscendIndexIVFRaBitQImpl::initFlatAtFp32()
{
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl initFlatAtFp32 started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
    assignIndex = CREATE_UNIQUE_PTR(::ascend::IndexIVFFlat, nlist, intf_->d, 1, -1);
    assignIndex->init();
    if (this->ivfConfig.useKmeansPP)
    {
        AscendClusteringConfig npuClusConf({ivfConfig.deviceList[0]}, ivfConfig.resourceSize);
        pQuantizerImpl->npuClus =
            std::make_shared<AscendClustering>(this->intf_->d, this->nlist, this->intf_->metric_type, npuClusConf);
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl initFlatAtFp32 finished.\n");
}

void AscendIndexIVFRaBitQImpl::checkParams() const
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
                               this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(),
                           "Unsupported nlists %d", this->nlist);
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(),
                           "Unsupported dimensions %d, should be in {64, 128, 256, 384, 512, 768, 1024, 2048}",
                           this->intf_->d);
    FAISS_THROW_IF_NOT_MSG(ivfConfig.deviceList.size() != 0, "deviceList is empty");
    FAISS_THROW_IF_NOT_MSG(
        ivfrabitqConfig.quantBits == 1 || ivfrabitqConfig.quantBits == 2 || ivfrabitqConfig.quantBits == 4,
        "RaBitQ quantBits must be 1, 2, or 4");
    FAISS_THROW_IF_NOT_MSG(ivfrabitqConfig.quantBits == 1 || this->intf_->metric_type == MetricType::METRIC_L2,
                           "Multi-bit RaBitQ POC only supports L2 metric");
    FAISS_THROW_IF_NOT_MSG(ivfrabitqConfig.quantBits != 2 || this->intf_->d <= 512,
                           "2-bit RaBitQ POC supports dimensions up to 512 (query LUT UB limit)");
    FAISS_THROW_IF_NOT_MSG(ivfrabitqConfig.quantBits != 4 || this->intf_->d <= 256,
                           "4-bit RaBitQ POC supports dimensions up to 256 (query LUT UB limit)");
}

std::vector<idx_t> AscendIndexIVFRaBitQImpl::update(idx_t n, const float* x, const idx_t* ids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    std::lock_guard<std::mutex> writerLock(writerMutex);
    FAISS_THROW_IF_NOT_MSG(attachedSrc_ == nullptr, "update conflicts with attached refine sources; reset first.");
    FAISS_THROW_IF_NOT_MSG(!copiedSources_ && srcIndexes.empty(),
                           "update conflicts with copied refine sources; reset first.");
    if (ivfrabitqConfig.needRefine)
    {
        getRefineSources();  // Reject missing originals before update's delete/add sequence.
    }
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "vector list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(ids != nullptr, "vector ID list is nullptr!");
    FAISS_THROW_IF_NOT_MSG(n > 0, "vector number must be greater than 0!");
    FAISS_THROW_IF_NOT_FMT(n < MAX_N, "n must be < %ld", MAX_N);
    FAISS_THROW_IF_NOT_MSG(n <= std::numeric_limits<int>::max(), "n exceeds int range");
    FAISS_THROW_IF_NOT_MSG(
        static_cast<size_t>(n) * this->intf_->d == static_cast<size_t>(std::distance(x, x + n * this->intf_->d)),
        "vector list size is not match!");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(n) == static_cast<size_t>(std::distance(ids, ids + n)),
                           "vector ID list size is not match!");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "AscendIndexIVFRaBitQ is not trained!");
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl update operation started: n=%ld.\n", n);

    std::vector<idx_t> noExistIds;
    std::vector<idx_t> existIds;
    std::vector<float> existVectors;
    idx_t noExistNum = 0;
    idx_t existNum = 0;

    // An id repeated within one batch must collapse to its last vector, otherwise the single delete
    // below would be followed by two adds of the same id.
    std::unordered_map<idx_t, idx_t> lastOccurrence;
    for (idx_t i = 0; i < n; ++i)
    {
        lastOccurrence[ids[i]] = i;
    }

    {
        std::lock_guard<std::mutex> lock(mapMutex);
        for (idx_t i = 0; i < n; ++i)
        {
            idx_t id = ids[i];
            if (lastOccurrence[id] != i)
            {
                continue;
            }
            if (idToDeviceMap.find(id) == idToDeviceMap.end())
            {
                noExistIds.push_back(id);
                noExistNum++;
                continue;
            }
            idx_t devId = idToDeviceMap[id];
            if (deviceInfos[devId].idSet.find(id) == deviceInfos[devId].idSet.end())
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
    }

    if (!noExistIds.empty())
    {
        APP_LOG_WARNING("The following %ld IDs do not exist: \n", noExistNum);
        for (idx_t i = 0; i < noExistNum; i++)
        {
            APP_LOG_WARNING("ID: %ld\n", noExistIds[i]);
        }
        APP_LOG_WARNING("Updating other vectors with ids\n");
    }
    if (existNum > 0)
    {
        deleteImpl(existNum, existIds.data());
        addPaged(existNum, existVectors.data(), existIds.data());
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQ update operation finished.\n");

    return noExistIds;
}

void AscendIndexIVFRaBitQImpl::addL1(int n, const float* x, std::unique_ptr<idx_t[]>& assign)
{
    // 使用npu能力加速add过程
    FAISS_THROW_IF_NOT_MSG(assignIndex != nullptr, "assignIndex is not init");
    FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
    ::ascend::AscendTensor<float, ::ascend::DIMS_2> codes(const_cast<float*>(x), {n, this->intf_->d});
    ::ascend::AscendTensor<idx_t, ::ascend::DIMS_2> indices(assign.get(), {n, 1});
    auto ret = assignIndex->assign(codes, indices);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "assign failed %d", ret);
}

void AscendIndexIVFRaBitQImpl::addImpl(int n, const float* x, const idx_t* ids) { addPaged(n, x, ids); }

void AscendIndexIVFRaBitQImpl::stageAdd(int n, const float* x, const idx_t* ids, AddStaging& staging)
{
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl stageAdd operation started: n=%d.\n", n);
    const size_t deviceCnt = indexConfig.deviceList.size();
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);
    addL1(n, x, assign);
    const size_t dim = static_cast<size_t>(intf_->d);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i)
    {
        const idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);
        auto inserted = staging.try_emplace(static_cast<int>(listId), deviceAddNumMap[listId]);
        auto& list = inserted.first->second;
        if (inserted.second)
        {
            for (size_t j = 0; j < deviceCnt; ++j)
            {
                list.occupancy[j] = static_cast<int>(getActualIndex(indexConfig.deviceList[j])->getListLength(listId));
            }
        }
        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; ++j)
        {
            if (list.occupancy[j] < list.occupancy[devIdx])
            {
                devIdx = j;
            }
        }
        auto& device = list.devices[devIdx];
        device.codes.insert(device.codes.end(), x + i * dim, x + (i + 1) * dim);
        device.ids.push_back(static_cast<ascend_idx_t>(ids[i]));
        ++list.occupancy[devIdx];  // Scratch only; authoritative counts change after publication.
    }
}

void AscendIndexIVFRaBitQImpl::copyVectorToDevice(int n, AddStaging& staging)
{
    const size_t deviceCnt = indexConfig.deviceList.size();
    const bool memDebug = ::ascend::MemDebugEnabled();
    const size_t every = memDebug ? ::ascend::MemDebugEvery() : 0;

    // reserve() allocates buckets, not nodes. Build both kinds of nodes before any daemon publication,
    // then transfer them under mapMutex without allocating after a successful upload.
    std::unordered_map<idx_t, idx_t> pendingMappings;
    std::vector<std::unordered_set<idx_t>> pendingIds(deviceCnt);
    pendingMappings.reserve(static_cast<size_t>(n));
    for (auto& centroid : staging)
    {
        for (size_t d = 0; d < deviceCnt; ++d)
        {
            for (ascend_idx_t id : centroid.second.devices[d].ids)
            {
                FAISS_THROW_IF_NOT_MSG(pendingMappings.emplace(static_cast<idx_t>(id), d).second,
                                       "duplicate id within staged add");
                pendingIds[d].insert(static_cast<idx_t>(id));
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        deviceInfos.resize(deviceCnt);
        idToDeviceMap.reserve(idToDeviceMap.size() + pendingMappings.size());
        for (const auto& entry : pendingMappings)
        {
            FAISS_THROW_IF_NOT_FMT(idToDeviceMap.find(entry.first) == idToDeviceMap.end(), "id %ld already exists",
                                   entry.first);
        }
        for (size_t d = 0; d < deviceCnt; ++d)
        {
            deviceInfos[d].idSet.reserve(deviceInfos[d].idSet.size() + pendingIds[d].size());
        }
    }

    if (memDebug)
    {
        ::ascend::MemDebugPrintf("[MemDebug] copyVectorToDevice start n=%d nlist=%d nonEmptyLists=%zu devices=%zu\n", n,
                                 this->nlist, staging.size(), deviceCnt);
        for (size_t d = 0; d < deviceCnt; ++d)
        {
            ::ascend::MemDebugPrintf("[MemDebug] copyVectorToDevice hostVecs deviceIdx=%zu deviceId=%d count=%zu\n", d,
                                     indexConfig.deviceList[d], pendingIds[d].size());
            ::ascend::LogHbm("copyVectorToDevice_start", indexConfig.deviceList[d]);
        }
    }

    auto addFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        size_t listsDone = 0;
        size_t vecsUploaded = 0;
        size_t bytesEst = 0;
        const size_t codeBytes = getAscendRaBitQCodePartSize(this->intf_->d, ivfrabitqConfig.quantBits);
        const size_t perVecBytes = codeBytes + 2 * sizeof(float) + sizeof(ascend_idx_t);

        if (memDebug)
        {
            FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
            ::ascend::LogHbm("copyVectorToDevice_device_begin", deviceId);
        }

        for (auto& centroid : staging)
        {
            const int listId = centroid.first;
            auto& device = centroid.second.devices[idx];
            const int num = static_cast<int>(device.ids.size());
            if (num == 0)
            {
                continue;
            }
            IndexParam<float, float, ascend_idx_t> param(deviceId, num, 0, 0);
            param.listId = listId;
            param.query = device.codes.data();
            param.label = device.ids.data();
            indexIVFRaBitQAdd(param);
            device.published = true;
            {
                std::lock_guard<std::mutex> lock(mapMutex);
                for (ascend_idx_t id : device.ids)
                {
                    idToDeviceMap.insert(pendingMappings.extract(static_cast<idx_t>(id)));
                    deviceInfos[idx].idSet.insert(pendingIds[idx].extract(static_cast<idx_t>(id)));
                }
                deviceAddNumMap[listId][idx] = centroid.second.occupancy[idx];
            }

            if (memDebug)
            {
                ++listsDone;
                vecsUploaded += static_cast<size_t>(num);
                bytesEst += static_cast<size_t>(num) * perVecBytes;
                if (every > 0 && (listsDone % every == 0))
                {
                    ::ascend::MemDebugPrintf(
                        "[MemDebug] upload device=%d listId=%d num=%d listsDone=%zu "
                        "vecsUploaded=%zu bytesEst=%zu\n",
                        deviceId, listId, num, listsDone, vecsUploaded, bytesEst);
                    ::ascend::LogHbm("copyVectorToDevice_progress", deviceId);
                }
            }
        }

        if (memDebug)
        {
            ::ascend::MemDebugPrintf(
                "[MemDebug] copyVectorToDevice device=%d finished lists=%zu vecs=%zu bytesEst=%zu\n", deviceId,
                listsDone, vecsUploaded, bytesEst);
            ::ascend::LogHbm("copyVectorToDevice_device_end", deviceId);
        }
    };
    std::exception_ptr failure;
    try
    {
        // The shared helper must drain every enqueued future, including on failure, before locals die.
        CALL_PARALLEL_FUNCTOR(deviceCnt, writePool, addFunctor);
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    idx_t published = 0;
    for (const auto& centroid : staging)
    {
        for (const auto& device : centroid.second.devices)
        {
            if (device.published)
            {
                published += static_cast<idx_t>(device.ids.size());
            }
        }
    }
    this->intf_->ntotal += published;
    if (published > 0)
    {
        InvalidateFilterCache();
    }
    if (failure)
    {
        std::rethrow_exception(failure);  // Successful lists stay published; no global rollback.
    }

    if (memDebug)
    {
        for (size_t d = 0; d < deviceCnt; ++d)
        {
            ::ascend::LogHbm("copyVectorToDevice_end", indexConfig.deviceList[d]);
        }
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl copyVectorToDevice operation finished.\n");
}

size_t AscendIndexIVFRaBitQImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

void AscendIndexIVFRaBitQImpl::addConcurrent(int n, const float* x, const idx_t* ids)
{
    // Refine searches borrow srcIndexes. Appending originals can reallocate it and temporarily make its
    // row count disagree with ntotal, so this mode retains the exclusive host lock across the whole add.
    // Ordinary searches use only device snapshots and can still overlap a writer.
    std::optional<std::unique_lock<std::shared_mutex>> exclusiveLock;
    std::optional<std::shared_lock<std::shared_mutex>> sharedLock;
    sharedLock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    if (ivfrabitqConfig.needRefine || copiedSources_ || !srcIndexes.empty())
    {
        sharedLock.reset();
        exclusiveLock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    }
    std::lock_guard<std::mutex> writerLock(writerMutex);
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl addConcurrent operation started.\n");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + n < MAX_N, "ntotal must be < %ld", MAX_N);

    CheckFilterTime(n, ids);

    std::vector<idx_t> generatedIds;
    {
        std::lock_guard<std::mutex> mapLock(mapMutex);
        if (ids == nullptr)
        {
            const bool keepSources = ivfrabitqConfig.needRefine || copiedSources_ || !srcIndexes.empty();
            generatedIds.resize(static_cast<size_t>(n));
            idx_t nextId = this->intf_->ntotal;
            for (int i = 0; i < n; ++i)
            {
                while (!keepSources && idToDeviceMap.find(nextId) != idToDeviceMap.end())
                {
                    FAISS_THROW_IF_NOT_MSG(nextId < std::numeric_limits<idx_t>::max(), "automatic ID overflow");
                    ++nextId;
                }
                generatedIds[i] = nextId;
                if (i + 1 < n)
                {
                    FAISS_THROW_IF_NOT_MSG(nextId < std::numeric_limits<idx_t>::max(), "automatic ID overflow");
                    ++nextId;
                }
            }
            ids = generatedIds.data();
        }

        // The base class does not check duplicate IDs; validate both batch and existing IDs here.
        std::unordered_set<idx_t> batchIds;
        for (int i = 0; i < n; ++i)
        {
            FAISS_THROW_IF_NOT_FMT(batchIds.insert(ids[i]).second, "duplicate id %ld within one add batch", ids[i]);
            FAISS_THROW_IF_NOT_FMT(idToDeviceMap.find(ids[i]) == idToDeviceMap.end(), "id %ld already exists", ids[i]);
        }
    }

    addPaged(n, x, ids);
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl addConcurrent operation finished.\n");
}

void AscendIndexIVFRaBitQImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(attachedSrc_ == nullptr, "add conflicts with attached refine sources; reset first.");
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl addPaged operation started.\n");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "x cannot be nullptr");
    const bool keepRefineSources = ivfrabitqConfig.needRefine || copiedSources_ || !srcIndexes.empty();
    if (keepRefineSources)
    {
        FAISS_THROW_IF_NOT_MSG(
            refineSourcesReady_ && srcIndexes.size() == static_cast<size_t>(intf_->ntotal) * intf_->d,
            "refine sources are not ready: loadRefineSources before adding vectors.");
        FAISS_THROW_IF_NOT_MSG(ids != nullptr, "refine requires explicit sequential IDs.");
        for (int i = 0; i < n; ++i)
        {
            FAISS_THROW_IF_NOT_MSG(ids[i] == intf_->ntotal + i, "refine requires sequential IDs starting at ntotal.");
        }
        srcIndexes.reserve(srcIndexes.size() + static_cast<size_t>(n) * static_cast<size_t>(intf_->d));
    }
    AddStaging staging;
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
                printf("AscendIndexIVFRaBitQImpl::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            stageAdd(curNum, x + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr, staging);
        }
    }
    else
    {
        if (this->intf_->verbose)
        {
            printf("AscendIndexIVFRaBitQImpl::add: adding 0:%d / %d\n", n, n);
        }
        stageAdd(n, x, ids, staging);
    }
    const idx_t previousTotal = intf_->ntotal;
    try
    {
        copyVectorToDevice(n, staging);
    }
    catch (...)
    {
        if (keepRefineSources && intf_->ntotal != previousTotal)
        {
            // Per-list publication need not be an ID prefix. Never expose partial originals as ready.
            copiedSources_ = true;
            refineSourcesReady_ = false;
        }
        throw;
    }
    if (keepRefineSources)
    {
        this->srcIndexes.insert(this->srcIndexes.end(), x,
                                x + static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d));
        copiedSources_ = true;
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl addPaged operation finished.\n");
}

void AscendIndexIVFRaBitQImpl::indexIVFRaBitQAdd(IndexParam<float, float, ascend_idx_t>& param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;
    const float* codes = param.query;
    const ascend_idx_t* ids = param.label;
    auto ret = pIndex->addVectors(param.listId, param.n, codes, static_cast<const ::ascend::Index::idx_t*>(ids));
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to add to ivf rabitq, ret: %d", ret);
}

std::shared_ptr<::ascend::Index> AscendIndexIVFRaBitQImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFRaBitQ  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::string metric = "L2";
    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT)
    {
        metric = "IP";
    }
    std::shared_ptr<::ascend::IndexIVF> index = std::make_shared<::ascend::IndexIVFRaBitQ>(
        nlist, this->intf_->d, nprobe, metric, indexConfig.resourceSize, ivfrabitqConfig.quantBits,
        ivfrabitqConfig.retiredMemoryHighWatermark);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf rabitq, result is %d", ret);

    APP_LOG_INFO("AscendIndexIVFRaBitQ createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFRaBitQImpl::updateCoarseCenter(std::vector<float>& centerData)
{
    size_t deviceCnt = indexConfig.deviceList.size();
    auto updateCoarseCenterFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->updateCoarseCenterImpl(centerData);
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to update coarsecenter, deviceId is %d, result is: %d\n", deviceId, ret);
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, writePool, updateCoarseCenterFunctor);
}

void AscendIndexIVFRaBitQImpl::randomOrthogonalGivens(int n, std::vector<float>& orthogonalMatrix)
{
    APP_LOG_INFO("AscendIndexIVFRaBitQ  create randomOrthogonalMatrix started\n");
    if (ivfrabitqConfig.useRandomOrthogonalMatrix)
    {
        orthonormalinit(orthogonalMatrix, ivfrabitqConfig.matrixSeed, n, n);
    }
    else
    {
        // 使用单位矩阵
        for (int i = 0; i < n; ++i)
        {
            orthogonalMatrix[i * n + i] = 1.0;
        }
    }
}

void AscendIndexIVFRaBitQImpl::uploadorthogonalMatrix(std::vector<float>& orthogonalMatrix)
{
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());
    for (int i = 0; i < deviceCnt; i++)
    {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = aclrtMemcpy(pIndex->OrthogonalMatrixOnDevice->data(),
                               pIndex->OrthogonalMatrixOnDevice->size() * sizeof(float), orthogonalMatrix.data(),
                               intf_->d * intf_->d * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,
                               "Failed to aclrtMemcpy orthogonalMatrix, deviceId is %d, result is: %d\n", deviceId,
                               ret);
    }
}

void AscendIndexIVFRaBitQImpl::train(idx_t n, const float* x, bool clearNpuData)
{
    APP_LOG_INFO("AscendIndexIVFRaBitQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained)
    {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    size_t trainNum = std::min({static_cast<size_t>(n), static_cast<size_t>(nlist) * 256, MAX_TRAINNUM});
    const float* trainData = x;
    std::vector<float> trainVector;

    if (static_cast<size_t>(n) > trainNum)
    {
        APP_LOG_INFO("AscendIndexIVFRaBitQImpl train: sampling %zu vectors from %ld for training\n", trainNum, n);
        trainVector.resize(trainNum * this->intf_->d);
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < static_cast<size_t>(n); ++i)
        {
            indices[i] = i;
        }
        std::mt19937 gen(static_cast<uint32_t>(this->ivfConfig.cp.seed));
        std::shuffle(indices.begin(), indices.end(), gen);
        for (size_t i = 0; i < trainNum; ++i)
        {
            size_t srcIdx = indices[i];
            std::copy(x + srcIdx * this->intf_->d, x + (srcIdx + 1) * this->intf_->d,
                      trainVector.begin() + i * this->intf_->d);
        }
        trainData = trainVector.data();
    }

    randomOrthogonalGivens(this->intf_->d, this->orthogonalMatrix);
    uploadorthogonalMatrix(this->orthogonalMatrix);
    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT)
    {
        APP_LOG_INFO("METRIC_INNER_PRODUCT must set spherical to true in cpu train case\n");
        this->ivfConfig.cp.spherical = true;
    }
    if (ivfConfig.useKmeansPP)
    {
        pQuantizerImpl->npuClus->verbose = this->intf_->verbose;
        std::vector<float> tmpCentroids(nlist * this->intf_->d);
        if (pQuantizerImpl->npuClus->GetNTotal() == 0)
        {
            pQuantizerImpl->npuClus->AddFp32(trainNum, trainData);
        }
        if (this->intf_->verbose)
        {
            printf("Ascend cluster start training %zu vectors\n", pQuantizerImpl->npuClus->GetNTotal());
        }
        if (ivfrabitqConfig.trainMode == AscendIndexIVFRaBitQConfig::TrainMode::Slice &&
            indexConfig.deviceList.size() > 1)
        {
            pQuantizerImpl->npuClus->DistributedTrainFp32(this->ivfConfig.cp.niter, tmpCentroids.data(),
                                                          indexConfig.deviceList, clearNpuData);
        }
        else
        {
            pQuantizerImpl->npuClus->TrainFp32(this->ivfConfig.cp.niter, tmpCentroids.data(), clearNpuData);
        }
        updateCoarseCenter(tmpCentroids);
        FAISS_THROW_IF_NOT(aclrtSetDevice(ivfConfig.deviceList[0]) == ACL_ERROR_NONE);
        ::ascend::AscendTensor<float, ::ascend::DIMS_2> centroidsTrained(tmpCentroids.data(), {nlist, intf_->d});
        assignIndex->addVectorsAsCentroid(centroidsTrained);
    }
    else
    {
        Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->intf_->verbose;
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer, "cpuQuantizer is not init.");
        clus.train(trainNum, trainData, *(pQuantizerImpl->cpuQuantizer));
        updateCoarseCenter(clus.centroids);
        ::ascend::AscendTensor<float, ::ascend::DIMS_2> centroidsTrained(clus.centroids.data(), {nlist, intf_->d});
        assignIndex->addVectorsAsCentroid(centroidsTrained);
    }
    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFRaBitQ train operation finished.\n");
}

void AscendIndexIVFRaBitQImpl::mergeSearchResultSingleQuery(idx_t qIdx, size_t devices,
                                                            std::vector<std::vector<float>>& dist,
                                                            std::vector<std::vector<ascend_idx_t>>& label, idx_t n,
                                                            idx_t k, size_t eachdeviceK, float* distances,
                                                            idx_t* labels,
                                                            std::function<bool(float, float)>& compFunc) const
{
    VALUE_UNUSED(n);
    idx_t num = 0;
    const idx_t offset = qIdx * k;
    const idx_t offsetDevice = qIdx * eachdeviceK;
    std::vector<int> posit(devices, 0);
    while (num < k)
    {
        size_t id = 0;
        while (posit[id] >= static_cast<int>(eachdeviceK))
        {
            id++;
        }
        float disMerged = dist[id][offsetDevice + posit[id]];
        ascend_idx_t labelMerged = label[id][offsetDevice + posit[id]];
        for (size_t j = id + 1; j < devices && posit[j] < static_cast<int>(eachdeviceK); j++)
        {
            idx_t pos = offsetDevice + posit[j];
            if (static_cast<idx_t>(label[j][pos]) != -1 && compFunc(dist[j][pos], disMerged))
            {
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

void AscendIndexIVFRaBitQImpl::mergeSearchResult(size_t devices, std::vector<std::vector<float>>& dist,
                                                 std::vector<std::vector<ascend_idx_t>>& label, idx_t n, idx_t k,
                                                 float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex mergeSearchResult operation started.\n");
    size_t eachdeviceK = dist[0].size() / n;
    FAISS_THROW_IF_NOT_FMT(devices * eachdeviceK >= static_cast<size_t>(k), "deviceNum %ld * %ld must be >= k %ld",
                           devices, eachdeviceK, k);
    std::function<bool(float, float)> compFunc = GetCompFunc();

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 100) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < n; i++)
    {
        mergeSearchResultSingleQuery(i, devices, dist, label, n, k, eachdeviceK, distances, labels, compFunc);
    }
    APP_LOG_INFO("AscendIndex mergeSearchResult operation finished.\n");
}

namespace
{

constexpr size_t kBitmapMaxBytes = 128ULL * 1024 * 1024;  // 128MB, covers up to ~1.07B IDs
constexpr size_t kParallelIdThreshold = 65536;
constexpr int kIdScanMaxThreads = 8;
// Caps Prefix Directory at 2^16 buckets to balance sparse selector lookup cost and payload size.
constexpr int kDefaultPrefixBits = 16;

bool PreferBitmap(int64_t maxId, size_t nIds, bool hasNegative)
{
    if (hasNegative || maxId < 0 || nIds == 0)
    {
        return false;
    }
    const size_t bitmapBytes = static_cast<size_t>(maxId / 8 + 1);
    const size_t sortedBytes = nIds * sizeof(int64_t);
    return bitmapBytes <= kBitmapMaxBytes && bitmapBytes <= sortedBytes;
}

bool OrBitmapBitChecked(uint8_t* bits, size_t nbytes, int64_t id);

void SetBitmapBit(uint8_t* bitmap, size_t nbytes, int64_t id) { (void)OrBitmapBitChecked(bitmap, nbytes, id); }

void InitBitmap(::ascend::RabitqIdFilterHost& out, int64_t maxId)
{
    const size_t bitmapBytes = static_cast<size_t>(maxId / 8 + 1);
    out.mode = aicpu::RABITQ_ID_FILTER_BITMAP;
    out.aux0 = static_cast<int64_t>(bitmapBytes) * 8;
    out.bitmapView = nullptr;
    out.sortedView = nullptr;
    out.viewBytes = 0;
    out.bitmap.assign(bitmapBytes, 0);
}

void SetEmptyFilter(::ascend::RabitqIdFilterHost& out)
{
    out.mode = aicpu::RABITQ_ID_FILTER_RANGE;
    out.aux0 = 0;
    out.aux1 = 0;
    out.bitmapView = nullptr;
    out.sortedView = nullptr;
    out.viewBytes = 0;
}

void AppendSortedUnique(std::vector<int64_t>& out, const idx_t* ids, size_t n)
{
    out.assign(ids, ids + n);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

size_t AlignUp(size_t value, size_t align) { return (value + align - 1) / align * align; }

int CeilLog2(uint64_t value)
{
    int bits = 0;
    uint64_t cur = 1;
    while (cur < value)
    {
        cur <<= 1;
        ++bits;
    }
    return bits;
}

void BuildPrefix(const std::vector<int64_t>& sortedIds, int requestedPrefixBits, std::vector<uint32_t>& offsets,
                 int64_t& prefixBits, int64_t& prefixShift, int64_t& prefixBucketCount)
{
    const uint64_t maxId = sortedIds.empty() ? 0 : static_cast<uint64_t>(std::max<int64_t>(0, sortedIds.back()));
    const int idBits = std::max(1, CeilLog2(maxId + 1));
    const int countBits = std::max(1, CeilLog2(static_cast<uint64_t>(sortedIds.size() + 1)));
    prefixBits = std::min<int64_t>({requestedPrefixBits, idBits, countBits});
    prefixShift = idBits - prefixBits;
    prefixBucketCount = 1LL << prefixBits;
    offsets.assign(static_cast<size_t>(prefixBucketCount) + 1, 0);

    size_t pos = 0;
    for (int64_t bucket = 0; bucket < prefixBucketCount; ++bucket)
    {
        offsets[static_cast<size_t>(bucket)] = static_cast<uint32_t>(pos);
        while (pos < sortedIds.size())
        {
            const uint64_t prefix = static_cast<uint64_t>(sortedIds[pos]) >> static_cast<uint64_t>(prefixShift);
            if (prefix != static_cast<uint64_t>(bucket))
            {
                break;
            }
            ++pos;
        }
    }
    offsets[static_cast<size_t>(prefixBucketCount)] = static_cast<uint32_t>(pos);
}

void BuildSortedPrefixPayload(::ascend::RabitqIdFilterHost& out)
{
    out.sortedPrefixPayload.clear();
    if (out.sortedIds.empty() || out.sortedIds.front() < 0)
    {
        return;
    }

    std::vector<uint32_t> prefixOffsets;
    int64_t prefixBits = 0;
    int64_t prefixShift = 0;
    int64_t prefixBucketCount = 0;
    BuildPrefix(out.sortedIds, kDefaultPrefixBits, prefixOffsets, prefixBits, prefixShift, prefixBucketCount);

    const size_t headerBytes = sizeof(aicpu::RabitqSortedPrefixPayloadHeader);
    const size_t sortedBytes = out.sortedIds.size() * sizeof(int64_t);
    const size_t prefixBytes = prefixOffsets.size() * sizeof(uint32_t);
    const size_t sortedOffset = AlignUp(headerBytes, alignof(int64_t));
    const size_t prefixOffset = AlignUp(sortedOffset + sortedBytes, alignof(uint32_t));

    out.sortedPrefixPayload.assign(prefixOffset + prefixBytes, 0);
    aicpu::RabitqSortedPrefixPayloadHeader header{};
    header.magic = aicpu::RABITQ_SORTED_PREFIX_MAGIC;
    header.sortedCount = static_cast<int64_t>(out.sortedIds.size());
    header.sortedOffsetBytes = static_cast<int64_t>(sortedOffset);
    header.prefixBits = prefixBits;
    header.prefixShift = prefixShift;
    header.prefixBucketCount = prefixBucketCount;
    header.prefixOffsetBytes = static_cast<int64_t>(prefixOffset);

    std::memcpy(out.sortedPrefixPayload.data(), &header, sizeof(header));
    std::memcpy(out.sortedPrefixPayload.data() + sortedOffset, out.sortedIds.data(), sortedBytes);
    std::memcpy(out.sortedPrefixPayload.data() + prefixOffset, prefixOffsets.data(), prefixBytes);
}

int IdScanThreadCount()
{
    return std::min(kIdScanMaxThreads, static_cast<int>(::ascend::CommonUtils::GetThreadMaxNums()));
}

void ScanIdStats(const idx_t* ids, size_t n, int64_t& maxId, bool& hasNegative)
{
    maxId = -1;
    int64_t hasNeg = 0;
#pragma omp parallel for reduction(max : maxId) reduction(max : hasNeg) if (n >= kParallelIdThreshold) \
    num_threads(IdScanThreadCount())
    for (size_t i = 0; i < n; ++i)
    {
        hasNeg = std::max(hasNeg, static_cast<int64_t>(ids[i] < 0 ? 1 : 0));
        maxId = std::max(maxId, ids[i]);
    }
    hasNegative = hasNeg != 0;
}

bool OrBitmapBitChecked(uint8_t* bits, size_t nbytes, int64_t id)
{
    if (id < 0)
    {
        return false;
    }
    const size_t byteIdx = static_cast<size_t>(id) >> 3;
    if (byteIdx >= nbytes)
    {
        return false;
    }
    bits[byteIdx] |= static_cast<uint8_t>(1u << (id & 7));
    return true;
}

void FillBitmap(std::vector<uint8_t>& bitmap, const idx_t* ids, size_t n)
{
    uint8_t* bits = bitmap.data();
    const size_t nbytes = bitmap.size();
    for (size_t i = 0; i < n; ++i)
    {
        SetBitmapBit(bits, nbytes, ids[i]);
    }
}

void SetSortedFromIds(::ascend::RabitqIdFilterHost& out, const idx_t* ids, size_t n)
{
    out.mode = aicpu::RABITQ_ID_FILTER_SORTED;
    out.sortedView = nullptr;
    out.bitmapView = nullptr;
    out.viewBytes = 0;
    AppendSortedUnique(out.sortedIds, ids, n);
    out.aux0 = static_cast<int64_t>(out.sortedIds.size());
    BuildSortedPrefixPayload(out);
}

void SetRangeFilter(::ascend::RabitqIdFilterHost& out, int64_t imin, int64_t imax)
{
    out.mode = aicpu::RABITQ_ID_FILTER_RANGE;
    out.aux0 = imin;
    out.aux1 = imax;
    out.sortedView = nullptr;
    out.bitmapView = nullptr;
    out.viewBytes = 0;
}

bool IdsFitUint32(int64_t id) { return id >= 0 && static_cast<uint64_t>(id) <= std::numeric_limits<uint32_t>::max(); }

bool TryRangeFromIds(const idx_t* ids, size_t n, ::ascend::RabitqIdFilterHost& out)
{
    if (n == 0)
    {
        SetRangeFilter(out, 0, 0);
        return true;
    }
    if (ids == nullptr || !IdsFitUint32(ids[0]) || !IdsFitUint32(ids[n - 1]))
    {
        return false;
    }
    if (!IdsFormArithmeticProgression(ids, n, 1))
    {
        return false;
    }
    SetRangeFilter(out, ids[0], ids[0] + static_cast<int64_t>(n));
    return true;
}

void MaterializeFromIdsTwoPass(const idx_t* ids, size_t n, ::ascend::RabitqIdFilterHost& out)
{
    int64_t maxId = -1;
    bool hasNegative = false;
    ScanIdStats(ids, n, maxId, hasNegative);
    if (PreferBitmap(maxId, n, hasNegative))
    {
        InitBitmap(out, maxId);
        FillBitmap(out.bitmap, ids, n);
        return;
    }
    SetSortedFromIds(out, ids, n);
}

void MaterializeFromIdSetTwoPass(const std::unordered_set<idx_t>& ids, ::ascend::RabitqIdFilterHost& out)
{
    int64_t maxId = -1;
    bool hasNegative = false;
    for (idx_t id : ids)
    {
        hasNegative = hasNegative || (id < 0);
        maxId = std::max(maxId, static_cast<int64_t>(id));
    }
    if (PreferBitmap(maxId, ids.size(), hasNegative))
    {
        InitBitmap(out, maxId);
        uint8_t* bits = out.bitmap.data();
        const size_t nbytes = out.bitmap.size();
        for (idx_t id : ids)
        {
            SetBitmapBit(bits, nbytes, id);
        }
        return;
    }
    std::vector<idx_t> tmp(ids.begin(), ids.end());
    SetSortedFromIds(out, tmp.data(), tmp.size());
}

void MaterializeFromIds(const idx_t* ids, size_t n, ::ascend::RabitqIdFilterHost& out)
{
    if (TryRangeFromIds(ids, n, out))
    {
        return;
    }
    if (FreezeRoaringIdxs(ids, n, out))
    {
        return;
    }
    out.resetKeepCapacity();
    MaterializeFromIdsTwoPass(ids, n, out);
}

void MaterializeFromIdSet(const std::unordered_set<idx_t>& ids, ::ascend::RabitqIdFilterHost& out)
{
    if (ids.empty())
    {
        SetEmptyFilter(out);
        return;
    }
    int64_t minId = std::numeric_limits<int64_t>::max();
    int64_t maxId = std::numeric_limits<int64_t>::min();
    bool fitUint32 = true;
    for (idx_t id : ids)
    {
        fitUint32 = fitUint32 && IdsFitUint32(id);
        minId = std::min(minId, static_cast<int64_t>(id));
        maxId = std::max(maxId, static_cast<int64_t>(id));
    }
    if (fitUint32 && static_cast<uint64_t>(maxId) - static_cast<uint64_t>(minId) + 1ULL == ids.size())
    {
        SetRangeFilter(out, minId, maxId + 1);
        return;
    }
    if (fitUint32)
    {
        std::vector<uint32_t> vals;
        vals.reserve(ids.size());
        for (idx_t id : ids)
        {
            vals.push_back(static_cast<uint32_t>(id));
        }
        std::sort(vals.begin(), vals.end());
        if (FreezeRoaringUint32Sorted(vals.data(), vals.size(), out))
        {
            return;
        }
    }
    out.resetKeepCapacity();
    MaterializeFromIdSetTwoPass(ids, out);
}

void MaterializeFromRoaring(const IDSelectorRoaring* roar, ::ascend::RabitqIdFilterHost& out)
{
    if (roar->live != nullptr)
    {
        FAISS_THROW_IF_NOT_MSG(FreezeRoaringLive(roar->live, out), "failed to freeze IDSelectorRoaring live bitmap");
        return;
    }
    AssertValidFrozenRoaring(roar->frozen, roar->n);
    out.mode = aicpu::RABITQ_ID_FILTER_ROARING;
    out.aux0 = static_cast<int64_t>(roar->n);
    out.bitmapView = roar->frozen;
    out.viewBytes = roar->n;
    out.sortedView = nullptr;
}

void MaterializeIdSelector(const IDSelector* sel, ::ascend::RabitqIdFilterHost& out)
{
    FAISS_THROW_IF_NOT_MSG(sel != nullptr, "IDSelector cannot be nullptr");

    out.resetKeepCapacity();
    bool negate = false;
    const IDSelector* inner = sel;
    if (const auto* notSel = dynamic_cast<const IDSelectorNot*>(sel))
    {
        negate = true;
        inner = notSel->sel;
        FAISS_THROW_IF_NOT_MSG(inner != nullptr, "IDSelectorNot.sel cannot be nullptr");
    }

    if (const auto* rangeSel = dynamic_cast<const IDSelectorRange*>(inner))
    {
        FAISS_THROW_IF_NOT_MSG(rangeSel->imin <= rangeSel->imax, "IDSelectorRange.imin must be <= imax");
        out.mode = aicpu::RABITQ_ID_FILTER_RANGE;
        out.aux0 = rangeSel->imin;
        out.aux1 = rangeSel->imax;
    }
    else if (const auto* batchSel = dynamic_cast<const IDSelectorBatch*>(inner))
    {
        MaterializeFromIdSet(batchSel->set, out);
    }
    else if (const auto* arraySel = dynamic_cast<const IDSelectorArray*>(inner))
    {
        FAISS_THROW_IF_NOT_MSG(arraySel->n == 0 || arraySel->ids != nullptr,
                               "IDSelectorArray.ids cannot be nullptr when n > 0");
        MaterializeFromIds(arraySel->ids, arraySel->n, out);
    }
    else if (const auto* bitmapSel = dynamic_cast<const IDSelectorBitmap*>(inner))
    {
        FAISS_THROW_IF_NOT_MSG(bitmapSel->n == 0 || bitmapSel->bitmap != nullptr,
                               "IDSelectorBitmap.bitmap cannot be nullptr when n > 0");
        // FAISS: n is bitmap byte length; id selected iff id/8 < n.
        out.mode = aicpu::RABITQ_ID_FILTER_BITMAP;
        out.aux0 = static_cast<int64_t>(bitmapSel->n) * 8;
        out.bitmapView = bitmapSel->bitmap;
        out.viewBytes = bitmapSel->n;
    }
    else if (const auto* roarSel = dynamic_cast<const IDSelectorRoaring*>(inner))
    {
        MaterializeFromRoaring(roarSel, out);
    }
    else
    {
        FAISS_THROW_MSG(
            "AscendIndexIVFRaBitQ search IDSelector only supports Range/Batch/Array/Bitmap/Roaring and "
            "IDSelectorNot of them");
    }

    out.negate = negate ? 1 : 0;
}

}  // namespace

void AscendIndexIVFRaBitQImpl::indexSearch(IndexParam<float, float, ascend_idx_t>& param) const
{
    indexSearch(param, nullptr);
}

void AscendIndexIVFRaBitQImpl::indexSearch(IndexParam<float, float, ascend_idx_t>& param,
                                           const ::ascend::RabitqIdFilterHost* idFilter, int searchNprobe) const
{
    const float* indexPtr = nullptr;
    if (ivfrabitqConfig.needRefine)
    {
        indexPtr = getRefineSources();
    }
    auto pIndex = getActualIndex(param.deviceId);
    auto ret = pIndex->searchImpl(param.n, param.query, param.k, param.distance, param.label, indexPtr, idFilter,
                                  searchNprobe);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to search index,deviceId is %d, result is: %d\n",
                           param.deviceId, ret);
}

void AscendIndexIVFRaBitQImpl::searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    searchImplFiltered(n, x, k, distances, labels, nullptr);
}

void AscendIndexIVFRaBitQImpl::searchImplFiltered(int n, const float* x, int k, float* distances, idx_t* labels,
                                                  const ::ascend::RabitqIdFilterHost* idFilter, int searchNprobe) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d, searchNprobe=%d.\n", n, k, searchNprobe);
    size_t deviceCnt = indexConfig.deviceList.size();
    int finalk = k;
    if (ivfrabitqConfig.needRefine)
    {
        k = finalk * ivfrabitqConfig.refineAlpha;
    }
    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        IndexParam<float, float, ascend_idx_t> param(deviceId, n, this->intf_->d, k);
        param.query = x;
        param.distance = dist[idx].data();
        param.label = label[idx].data();
        indexSearch(param, idFilter, searchNprobe);
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, readPool, searchFunctor);
    searchPostProcess(deviceCnt, dist, label, n, finalk, distances, labels);
}

const ::ascend::RabitqIdFilterHost* AscendIndexIVFRaBitQImpl::getCachedFilter(const IDSelector* sel) const
{
    if (sel == nullptr)
    {
        return nullptr;
    }
    // Caller (searchWithSelector) must hold filterCacheMutex for the rest of the search.
    const FilterCacheKey key = MakeFilterCacheKey(sel);
    const int hit = FindFilterCacheHit(key);
    if (hit >= 0)
    {
        filterSlotLru = hit;
        return &filterSlots[hit].filter;
    }
    const int victim = AllocFilterCacheSlot();
    MaterializeIdSelector(sel, filterSlots[victim].filter);
    ++cachedFilterGeneration;
    filterSlots[victim].filter.generation = cachedFilterGeneration;
    filterSlots[victim].key = key;
    filterSlotLru = victim;
    return &filterSlots[victim].filter;
}

AscendIndexIVFRaBitQImpl::FilterCacheKey AscendIndexIVFRaBitQImpl::MakeFilterCacheKey(const IDSelector* sel)
{
    FilterCacheKey key;
    key.selPtr = sel;
    const IDSelector* inner = sel;
    if (const auto* notSel = dynamic_cast<const IDSelectorNot*>(sel))
    {
        key.negate = 1;
        inner = notSel->sel;
        FAISS_THROW_IF_NOT_MSG(inner != nullptr, "IDSelectorNot.sel cannot be nullptr");
    }
    if (const auto* arraySel = dynamic_cast<const IDSelectorArray*>(inner))
    {
        key.kind = FilterCacheKey::Kind::Array;
        key.payload = arraySel->ids;
        key.n = arraySel->n;
        return key;
    }
    if (const auto* bitmapSel = dynamic_cast<const IDSelectorBitmap*>(inner))
    {
        key.kind = FilterCacheKey::Kind::Bitmap;
        key.payload = bitmapSel->bitmap;
        key.n = bitmapSel->n;
        return key;
    }
    if (const auto* roarSel = dynamic_cast<const IDSelectorRoaring*>(inner))
    {
        key.kind = FilterCacheKey::Kind::Roaring;
        if (roarSel->live != nullptr)
        {
            key.payload = roarSel->live;
            key.n = 0;
        }
        else
        {
            key.payload = roarSel->frozen;
            key.n = roarSel->n;
        }
        return key;
    }
    key.kind = FilterCacheKey::Kind::Object;
    if (const auto* rangeSel = dynamic_cast<const IDSelectorRange*>(inner))
    {
        key.contentHash = static_cast<uint64_t>(rangeSel->imin);
        key.n = static_cast<size_t>(rangeSel->imax);
    }
    return key;
}

bool AscendIndexIVFRaBitQImpl::IsFilterCacheHit(const FilterCacheKey& cached, const FilterCacheKey& key)
{
    if (cached.kind != key.kind || cached.negate != key.negate || cached.contentHash != key.contentHash)
    {
        return false;
    }
    if (key.kind == FilterCacheKey::Kind::Object)
    {
        return cached.selPtr == key.selPtr && cached.n == key.n;
    }
    return cached.payload == key.payload && cached.n == key.n;
}

bool AscendIndexIVFRaBitQImpl::IsFilterSlotEmpty(const FilterCacheKey& key)
{
    return key.selPtr == nullptr && key.payload == nullptr;
}

int AscendIndexIVFRaBitQImpl::FindFilterCacheHit(const FilterCacheKey& key) const
{
    for (int i = 0; i < kFilterCacheSlots; ++i)
    {
        if (IsFilterCacheHit(filterSlots[i].key, key))
        {
            return i;
        }
    }
    return -1;
}

int AscendIndexIVFRaBitQImpl::AllocFilterCacheSlot() const
{
    for (int i = 0; i < kFilterCacheSlots; ++i)
    {
        if (IsFilterSlotEmpty(filterSlots[i].key))
        {
            return i;
        }
    }
    return 1 - filterSlotLru;
}

void AscendIndexIVFRaBitQImpl::InvalidateFilterCache() const
{
    std::lock_guard<std::mutex> guard(filterCacheMutex);
    ++cachedFilterGeneration;
    for (int i = 0; i < kFilterCacheSlots; ++i)
    {
        filterSlots[i].key = FilterCacheKey{};
        filterSlots[i].filter.resetKeepCapacity();
        filterSlots[i].filter.generation = cachedFilterGeneration;
    }
    filterSlotLru = 0;
}

void AscendIndexIVFRaBitQImpl::invalidateFilterCache() const { InvalidateFilterCache(); }

void AscendIndexIVFRaBitQImpl::searchWithSelector(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
                                                  const IDSelector* sel, int searchNprobe) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVFRaBitQ searchWithSelector started: n=%ld, k=%ld, hasSel=%d, searchNprobe=%d.\n", n, k,
                 sel != nullptr ? 1 : 0, searchNprobe);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");

    std::unique_lock<std::mutex> cacheLock(filterCacheMutex, std::defer_lock);
    if (sel != nullptr)
    {
        cacheLock.lock();
    }
    const ::ascend::RabitqIdFilterHost* filterPtr = getCachedFilter(sel);

    const size_t tileSize = getSearchPagedSize(static_cast<int>(n), static_cast<int>(k));
    for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize)
    {
        const size_t curNum = std::min(tileSize, static_cast<size_t>(n) - i);
        searchImplFiltered(static_cast<int>(curNum), x + i * static_cast<size_t>(this->intf_->d), static_cast<int>(k),
                           distances + i * static_cast<size_t>(k), labels + i * static_cast<size_t>(k), filterPtr,
                           searchNprobe);
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQ searchWithSelector finished.\n");
}

void AscendIndexIVFRaBitQImpl::deleteConcurrent(int n, const idx_t* ids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    std::lock_guard<std::mutex> writerLock(writerMutex);
    deleteImpl(n, ids);
}

void AscendIndexIVFRaBitQImpl::deleteImpl(int n, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(attachedSrc_ == nullptr, "delete conflicts with attached refine sources; reset first.");
    FAISS_THROW_IF_NOT_MSG(!copiedSources_ && srcIndexes.empty(),
                           "delete conflicts with copied refine sources; reset first.");
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl deleteImpl operation started: n=%d.\n", n);
    const size_t deviceCnt = indexConfig.deviceList.size();
    std::unordered_map<int, std::vector<idx_t>> idMap;
    std::unordered_set<idx_t> seen;
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        for (int i = 0; i < n; ++i)
        {
            const idx_t id = ids[i];
            if (!seen.insert(id).second)
            {
                continue;
            }
            auto it = idToDeviceMap.find(id);
            if (it == idToDeviceMap.end())
            {
                continue;  // Missing IDs are not deletions; do not use findDeviceId's legacy fallback.
            }
            const idx_t devIdx = it->second;
            if (devIdx >= 0 && devIdx < static_cast<idx_t>(deviceCnt))
            {
                idMap[static_cast<int>(devIdx)].push_back(id);
            }
        }
    }

    int deleted = 0;
    std::exception_ptr failure;
    try
    {
        for (auto& device : idMap)
        {
            auto& deleteIds = device.second;
            std::vector<ascend_idx_t> ascendIds(deleteIds.begin(), deleteIds.end());
            IndexParam<void, void, ascend_idx_t> param(indexConfig.deviceList[device.first], deleteIds.size(), 0, 0);
            param.label = ascendIds.data();
            deleteFromIVFRaBitQ(param);
            removeIdMapping(deleteIds);
            this->intf_->ntotal -= static_cast<idx_t>(deleteIds.size());
            deleted += static_cast<int>(deleteIds.size());
            auto pIndex = getActualIndex(indexConfig.deviceList[device.first]);
            for (int listId = 0; listId < nlist; ++listId)
            {
                deviceAddNumMap[listId][device.first] = static_cast<int>(pIndex->getListLength(listId));
            }
        }
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    if (deleted > 0)
    {
        InvalidateFilterCache();
    }
    if (failure)
    {
        std::rethrow_exception(failure);
    }
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl deleteImpl operation finished.\n");
}

void AscendIndexIVFRaBitQImpl::deleteFromIVFRaBitQ(IndexParam<void, void, ascend_idx_t>& param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;

    const ascend_idx_t* ids = param.label;

    auto ret = pIndex->removeIds(param.n, static_cast<const ::ascend::Index::idx_t*>(ids));

    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to delete from ivf RaBitQ, ret: %d", ret);
}

idx_t AscendIndexIVFRaBitQImpl::findDeviceId(idx_t id)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    auto it = idToDeviceMap.find(id);
    if (it != idToDeviceMap.end())
    {
        return it->second;
    }

    APP_LOG_WARNING("ID %ld not found in device mapping, attempting to find in device data\n", id);

    size_t deviceCnt = indexConfig.deviceList.size();
    idx_t fallbackDeviceId = id % deviceCnt;
    APP_LOG_WARNING("ID %ld not found, using fallback deviceId: %ld\n", id, fallbackDeviceId);
    return fallbackDeviceId;
}

void AscendIndexIVFRaBitQImpl::updateIdMapping(const ascend_idx_t* ids, int deviceId, int num)
{
    std::lock_guard<std::mutex> lock(mapMutex);
    for (int i = 0; i < num; i++)
    {
        idx_t id = ids[i];

        idToDeviceMap[id] = deviceId;

        if (static_cast<idx_t>(deviceInfos.size()) <= deviceId)
        {
            deviceInfos.resize(deviceId + 1);
        }
        deviceInfos[deviceId].idSet.insert(id);
    }
    APP_LOG_DEBUG("Updated batch mapping for %zu IDs\n", num);
}

void AscendIndexIVFRaBitQImpl::removeIdMapping(const std::vector<idx_t>& ids)
{
    std::lock_guard<std::mutex> lock(mapMutex);

    for (idx_t id : ids)
    {
        auto it = idToDeviceMap.find(id);
        if (it != idToDeviceMap.end())
        {
            idx_t devId = it->second;

            if (devId < static_cast<idx_t>(deviceInfos.size()))
            {
                deviceInfos[devId].idSet.erase(id);
            }
            idToDeviceMap.erase(it);
        }
    }
    APP_LOG_DEBUG("Removed batch mapping for %zu IDs\n", ids.size());
}

void AscendIndexIVFRaBitQImpl::threadUnsafeReset()
{
    const size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<::ascend::IndexIVFRaBitQ*> devices(deviceCnt);
    std::vector<idx_t> deviceTotals(deviceCnt);
    // Preflight every reader before resetting any device. SnapshotGuard is the public drain/recovery
    // boundary; the host's exclusive lock prevents new searches or mutations between these steps.
    for (size_t d = 0; d < deviceCnt; ++d)
    {
        devices[d] = getActualIndex(indexConfig.deviceList[d]);
        ::ascend::IndexIVFRaBitQ::SnapshotGuard guard(*devices[d]);
        deviceTotals[d] = static_cast<idx_t>(guard.get().total);
        const auto ret = guard.finish();
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "reader stream drain failed before reset, device=%d, ret=%d",
                               indexConfig.deviceList[d], ret);
    }
    for (size_t d = 0; d < deviceCnt; ++d)
    {
        FAISS_THROW_IF_NOT(aclrtSetDevice(indexConfig.deviceList[d]) == ACL_ERROR_NONE);
        const auto ret = devices[d]->reset();
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to reset index, device id: %d, result is %d",
                               indexConfig.deviceList[d], ret);
        // Writer drain/reset may still fail after another device succeeded. Commit each successful
        // device immediately, so throwing never leaves host metadata claiming its deleted rows exist.
        {
            std::lock_guard<std::mutex> lock(mapMutex);
            for (auto it = idToDeviceMap.begin(); it != idToDeviceMap.end();)
            {
                if (it->second == static_cast<idx_t>(d))
                {
                    it = idToDeviceMap.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            if (d < deviceInfos.size())
            {
                deviceInfos[d].idSet.clear();
            }
            for (auto& list : deviceAddNumMap)
            {
                list[d] = 0;
            }
        }
        intf_->ntotal -= deviceTotals[d];
        if (deviceTotals[d] != 0 &&
            (ivfrabitqConfig.needRefine || copiedSources_ || attachedSrc_ != nullptr || !srcIndexes.empty()))
        {
            refineSourcesReady_ = false;
        }
        InvalidateFilterCache();
    }
    intf_->ntotal = 0;
    attachedSrc_ = nullptr;
    attachedSrcRows_ = 0;
    srcIndexes.clear();
    copiedSources_ = false;
    refineSourcesReady_ = true;
}

size_t AscendIndexIVFRaBitQImpl::getAddElementSize() const { return static_cast<size_t>(intf_->d) * sizeof(float); }

void AscendIndexIVFRaBitQImpl::loadRefineSources(const float* x, idx_t n)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    FAISS_THROW_IF_NOT_MSG(attachedSrc_ == nullptr, "loadRefineSources conflicts with attached refine sources.");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "refine source x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(n == this->intf_->ntotal, "refine source n (%ld) must equal ntotal (%ld)", n,
                           this->intf_->ntotal);
    checkRefineIds();
    std::vector<float> sources(x, x + static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d));
    srcIndexes.swap(sources);
    copiedSources_ = true;
    refineSourcesReady_ = true;
}

void AscendIndexIVFRaBitQImpl::attachRefineSources(const float* x, idx_t n)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "refine source x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(n == this->intf_->ntotal, "refine source n (%ld) must equal ntotal (%ld)", n,
                           this->intf_->ntotal);
    FAISS_THROW_IF_NOT_MSG(!copiedSources_ && srcIndexes.empty(),
                           "attachRefineSources conflicts with copied refine sources; "
                           "use loadRefineSources or attachRefineSources, not both.");
    checkRefineIds();
    attachedSrc_ = x;  // 零拷贝：仅保存指针，生命周期由调用方保证
    attachedSrcRows_ = n;
    refineSourcesReady_ = true;
}

void AscendIndexIVFRaBitQImpl::checkRefineIds()
{
    std::lock_guard<std::mutex> lock(mapMutex);
    FAISS_THROW_IF_NOT_MSG(idToDeviceMap.size() == static_cast<size_t>(intf_->ntotal),
                           "refine requires unique IDs covering [0, ntotal).");
    for (const auto& entry : idToDeviceMap)
    {
        FAISS_THROW_IF_NOT_MSG(entry.first >= 0 && entry.first < intf_->ntotal,
                               "refine requires unique IDs covering [0, ntotal).");
    }
}

const float* AscendIndexIVFRaBitQImpl::getRefineSources() const
{
    FAISS_THROW_IF_NOT_MSG(refineSourcesReady_, "refine sources are not ready: reload sources or reset before search.");
    if (attachedSrc_ != nullptr)
    {
        FAISS_THROW_IF_NOT_MSG(attachedSrcRows_ == intf_->ntotal,
                               "attached refine source row count must match ntotal.");
        return attachedSrc_;
    }
    FAISS_THROW_IF_NOT_MSG(refineSourcesReady_ && srcIndexes.size() == static_cast<size_t>(intf_->ntotal) * intf_->d,
                           "refine sources are not ready: loadRefineSources or attachRefineSources before search.");
    return srcIndexes.data();
}

// copy from cpu index
void AscendIndexIVFRaBitQImpl::copyFrom(const faiss::IndexIVFRaBitQ* index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl copyFrom operation started.");

    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Input index is nullptr");
    FAISS_THROW_IF_NOT_MSG(this->intf_ != nullptr, "Internal interface is nullptr");
    FAISS_THROW_IF_NOT_MSG(index->d == this->intf_->d, "Dimension mismatch");
    FAISS_THROW_IF_NOT_MSG(index->nlist == static_cast<size_t>(nlist), "nlist mismatch");
    FAISS_THROW_IF_NOT_MSG(indexConfig.deviceList.size() > 0, "Device list is empty");

    const size_t codePartBytesPerVec = getAscendRaBitQCodePartSize(intf_->d, ivfrabitqConfig.quantBits);
    const size_t payloadSize = getAscendRaBitQPayloadSize(intf_->d, ivfrabitqConfig.quantBits);
    const size_t expectedCodeSize = getFaissRaBitQCarrierSize(intf_->d, ivfrabitqConfig.quantBits);
    FAISS_THROW_IF_NOT_FMT(static_cast<int>(index->rabitq.nb_bits) == ivfrabitqConfig.quantBits,
                           "RaBitQ quantBits mismatch: file has %zu bits, but the Ascend index requests %d",
                           static_cast<size_t>(index->rabitq.nb_bits), ivfrabitqConfig.quantBits);
    FAISS_THROW_IF_NOT_FMT(index->rabitq.compute_code_size(intf_->d, index->rabitq.nb_bits) == expectedCodeSize,
                           "Unsupported Faiss RaBitQ carrier layout for quantBits=%d and dim=%d",
                           ivfrabitqConfig.quantBits, intf_->d);
    FAISS_THROW_IF_NOT_FMT(
        index->code_size == expectedCodeSize,
        "RaBitQ code size mismatch: file has %zu bytes/vector, but quantBits=%d and dim=%d require %zu. "
        "Load the index with the same quantBits used when it was saved.",
        index->code_size, ivfrabitqConfig.quantBits, intf_->d, expectedCodeSize);
    FAISS_THROW_IF_NOT_MSG(index->invlists != nullptr, "Input index has no inverted lists");
    FAISS_THROW_IF_NOT_FMT(index->invlists->code_size == expectedCodeSize,
                           "Inverted-list code size mismatch: got %zu, expected %zu", index->invlists->code_size,
                           expectedCodeSize);
    if (ivfrabitqConfig.quantBits > 1)
    {
        bool foundCode = false;
        bool validCarrier = false;
        for (int listId = 0; listId < nlist; ++listId)
        {
            if (index->invlists->list_size(listId) == 0) continue;
            const uint8_t* firstCode = index->invlists->get_codes(listId);
            validCarrier = isAscendMultiBitCarrier(firstCode + payloadSize);
            index->invlists->release_codes(listId, firstCode);
            foundCode = true;
            break;
        }
        FAISS_THROW_IF_NOT_MSG(!foundCode || validCarrier,
                               "Input is a native Faiss multi-bit RaBitQ index, not an Ascend carrier");
    }

    FAISS_THROW_IF_NOT_MSG(index->quantizer != nullptr && index->is_trained,
                           "Index must be trained and have quantizer");
    auto flat = dynamic_cast<faiss::IndexFlat*>(index->quantizer);
    FAISS_THROW_IF_NOT_MSG(flat != nullptr, "Quantizer must be IndexFlat");
    FAISS_THROW_IF_NOT_MSG(
        flat->ntotal == nlist && flat->codes.size() / sizeof(float) == static_cast<size_t>(nlist) * intf_->d,
        "Quantizer must contain nlist complete centroids");
    FAISS_THROW_IF_NOT_MSG(index->metric_type == intf_->metric_type, "Invalid metric_type");
    FAISS_THROW_IF_NOT_FMT(index->nprobe > 0 && index->nprobe <= index->nlist, "Invalid number of nprobe %zu",
                           index->nprobe);
    FAISS_THROW_IF_NOT_FMT(index->ntotal >= 0 && index->ntotal < MAX_N, "Ntotal must be >= 0 and < %ld", MAX_N);
    threadUnsafeReset();  // Complete reset/reconciliation before installing any replacement metadata.
    intf_->is_trained = false;
    try
    {
        AscendIndexIVFImpl::copyFrom(index);
    }
    catch (...)
    {
        intf_->ntotal = 0;  // Base metadata import can fail after setting the incoming total.
        intf_->is_trained = false;
        throw;
    }
    intf_->ntotal = 0;          // Only committed encoded publications below contribute to the loaded count.
    intf_->is_trained = false;  // Base import does not establish the RaBitQ device setup.

    // Orthogonal matrix: FAISS has no corresponding storage interface,
    // regenerate with the same seed configuration
    orthogonalMatrix.resize(intf_->d * intf_->d);
    if (ivfrabitqConfig.useRandomOrthogonalMatrix)
    {
        try
        {
            randomOrthogonalGivens(intf_->d, orthogonalMatrix);
        }
        catch (const std::exception& e)
        {
            FAISS_THROW_IF_NOT_FMT(false, "Failed to generate random orthogonal matrix: %s", e.what());
        }
    }
    else
    {
        std::fill(orthogonalMatrix.begin(), orthogonalMatrix.end(), 0.0f);
        for (int i = 0; i < intf_->d; ++i)
        {
            orthogonalMatrix[i * intf_->d + i] = 1.0f;
        }
    }
    uploadorthogonalMatrix(orthogonalMatrix);

    // copy centroids
    const float* centroids = flat->get_xb();
    std::vector<float> hostCentroids(centroids, centroids + nlist * intf_->d);

    updateCoarseCenter(hostCentroids);
    size_t dim = intf_->d;
    ::ascend::AscendTensor<float, ::ascend::DIMS_2> centroidsTrained(hostCentroids.data(),
                                                                     {nlist, static_cast<int>(dim)});
    const auto ret = assignIndex->addVectorsAsCentroid(centroidsTrained);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to add assignment centroids, ret=%d", ret);
    intf_->is_trained = true;  // Fully configured even if a later encoded publication fails.

    // copy inverted lists
    size_t deviceCnt = indexConfig.deviceList.size();
    size_t codeSize = index->code_size;

    for (int listId = 0; listId < nlist; ++listId)
    {
        size_t listLen = index->invlists->list_size(listId);
        if (listLen == 0) continue;

        const faiss::idx_t* ids = index->invlists->get_ids(listId);
        const uint8_t* codes = index->invlists->get_codes(listId);

        // ArrayInvertedLists storage format: [vec0: code+L2+L1][vec1: code+L2+L1]...
        size_t factorBytesPerVec = sizeof(float);  // Factor bytes per vector

        // Create temporary buffers for each device to store encoded data and IDs
        std::vector<std::vector<uint8_t>> perDeviceCodes(deviceCnt);
        std::vector<std::vector<faiss::idx_t>> perDeviceIds(deviceCnt);
        // Record vector indices assigned to each device
        std::vector<std::vector<size_t>> perDeviceVecIndices(deviceCnt);
        auto occupancy = deviceAddNumMap[listId];

        for (size_t j = 0; j < listLen; ++j)
        {
            // Select device based on load balancing
            size_t devIdx = 0;
            for (size_t d = 1; d < deviceCnt; ++d)
            {
                if (occupancy[d] < occupancy[devIdx])
                {
                    devIdx = d;
                }
            }
            ++occupancy[devIdx];
            perDeviceVecIndices[devIdx].push_back(j);
            perDeviceIds[devIdx].push_back(ids[j]);
        }

        // Build per-device encoded data in block format: [all codes][all L2][all L1]
        for (size_t devIdx = 0; devIdx < deviceCnt; ++devIdx)
        {
            if (perDeviceVecIndices[devIdx].empty()) continue;

            size_t numVecs = perDeviceVecIndices[devIdx].size();
            size_t totalCodeBytes = numVecs * codePartBytesPerVec;
            size_t totalFactorBytes = numVecs * factorBytesPerVec;
            perDeviceCodes[devIdx].resize(totalCodeBytes + 2 * totalFactorBytes);

            // Extract data from per-vector format
            uint8_t* dstCodesPtr = perDeviceCodes[devIdx].data();
            uint8_t* dstL2Ptr = dstCodesPtr + totalCodeBytes;
            uint8_t* dstL1Ptr = dstL2Ptr + totalFactorBytes;

            for (size_t k = 0; k < numVecs; ++k)
            {
                size_t srcIdx = perDeviceVecIndices[devIdx][k];
                const uint8_t* vecPtr = codes + srcIdx * codeSize;
                // Extract compressed code
                auto ret =
                    memcpy_s(dstCodesPtr + k * codePartBytesPerVec, codePartBytesPerVec, vecPtr, codePartBytesPerVec);
                FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for codes, ret=%d", ret);
                // Extract L2 factor
                ret = memcpy_s(dstL2Ptr + k * factorBytesPerVec, factorBytesPerVec, vecPtr + codePartBytesPerVec,
                               factorBytesPerVec);
                FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for L2, ret=%d", ret);
                // Extract L1 factor
                ret = memcpy_s(dstL1Ptr + k * factorBytesPerVec, factorBytesPerVec,
                               vecPtr + codePartBytesPerVec + factorBytesPerVec, factorBytesPerVec);
                FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for L1, ret=%d", ret);
            }
        }

        for (size_t devIdx = 0; devIdx < deviceCnt; ++devIdx)
        {
            if (perDeviceCodes[devIdx].empty()) continue;

            int deviceId = indexConfig.deviceList[devIdx];
            auto pIndex = getActualIndex(deviceId);
            size_t numVecs = perDeviceIds[devIdx].size();
            std::unordered_map<idx_t, idx_t> pendingMappings;
            std::unordered_set<idx_t> pendingIds(perDeviceIds[devIdx].begin(), perDeviceIds[devIdx].end());
            for (idx_t id : pendingIds)
            {
                pendingMappings.emplace(id, devIdx);
            }
            {
                std::lock_guard<std::mutex> lock(mapMutex);
                deviceInfos.resize(deviceCnt);
                idToDeviceMap.reserve(idToDeviceMap.size() + pendingMappings.size());
                deviceInfos[devIdx].idSet.reserve(deviceInfos[devIdx].idSet.size() + pendingIds.size());
            }

            auto ret = pIndex->addEncodedVectors(listId, numVecs, perDeviceCodes[devIdx].data(),
                                                 reinterpret_cast<const uint64_t*>(perDeviceIds[devIdx].data()));

            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                                   "Failed to add encoded vectors to device %d, list %d, ret=%d", deviceId, listId,
                                   ret);
            {
                std::lock_guard<std::mutex> lock(mapMutex);
                idToDeviceMap.merge(pendingMappings);
                deviceInfos[devIdx].idSet.merge(pendingIds);
                deviceAddNumMap[listId][devIdx] = occupancy[devIdx];
            }
            this->intf_->ntotal += static_cast<idx_t>(numVecs);
            refineSourcesReady_ = false;  // Encoded imports have no originals, including partial loads.
            InvalidateFilterCache();
        }
        index->invlists->release_ids(listId, ids);
        index->invlists->release_codes(listId, codes);
    }

    FAISS_THROW_IF_NOT_MSG(this->intf_->ntotal == index->ntotal, "Imported vector count does not match ntotal");
    this->intf_->is_trained = index->is_trained;
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl copyFrom operation finished.");
}

// copy to cpu index
void AscendIndexIVFRaBitQImpl::copyTo(faiss::IndexIVFRaBitQ* index) const
{
    APP_LOG_INFO("AscendIndexIVFRaBitQImpl copyTo operation started.");

    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Output index is nullptr");
    FAISS_THROW_IF_NOT_MSG(indexConfig.deviceList.size() > 0, "Device list is empty");

    index->d = intf_->d;
    index->metric_type = intf_->metric_type;
    index->nlist = nlist;
    index->nprobe = nprobe;
    index->is_trained = intf_->is_trained;
    index->ntotal = intf_->ntotal;

    auto firstDevIdx = getActualIndex(indexConfig.deviceList[0]);
    size_t devCodeSize = firstDevIdx->getCodeSize();
    const size_t codePartBytesPerVec = getAscendRaBitQCodePartSize(intf_->d, ivfrabitqConfig.quantBits);
    const size_t payloadSize = getAscendRaBitQPayloadSize(intf_->d, ivfrabitqConfig.quantBits);
    const size_t carrierCodeSize = getFaissRaBitQCarrierSize(intf_->d, ivfrabitqConfig.quantBits);
    FAISS_THROW_IF_NOT_FMT(devCodeSize == payloadSize,
                           "Device RaBitQ code size mismatch: got %zu bytes/vector, expected %zu for quantBits=%d",
                           devCodeSize, payloadSize, ivfrabitqConfig.quantBits);
    index->code_size = carrierCodeSize;

    index->rabitq.d = intf_->d;
    index->rabitq.metric_type = intf_->metric_type;
    index->rabitq.nb_bits = static_cast<size_t>(ivfrabitqConfig.quantBits);
    index->rabitq.code_size = carrierCodeSize;
    FAISS_THROW_IF_NOT_FMT(index->rabitq.compute_code_size(intf_->d, index->rabitq.nb_bits) == carrierCodeSize,
                           "Unsupported Faiss RaBitQ carrier layout for quantBits=%d and dim=%d",
                           ivfrabitqConfig.quantBits, intf_->d);

    // copy centroids
    std::vector<float> hostCentroids;
    auto ret = firstDevIdx->getCentroids(hostCentroids);
    FAISS_THROW_IF_NOT_MSG(ret == ::ascend::APP_ERR_OK, "Failed to get centroids from device");
    FAISS_THROW_IF_NOT_MSG(hostCentroids.size() == static_cast<size_t>(nlist) * static_cast<size_t>(intf_->d),
                           "Centroid size mismatch");

    faiss::IndexFlat* quantizer = nullptr;
    if (index->metric_type == faiss::METRIC_INNER_PRODUCT)
    {
        quantizer = new faiss::IndexFlatIP(intf_->d);
    }
    else
    {
        quantizer = new faiss::IndexFlatL2(intf_->d);
    }
    quantizer->add(nlist, hostCentroids.data());
    if (index->own_fields)
    {
        delete index->quantizer;
    }
    index->quantizer = quantizer;
    index->quantizer_trains_alone = 0;
    index->own_fields = true;
    index->direct_map.clear();

    // copy inverted lists
    auto newInvlists = std::make_unique<faiss::ArrayInvertedLists>(nlist, carrierCodeSize);
    size_t deviceCnt = indexConfig.deviceList.size();
    int totalCopied = 0;

    for (int listId = 0; listId < nlist; ++listId)
    {
        // calculate the total length of this list on all devices
        size_t totalLen = 0;
        for (size_t devIdx = 0; devIdx < deviceCnt; ++devIdx)
        {
            auto pIndex = getActualIndex(indexConfig.deviceList[devIdx]);
            totalLen += pIndex->getListLength(listId);
        }
        if (totalLen == 0) continue;

        // getListRawCodes return format: [all codes][all L2][all L1]
        size_t factorBytesPerVec = sizeof(float);  // Factor bytes per vector

        size_t totalCodeBytes = totalLen * codePartBytesPerVec;
        size_t totalFactorBytes = totalLen * factorBytesPerVec;

        std::vector<faiss::idx_t> allIds(totalLen);
        std::vector<uint8_t> allCodesPart(totalCodeBytes);
        std::vector<uint8_t> allL2Part(totalFactorBytes);
        std::vector<uint8_t> allL1Part(totalFactorBytes);

        size_t idOffset = 0;
        size_t codeOffset = 0;
        size_t l2Offset = 0;
        size_t l1Offset = 0;

        for (size_t devIdx = 0; devIdx < deviceCnt; ++devIdx)
        {
            int deviceId = indexConfig.deviceList[devIdx];
            auto pIndex = getActualIndex(deviceId);
            size_t devLen = pIndex->getListLength(listId);
            if (devLen == 0) continue;

            // get IDs
            std::vector<faiss::idx_t> devIds;
            ret = pIndex->getListIds(listId, devIds);
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to get IDs from device %d", deviceId);
            FAISS_THROW_IF_NOT_FMT(devIds.size() == devLen, "Device %d list %d ID count mismatch", deviceId, listId);

            // get compressed codes
            std::vector<uint8_t> devCodes;
            ret = pIndex->getListRawCodes(listId, devCodes);
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to get raw codes from device %d", deviceId);
            FAISS_THROW_IF_NOT_FMT(devCodes.size() == devLen * devCodeSize, "Device %d list %d codes size mismatch",
                                   deviceId, listId);

            // Parse block format: [codes][L2][L1]
            const uint8_t* devCodesPtr = devCodes.data();
            const uint8_t* devL2Ptr = devCodesPtr + devLen * codePartBytesPerVec;
            const uint8_t* devL1Ptr = devL2Ptr + devLen * factorBytesPerVec;

            // Use memcpy_s to copy data
            ret = memcpy_s(allIds.data() + idOffset, (totalLen - idOffset) * sizeof(faiss::idx_t), devIds.data(),
                           devIds.size() * sizeof(faiss::idx_t));
            FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for ids, ret=%d", ret);
            idOffset += devIds.size();

            ret = memcpy_s(allCodesPart.data() + codeOffset, totalCodeBytes - codeOffset, devCodesPtr,
                           devLen * codePartBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for codes, ret=%d", ret);
            codeOffset += devLen * codePartBytesPerVec;

            ret = memcpy_s(allL2Part.data() + l2Offset, totalFactorBytes - l2Offset, devL2Ptr,
                           devLen * factorBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for L2, ret=%d", ret);
            l2Offset += devLen * factorBytesPerVec;

            ret = memcpy_s(allL1Part.data() + l1Offset, totalFactorBytes - l1Offset, devL1Ptr,
                           devLen * factorBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed for L1, ret=%d", ret);
            l1Offset += devLen * factorBytesPerVec;
        }

        // Convert to per-vector format: [vec0: code+L2+L1][vec1: code+L2+L1]...
        std::vector<uint8_t> allCodes(totalLen * carrierCodeSize, 0);
        for (size_t i = 0; i < totalLen; ++i)
        {
            uint8_t* carrier = allCodes.data() + i * carrierCodeSize;
            auto memcpyRet =
                memcpy_s(carrier, carrierCodeSize, allCodesPart.data() + i * codePartBytesPerVec, codePartBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(memcpyRet == 0, "memcpy_s failed for codes interleave, ret=%d", memcpyRet);

            memcpyRet = memcpy_s(carrier + codePartBytesPerVec, carrierCodeSize - codePartBytesPerVec,
                                 allL2Part.data() + i * factorBytesPerVec, factorBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(memcpyRet == 0, "memcpy_s failed for L2 interleave, ret=%d", memcpyRet);

            memcpyRet = memcpy_s(carrier + codePartBytesPerVec + factorBytesPerVec,
                                 carrierCodeSize - codePartBytesPerVec - factorBytesPerVec,
                                 allL1Part.data() + i * factorBytesPerVec, factorBytesPerVec);
            FAISS_THROW_IF_NOT_FMT(memcpyRet == 0, "memcpy_s failed for L1 interleave, ret=%d", memcpyRet);
            if (ivfrabitqConfig.quantBits > 1)
            {
                writeMultiBitCarrierTrailer(carrier + payloadSize);
            }
        }

        // Add to inverted lists
        newInvlists->add_entries(listId, totalLen, allIds.data(), allCodes.data());
        totalCopied += totalLen;
    }

    index->replace_invlists(newInvlists.release(), true);
    index->own_fields = true;

    FAISS_THROW_IF_NOT_FMT(totalCopied == intf_->ntotal, "Total vectors mismatch: expected %ld, got %ld", intf_->ntotal,
                           static_cast<idx_t>(totalCopied));

    APP_LOG_INFO("AscendIndexIVFRaBitQImpl copyTo operation finished.\n");
}
}  // namespace ascend
}  // namespace faiss
