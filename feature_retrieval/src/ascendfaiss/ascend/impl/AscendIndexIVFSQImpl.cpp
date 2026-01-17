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


#include "AscendIndexIVFSQImpl.h"

#include <algorithm>

#include <faiss/utils/distances.h>
#include "ascend/utils/AscendIVFAddInfo.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascendhost/include/index/IndexIVFSQIPAicpu.h"
#include "ascendhost/include/index/IndexIVFSQL2Aicpu.h"

namespace faiss {
namespace ascend {
// For copy index we append information to sq.trained, but standard sq trained size is 2
const int SQ_VAL_SIZE = 2;

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 512;
const size_t MAX_RESIDUAL_VAL = 100000;
const size_t SEED = 1234;
// Default nlist in case of nullptr index
const size_t DEFAULT_NLIST = 1024;

// The value range of dim
const std::vector<int> DIMS = { 64, 128, 256, 384, 512 };

// The value range of nlist
const std::vector<int> NLISTS = { 1024, 2048, 4096, 8192, 16384, 32768 };

AscendIndexIVFSQImpl::AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, const faiss::IndexIVFScalarQuantizer *index,
    AscendIndexIVFSQConfig config)
    : AscendIndexIVFImpl(intf, ((index == nullptr) ? DEFAULT_DIM : index->d),
    ((index == nullptr) ? faiss::METRIC_L2 : index->metric_type),
    ((index == nullptr) ? DEFAULT_NLIST : index->nlist), config), intf_(intf), ivfsqConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    initFlatAT();
    copyFrom(index);
}

AscendIndexIVFSQImpl::AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, int dims, int nlist,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric,
    bool encodeResidual, AscendIndexIVFSQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config),
      intf_(intf),
      ivfsqConfig(config),
      byResidual(encodeResidual)
{
    initFlatAT();
    sq = faiss::ScalarQuantizer(dims, qtype);

    checkParams();

    initIndexes();

    initDeviceAddNumMap();

    this->intf_->is_trained = false;
}

// Construct an empty index when AscendIndexIVFSQ is parent class for custom classes
AscendIndexIVFSQImpl::AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, int dims, int nlist, bool dummy,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric,
    bool encodeResidual, AscendIndexIVFSQConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config),
      intf_(intf),
      ivfsqConfig(config),
      byResidual(encodeResidual)
{
    VALUE_UNUSED(dummy);
    initFlatAT();

    sq = faiss::ScalarQuantizer(dims, qtype);

    // Params for children classes can use
    checkParams();

    initDeviceAddNumMap();

    this->intf_->is_trained = false;
}

AscendIndexIVFSQImpl::~AscendIndexIVFSQImpl() {}

void AscendIndexIVFSQImpl::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    copyFromInner(index);
}

void AscendIndexIVFSQImpl::copyFromInner(const faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQ copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVFImpl::copyFrom(index);

    sq = index->sq;
    byResidual = index->by_residual;
    checkParams();
    clearIndexes();
    initIndexes();
    initDeviceAddNumMap();

    // The other index might not be trained
    if (!index->is_trained) {
        return;
    }

    this->intf_->is_trained = true;

    // Copy our lists as well. The product quantizer must have data in it
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() > 0, "index is trained but trained is empty.");

    updateDeviceCoarseCenter();
    updateDeviceSQTrainedValue();

    // copy cpu index's codes to ascend device
    copyCodes(index);
    APP_LOG_INFO("AscendIndexIVFSQ copyFrom operation finished.\n");
}

void AscendIndexIVFSQImpl::copyTo(faiss::IndexIVFScalarQuantizer *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQ copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVFImpl::copyTo(index);
    index->by_residual = byResidual;
    index->sq = sq;
    index->code_size = sq.code_size;
    index->sq.trained.resize(SQ_VAL_SIZE * sq.d);

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->intf_->is_trained && this->intf_->ntotal > 0) {
        // use for(deviceList) rather than for(auto& index : indexMap),
        // to ensure merged codes and ids in sequence
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
            int deviceId = indexConfig.deviceList[i];
            indexIVFFastGetListCodes(deviceId, nlist, ivf);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQ copyTo operation finished.\n");
}

void AscendIndexIVFSQImpl::checkInvertedLists(const faiss::IndexIVFScalarQuantizer *index)
{
    auto arrayIvf = dynamic_cast<ArrayInvertedLists *>(index->invlists);
    FAISS_THROW_IF_NOT_MSG(arrayIvf != nullptr, "invlists of index invalid.");
    FAISS_THROW_IF_NOT_MSG(arrayIvf->codes.size() == arrayIvf->nlist, "codes size error.");
    FAISS_THROW_IF_NOT_MSG(arrayIvf->ids.size() == arrayIvf->nlist, "ids size error.");
    FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(this->nlist) == arrayIvf->nlist,
                           "nlist of index->invlists must equal to this->nlist");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->code_size) != DIMS.end(),
                           "index code_size invalid");
    FAISS_THROW_IF_NOT_MSG(index->code_size == index->sq.code_size, "index->sq code_size not equal index code_size");
}

void AscendIndexIVFSQImpl::copyCodes(const faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQ copyCodes operation started.\n");
    auto arrayIvf = dynamic_cast<ArrayInvertedLists *>(index->invlists);
    checkInvertedLists(index);
    const InvertedLists *ivf = index->invlists;
    size_t nlist = ivf->nlist;
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<std::vector<int>> offsumMap(nlist, std::vector<int>(deviceCnt, 0));
    std::vector<std::vector<float>> precomputeVals(nlist);
    std::vector<size_t> listSizes(nlist, 0);
    // seperate codes to every device
#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        FAISS_THROW_IF_NOT_FMT(listSize * index->d == arrayIvf->codes[i].size(),
                               "list_size(%zu) not equl codes[%zu] size.", i, i);
        listSizes[i] = listSize;

        // Ascend index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(listSize <= (size_t)std::numeric_limits<int>::max(),
            "Ascend inverted list can only support %zu entries; %zu found", (size_t)std::numeric_limits<int>::max(),
            listSize);

        int listNumPerDevice = listSize / deviceCnt;
        int listNumLast = listSize % deviceCnt;
        for (size_t j = 0; j < deviceCnt; j++) {
            deviceAddNumMap[i][j] += listNumPerDevice;
        }

        for (int j = 0; j < listNumLast; j++) {
            deviceAddNumMap[i][j] += 1;
        }

        int offset = 0;
        for (size_t j = 0; j < deviceCnt; j++) {
            offsumMap[i][j] = offset;
            offset += deviceAddNumMap[i][j];
        }
        precomputeVals[i].resize(listSize, 0.0f);
        if (ivf->get_codes(i) != nullptr) {
            calcPrecompute(ivf->get_codes(i), precomputeVals[i].data(), listSize);
        }
    }

    auto maxListSize = listSizes.empty() ? 0 : *(std::max_element(listSizes.begin(), listSizes.end()));
    if (maxListSize == 0) {
        return;
    }

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, 0, 0, 0);
        param.listId= idx;
        indexIVFSQFastAdd(param, offsumMap, deviceAddNumMap, precomputeVals, ivf);
    };

    // parallel adding codes to every device
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    APP_LOG_INFO("AscendIndexIVFSQ copyCodes operation finished.\n");
}

void AscendIndexIVFSQImpl::train(idx_t n, const float *x)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    // train L1 IVF quantizer
    trainQuantizer(n, x);
    updateDeviceCoarseCenter();

    // train L2 quantizer
    trainResidualQuantizer(n, x);

    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFSQ train operation finished.\n");
}

void AscendIndexIVFSQImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    //    v_i = (code_i + 0.5) / 255 * vdiff_i + vmin_i (reconstruct code to float)
    //    distance = || x - v ||^2
    //             = x_i^2 - 2 * (x_i * v_i) + v_i^2
    //               --------------------------    -------
    //                       term 1                 term 2
    //    term2 can be precomputed when adding database
    APP_LOG_INFO("AscendIndexIVFSQ addImpl operation started: n=%d.\n", n);
    const float *xi = x;
    size_t deviceCnt = indexConfig.deviceList.size();

    // calculate which list the vectors belongs to
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);

    if (this->ivfsqConfig.useKmeansPP) {
        pQuantizerImpl->npuQuantizer->assign(n, x, assign.get());
    } else {
        pQuantizerImpl->cpuQuantizer->assign(n, x, assign.get());
    }

    std::unique_ptr<float[]> tmpPtr;
    if (byResidual) {
        // get the residuals with the l1 coarse centroids
        float *residuals = new (std::nothrow) float[static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d)];
        FAISS_THROW_IF_NOT_MSG(residuals != nullptr, "Memory allocation fail for residuals");
        pQuantizerImpl->cpuQuantizer->compute_residual_n(n, x, residuals, assign.get());
        tmpPtr.reset(residuals);
        xi = residuals;
    }

    // compute the sq codes
    
    std::unique_ptr<uint8_t[]> codes = std::make_unique<uint8_t[]>(static_cast<size_t>(n) * sq.code_size);
    sq.compute_codes(xi, codes.get(), n);
    // precompute values(term2), reuse residuals mem if byResidual
    std::unique_ptr<float[]> precomputeVals = std::make_unique<float[]>(n);
    std::unique_ptr<float[]> del2 = std::make_unique<float[]>(static_cast<size_t>(n) *
            static_cast<size_t>(this->intf_->d));
    calcPrecompute(codes.get(), precomputeVals.get(), n, del2.get());

    // list id -> # being added,
    // combine the codes(codes assigned to the same IVFList) together
    std::unordered_map<int, AscendIVFAddInfo> assignCounts;
    for (int i = 0; i < n; i++) {
        idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(codes.get() + i * sq.code_size, ids + i, precomputeVals.get() + i);
            continue;
        }

        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }

        assignCounts.emplace(listId, AscendIVFAddInfo(devIdx, deviceCnt, sq.code_size));
        assignCounts.at(listId).Add(codes.get() + i * sq.code_size, ids + i, precomputeVals.get() + i);
    }

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        for (auto &centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0) {
                continue;
            }

            uint8_t *codePtr = nullptr;
            ascend_idx_t *idPtr = nullptr;
            float *precompPtr = nullptr;
            centroid.second.GetCodeAndIdPtr(idx, &codePtr, &idPtr, &precompPtr);
            IndexParam<uint8_t, uint8_t, ascend_idx_t> param(deviceId, num, 0, 0);
            param.listId = listId;
            param.query = codePtr;
            param.label = idPtr;
            indexIVFSQAdd(param, precompPtr);
            deviceAddNumMap[listId][idx] += num;
        }
    };

    // parallel adding codes to every device
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);

    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexIVFSQ addImpl operation finished.\n");
}

void AscendIndexIVFSQImpl::checkParams() const
{
    // only support L2
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
        this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // only support SQ8
    FAISS_THROW_IF_NOT_MSG(sq.qtype == faiss::ScalarQuantizer::QT_8bit, "Unsupported qtype");

    FAISS_THROW_IF_NOT_MSG(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(), "Unsupported nlist");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(), "Unsupported dims");
}

std::shared_ptr<::ascend::Index> AscendIndexIVFSQImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFSQ  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::Index> index = nullptr;
    switch (this->intf_->metric_type) {
        case MetricType::METRIC_INNER_PRODUCT:
            index = std::make_shared<::ascend::IndexIVFSQIPAicpu>(nlist, this->intf_->d,
                                                                  byResidual, nprobe, indexConfig.resourceSize);
            break;
        case MetricType::METRIC_L2:
            index = std::make_shared<::ascend::IndexIVFSQL2Aicpu>(nlist, this->intf_->d,
                                                                  byResidual, nprobe, indexConfig.resourceSize);
            break;
        default:
            ASCEND_THROW_MSG("Unsupported metric type\n");
    }
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf sq, result is %d", ret);

    APP_LOG_INFO("AscendIndexIVFSQ createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFSQImpl::trainResidualQuantizer(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSQ trainResidualQuantizer operation started: n=%ld.\n", n);
    if (this->intf_->verbose) {
        printf("training scalar quantizer on %ld vectors in %dD\n", n, this->intf_->d);
    }
    const float* tmpX = x;
    std::unique_ptr<const float[]> tmpPtr;
    if (this->byResidual) {
        x = fvecs_maybe_subsample(this->intf_->d, (size_t*)&n, MAX_RESIDUAL_VAL, x, this->intf_->verbose, SEED);
        if (tmpX != x) {
            tmpPtr.reset(x);
        }

        std::vector<idx_t> idx(n);
        this->pQuantizerImpl->cpuQuantizer->assign(n, x, idx.data());

        std::vector<float> residuals(n * this->intf_->d);
        this->pQuantizerImpl->cpuQuantizer->compute_residual_n(n, x, residuals.data(), idx.data());
        sq.train(n, residuals.data());
    } else {
        sq.train(n, x);
    }
    updateDeviceSQTrainedValue();
    APP_LOG_INFO("AscendIndexIVFSQ trainResidualQuantizer operation finished.\n");
}

void AscendIndexIVFSQImpl::updateDeviceSQTrainedValue()
{
    // convert trained value to fp16, *2 means vmin and vdiff
    // trained size equal to sq.d whatever index type
    APP_LOG_INFO("AscendIndexIVFSQ updateDeviceSQTrainedValue operation started.\n");
    std::vector<uint16_t> trainedFp16(sq.d * 2); // 2 reps vmin and vdiff
    uint16_t *vmin = trainedFp16.data();
    uint16_t *vdiff = trainedFp16.data() + sq.d;

    switch (sq.qtype) {
        case faiss::ScalarQuantizer::QT_8bit:
            transform(begin(sq.trained), end(sq.trained), begin(trainedFp16),
                [](float temp) { return fp16(temp).data; });
            break;
        default:
            FAISS_THROW_FMT("not supportted qtype(%d).", sq.qtype);
            break;
    }

    for (auto &index : indexes) {
            indexIVFSQUpdateTrainedValue(index.first, sq.d, vmin, vdiff);
    }
    APP_LOG_INFO("AscendIndexIVFSQ updateDeviceSQTrainedValue operation finished.\n");
}

void AscendIndexIVFSQImpl::calcPrecompute(const uint8_t *codes, float *compute, size_t n, float *xMem)
{
    APP_LOG_INFO("AscendIndexIVFSQ calcPrecompute operation started.\n");
    float *x = xMem;
    std::unique_ptr<float[]> tmpPtr;
    if (!x) {
        x = new (std::nothrow) float[n * static_cast<size_t>(this->intf_->d)];
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "Memory allocation fail for x");
        tmpPtr.reset(x);
    }
    sq.decode(codes, x, n);

    fvec_norms_L2sqr(compute, x, this->intf_->d, n);
    APP_LOG_INFO("AscendIndexIVFSQ calcPrecompute operation finished.\n");
}

size_t AscendIndexIVFSQImpl::getAddElementSize() const
{
    // element size: codesize + sizeof(ID) + sizeof(preCompute)
    return sq.code_size + sizeof(ascend_idx_t) + sizeof(float);
}

void AscendIndexIVFSQImpl::indexIVFSQFastAdd(IndexParam<uint16_t, uint16_t, ascend_idx_t> param,
                                             std::vector<std::vector<int>> &offsumMap,
                                             std::vector<std::vector<int>> &deviceAddNumMap,
                                             std::vector<std::vector<float>> &precomputeVals,
                                             const InvertedLists *ivf)
{
    int idx = param.listId;
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;
    int nlist = ivf->nlist;
    int codeSize = ivf->code_size;
    for (int i = 0; i < nlist; i++) {
        int num = deviceAddNumMap[i][idx];
        uint8_t *codes = const_cast<uint8_t *>(ivf->get_codes(i)) + offsumMap[i][idx] * codeSize;
        float *precompute = const_cast<float *>(precomputeVals[i].data()) + offsumMap[i][idx];
        if (ivf->get_ids(i) == nullptr) {
            continue;
        }
        faiss::idx_t *ids = const_cast<faiss::idx_t *>(ivf->get_ids(i)) + offsumMap[i][idx];
        auto ret = pIndex->addVectors(i, num, codes, reinterpret_cast<::ascend::Index::idx_t *>(ids), precompute);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to fast add to ivf sq, listId %d, ret: %d", i, ret);
    }
}

void AscendIndexIVFSQImpl::indexIVFSQAdd(IndexParam<uint8_t, uint8_t, ascend_idx_t> param, const float *precomputedVal)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;
    const uint8_t *codes = param.query;
    const ascend_idx_t *ids = param.label;
    auto ret = pIndex->addVectors(param.listId, param.n, codes,
                                  static_cast<const ::ascend::Index::idx_t *>(ids),
                                  precomputedVal);
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to add to ivf sq, ret: %d", ret);
}

void AscendIndexIVFSQImpl::indexIVFSQUpdateTrainedValue(int deviceId, int dim, uint16_t *vmin, uint16_t *vdiff) const
{
    using namespace ::ascend;
    AscendTensor<float16_t, DIMS_1> vminTensor(vmin, { dim });
    AscendTensor<float16_t, DIMS_1> vdiffTensor(vdiff, { dim });
    auto *pIndex = getActualIndex(deviceId);
    pIndex->updateTrainedValue(vminTensor, vdiffTensor);
}

} // ascend
} // faiss
