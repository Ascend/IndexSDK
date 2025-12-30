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

 
#include "ascend/custom/impl/AscendIndexIVFSQCImpl.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <numeric>
#include <thread>
#include <sys/time.h>

#include <faiss/index_io.h>

#include "faiss/utils/distances.h"
#include "faiss/utils/random.h"
#include "faiss/utils/sorting.h"
#include "faiss/utils/utils.h"
#include "ascend/utils/AscendIVFAddInfo.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascend/utils/fp16.h"
#include "common/utils/CommonUtils.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
// Default params in case of nullptr index
const int DEFAULT_DIM_IN = 256;
const int DEFAULT_DIM_OUT = 64;
const int DEFAULT_NLIST = 8192;

// Define params for finetune before add
const int FINETUNE_NITER = 1;
const int FINETUNE_MIN_POINTS = 16;
const int FINETUNE_MAX_POINTS = 512;

// For copy index we append information to sq.trained, but standard sq trained size is 2
const int SQ_VAL_SIZE = 2;
const float EPSILON = 1e-6;

// The value range of dimIn
const std::vector<int> DIMS_IN = { 64, 128, 256, 384, 512, 1024 };

// The value range of dimOut
const std::vector<int> DIMS_OUT = { 32, 64, 96, 128 };

extern "C" {
/* declare BLAS functions */
int sgemm_(const char *transa, const char *transb, int *m, int *n, int *k, const float *alpha, const float *a, int *lda,
    const float *b, int *ldb, float *beta, float *c, int *ldc);
}

AscendIndexIVFSQCImpl::AscendIndexIVFSQCImpl(AscendIndexIVFSQ *intf, int dimIn, int dimOut, int nlist, bool dummy,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric, AscendIndexIVFSQConfig config)
    : AscendIndexIVFSQFuzzyImpl(intf, dimIn, nlist, false, qtype, metric, config),
      dimIn(dimIn),
      dimOut(dimOut),
      ivfsqcConfig(config)
{
    VALUE_UNUSED(dummy);
    sq = faiss::ScalarQuantizer(dimOut, qtype);
    // Params for children classes can use
    checkParams();

    ratio = static_cast<size_t>(dimIn) / static_cast<size_t>(dimOut);

    sq.trained.resize(SQ_VAL_SIZE * sq.d);
}

AscendIndexIVFSQCImpl::~AscendIndexIVFSQCImpl() {}

void AscendIndexIVFSQCImpl::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQC copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVFSQFuzzyImpl::copyFrom(index);
    updateCompressValue();
    APP_LOG_INFO("AscendIndexIVFSQC copyFrom operation finished.\n");
}

void AscendIndexIVFSQCImpl::initTrainedValue()
{
    APP_LOG_INFO("AscendIndexIVFSQC initTrainedValue operation started.\n");
    compressValue.resize(ratio);
    for (size_t i = 0; i < ratio; i++) {
        compressValue[i].resize(dimOut);
    }
    compressIndex.resize(dimOut);
    for (int i = 0; i < dimOut; i++) {
        compressIndex[i].resize(ratio);
    }
    APP_LOG_INFO("AscendIndexIVFSQC initTrainedValue operation finished.\n");
}
/**
 * append compressValue to sq.trained to implement Copy Ascend Index to CPU
 */
void AscendIndexIVFSQCImpl::appendCompressData()
{
    APP_LOG_INFO("AscendIndexIVFSQC appendCompressData operation started.\n");
    for (size_t i = 0; i < compressValue.size(); ++i) {
        sq.trained.insert(sq.trained.end(), compressValue[i].begin(), compressValue[i].end());
    }

    for (size_t i = 0; i < compressIndex.size(); ++i) {
        for (size_t j = 0; j < compressIndex[i].size(); ++j) {
            sq.trained.push_back((float(compressIndex[i][j])));
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQC appendCompressData operation finished.\n");
}

/**
 * split compressValue from sq.trained to implement Copy CPU Index to Ascend
 */
void AscendIndexIVFSQCImpl::splitCompressData()
{
    APP_LOG_INFO("AscendIndexIVFSQC splitCompressData operation started.\n");
    // splitCompressData, compressData is in last compressSize of sq.trained
    size_t compressSize = ratio * static_cast<size_t>(dimOut) + static_cast<size_t>(dimOut) * ratio;

    // compressTag has 2 meanings: 1: compressData start id; 2: after split compressData, counts of data remaining
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() >= compressSize,
        "sq.trained.size should not be less than compressSize");
    size_t compressTag = sq.trained.size() - compressSize;
    FAISS_THROW_IF_NOT_MSG(compressTag >= (size_t)SQ_VAL_SIZE * static_cast<size_t>(dimOut),
        "compressTag should not be less than SQ_VAL_SIZE * dimOut");

    for (size_t i = compressTag; i < compressTag + ratio * static_cast<size_t>(dimOut); i++) {
        size_t row = (i - compressTag) / static_cast<size_t>(dimOut);
        size_t col = (i - compressTag) % static_cast<size_t>(dimOut);
        compressValue[row][col] = sq.trained[i];
    }

    for (size_t i = compressTag + ratio * static_cast<size_t>(dimOut); i < sq.trained.size(); i++) {
        size_t row = (i - (compressTag + ratio * static_cast<size_t>(dimOut))) / (ratio);
        size_t col = (i - (compressTag + ratio * static_cast<size_t>(dimOut))) % (ratio);
        compressIndex[row][col] = (size_t)sq.trained[i];
    }
    sq.trained.resize(compressTag);
    APP_LOG_INFO("AscendIndexIVFSQC splitCompressData operation started.\n");
}

/**
 * split Trained
 */
void AscendIndexIVFSQCImpl::splitTrained()
{
    AscendIndexIVFSQFuzzyImpl::splitTrained();
    splitCompressData();
}

void AscendIndexIVFSQCImpl::appendTrained()
{
    appendCompressData();
    AscendIndexIVFSQFuzzyImpl::appendTrained();
}

/**
 * Obtains index information to be Compressed during dimension reduction.
 */
void AscendIndexIVFSQCImpl::computeCompressIndex(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSQC computeCompressIndex operation started.\n");
    if (this->intf_->verbose) {
        printf("training computeCompressIndex on correlation matrix \n");
    }
    if (!this->ivfsqcConfig.useKmeansPP) {
        float alpha = 1.0;
        float beta = 0.0;
        int nInt = (int)n;
        sgemm_("N", "T", &dimIn, &dimIn, &nInt, &alpha, x, &dimIn, x, &dimIn, &beta, this->correlation.data(), &dimIn);
    }

    // Compute Compress index
    // Saved which several columns will be squeezed as one column
    std::vector<int> flag(dimIn, 1);
    compressIndex.resize(0);
    for (int i = 0; i < dimIn; i++) {
        // Already chosen as other dim compress index
        if (flag[i] != 1) {
            continue;
        }
        std::vector<size_t> ratioIndex; // which "ratio" dimensions will be squeezed
        std::vector<idx_t> idx(dimIn);
        fvec_argsort(dimIn, this->correlation.data() + i * dimIn, (size_t *)idx.data());
        std::reverse(idx.begin(), idx.end());
        ratioIndex.push_back(i);
        flag[i] = 0;
        size_t addCols = 1;
        for (int j = 1; j < dimIn; j++) {
            if (flag[idx[j]] != 1) {
                continue;
            }
            ratioIndex.push_back(idx[j]);
            flag[idx[j]] = 0;
            addCols += 1;
            if (addCols == ratio) {
                break;
            }
        }
        compressIndex.push_back(ratioIndex);
    }
    APP_LOG_INFO("AscendIndexIVFSQC computeCompressIndex operation finished.\n");
}

/**
 * Obtains parameters used for dimension reduction.
 */
void AscendIndexIVFSQCImpl::computeCompressValue(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSQC computeCompressValue operation started.\n");
    if (this->intf_->verbose) {
        printf("training computeCompressValue on %ld vectors in %dD\n", n, this->intf_->d);
    }
    // 1. add up every `ratio` numbers
    std::vector<std::vector<float>> sumX(n, std::vector<float>(dimOut, 0));
    float sum = 0.0;
#pragma omp parallel for reduction(+ : sum) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < (size_t)n; ++i) {
        const float *pX = x + i * dimIn;
        sum = 0.0;
        for (int j = 0; j < dimOut; ++j) {
            for (size_t k = 0; k < ratio; ++k) {
                sum += *(pX + compressIndex[j][k]);
            }
            sumX[i][j] = sum;
        }
    }

    // 2. compute compress value
    float compress = 0.0;
#pragma omp parallel for reduction(+ : compress) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < ratio; ++i) {
        for (int j = 0; j < dimOut; ++j) {
            const float *pX = x + compressIndex[j][i];
            compress = 0.0;
            for (idx_t k = 0; k < n; ++k) {
                compress += (sumX[k][j] - *(pX + k * dimIn));
            }
            compressValue[i][j] = compress / n * 1.0;
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQC computeCompressValue operation finished.\n");
}

/**
 * use the compressValue and compressIndex to compress `x` from `dimIn` to `dimOut`
 */
void AscendIndexIVFSQCImpl::compress(idx_t n, const float *x, float *res) const
{
    // Assuming that the data is n-dimensional, expressed as (x1, x2, ..., xn),
    // the compressing value:
    // A = np.mean(x[compressIndex[2]] + x[compressIndex[3] + x[compressIndex[4])
    // B = np.mean(x[compressIndex[1]] + x[compressIndex[3] + x[compressIndex[4])
    // C = np.mean(x[compressIndex[1]] + x[compressIndex[2] + x[compressIndex[4])
    // D = np.mean(x[compressIndex[1]] + x[compressIndex[2] + x[compressIndex[3])
    // the formula for compressing into one-dimensional x is:
    // x = x[compressIndex[1]] + x[compressIndex[2]
    // + x[compressIndex[3] + x[compressIndex[4]
    // - x[compressIndex[1]] * A - x[compressIndex[2] * B
    // - x[compressIndex[3] * C - x[compressIndex[4] * D .
    APP_LOG_INFO("AscendIndexIVFSQC compress operation started.\n");
#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < size_t(n); ++i) {
        const float *xSlice = x + i * dimIn;
        for (int j = 0; j < dimOut; ++j) {
            float *resSlice = res + i * dimOut;
            std::vector<float> tempVec(dimOut);
            for (size_t k = 0; k < ratio; ++k) {
                resSlice[j] += *(xSlice + compressIndex[j][k]) * (1 - compressValue[k][j]);
                tempVec[j] = resSlice[j];
            }
            resSlice[j] = tempVec[j] / ratio;
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQC compress operation finished.\n");
}

/**
 * Normalize the data `x`
 */
void AscendIndexIVFSQCImpl::normalize(idx_t n, float *x) const
{
    APP_LOG_INFO("AscendIndexIVFSQC start to normalize with %ld vector(s).\n", n);
#pragma omp parallel for num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < n; ++i) {
        float *slice = x + i * dimOut;
        float denominator = 0;
        for (int j = 0; j < dimOut; ++j) {
            denominator += (slice[j] * slice[j]);
        }
        if ((denominator <= EPSILON) && (denominator >= -EPSILON)) {
            continue;
        }
        const float inv_nr = 1.0 / sqrt(denominator);
        for (int j = 0; j < dimOut; ++j) {
            slice[j] *= inv_nr;
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQC normalize operation finished.\n");
}

void AscendIndexIVFSQCImpl::trainCompress(idx_t n, const float *x)
{
    computeCompressIndex(n, x);
    computeCompressValue(n, x);

    // 3. compress and normalize the train data
    std::vector<float> compressX(n * dimOut);
    compress(n, x, compressX.data());
    normalize(n, const_cast<float *>(compressX.data()));

    // 4. train L2 quantizer
    trainScalarQuantizer(n, compressX.data());

    updateDeviceSQTrainedValue();
    updateCompressValue();
}

void AscendIndexIVFSQCImpl::train(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSQC start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    if (pQuantizerImpl->npuClus != nullptr) {
        pQuantizerImpl->npuClus->Add(n, x);
    }

    // Compute Dim Matrix
    // Saved the similarity among columns
    correlation.resize(this->intf_->d * this->intf_->d);
    if (pQuantizerImpl->npuClus != nullptr) {
        pQuantizerImpl->npuClus->ComputeCorr(correlation.data(), false);
    }

    const int poolSize = 2;
    std::shared_ptr<AscendThreadPool> pools = std::make_shared<AscendThreadPool>(poolSize);
    std::vector<std::future<void>> functorRet;
    // 1. train L1 IVF quantizer
    functorRet.emplace_back(pools->Enqueue(&AscendIndexIVFSQCImpl::trainQuantizer, this, n, x, true));
    // 2. compute the compress value
    functorRet.emplace_back(pools->Enqueue(&AscendIndexIVFSQCImpl::trainCompress, this, n, x));

    int threadWait = 0;
    try {
        for_each(functorRet.begin(), functorRet.end(), [&](std::future<void> &ret) {
            threadWait++;
            ret.get();
        });
    } catch (std::exception &e) {
        for_each(functorRet.begin() + threadWait, functorRet.end(),
            [](std::future<void> &ret) { ret.wait(); });
        ASCEND_THROW_FMT("wait for train functor failed %s", e.what());
    }

    updateDeviceCoarseCenter();

    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFSQC train operation finished.\n");
}

void AscendIndexIVFSQCImpl::split(int n, const float *x)
{
    // calculate which list the vectors belongs to
    APP_LOG_INFO("AscendIndexIVFSQC split operation started: n=%d.\n", n);
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);
    pQuantizerImpl->cpuQuantizer->assign(n, x, assign.get());

    std::unordered_map<int, std::vector<int>> assignResult;
    std::vector<int> elemCount(nlist);

    for (int i = 0; i < n; i++) {
        idx_t listId = assign[i];
        auto it = assignResult.find(listId);
        if (it != assignResult.end()) {
            it->second.push_back(i);
            elemCount[listId] += 1;
            continue;
        }
        assignResult.emplace(listId, std::vector<int>(1, i));
        elemCount[listId] = 1;
    }

    float mean = (float)n / nlist;

    std::vector<float> diff(elemCount.size());
    std::transform(elemCount.begin(), elemCount.end(), diff.begin(), std::bind2nd(std::minus<float>(), mean));
    float sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sqSum / elemCount.size());

    float upThreshold = std::max(2 * mean, mean + (float)2.5 * stdev); // 2 times the mean or mean plus 2.5 times stdev

    std::vector<idx_t> idx(nlist);
    std::vector<idx_t> idxSmall;
    fvec_argsort(nlist, (float *)elemCount.data(), (size_t *)idx.data());

    idxSmall.assign(idx.begin(), idx.end());
    std::reverse(idx.begin(), idx.end());

    std::vector<int> largeList;
    std::vector<int> splitTime;

    for (int i = 0; i < nlist; i++) {
        if (elemCount[idx[i]] > upThreshold) {
            largeList.push_back(idx[i]);
            double tmpTime = floor(elemCount[idx[i]] / mean);
            splitTime.push_back((int)tmpTime);
        } else {
            break;
        }
    }

    if (this->intf_->verbose) {
        printf("Totally %zu large buckets detected\n", largeList.size());
    }
    size_t numMerge = static_cast<size_t>(std::accumulate(splitTime.begin(), splitTime.end(), 0)) - largeList.size();

    std::vector<int> popIndex;
    popIndex.insert(popIndex.end(), largeList.begin(), largeList.end());
    popIndex.insert(popIndex.end(), idxSmall.begin(), idxSmall.begin() + numMerge);

    int popCnt = 0;
    for (size_t i = 0; i < largeList.size(); i++) {
        std::vector<float> partX;
        for (auto j = 0; j < elemCount[largeList[i]]; j++) {
            partX.insert(partX.end(), x + assignResult[largeList[i]][j] * dimIn,
                x + assignResult[largeList[i]][j] * dimIn + dimIn);
        }
        std::unique_ptr<IndexFlat> partQuantizer(new IndexFlatL2(dimIn));
        Clustering clus(this->intf_->d, splitTime[i], this->finetuneConfig.cp);
        clus.train(elemCount[largeList[i]], partX.data(), *partQuantizer);
        for (auto j = 0; j < splitTime[i]; j++) {
            std::copy(partQuantizer->get_xb() + j * dimIn, partQuantizer->get_xb() + j * dimIn + dimIn,
                pQuantizerImpl->cpuQuantizer->get_xb() + popIndex[popCnt] * dimIn);
            popCnt += 1;
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQC split operation finished.\n");
}


void AscendIndexIVFSQCImpl::fineTune(size_t n, const float *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    APP_LOG_ERROR("fineTune is not implemented, and will be deprecated.");
}

void AscendIndexIVFSQCImpl::checkParams() const
{
    AscendIndexIVFSQImpl::checkParams();
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS_IN.begin(), DIMS_IN.end(), this->dimIn) != DIMS_IN.end(),
        "Unsupported dimIn");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS_OUT.begin(), DIMS_OUT.end(), this->dimOut) != DIMS_OUT.end(),
        "Unsupported dimOut");
    FAISS_THROW_IF_NOT_MSG(dimIn % dimOut == 0, "The dimIn must be an integer multiple of the dimOut");
    FAISS_THROW_IF_NOT_MSG(dimIn > dimOut, "The dimIn must be greater than the dimOut");
}

std::shared_ptr<::ascend::Index> AscendIndexIVFSQCImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFSQC  createIndex operation started, device id: %d\n", deviceId);
    VALUE_UNUSED(deviceId);
    APP_LOG_INFO("AscendIndexIVFSQC createIndex operation finished.\n");
    return nullptr;
}

void AscendIndexIVFSQCImpl::trainScalarQuantizer(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSQC trainScalarQuantizer operation started.\n");
    if (this->intf_->verbose) {
        printf("training scalar quantizer on %ld vectors in %dD\n", n, this->dimOut);
    }

    // The input is already guaranteed to be on the CPU
    sq.train(n, x);
    APP_LOG_INFO("AscendIndexIVFSQC trainScalarQuantizer operation finished.\n");
}

void AscendIndexIVFSQCImpl::updateCompressValue()
{
    APP_LOG_INFO("AscendIndexIVFSQC updateCompressValue operation started.\n");
    std::vector<float> vcompress;
    for (size_t i = 0; i < compressValue.size(); ++i) {
        vcompress.insert(vcompress.end(), compressValue[i].begin(), compressValue[i].end());
    }

    std::vector<int> vcompressIndex;
    for (size_t i = 0; i < compressIndex.size(); ++i) {
        vcompressIndex.insert(vcompressIndex.end(), compressIndex[i].begin(), compressIndex[i].end());
    }
    using namespace ::ascend;
    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        AscendTensor<float, DIMS_2> vcompressTensor({ dimIn / dimOut, dimOut });
        auto ret = aclrtMemcpy(vcompressTensor.data(), vcompressTensor.getSizeInBytes(), vcompress.data(),
                               vcompressTensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy vcompress error %d", ret);
        AscendTensor<int, DIMS_2> vcompressIndexTensor({ dimOut, dimIn / dimOut });
        ret = aclrtMemcpy(vcompressIndexTensor.data(), vcompressIndexTensor.getSizeInBytes(), vcompressIndex.data(),
                          vcompressIndexTensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy vcompressIndex error %d", ret);
        pIndex->updateCompressValue(vcompressTensor, vcompressIndexTensor);
    }
    APP_LOG_INFO("AscendIndexIVFSQC updateCompressValue operation finished.\n");
}
} // ascend
} // faiss
