/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "impl/AscendIndexCagraImpl.h"

#include "ascenddaemon/utils/AscendOpDesc.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/Random.h"
#include "common/ErrorCode.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/DataType.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

using namespace ascend;
namespace faiss
{
namespace ascend
{
namespace
{

const uint32_t RANDOM_SEED = 1234;
const int LOG_INTERVAL = 10;
const int HASH_SIZE = 2;
const int PRE_COMPUTE_SIZE = 4;
std::vector<int> ASCEND_CAGRA_SEARCH_BATCHES = {64, 32, 16, 8, 4, 2, 1};
}  // namespace

AscendIndexCagraImpl::AscendIndexCagraImpl(int dim, int topK, const std::vector<int> &deviceList)
    : dim(dim), topK(topK), deviceList(deviceList)
{
}

AscendIndexCagraImpl::~AscendIndexCagraImpl() {}

APP_ERROR AscendIndexCagraImpl::Init(int graphDegree, int dataNum)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Init start");

    deviceId = deviceList[0];
    ASCEND_THROW_IF_NOT(deviceId >= 0);
    this->searchBatchSizes = ASCEND_CAGRA_SEARCH_BATCHES;
    auto ret = aclrtSetDevice(deviceId);
    APP_LOG_INFO("AscendIndexCagraImpl::Init on device(%d).\n", deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    pResources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    pResources->setTempMemory(ascendResourceSize);
    pResources->initialize();
    int queryBit = 6;
    int baseBit = 6;
    int offset = (dim + 7) / 8;
    this->rotatedSize = offset * queryBit;
    this->codeSize = offset * baseBit + 16;
    this->dataNum = dataNum;
    this->degree = graphDegree;

    ret = ResetCagraSearchOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "ResetCagraSearchOp init failed !!!");

    APP_LOG_INFO("AscendIndexCagraImpl::Init finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::Add(const uint32_t *graph, const uint32_t *hash, const float *data)
{
    AscendTensor<uint32_t, DIMS_2> graphDevice({this->dataNum, this->degree});
    auto ret = aclrtMemcpy(graphDevice.data(), graphDevice.getSizeInBytes(), graph,
                           this->dataNum * this->degree * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy graphDevice to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> hashDevice({this->dataNum, HASH_SIZE});
    ret = aclrtMemcpy(hashDevice.data(), hashDevice.getSizeInBytes(), hash,
                      this->dataNum * HASH_SIZE * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy hashDevice to device fail(%d)!", ret);

    AscendTensor<float, DIMS_2> dataDevice({this->dataNum, this->dim});
    ret = aclrtMemcpy(dataDevice.data(), dataDevice.getSizeInBytes(), data, this->dataNum * this->dim * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy dataDevice to device fail(%d)!", ret);
    g_graphDevice = std::move(graphDevice);
    g_hashDevice = std::move(hashDevice);
    g_dataDevice = std::move(dataDevice);
    return APP_ERR_OK;
}

APP_ERROR FloatToBytes(float val, uint8_t *bytes)
{
    uint32_t bits;
    int ret = memcpy_s(&bits, sizeof(bits), &val, sizeof(float));
    APPERR_RETURN_IF_NOT_FMT((ret == 0), APP_ERR_INNER_ERROR, "memcpy_s failed in FloatToBytes(%d)!", ret);
    bytes[0] = static_cast<uint8_t>(bits & 0xFFu);
    bytes[1] = static_cast<uint8_t>((bits >> 8) & 0xFFu);
    bytes[2] = static_cast<uint8_t>((bits >> 16) & 0xFFu);
    bytes[3] = static_cast<uint8_t>((bits >> 24) & 0xFFu);
    return APP_ERR_OK;
}

APP_ERROR PackbitsLittle(const uint8_t *bits, int n, uint8_t *packed)
{
    int nBytes = (n + 7) / 8;
    int ret = memset_s(packed, nBytes, 0, nBytes);
    APPERR_RETURN_IF_NOT_FMT((ret == 0), APP_ERR_INNER_ERROR, "memset_s failed in PackbitsLittle(%d)!", ret);
    for (int i = 0; i < n; ++i)
        if (bits[i] & 1u) packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    return APP_ERR_OK;
}

std::vector<float> AscendIndexCagraImpl::GenerateOrthogonalMatrix(int D, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> mat(D * D);
    for (int i = 0; i < D * D; ++i) mat[i] = dist(rng);

    std::vector<float> Q(D * D);
    for (int i = 0; i < D; ++i)
    {
        // 复制当前列
        for (int j = 0; j < D; ++j) Q[i * D + j] = mat[i * D + j];
        // 减去之前所有投影
        for (int k = 0; k < i; ++k)
        {
            float proj = 0.0f;
            for (int j = 0; j < D; ++j) proj += Q[i * D + j] * Q[k * D + j];
            for (int j = 0; j < D; ++j) Q[i * D + j] -= proj * Q[k * D + j];
        }
        // 归一化
        float norm = 0.0f;
        for (int j = 0; j < D; ++j) norm += Q[i * D + j] * Q[i * D + j];
        norm = std::sqrt(norm);
        if (norm < 1e-10f) continue;
        for (int j = 0; j < D; ++j) Q[i * D + j] /= norm;
    }
    return Q;
}

APP_ERROR AscendIndexCagraImpl::QuantizeData(int nq, const float *queryData, int ntotal, const float *baseData)
{
    APP_LOG_INFO("AscendIndexCagraImpl::QuantizeData start");
    int queryBit = 6;  // 量化到6bit
    int baseBit = 6;   // 量化到6bit
    int seed = 42;
    const int offset = (dim + 7) / 8;
    // 1) 随机正交矩阵
    std::vector<float> P = GenerateOrthogonalMatrix(dim, seed);
    std::vector<float> invP = MatTranspose(P, dim, dim);

    // 2) 质心：沿 ntotal 方向求每维均值
    std::vector<float> centroid(dim, 0.0f);
    for (int j = 0; j < dim; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < ntotal; ++i) sum += baseData[i * dim + j];
        centroid[j] = sum / static_cast<float>(ntotal);
    }

    // 3) 量化 base
    std::vector<float> baseCentered(ntotal * dim);
    for (int i = 0; i < ntotal; ++i)
        for (int j = 0; j < dim; ++j) baseCentered[i * dim + j] = baseData[i * dim + j] - centroid[j];

    std::vector<float> oPrime = MatMul(baseCentered, invP, ntotal, dim, dim);

    std::vector<float> maxAbsBase(ntotal);
    for (int i = 0; i < ntotal; ++i)
    {
        float m = 0.0f;
        for (int j = 0; j < dim; ++j) m = std::fmax(m, std::fabs(oPrime[i * dim + j]));
        maxAbsBase[i] = m;
    }

    const float halfRangeBase = static_cast<float>((1 << (baseBit - 1)) - 1);
    const float shiftBase = static_cast<float>(1 << (baseBit - 1));
    const float maxValBase = static_cast<float>((1 << baseBit) - 1);

    std::vector<float> scaleX(ntotal);
    for (int i = 0; i < ntotal; ++i) scaleX[i] = (maxAbsBase[i] == 0.0f) ? 1.0f : maxAbsBase[i] / halfRangeBase;

    std::vector<uint8_t> xQuant(ntotal * dim);
    std::vector<float> sumXQuant(ntotal, 0.0f);
    for (int i = 0; i < ntotal; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float v = std::round(oPrime[i * dim + j] / scaleX[i] + shiftBase);
            if (v < 0.0f) v = 0.0f;
            if (v > maxValBase) v = maxValBase;
            xQuant[i * dim + j] = static_cast<uint8_t>(v);
            sumXQuant[i] += v;
        }
    }

    std::vector<uint8_t> codes(ntotal * this->codeSize, 0);
    std::vector<uint8_t> bitBuf(dim);
    int ret = 0;
    for (int i = 0; i < ntotal; ++i)
    {
        for (int p = 0; p < baseBit; ++p)
        {
            for (int j = 0; j < dim; ++j) bitBuf[j] = static_cast<uint8_t>((xQuant[i * dim + j] >> p) & 1);
            ret = PackbitsLittle(bitBuf.data(), dim, codes.data() + i * this->codeSize + p * offset);
            APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "PackbitsLittle failed");
        }
        const int pos = offset * baseBit;
        float orCL2sqr = 0.0f;
        for (int j = 0; j < dim; ++j)
        {
            float diff = baseData[i * dim + j] - centroid[j];
            orCL2sqr += diff * diff;
        }
        ret = FloatToBytes(orCL2sqr, codes.data() + i * this->codeSize + pos);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "orCL2sqr FloatToBytes failed");
        ret = FloatToBytes(shiftBase, codes.data() + i * this->codeSize + pos + 4);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "shiftBase FloatToBytes failed");
        ret = FloatToBytes(scaleX[i], codes.data() + i * this->codeSize + pos + 8);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "scaleX FloatToBytes failed");
        ret = FloatToBytes(sumXQuant[i], codes.data() + i * this->codeSize + pos + 12);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sumXQuant FloatToBytes failed");
    }

    // 4) 量化 query
    std::vector<float> queryCentered(nq * dim);
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < dim; ++j) queryCentered[i * dim + j] = queryData[i * dim + j] - centroid[j];

    std::vector<float> qPrime = MatMul(queryCentered, invP, nq, dim, dim);

    std::vector<float> maxAbsQ(nq);
    for (int i = 0; i < nq; ++i)
    {
        float m = 0.0f;
        for (int j = 0; j < dim; ++j) m = std::fmax(m, std::fabs(qPrime[i * dim + j]));
        maxAbsQ[i] = m;
    }

    const float halfRangeQ = static_cast<float>((1 << (queryBit - 1)) - 1);
    const float shiftQuery = static_cast<float>(1 << (queryBit - 1));
    const float maxValQ = static_cast<float>((1 << queryBit) - 1);

    std::vector<float> scaleQ(nq);
    for (int i = 0; i < nq; ++i) scaleQ[i] = (maxAbsQ[i] == 0.0f) ? 1.0f : maxAbsQ[i] / halfRangeQ;

    std::vector<uint8_t> qQuant(nq * dim);
    std::vector<float> sumQQuant(nq, 0.0f);
    for (int i = 0; i < nq; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float v = std::round(qPrime[i * dim + j] / scaleQ[i] + shiftQuery);
            if (v < 0.0f) v = 0.0f;
            if (v > maxValQ) v = maxValQ;
            qQuant[i * dim + j] = static_cast<uint8_t>(v);
            sumQQuant[i] += v;
        }
    }

    std::vector<uint8_t> rotatedQq(nq * this->rotatedSize, 0);
    std::vector<float> precomputeAll(nq * PRE_COMPUTE_SIZE, 0.0f);
    for (int n = 0; n < nq; ++n)
    {
        for (int p = 0; p < queryBit; ++p)
        {
            for (int j = 0; j < dim; ++j) bitBuf[j] = static_cast<uint8_t>((qQuant[n * dim + j] >> p) & 1);
            PackbitsLittle(bitBuf.data(), dim, rotatedQq.data() + n * this->rotatedSize + p * offset);
        }
        float qrToCL2sqr = 0.0f;
        for (int j = 0; j < dim; ++j)
        {
            float diff = queryData[n * dim + j] - centroid[j];
            qrToCL2sqr += diff * diff;
        }
        precomputeAll[n * PRE_COMPUTE_SIZE + 0] = qrToCL2sqr;
        precomputeAll[n * PRE_COMPUTE_SIZE + 1] = shiftQuery;
        precomputeAll[n * PRE_COMPUTE_SIZE + 2] = scaleQ[n];
        precomputeAll[n * PRE_COMPUTE_SIZE + 3] = sumQQuant[n];
    }

    AscendTensor<uint8_t, DIMS_1> codeDev({this->dataNum * this->codeSize});
    ret = aclrtMemcpy(codeDev.data(), codeDev.getSizeInBytes(), codes.data(),
                      this->dataNum * this->codeSize * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy codeDev to device fail(%d)!", ret);

    g_precompute = std::move(precomputeAll);
    g_code = std::move(codeDev);
    g_rotated = std::move(rotatedQq);

    APP_LOG_INFO("AscendIndexCagraImpl::QuantizeData finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::Search(int n, const float *queryData, int topK, float *dists, uint32_t *labels)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Search start");

    size_t size = this->searchBatchSizes.size();
    int searched = 0;
    for (size_t i = 0; i < size; i++)
    {
        int batchSize = this->searchBatchSizes[i];
        if ((n - searched) >= batchSize)
        {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++)
            {
                auto ret = SearchImpl(batchSize, queryData + searched * this->dim, topK,
                                      g_precompute.data() + searched * PRE_COMPUTE_SIZE,
                                      g_rotated.data() + searched * this->rotatedSize, labels + searched * topK,
                                      dists + searched * topK);
                APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                                         "AscendIndexCagraImpl SearchImpl failed(%d)", ret);
                searched += batchSize;
            }
        }
    }

    APP_LOG_INFO("AscendIndexCagraImpl::Search finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchImpl(int n, const float *queryData, int topK, float *precompute, uint8_t *rotated,
                                           uint32_t *labels, float *dists)
{
    APP_LOG_INFO("AscendIndexCagraImpl::SearchImpl operation start.\n");

    AscendTensor<float, DIMS_2> queries({n, this->dim});
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), queryData, n * this->dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queries to device fail(%d)!", ret);

    AscendTensor<float, DIMS_1> preCodeDev({n * PRE_COMPUTE_SIZE});
    ret = aclrtMemcpy(preCodeDev.data(), preCodeDev.getSizeInBytes(), precompute, n * PRE_COMPUTE_SIZE * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy preCodeDev to device fail(%d)!", ret);

    AscendTensor<uint8_t, DIMS_1> rotatedDev({n * this->rotatedSize});
    ret = aclrtMemcpy(rotatedDev.data(), rotatedDev.getSizeInBytes(), rotated, n * this->rotatedSize * sizeof(uint8_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy rotatedDev to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> outIndices({n, topK});
    AscendTensor<float, DIMS_2> outDistances({n, topK});

    ret = SearchPaged(queries, preCodeDev, rotatedDev, outIndices, outDistances);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "AscendIndexCagraImpl SearchPaged failed(%d)",
                             ret);

    ret = aclrtMemcpy(dists, n * topK * sizeof(float), outDistances.data(), outDistances.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);

    ret = aclrtMemcpy(labels, n * topK * sizeof(uint32_t), outIndices.data(), outIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outIndices back to host(%d)", ret);
    APP_LOG_INFO("AscendIndexCagraImpl::SearchImpl operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchPaged(AscendTensor<float, DIMS_2> &queries,
                                            AscendTensor<float, DIMS_1> &preCompute,
                                            AscendTensor<uint8_t, DIMS_1> &rotated,
                                            AscendTensor<uint32_t, DIMS_2> &outIndices,
                                            AscendTensor<float, DIMS_2> &outDistances)
{
    APP_LOG_INFO("AscendIndexCagraImpl::SearchPaged operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    ComputeBlockDist(queries, g_graphDevice, g_hashDevice, g_dataDevice, preCompute, g_code, rotated, outIndices,
                     outDistances, stream);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    APP_LOG_INFO("AscendIndexCagraImpl::SearchPaged operation end.\n");
    return APP_ERR_OK;
}

void AscendIndexCagraImpl::ComputeBlockDist(AscendTensor<float, DIMS_2> &queryTensor,
                                            AscendTensor<uint32_t, DIMS_2> &graphDevice,
                                            AscendTensor<uint32_t, DIMS_2> &hashDevice,
                                            AscendTensor<float, DIMS_2> &data, AscendTensor<float, DIMS_1> &preCompute,
                                            AscendTensor<uint8_t, DIMS_1> &preCode,
                                            AscendTensor<uint8_t, DIMS_1> &rotated,
                                            AscendTensor<uint32_t, DIMS_2> &outIndices,
                                            AscendTensor<float, DIMS_2> &outDistances, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->cagraSearchOp.find(batchSize) != this->cagraSearchOp.end())
    {
        op = this->cagraSearchOp[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(graphDevice.data(), graphDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(hashDevice.data(), hashDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(data.data(), data.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(preCompute.data(), preCompute.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(preCode.data(), preCode.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(rotated.data(), rotated.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outIndices.data(), outIndices.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR AscendIndexCagraImpl::ResetCagraSearchOp()
{
    APP_LOG_INFO("AscendIndexCagraImpl::ResetCagraSearchOp operation started.\n");
    auto cagraSearchOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("CagraRabitq");
        std::vector<int64_t> queryShape({batch, this->dim});
        std::vector<int64_t> knnShape({this->dataNum, this->degree});
        std::vector<int64_t> hashShape({this->dataNum, HASH_SIZE});
        std::vector<int64_t> ptrShape({this->dataNum, this->dim});
        std::vector<int64_t> preCompute({batch * PRE_COMPUTE_SIZE});
        std::vector<int64_t> preCode({this->dataNum * this->codeSize});
        std::vector<int64_t> rotated({batch * this->rotatedSize});

        std::vector<int64_t> DistShape({batch, this->topK});
        std::vector<int64_t> IndiceShape({batch, this->topK});

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, knnShape.size(), knnShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, hashShape.size(), hashShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, ptrShape.size(), ptrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, preCompute.size(), preCompute.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, preCode.size(), preCode.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, rotated.size(), rotated.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, DistShape.size(), DistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, IndiceShape.size(), IndiceShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : this->searchBatchSizes)
    {
        cagraSearchOp[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(cagraSearchOpReset(cagraSearchOp[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed !!!");
    }

    APP_LOG_INFO("AscendIndexCagraImpl::ResetCagraSearchOp operation end.\n");
    return APP_ERR_OK;
}

}  // namespace ascend
}  // namespace faiss
