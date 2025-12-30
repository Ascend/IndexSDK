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


#include "ascend/custom/impl/AscendIndexIVFSPSQImpl.h"

#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <numeric>
#include <fcntl.h>
#include <omp.h>
#include <map>
#include <memory>

#include <ascendsearch/ascend/utils/AscendIVFAddInfo.h>
#include <common/threadpool/AscendThreadPool.h>
#include <ascendsearch/ascend/utils/AscendUtils.h>
#include <ascendsearch/ascend/utils/fp16.h>
#include <ascendsearch/ascend/rpc/AscendRpc.h>
#include <ascendsearch/ascend/rpc/AscendRpcIndexIVFSPSQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>

#include "ascenddaemon/utils/IoUtil.h"

namespace faiss {
namespace ascendSearch {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 512;

// The value range of dim
const std::vector<int> DIMS = {64, 128, 256, 512, 768};

const int USE_TRANSFORM_THRES = 48;

const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;

// Default size for which we get base
const size_t GET_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of get
const size_t GET_VEC_SIZE = UNIT_VEC_SIZE * KB;

// Default size for which we page add or search
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of add
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

const int FILTER_SIZE = 6;

const int HANDLE_BATCH_MAX_VALUE = 240;

const int HANDLE_BATCH_MIN_VALUE = 16;

// The value range of nlist
const std::vector<int> NLISTS = { 256, 512, 1024, 2048, 4096, 8192, 16384 };

// The multiples of dim2
const int MULTIPLES_DIM2 = 16;

// Default dim in case of nullptr index
const int64_t CODE_BOOK_MAX_FILE_SIZE = 3LL * 1024 * 1024 * 1024; // 3GB

// 设备数量上限
const uint32_t MAX_DEVICEID = 1024;

// searchListSize, nProbe, handleBatch的公约数，以16为单位递增
const int CONFIG_PARAM_BASE_FACTOR = 16;

// dim2 (nonzeroNum) 上限
const int MAX_NONZERO_NUM = 128;

// 大维度数据，超过此维度需要对nlist大小进行限制
const std::vector<int> HIGH_DIMS = { 512, 768 };

// 对于大维度数据，限制nlist的最大大小，防止码本大小过大，训练时间过长
const int HIGH_DIMS_NLIST_LIMIT = 2048;

// During addVector step, we can decide whether to use NPU to add Vectors based on add batch size
const int USE_NPU_ADD_LIMIT = 100000;

} // namespace

AscendIndexIVFSPSQImpl::AscendIndexIVFSPSQImpl(int dims, int dims2, int k, int nlist,
                                               AscendIndex *intf,
                                               faiss::ScalarQuantizer::QuantizerType qType,
                                               faiss::MetricType metric, bool encodeResidual,
                                               AscendIndexIVFSPSQConfig config)
    : AscendIndexImpl(dims, metric, config, intf),
    spSq(dims2, qType), spSqConfig(config), dims2(dims2),
    byResidual(encodeResidual), nCentroid(0), k(k), nlist(nlist), oriFeature(false)
{
    codeBook = std::make_shared<std::vector<float>>();
    checkParams();
    initRpcCtx();
    initDeviceAddNumMap();
    this->intf_->is_trained = false;

    // initial idxDeviceMap mem space
    int deviceNum = indexConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);
}

AscendIndexIVFSPSQImpl::~AscendIndexIVFSPSQImpl() {}

int AscendIndexIVFSPSQImpl::CreateIndex(rpcContext ctx)
{
    int indexId;
    IndexIVFSPSQParameter spSqIvfParameter(this->intf_->d, dims2, k, spSq.qtype, this->intf_->metric_type,
                                           spSqConfig.resourceSize, spSqConfig.slim,
                                           spSqConfig.filterable, nlist, byResidual,
                                           spSqConfig.handleBatch, spSqConfig.nprobe,
                                           spSqConfig.searchListSize);
    auto ret = RpcCreateIndexIVFSPSQ(ctx, indexId, spSqIvfParameter);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexSP failed(%d).", ret);

    return indexId;
}

void AscendIndexIVFSPSQImpl::initDeviceAddNumMap()
{
    APP_LOG_INFO("AscendIndexIVF initDeviceAddNumMap operation started.\n");
    deviceAddNumMap.clear();
    deviceAddNumMap.resize(this->nlist);
    for (int i = 0; i < this->nlist; i++) {
        deviceAddNumMap[i] = std::vector<int>(indexConfig.deviceList.size(), 0);
    }
    APP_LOG_INFO("AscendIndexIVF initDeviceAddNumMap operation finished.\n");
}


void AscendIndexIVFSPSQImpl::checkParams()
{
    // only support L2 and INNER_PRODUCT
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2,
                           "Unsupported metric type");

    // only support SQ8
    FAISS_THROW_IF_NOT_MSG(this->spSq.qtype == faiss::ScalarQuantizer::QT_8bit ||
                           spSq.qtype == faiss::ScalarQuantizer::QT_8bit_uniform,
                           "Unsupported qtype");

    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_MSG(std::find(NLISTS.begin(), NLISTS.end(), nlist) != NLISTS.end(), "Unsupported nlists");

    if (std::find(HIGH_DIMS.begin(), HIGH_DIMS.end(), this->intf_->d) != HIGH_DIMS.end()) {
        FAISS_THROW_IF_NOT_FMT(this->nlist <= HIGH_DIMS_NLIST_LIMIT,
            "nlist should be <= %d if dims >= %d; yours is %d",
            HIGH_DIMS_NLIST_LIMIT, HIGH_DIMS[0], this->nlist);
    }

    FAISS_THROW_IF_NOT_MSG(this->dims2 <= this->intf_->d &&
        this->dims2 <= MAX_NONZERO_NUM &&
        this->dims2 % MULTIPLES_DIM2 == 0,
        "Unsupported nonzeroNum");

    FAISS_THROW_IF_NOT_MSG(this->nlist == this->k, "k must be equal to nlist");

    // 对deviceList做校验
    FAISS_THROW_IF_NOT_MSG(spSqConfig.deviceList.size() == 1, "the size of deviceList must be 1.");
    uint32_t deviceId = static_cast<uint32_t>(spSqConfig.deviceList.front());
    FAISS_THROW_IF_NOT_FMT(deviceId < MAX_DEVICEID, "device id %u should be in [0, %u)",
        deviceId, MAX_DEVICEID);

    // 对config的所有参数进行校验
    FAISS_THROW_IF_NOT_MSG(spSqConfig.nprobe > 0 && spSqConfig.nprobe <= this->nlist,
        "nprobe must be greater than 0, and be less than or equal to nlist");
    FAISS_THROW_IF_NOT_FMT(spSqConfig.nprobe % CONFIG_PARAM_BASE_FACTOR == 0,
        "nprobe need to be a multiple of %d.", CONFIG_PARAM_BASE_FACTOR);
    FAISS_THROW_IF_NOT_FMT(spSqConfig.handleBatch % CONFIG_PARAM_BASE_FACTOR == 0 &&
        spSqConfig.handleBatch >= HANDLE_BATCH_MIN_VALUE &&
        spSqConfig.handleBatch <= HANDLE_BATCH_MAX_VALUE,
        "handleBatch must be in multiples of %d, and between %d and %d.",
        CONFIG_PARAM_BASE_FACTOR, HANDLE_BATCH_MIN_VALUE, HANDLE_BATCH_MAX_VALUE);
    FAISS_THROW_IF_NOT_FMT(spSqConfig.searchListSize % CONFIG_PARAM_BASE_FACTOR == 0,
        "searchListSize must be in multiples of %d.", CONFIG_PARAM_BASE_FACTOR);
}

void AscendIndexIVFSPSQImpl::checkParamsSame(AscendIndexImpl& index)
{
    try {
        AscendIndexIVFSPSQImpl& spSqIndex = dynamic_cast<AscendIndexIVFSPSQImpl&>(index);

        FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == spSqIndex.intf_->metric_type,
            "the metric type must be same");
        FAISS_THROW_IF_NOT_MSG(this->spSq.qtype == spSqIndex.spSq.qtype, "the qtype must be same");
        FAISS_THROW_IF_NOT_MSG(this->intf_->d == spSqIndex.intf_->d, "the dim must be same.");
        FAISS_THROW_IF_NOT_MSG(this->spSqConfig.deviceList == spSqIndex.spSqConfig.deviceList,
                               "the deviceList must be same.");
        FAISS_THROW_IF_NOT_MSG(this->codeBook == spSqIndex.codeBook, "codebook must be the same.");
        spSqIndex.checkAndSetAddFinish();
    } catch (std::bad_cast &e) {
        FAISS_THROW_MSG("the type of index is not same");
    }
}

void AscendIndexIVFSPSQImpl::loadAllData(const char *dataPath)
{
    FAISS_THROW_IF_NOT_MSG((dataPath != nullptr && strlen(dataPath) != 0), "dataPath is invalid.");
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation started.\n");
    for (auto &index : indexMap) {
        uint32_t size = 0;
        codeBook->resize(this->dims2*this->nlist*this->intf_->d);
        spSq.trained.resize(this->dims2 + this->dims2);
        RpcError ret = RpcIndexIVFSPSQLoadAllData(index.first, index.second, dataPath,
            codeBook->data(), spSq.trained.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQLoadAllData failed(%d).", ret);
        RpcIndexIVFSPSQGetBaseSize(index.first, index.second, size);
        this->intf_->ntotal = size;
        this->intf_->is_trained = true;
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::checkSharedCodebookParams(const AscendIndexIVFSPSQImpl &loadedIndex) const
{
    FAISS_THROW_IF_NOT_MSG(loadedIndex.codebookFinished,
        "loadedIndex itself does not have its codebook added.\n");
    FAISS_THROW_IF_NOT_FMT(loadedIndex.nlist == this->nlist,
        "nlist mismatch: loadedIndex's nlist = %d, current index's nlist = %d\n", loadedIndex.nlist, this->nlist);
    FAISS_THROW_IF_NOT_FMT(loadedIndex.dims2 == this->dims2,
        "nonZeroNum mismatch: loadedIndex's nonZeroNum = %d, current index's nonZeroNum = %d\n",
        loadedIndex.dims2, this->dims2);
    FAISS_THROW_IF_NOT_FMT(loadedIndex.intf_->d == this->intf_->d,
        "dim mismatch: loadedIndex's dim = %d, current index's dim = %d\n", loadedIndex.intf_->d, this->intf_->d);
    // 校验 loadedIndex和当前index的所有deviceId是否匹配; 当前版本deviceList应该只有一个id
    FAISS_THROW_IF_NOT_MSG(this->indexConfig.deviceList.size() == loadedIndex.indexConfig.deviceList.size(),
        "AscendIndexIVFSPSQImpl addCodeBook: mismatched device counts between loaded index and current index.\n");
    for (size_t i = 0; i < this->indexConfig.deviceList.size(); i++) {
        int deviceId = indexConfig.deviceList[i];
        FAISS_THROW_IF_NOT_MSG(deviceId == loadedIndex.indexConfig.deviceList[i],
            "AscendIndexIVFSPSQImpl addCodeBook: mismatched device id between loaded index and current index.\n");
    }
}

void AscendIndexIVFSPSQImpl::loadAllData(const char *dataPath, const AscendIndexIVFSPSQImpl &loadedIndex)
{
    FAISS_THROW_IF_NOT_MSG((dataPath != nullptr && strlen(dataPath) != 0), "dataPath is invalid.");
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation started.\n");
    checkSharedCodebookParams(loadedIndex);
    codeBook = loadedIndex.codeBook;
    spSq.trained = loadedIndex.spSq.trained;
    for (auto &index : indexMap) {
        uint32_t size = 0;
        int deviceId = 0;
        for (const auto &deviceCtx: contextMap) { // get the corresponding device id of the context in current index
            if (deviceCtx.second == index.first) {
                deviceId = deviceCtx.first;
                break;
            }
        }
        auto ctxLoaded = loadedIndex.contextMap.at(deviceId); // get the ctx of loadedIndex with deviceId
        auto indexIdLoaded = loadedIndex.indexMap.at(ctxLoaded); // get the indexId of the loadedIndex with the context
        RpcError ret = RpcIndexIVFSPSQLoadAllData(index.first, index.second, ctxLoaded, indexIdLoaded, dataPath);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQLoadAllData failed(%d).", ret);
        RpcIndexIVFSPSQGetBaseSize(index.first, index.second, size);
        this->intf_->ntotal = size;
        this->intf_->is_trained = true;
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::loadAllData(const uint8_t* data, size_t dataLen)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation started.\n");
    for (auto &index : indexMap) {
        uint32_t size = 0;
        codeBook->resize(this->dims2 * this->nlist * this->intf_->d);
        spSq.trained.resize(this->dims2 + this->dims2);
        RpcError ret = RpcIndexIVFSPSQLoadAllData(index.first, index.second, data, dataLen,
            codeBook->data(), spSq.trained.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQLoadAllData failed(%d).", ret);
        RpcIndexIVFSPSQGetBaseSize(index.first, index.second, size);
        this->intf_->ntotal = size;
        this->intf_->is_trained = true;
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::loadAllData(const uint8_t* data, size_t dataLen, const AscendIndexIVFSPSQImpl &loadedIndex)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation started.\n");
    checkSharedCodebookParams(loadedIndex);
    codeBook = loadedIndex.codeBook;
    spSq.trained = loadedIndex.spSq.trained;
    for (auto &index : indexMap) {
        uint32_t size = 0;
        int deviceId = 0;
        for (const auto &deviceCtx: contextMap) { // get the corresponding device id of the context in current index
            if (deviceCtx.second == index.first) {
                deviceId = deviceCtx.first;
                break;
            }
        }
        auto ctxLoaded = loadedIndex.contextMap.at(deviceId); // get the ctx of loadedIndex with deviceId
        auto indexIdLoaded = loadedIndex.indexMap.at(ctxLoaded); // get the indexId of the loadedIndex with the context
        RpcError ret = RpcIndexIVFSPSQLoadAllData(index.first, index.second, ctxLoaded, indexIdLoaded, data, dataLen);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQLoadAllData failed(%d).", ret);
        RpcIndexIVFSPSQGetBaseSize(index.first, index.second, size);
        this->intf_->ntotal = size;
        this->intf_->is_trained = true;
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::loadCodeBookOnly(const uint8_t* data, size_t dataLen)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadCodeBookOnly operation started.\n");
    // 由于对offset的填充始终都是0到(nlist - 1)的整数(参考addCodeBook接口)，此处为了减少冗余逻辑，直接对offset直接赋值
    // 如果之后对offset逻辑不同，此处需要修改!
    codeOffset.resize(nlist, 0);
    std::iota(codeOffset.begin(), codeOffset.end(), 0);
    codeBook->resize(this->dims2 * this->nlist * this->intf_->d);
    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFSPSQLoadCodeBook(index.first, index.second, data, dataLen, codeBook->data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQ loadCodeBookOnly failed(%d).", ret);
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl loadCodeBookOnly operation finished.\n");
}

void AscendIndexIVFSPSQImpl::saveAllData(const char *dataPath)
{
    FAISS_THROW_IF_NOT_MSG((dataPath != nullptr && strlen(dataPath) != 0), "dataPath is invalid.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    APP_LOG_INFO("AscendIndexIVFSPSQImpl saveAllData operation started.\n");
    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFSPSQSaveAllData(index.first, index.second,
            dataPath, codeBook->data(), spSq.trained.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQASaveAllData failed(%d).", ret);
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl saveAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::saveAllData(uint8_t* &data, size_t &dataLen)
{
    FAISS_THROW_IF_NOT_MSG(codebookFinished, "Index does not have codebook added.\n");
    APP_LOG_INFO("AscendIndexIVFSPSQImpl saveAllData operation started.\n");
    for (auto &index : indexMap) {
        if (!this->intf_->is_trained) { // 仅有码本但未添加底库，因此仅序列化一个仅有码本的索引
            RpcError ret = RpcIndexIVFSPSQSaveCodeBook(index.first, index.second,
                data, dataLen, codeBook->data());
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQASaveCodeBook failed(%d).", ret);
        } else {
            RpcError ret = RpcIndexIVFSPSQSaveAllData(index.first, index.second,
                data, dataLen, codeBook->data(), spSq.trained.data());
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQASaveAllData failed(%d).", ret);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl saveAllData operation finished.\n");
}

void AscendIndexIVFSPSQImpl::setNumProbes(int nprobes)
{
    APP_LOG_INFO("AscendIndexIVF setNumProbes operation started.\n");
    FAISS_THROW_IF_NOT_MSG(nprobes > 0, "nprobe must be greater than 0");
    FAISS_THROW_IF_NOT_MSG(nprobes <= this->nlist, "nprobe must be less than or equal to nlist");
    this->nprobe = nprobes;
    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFSPSQUpdateNprobe(index.first, index.second, nprobes);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Set NumProbes failed(%d).", ret);
    }
    APP_LOG_INFO("AscendIndexIVF setNumProbes operation finished.\n");
}

void AscendIndexIVFSPSQImpl::train(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");
        return;
    }

    spSq.train(n, x);
    updateDeviceSPSQTrainedValue();
    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl train operation finished.\n");
}

void AscendIndexIVFSPSQImpl::updateDeviceSPSQTrainedValue()
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl updateDeviceSPSQTrainedValue operation started.\n");
    // convert trained value to fp16, contain vmin and vdiff, so need *2
    std::vector<uint16_t> trainedFp16(this->dims2 * 2);
    uint16_t *vmin = trainedFp16.data();
    uint16_t *vdiff = trainedFp16.data() + this->dims2;

    switch (spSq.qtype) {
        case faiss::ScalarQuantizer::QT_8bit:
            transform(spSq.trained.begin(), spSq.trained.end(), trainedFp16.begin(),
                      [](float temp) { return fp16(temp).data; });
            break;
        case faiss::ScalarQuantizer::QT_8bit_uniform:
            for (int i = 0; i < this->dims2; i++) {
                *(vmin + i) = fp16(spSq.trained[0]).data;
                *(vdiff + i) = fp16(spSq.trained[1]).data;
            }
            break;
        default:
            FAISS_THROW_FMT("not supportted qtype(%d).", spSq.qtype);
            break;
    }

    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFSPSQUpdateTrainedValue(index.first, index.second, this->dims2, vmin, vdiff, true);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update trained value failed(%d).", ret);
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl updateDeviceSPSQTrainedValue operation finished.\n");
}

void AscendIndexIVFSPSQImpl::searchImpl(int n, const float* x, int k, float* distances,
    idx_t* labels) const
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl searchImpl operation started: n=%d, k=%d.\n", n, k);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "x cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distance cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    checkAndSetAddFinish();
    size_t deviceCnt = indexConfig.deviceList.size();

    // if n is small, transform is better than tik ops
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret;
        ret = RpcIndexSearch(ctx, indexId, n, this->intf_->d, k, query.data(),
                             distHalf[idx].data(), label[idx].data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]),
                  begin(dist[idx]), [](uint16_t temp) { return (float)fp16(temp); });
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndex searchImpl operation finished.\n");
}

void AscendIndexIVFSPSQImpl::search_with_masks(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_masks operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "x cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distance cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(mask, "mask cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    checkAndSetAddFinish();

    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        // +7 for div up to 8
        RpcError ret = RpcIndexSearchFilter(ctx, indexId, n, this->intf_->d, k, query.data(), distHalf[idx].data(),
            label[idx].data(), (this->intf_->ntotal + 7) / 8, static_cast<const uint8_t *>(mask));
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search filter implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                  [](uint16_t temp) { return (float)fp16(temp); });
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_masks operation finished.\n");
}

void AscendIndexIVFSPSQImpl::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexIVF searchPostProcess operation started.\n");
    mergeSearchResult(devices, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexIVF searchPostProcess operation finished.\n");
}

void AscendIndexIVFSPSQImpl::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_filter operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "x cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distance cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(filters, "mask cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(indexConfig.filterable, "search filter not support as sqconfig.filterable = false");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    checkAndSetAddFinish();

    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        RpcError ret = RpcIndexSearchFilter(ctx, indexId, n, this->intf_->d, k, query.data(), distHalf[idx].data(),
                                            label[idx].data(), n * FILTER_SIZE, static_cast<const uint32_t *>(filters));
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search filter implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                  [](uint16_t temp) { return (float)fp16(temp); });
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_filter operation finished.\n");
}

void AscendIndexIVFSPSQImpl::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, float *l1distances, const void *filters) const
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_filter operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "x cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distance cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(filters, "mask cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(indexConfig.filterable, "search filter not support as sqconfig.filterable = false");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    checkAndSetAddFinish();

    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });
    std::vector<uint16_t> l1dists(n * this->nCentroid, 0);
    transform(l1distances, l1distances + n * this->nCentroid,
        begin(l1dists), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexSearchFilter(ctx, indexId, n, this->intf_->d, k, query.data(), distHalf[idx].data(),
                                            label[idx].data(), n * FILTER_SIZE, static_cast<const uint32_t *>(filters));
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search filter implement failed(%d).", ret);

        // convert result data from fp16 to float

        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                  [](uint16_t temp) { return (float)fp16(temp); });
        if (*l1distances > 0) {
        } else {
            transform(begin(l1dists), end(l1dists), l1distances, [](uint16_t temp) { return (float)fp16(temp);});
        }
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);

    APP_LOG_INFO("AscendIndexIVFSPSQImpl search_with_filter operation finished.\n");
}

void AscendIndexIVFSPSQImpl::reset()
{
    APP_LOG_INFO("AscendIndexIVFSPSQ reset operation started.\n");
    if (this->intf_->is_trained) {
        for (auto &data : indexMap) {
            RpcError ret = RpcIndexReset(data.first, data.second);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset Index failed(%d).", ret);
        }
        initDeviceAddNumMap();
        this->intf_->ntotal = 0;
        this->intf_->is_trained = false;
    }

    APP_LOG_INFO("AscendIndexIVFSPSQ reset operation finished.\n");
}

void AscendIndexIVFSPSQImpl::trainCodeBook(const AscendIndexCodeBookTrainerConfig &codeBookTrainerConfig)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl trainCodeBook operation started.\n");

    RpcIndexCodeBookTrainerConfig rpcConfig;
    rpcConfig.numIter = codeBookTrainerConfig.numIter;
    rpcConfig.device = codeBookTrainerConfig.device;
    rpcConfig.ratio = codeBookTrainerConfig.ratio;
    rpcConfig.batchSize = codeBookTrainerConfig.batchSize;
    rpcConfig.codeNum = codeBookTrainerConfig.codeNum;
    rpcConfig.codeBookOutputDir = codeBookTrainerConfig.codeBookOutputDir;
    rpcConfig.learnDataPath = codeBookTrainerConfig.learnDataPath;
    rpcConfig.memLearnData = codeBookTrainerConfig.memLearnData;
    rpcConfig.memLearnDataSize = codeBookTrainerConfig.memLearnDataSize;
    rpcConfig.verbose = codeBookTrainerConfig.verbose;
    for (auto &index : indexMap) {
        if (codeBookTrainerConfig.trainAndAdd) {
            std::vector<float> codeBookData(this->intf_->d * dims2 * nlist);
            RpcError ret = RpcIndexIVFSPSQTrainCodeBook(index.first, index.second, rpcConfig, codeBookData.data());
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQGetCodeBook failed(%d).", ret);
            std::vector<faiss::idx_t> offset(nlist, 0);
            std::iota(offset.begin(), offset.end(), 0);
            AscendIndexIVFSPSQImpl::addCodeBook(nlist * dims2, this->intf_->d, codeBookData.data(), offset.data());
        } else {
            RpcError ret = RpcIndexIVFSPSQTrainCodeBook(index.first, index.second, rpcConfig);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQTrainCodeBook failed(%d).", ret);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl trainCodeBook operation finished.\n");
}

void AscendIndexIVFSPSQImpl::addCodeBook(int n, int dim, const float *x, idx_t *offset)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook started with %d vector(s).\n", n);

    if (n != this->nlist * this->dims2) {
        FAISS_THROW_MSG("codeBook Size is not correct! Please check your codebook or call addCodeBook first!");
    }
    FAISS_THROW_IF_NOT_MSG(dim == this->intf_->d, "dim must be equal to dimension of origin data");
    FAISS_THROW_IF_NOT(n > 0);

    if (!codebookFinished) {
        codeBook->resize(n * dim);
        codeOffset.resize(n / dims2);
        auto err = memcpy_s(codeBook->data(), n * dim * sizeof(float), x, n * dim * sizeof(float));
        FAISS_THROW_IF_NOT(err == EOK);
        err = memcpy_s(codeOffset.data(), n / dims2 * sizeof(idx_t), offset, n / dims2 * sizeof(idx_t));
        FAISS_THROW_IF_NOT(err == EOK);
    }

    size_t devIdx = 0;
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    for (size_t i = 1; i < deviceCnt; i++) {
        if (idxDeviceMap[i].size() < idxDeviceMap[devIdx].size()) {
            devIdx = i;
            break;
        }
    }
    for (size_t i = 0; i < deviceCnt; i++) {
        addMap[i] += n / deviceCnt;
    }
    for (size_t i = 0; i < n % deviceCnt; i++) {
        addMap[devIdx % deviceCnt] += 1;
        devIdx += 1;
    }

    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook operation add: %d.\n", num * dim);

        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        std::vector<uint16_t> xb(num * dim);
        transform(x + offsum * dim, x + (offsum + num) * dim,
            std::begin(xb), [](float temp) { return fp16(temp).data; });

        RpcError ret = RpcIndexIVFSPSQAddCodeBook(ctx, indexId, num, dim, this->dims2, xb.data(), offset);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add code book failed(%d).", ret);

        offsum += num;
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook finished.\n");
}

void AscendIndexIVFSPSQImpl::getBaseImpl(int, int, int, char *) const
{
}

void AscendIndexIVFSPSQImpl::addCodeBook(const AscendIndexIVFSPSQImpl &loadedIndex)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook started with an "
            "already-loaded AscendIndexIVFSPSQImpl instance.\n");
    checkSharedCodebookParams(loadedIndex);
    this->codeBook = loadedIndex.codeBook;
    this->codeOffset = loadedIndex.codeOffset;

    size_t deviceCnt = indexConfig.deviceList.size();
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        rpcContext ctxLoaded = loadedIndex.contextMap.at(deviceId);
        int indexIdLoaded = loadedIndex.indexMap.at(ctxLoaded);
        RpcError ret = RpcIndexIVFSPSQAddCodeBook(ctx, indexId, ctxLoaded, indexIdLoaded);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add code book failed(%d).", ret);
    }
    codebookFinished = true;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook finished.\n");
}

void AscendIndexIVFSPSQImpl::add(idx_t n, const float* feature,
    const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl add operation started.\n");
    oriFeature = true;
    add_with_ids(n, feature, ids);
    APP_LOG_INFO("AscendIndexIVFSPSQImpl add operation finished.\n");
}

void AscendIndexIVFSPSQImpl::addFinish() const
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl add_finish operation started.\n");
    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFSPSQAddFinish(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexIVFSPSQAddFinish failed(%d).", ret);
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl add_finish operation finished.\n");
}

void AscendIndexIVFSPSQImpl::getCodeword(int n, const float *feature, float *, idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl getCodeword operation started: n=%d.\n", n);
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<std::vector<uint16_t>> codeWords(deviceCnt, std::vector<uint16_t>(n * dims2, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexIVFSPSQGetCodeWord(ctx, indexId, n, this->intf_->d, feature,
                                                  codeWords[idx].data(), ids);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "getCodeword failed(%d).", ret);
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    APP_LOG_INFO("AscendIndex getCodeword operation finished.\n");
}

void AscendIndexIVFSPSQImpl::add_with_ids(idx_t n, const float* x,
    const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl add_with_ids operation started.\n");
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + n < MAX_N, "ntotal must be < %ld", MAX_N);

    std::vector<idx_t> tmpIds;
    if (ids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<idx_t>(n);

        for (idx_t i = 0; i < n; ++i) {
            tmpIds[i] = this->intf_->ntotal + i;
        }

        ids = tmpIds.data();
    }

    addPaged(n, x, ids);

    APP_LOG_INFO("AscendIndexIVFSPSQImpl add_with_ids operation finished.\n");
}


void AscendIndexIVFSPSQImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addPaged operation started.\n");
    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    if (totalSize > ADD_PAGE_SIZE || static_cast<size_t>(n) > ADD_VEC_SIZE) {
        size_t tileSize = getAddPagedSize(n);

        for (size_t i = 0; i < (size_t)n; i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose) {
                printf("AscendIndexIVFSPSQImpl::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            if (oriFeature) {
                addCodeImpl(curNum, x + i * (size_t)this->intf_->d,
                            ids ? (ids + i) : nullptr);
            } else {
                addImpl(curNum, x + i * (size_t)dims2,
                        ids ? (ids + i) : nullptr);
            }
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndexIVFSPSQImpl::add: adding %zu:%zu / %d\n",
                this->intf_->ntotal, this->intf_->ntotal + n, n);
        }
        if (oriFeature) {
            addCodeImpl(n, x, ids);
        } else {
            addImpl(n, x, ids);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addPaged operation finished.\n");
}

void AscendIndexIVFSPSQImpl::addCodeImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeImpl operation started with %d vector(s).\n", n);
    FAISS_THROW_IF_NOT(n > 0);

    if (codeBook->size() != static_cast<uint32_t>(nlist * this-> dims2 * this->intf_->d)) {
        FAISS_THROW_MSG("codeBook Size is not correct! Please check your codebook or call addCodeBook first!");
    }

    bool useNPU = (n >= USE_NPU_ADD_LIMIT) ? true : false;

    // 1. compute addMap
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // calculate which list the vectors belongs to
    std::vector<idx_t> assign(n, 0);

    std::vector<float> codeWord(n * dims2);
    getCodeword(n, x, codeWord.data(), assign.data());

    std::map<int, std::vector<idx_t>> idsCounts;
    std::map<int, std::vector<float>> codeCounts;
    for (int i = 0; i < n; i++) {
        idx_t listId = assign[i];

        FAISS_THROW_IF_NOT(listId >= 0 && listId < nlist);
        idsCounts[listId].emplace_back(ids[i]);
        size_t originLen = codeCounts[listId].size();
        codeCounts[listId].resize(originLen + this->intf_->d);

        auto err = memcpy_s(codeCounts[listId].data() + originLen, this->intf_->d * sizeof(float),
            x + i * this->intf_->d, this->intf_->d * sizeof(float));
        FAISS_THROW_IF_NOT(err == EOK);
    }

    std::vector<float> codeWordCounts(n * this->dims2);
    
    std::vector<int> idsCountsKeys;
    for (const auto& pair : idsCounts) {
        idsCountsKeys.push_back(pair.first);
    }
    std::vector<int> accumOffset(idsCountsKeys.size(), 0);
    std::vector<int> accumOffsetCodeWord(idsCountsKeys.size(), 0);
    for (size_t i = 1; i < idsCountsKeys.size(); ++i) {
        std::vector<idx_t>& bucket = idsCounts[idsCountsKeys[i - 1]];
        accumOffset[i] = accumOffset[i - 1] + static_cast<int>(bucket.size());
        accumOffsetCodeWord[i] = accumOffset[i] * this->dims2;
    }

#pragma omp parallel for
    for (size_t i = 0; i < idsCountsKeys.size(); ++i) {
        int listId = idsCountsKeys[i];
        std::vector<idx_t>& bucket = idsCounts[listId];
        int bucketSize = static_cast<int>(bucket.size());
        float *codeBookData = codeBook->data() + listId * this->dims2 * this->intf_->d;
        float *codeData = codeWordCounts.data() + accumOffsetCodeWord[i];
        matMul(codeData, codeCounts[listId].data(), codeBookData,
               bucketSize, this->intf_->d, this->dims2, true);
    }

    const float *xi = codeWordCounts.data();
    if (!this->intf_->is_trained) {
        train(n, xi);
    }

    // 2. compute the spSq codes
    auto codes = std::make_unique<uint8_t[]>(n * spSq.code_size);
    spSq.compute_codes(xi, codes.get(), n);
    // 3. compute the preCompute values
    auto precomputeVals = std::make_unique<float[]>(n);

#pragma omp parallel for
    for (size_t i = 0; i < idsCountsKeys.size(); ++i) {
        int listId = idsCountsKeys[i];
        std::vector<idx_t>& bucket = idsCounts[listId];
        int bucketSize = static_cast<int>(bucket.size());
        float *codeBookData = codeBook->data() + listId * this->dims2 * this->intf_->d;
        float *codeData = codeWordCounts.data() + accumOffset[i] * this->dims2;
        std::vector<float> cbData(bucketSize * this->intf_->d); // n*256
        matMul(cbData.data(), codeData, codeBookData,
               bucketSize, this->dims2, this->intf_->d);
        std::vector<float> preCompute(bucketSize);
        fvec_norms_L2sqr(preCompute.data(), cbData.data(), this->intf_->d, bucketSize);
        auto err = memcpy_s(precomputeVals.get() + accumOffset[i], bucketSize * sizeof(float),
            preCompute.data(), bucketSize * sizeof(float));
        FAISS_THROW_IF_NOT(err == EOK);
    }

    int deviceId = indexConfig.deviceList[0];
    rpcContext ctx = contextMap[deviceId];
    int indexId = indexMap[ctx];
#pragma omp parallel for if (!useNPU)
    for (size_t i = 0; i < idsCountsKeys.size(); ++i) {
        int listId = idsCountsKeys[i];
        std::vector<idx_t>& bucket = idsCounts[listId];
        int bucketSize = static_cast<int>(bucket.size());
        if (bucketSize == 0) {
            continue;
        }
        uint8_t *codePtr = codes.get() + accumOffset[i] * spSq.code_size;
        ascend_idx_t *idPtr = reinterpret_cast<ascend_idx_t *>(bucket.data());
        float *precompPtr = precomputeVals.get() + accumOffset[i];

        RpcError ret = RpcIndexIVFSPSQAddWithIds(ctx, indexId, bucketSize,
            spSq.code_size, listId, codePtr, idPtr, precompPtr, useNPU);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
        deviceAddNumMap[listId][0] += bucketSize;
    }

    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addImpl operation finished.\n");
}

void AscendIndexIVFSPSQImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

void AscendIndexIVFSPSQImpl::matMul(float *dst, const float *c, const float *b,
    size_t n, size_t k, size_t m, bool transpose)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            double res = 0;
            for (size_t p = 0; p < k; p++) {
                if (transpose) {
                    res += c[p + i * k] * b[p + k * j];
                } else {
                    res += c[p + i * k] * b[p * m + j];
                }
            }
            dst[j + i * m] = res;
        }
    }
}

size_t AscendIndexIVFSPSQImpl::getAddElementSize() const
{
    return this->spSq.code_size * sizeof(uint8_t);
}

size_t AscendIndexIVFSPSQImpl::getBaseElementSize() const
{
    return this->getAddElementSize();
}

size_t AscendIndexIVFSPSQImpl::remove_ids(const std::vector<idx_t> sel)
{
    // remove底库ID，预计算
    APP_LOG_INFO("AscendIndexIVFSPSQImpl remove_ids operation started.\n");
    if (!this->intf_->is_trained) {
        return 0;
    }
    if (sel.size() == 0) {
        APP_LOG_INFO("remove vec size is 0, no vectors will be removed!\n");
        return 0;
    }
    size_t deviceCnt = indexConfig.deviceList.size();
    uint32_t removeCnt = 0;
    
    // remove vector from device
    size_t removeSize = sel.size();
    FAISS_THROW_IF_NOT_FMT(removeSize <= static_cast<size_t>(this->intf_->ntotal),
        "the size of removed codes should be in range [0, %ld], actual=%zu.",
        this->intf_->ntotal, removeSize);
    std::vector<ascend_idx_t> removeBatch(removeSize);
    transform(begin(sel), end(sel), begin(removeBatch),
              [](idx_t temp) { return (ascend_idx_t)temp; });

#pragma omp parallel for reduction(+ : removeCnt)
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        
        RpcError ret = RpcIndexIVFSPSQRemoveIds(ctx, indexId, removeBatch.size(),
            removeBatch.data(), &removeCnt);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID selector failed(%d).", ret);
    }

    // update vector nums of deviceAddNumMap
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap[deviceId];
        int indexId = indexMap.at(ctx);
        for (int listId = 0; listId < nlist; listId++) {
            uint32_t len = 0;
            RpcError ret = RpcIndexIVFSPSQGetListLength(ctx, indexId, listId, len);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE,
                "IVFSPSQ get list length failed(%d).", ret);
            deviceAddNumMap[listId][i] = len;
        }
    }
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal >= removeCnt,
        "removeCnt should be in range [0, %ld], actual=%d.", this->intf_->ntotal, removeCnt);
    this->intf_->ntotal -= static_cast<faiss::idx_t>(removeCnt);
    APP_LOG_INFO("AscendIndexIVFSPSQImpl remove_ids operation finished.\n");
    return (size_t)removeCnt;
}

size_t AscendIndexIVFSPSQImpl::remove_ids(const idx_t minRange, const idx_t maxRange)
{
    // remove底库ID，预计算
    APP_LOG_INFO("AscendIndexIVFSPSQImpl remove_ids operation started.\n");
    if (!this->intf_->is_trained) {
        return 0;
    }
    FAISS_THROW_IF_NOT_MSG(maxRange >= minRange, "maxRange should be larger than minRange!");
    size_t deviceCnt = indexConfig.deviceList.size();
    uint32_t removeCnt = 0;

#pragma omp parallel for reduction(+ : removeCnt)
        for (size_t i = 0; i < deviceCnt; i++) {
            int deviceId = indexConfig.deviceList[i];
            rpcContext ctx = contextMap[deviceId];
            int indexId = indexMap.at(ctx);
            RpcError ret = RpcIndexIVFSPSQRemoveRangeIds(ctx, indexId, minRange, maxRange, &removeCnt);

            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID range failed(%d).", ret);
        }

    // update vector nums of deviceAddNumMap
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap[deviceId];
        int indexId = indexMap.at(ctx);
        for (int listId = 0; listId < nlist; listId++) {
            uint32_t len = 0;
            RpcError ret = RpcIndexIVFSPSQGetListLength(ctx, indexId, listId, len);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "IVFSPSQ get list length failed(%d).", ret);
            deviceAddNumMap[listId][i] = len;
        }
    }
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal >= removeCnt,
        "removeCnt should be in range [0, %ld], actual=%d.", this->intf_->ntotal, removeCnt);
    this->intf_->ntotal -= static_cast<faiss::idx_t>(removeCnt);
    APP_LOG_INFO("AscendIndexIVFSPSQImpl remove_ids operation finished.\n");
    return (size_t)removeCnt;
}

void AscendIndexIVFSPSQImpl::addCodeBook(const std::string &path)
{
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook begin.\n");
    std::vector<char> allData;

    struct stat codeBookStat;
    if (path.c_str() && lstat(path.c_str(), &codeBookStat) == 0) {
        if (!S_ISREG(codeBookStat.st_mode)) {
            FAISS_THROW_MSG("codebook file is not a regular file.\n");
        } else if (codeBookStat.st_size > CODE_BOOK_MAX_FILE_SIZE || codeBookStat.st_uid != geteuid()) {
            FAISS_THROW_FMT("codebook file is too big(bigger than %ld) or its owner is not the execution user.\n",
                CODE_BOOK_MAX_FILE_SIZE);
        }
        APP_LOG_INFO("start to read codebook.\n");

        ::ascendSearch::FSPIOReader codeBookReader(path);
        size_t dataLen = codeBookReader.GetFileSize();
        if (dataLen != 19 + 64 + this->intf_->d * dims2 * nlist * sizeof(float)) { // 19 and 64 is len of data head
            FAISS_THROW_MSG("codebook length is not correct!");
        }
        allData.resize(dataLen, true);
        codeBookReader.ReadAndCheck(allData.data(), dataLen);

        char *allDataPtr = allData.data();
        char fourcc[4] = {'C', 'D', 'B', 'K'};
        // 4 means CDBK
        for (int i = 0; i < 4; i++) {
            if (*(allDataPtr + i) != fourcc[i]) {
                FAISS_THROW_MSG("codebook format is not correct!"); // raise error
            }
        }
        // 4 means start of versionMajor
        uint8_t *versionMajor = reinterpret_cast<uint8_t*>(allDataPtr + 4);
        // 5 means start of versionMajor
        uint8_t *versionMedium = reinterpret_cast<uint8_t*>(allDataPtr + 5);
        // 6 means start of versionMajor
        uint8_t *versionMinor = reinterpret_cast<uint8_t*>(allDataPtr + 6);
        if ((*versionMajor != 1) || (*versionMedium != 0) || (*versionMinor != 0)) {
            FAISS_THROW_MSG("codebook version is not correct!"); // raise error
        }
        int *dim_ = reinterpret_cast<int*>(allDataPtr + 7); // dim
        int *dim2_ = reinterpret_cast<int*>(allDataPtr + 11); // dim2
        int *ncentroids_ = reinterpret_cast<int*>(allDataPtr + 15); // ncentroids
        if ((*dim_ != this->intf_->d) || (*dim2_ != dims2) || (*ncentroids_ != nlist)) {
            FAISS_THROW_MSG("codebook shape is not correct!"); // raise error
        }

        std::vector<float> codeBookData(this->intf_->d * dims2 * nlist);
        // 19 + 64 means the start of real codebook
        auto err = memcpy_s(codeBookData.data(), this->intf_->d * dims2 * nlist * sizeof(float),
            reinterpret_cast<float*>(allDataPtr + 19 + 64), this->intf_->d * dims2 * nlist * sizeof(float));
        FAISS_THROW_IF_NOT(err == EOK);

        // codes offset
        std::vector<faiss::idx_t> offset(nlist, 0);
        std::iota(offset.begin(), offset.end(), 0);

        AscendIndexIVFSPSQImpl::addCodeBook(nlist * dims2, this->intf_->d, codeBookData.data(), offset.data());
    } else {
        FAISS_THROW_MSG("add codebook error!");
    }
    APP_LOG_INFO("AscendIndexIVFSPSQImpl addCodeBook finished.\n");
}

int AscendIndexIVFSPSQImpl::getDims() const
{
    return intf_->d;
}

int AscendIndexIVFSPSQImpl::getDims2() const
{
    return dims2;
}

int AscendIndexIVFSPSQImpl::getNumList() const
{
    return nlist;
}

bool AscendIndexIVFSPSQImpl::getFilterable() const
{
    return spSqConfig.filterable;
}

/* This method checks whether addFinish has been called and call it if necessary */
void AscendIndexIVFSPSQImpl::checkAndSetAddFinish() const
{
    bool isAddFinish = false;
    for (auto &index : indexMap) {
        int ret = RpcIndexGetAddFinish(index.first, index.second, isAddFinish);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexGetAddFinish failed(%d).", ret);
        if (!isAddFinish) {
            APP_LOG_INFO("addFinish has not been called, so we call it before searching.\n");
            ret = RpcIndexIVFSPSQAddFinish(index.first, index.second);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "RpcIndexAddFinish failed(%d).", ret);
        }
    }
}

} // ascend
} // faiss
