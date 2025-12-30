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


#include "AscendIndexIVFSPImpl.h"

#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <string>
#include <omp.h>
#include "common/threadpool/AscendThreadPool.h"
#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/AscendUtils.h"

namespace faiss {
namespace ascend {
namespace {
// Default dim in case of nullptr index
const int64_t CODE_BOOK_MAX_FILE_SIZE = 0x80000000; // 2048MB
const int NPROBE_BASE_FACTOR = 16;
}

using faiss::ascendSearch::AscendIndexIVFSPSQ;

std::vector<std::shared_mutex> AscendIndexIVFSPImpl::mtxVec(::ascend::MAX_DEVICEID);

AscendIndexIVFSPImpl::AscendIndexIVFSPImpl(std::shared_ptr<AscendIndexIVFSP> intf,
    std::shared_ptr<AscendIndexIVFSPSQ> ivfspsq, int dims, int nonzeroNum,
    int nlist, faiss::ScalarQuantizer::QuantizerType, faiss::MetricType metric,
    const AscendIndexIVFSPConfig &config)
    : AscendIndexImpl(dims, metric, config, intf.get(), false), ivfspNonzeroNum(nonzeroNum), ivfspNList(nlist),
    ivfspConfig(config), pIVFSPSQ(ivfspsq)
{
    intf->is_trained = true;
    this->intf_ = intf.get();
}

AscendIndexIVFSPImpl::AscendIndexIVFSPImpl(AscendIndexIVFSP *intf, int dims, int nonzeroNum,
    int nlist, faiss::ScalarQuantizer::QuantizerType qType,
    faiss::MetricType metric, const AscendIndexIVFSPConfig &config)
    : AscendIndexImpl(dims, metric, config, intf, false), ivfspNonzeroNum(nonzeroNum), ivfspNList(nlist),
    ivfspConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(qType == ScalarQuantizer::QuantizerType::QT_8bit,
        "only support ScalarQuantizer::QuantizerType::QT_8bit.");
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_L2, "only support MetricType::METRIC_L2.");
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size() == 1, "the size of deviceList must be 1.");
    FAISS_THROW_IF_NOT_MSG(config.nprobe > 0 && config.nprobe <= nlist,
        "config.nprobe must be greater than 0, and be less than or equal to nlist");
    FAISS_THROW_IF_NOT_FMT(config.nprobe % NPROBE_BASE_FACTOR == 0,
        "config.nprobe need to be a multiple of %d.", NPROBE_BASE_FACTOR);

    // 前面保证了deviceList至少有一个数
    uint32_t deviceId = static_cast<uint32_t>(config.deviceList.front());
    FAISS_THROW_IF_NOT_FMT(deviceId < ::ascend::MAX_DEVICEID, "device id %u should be in [0, %u)",
        deviceId, ::ascend::MAX_DEVICEID);
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[deviceId]);

    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig(config.deviceList, config.resourceSize);
    ivfspsqConfig.filterable = config.filterable;
    ivfspsqConfig.handleBatch = config.handleBatch;
    ivfspsqConfig.nprobe = config.nprobe;
    ivfspsqConfig.searchListSize = config.searchListSize;

    pIVFSPSQ = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(dims, nonzeroNum, nlist, nlist, qType,
       metric, false, ivfspsqConfig);

    intf->is_trained = false;
    this->intf_ = intf;
}

void AscendIndexIVFSPImpl::addCodeBook(const AscendIndexIVFSPImpl &codeBookSharedImpl)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("start to share codebook.\n");
    // pIVFSPSQ在构造中创造，无需判断非空
    pIVFSPSQ->addCodeBook(*codeBookSharedImpl.pIVFSPSQ);
    intf_->is_trained = true;
    APP_LOG_INFO("finished to share codebook.\n");
}

void AscendIndexIVFSPImpl::addCodeBook(const char *codeBookPath)
{
    FAISS_THROW_IF_NOT_MSG((codeBookPath != nullptr), "codeBookPath can not be nullptr");
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("start to read codebook.\n");
    std::string cdbkFilePath(codeBookPath);
    pIVFSPSQ->addCodeBook(cdbkFilePath);
    intf_->is_trained = true;
    APP_LOG_INFO("finished to read codebook.\n");
}

AscendIndexIVFSPImpl::~AscendIndexIVFSPImpl()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    pIVFSPSQ.reset();
}

void AscendIndexIVFSPImpl::add(idx_t n, const float *x)
{
    add_with_ids(n, x, nullptr);
}

void AscendIndexIVFSPImpl::add_with_ids(idx_t n, const float *x, const idx_t *ids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("AscendIndexIVFSP add_with_ids operation started with %d vector(s).\n", n);
    FAISS_THROW_IF_NOT(n > 0);
    pIVFSPSQ->add(n, x, ids);
    this->intf_->ntotal = pIVFSPSQ->ntotal;
    APP_LOG_INFO("AscendIndexIVFSP add_with_ids operation finished.\n");
}

void AscendIndexIVFSPImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

std::shared_ptr<::ascend::Index> AscendIndexIVFSPImpl::createIndex(int)
{
    FAISS_THROW_MSG("not implement now");
    return nullptr;
}

size_t AscendIndexIVFSPImpl::remove_ids(const faiss::IDSelector &sel)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("AscendIndexIVFSP removeImpl operation started.\n");

    size_t nremove = 0;
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        std::vector<idx_t> removeBatch(rangeSel->set.begin(), rangeSel->set.end());
        nremove = pIVFSPSQ->remove_ids(removeBatch);
    } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
        nremove = pIVFSPSQ->remove_ids(rangeSel->imin, rangeSel->imax);
    } else {
        APP_LOG_WARNING("Invalid IDSelector.\n");
        return 0;
    }

    this->intf_->ntotal = pIVFSPSQ->ntotal;
    APP_LOG_INFO("AscendIndexIVFSP removeImpl operation finished.\n");
    return nremove;
}

void AscendIndexIVFSPImpl::reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("AscendIndexIVFSP reset operation started.\n");
    pIVFSPSQ->reset();
    this->intf_->ntotal = pIVFSPSQ->ntotal;
    APP_LOG_INFO("AscendIndexIVFSP reset operation finished.\n");
}

void AscendIndexIVFSPImpl::loadAllData(const char *dataPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    APP_LOG_INFO("AscendIndexIVFSP loadAllData operation started.\n");
    pIVFSPSQ->loadAllData(dataPath);
    this->intf_->ntotal = pIVFSPSQ->ntotal;
    // 加载保存的落盘数据是含有码本信息的，所以置is_trained = true
    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFSP loadAllData operation finished.\n");
}

void AscendIndexIVFSPImpl::saveAllData(const char *dataPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    pIVFSPSQ->saveAllData(dataPath);
}

void AscendIndexIVFSPImpl::trainCodeBook(const AscendIndexCodeBookInitParams &codeBookInitParams) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    faiss::ascendSearch::AscendIndexCodeBookTrainerConfig config;
    config.numIter = codeBookInitParams.numIter;
    config.device = codeBookInitParams.device;
    config.ratio = codeBookInitParams.ratio;
    config.batchSize = codeBookInitParams.batchSize;
    config.codeNum = codeBookInitParams.codeNum;
    config.verbose = codeBookInitParams.verbose;
    config.codeBookOutputDir = codeBookInitParams.codeBookOutputDir;
    config.learnDataPath = codeBookInitParams.learnDataPath;
    pIVFSPSQ->trainCodeBook(config);
}

void AscendIndexIVFSPImpl::trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams
    &codeBookInitFromMemParams) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    faiss::ascendSearch::AscendIndexCodeBookTrainerConfig config;
    config.numIter = codeBookInitFromMemParams.numIter;
    config.device = codeBookInitFromMemParams.device;
    config.ratio = codeBookInitFromMemParams.ratio;
    config.batchSize = codeBookInitFromMemParams.batchSize;
    config.codeNum = codeBookInitFromMemParams.codeNum;
    config.verbose = codeBookInitFromMemParams.verbose;
    config.codeBookOutputDir = codeBookInitFromMemParams.codeBookOutputDir;
    config.memLearnData = codeBookInitFromMemParams.memLearnData;
    config.memLearnDataSize = codeBookInitFromMemParams.memLearnDataSize;
    config.trainAndAdd = codeBookInitFromMemParams.isTrainAndAdd;
    pIVFSPSQ->trainCodeBook(config);
    if (codeBookInitFromMemParams.isTrainAndAdd) {
        intf_->is_trained = true;
    }
}


void AscendIndexIVFSPImpl::CheckIndexParams(IndexImplBase &index, bool checkFilterable) const
{
    try {
        AscendIndexIVFSPImpl& ivfspImpl = dynamic_cast<AscendIndexIVFSPImpl&>(index);
        FAISS_THROW_IF_NOT_MSG(this->intf_->d == ivfspImpl.intf_->d, "the dim must be same.");
        FAISS_THROW_IF_NOT_MSG(this->ivfspConfig.deviceList == ivfspImpl.ivfspConfig.deviceList,
                               "the deviceList must be same.");
        FAISS_THROW_IF_NOT_FMT(ivfspImpl.ivfspConfig.deviceList.size() == 1,
                               "the size of deviceList (%zu) must be 1.", ivfspImpl.ivfspConfig.deviceList.size());
        FAISS_THROW_IF_NOT_MSG(this->ivfspNonzeroNum == ivfspImpl.ivfspNonzeroNum, "the nonzeroNum must be same.");
        FAISS_THROW_IF_NOT_MSG(this->ivfspNList == ivfspImpl.ivfspNList, "the nlist must be same.");
        if (checkFilterable) {
            FAISS_THROW_IF_NOT_MSG(ivfspConfig.filterable, "the index does not support filterable");
        }
    } catch (std::bad_cast &e) {
        FAISS_THROW_MSG("the type of index is not AscendIndexIVFSP.");
    }
}

void AscendIndexIVFSPImpl::setNumProbes(int nprobes)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    // IVFSP设置桶的数量nprobes为16的倍数是为了减少算子下发次数，提升性能
    FAISS_THROW_IF_NOT_FMT(nprobes % NPROBE_BASE_FACTOR == 0,
        "nprobes(%d) need to be a multiple of %d.", nprobes, NPROBE_BASE_FACTOR);
    pIVFSPSQ->setNumProbes(nprobes);
}

void AscendIndexIVFSPImpl::search(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    pIVFSPSQ->search(n, x, k, distances, labels);
}

void AscendIndexIVFSPImpl::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    pIVFSPSQ->search_with_filter(n, x, k, distances, labels, filters);
}

// only faiss::ascend::AscendIndexIVFSP
void AscendIndexIVFSPImpl::SearchMultiIndex(int deviceId, std::vector<faiss::ascendSearch::AscendIndex*> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[deviceId]);
    faiss::ascendSearch::AscendIndexIVFSPSQ::SearchMultiIndex(indexes, n, x, k, distances, labels, merged);
}

void AscendIndexIVFSPImpl::SearchWithFilterMultiIndex(int deviceId,
    std::vector<faiss::ascendSearch::AscendIndex*> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[deviceId]);
    faiss::ascendSearch::AscendIndexIVFSPSQ::SearchWithFilterMultiIndex(indexes, n, x, k,
        distances, labels, filters, merged);
}

void AscendIndexIVFSPImpl::SearchWithFilterMultiIndex(int deviceId,
    std::vector<faiss::ascendSearch::AscendIndex*> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, void *filters[], bool merged)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[deviceId]);
    faiss::ascendSearch::AscendIndexIVFSPSQ::SearchWithFilterMultiIndex(indexes, n, x, k,
        distances, labels, filters, merged);
}

faiss::ascendSearch::AscendIndexIVFSPSQ* AscendIndexIVFSPImpl::GetIVFSPSQPtr() const
{
    return this->pIVFSPSQ.get();
}

std::shared_ptr<AscendIndexIVFSPImpl> AscendIndexIVFSPImpl::loadAllData(std::shared_ptr<AscendIndexIVFSP> intf,
    const AscendIndexIVFSPConfig &config, const uint8_t *data, size_t dataLen,
    const std::shared_ptr<AscendIndexIVFSPImpl> codeBookSharedIdx)
{
    auto deviceNum = config.deviceList.size();
    FAISS_THROW_IF_NOT_FMT(deviceNum == 1, "the size of deviceList (%zu) must be 1.", deviceNum);
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[config.deviceList.front()]);
    std::shared_ptr<AscendIndexIVFSPSQ> ivfspsq;
    if (codeBookSharedIdx == nullptr) {
        ivfspsq = AscendIndexIVFSPSQ::createAndLoadData(data, dataLen, config.deviceList, config.resourceSize);
    } else {
        FAISS_THROW_IF_NOT_MSG(codeBookSharedIdx->pIVFSPSQ != nullptr, "pIVFSPSQ is nullptr!");
        ivfspsq = AscendIndexIVFSPSQ::createAndLoadData(data, dataLen, config.deviceList,
            config.resourceSize, *codeBookSharedIdx->pIVFSPSQ);
    }
    FAISS_THROW_IF_NOT_MSG(ivfspsq != nullptr, "AscendIndexIVFSPSQ::createAndLoadData failed!");

    // 构造ivfsp的参数需要从文件中恢复出来，只能通过底层恢复后获取
    AscendIndexIVFSPConfig newConfig = config;
    int dims = ivfspsq->getDims();
    int nlist = ivfspsq->getNumList();
    int nonzeroNum = ivfspsq->getDims2();
    newConfig.filterable = ivfspsq->getFilterable();
    faiss::MetricType metric = ivfspsq->getMetric();
    faiss::ScalarQuantizer::QuantizerType qType = ivfspsq->getQuantizerType();
    return std::make_shared<AscendIndexIVFSPImpl>(intf, ivfspsq, dims, nonzeroNum, nlist, qType, metric,
        newConfig);
}

void AscendIndexIVFSPImpl::saveAllData(uint8_t *&data, size_t &dataLen) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtxVec[ivfspConfig.deviceList.front()]);
    pIVFSPSQ->saveAllData(data, dataLen);
}

}  // namespace ascend
}  // namespace faiss
