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


#include <ascendsearch/ascend/custom/AscendIndexIVFSPSQ.h>
#include <ascendsearch/ascend/custom/impl/AscendIndexIVFSPSQImpl.h>
#include <ascendsearch/ascend/utils/AscendIVFAddInfo.h>
#include <common/threadpool/AscendThreadPool.h>
#include <ascendsearch/ascend/utils/AscendUtils.h>
#include <ascendsearch/ascend/utils/fp16.h>
#include <ascendsearch/ascend/rpc/AscendRpc.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <ascendsearch/ascend/AscendMultiIndexSearch.h>

namespace faiss {
namespace ascendSearch {

namespace {
constexpr size_t LOCAL_SECUREC_MEM_MAX_LEN = 0x7fffffffUL; // max buffer size secure_c supports (2GB)
constexpr int MAGIC_NUMBER_LEN = 4; // 序列化头魔术字长度为4
}

template<typename T>
void CopyDataForLoadSQ(T* dest, const uint8_t* src, size_t sizeBytes, size_t &offset, size_t dataLen)
{
    FAISS_THROW_IF_NOT_MSG(dataLen >= offset + sizeBytes, "memcpy error: insufficient data length");
    int err = memcpy_s(dest, std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset), src + offset, sizeBytes);
    FAISS_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    offset += sizeBytes;
}

/**
 * @brief 通过序列化存入data指针中的数据，创建一个AscendIndexIVFSPSQ实例
 *
 * @param data 填充着创建AscendIndexIVFSPSQ索引内容的数据
 * @param dataLen 用户输入的data内数据长度，用于校验
 * @param devices
 * @param resourceSize
 * @return std::shared_ptr<AscendIndexIVFSPSQ> AscendIndexIVFSPSQ智能指针
 */
std::shared_ptr<AscendIndexIVFSPSQ> createIVFSPSQInstanceFromData(const uint8_t* data, size_t dataLen,
    const std::vector<int> &devices, int64_t resourceSize)
{
    // 对输入进行基本的空指针和长度校验
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "data is not nullptr.");
    FAISS_THROW_IF_NOT_MSG(dataLen != 0, "dataLen must be non-zero.");

    size_t offset = 0;
    int dims = 0;
    int dim2 = 0;
    int numLists = 0;
    bool filterable = false;
    int handleBatch = 0;
    int nprobe = 0;
    int searchListSize = 0;

    char foureccBuffer[MAGIC_NUMBER_LEN];
    CopyDataForLoadSQ(foureccBuffer, data, sizeof(foureccBuffer), offset, dataLen);
    std::string foureccBufferString (foureccBuffer, MAGIC_NUMBER_LEN);

    const std::unordered_set<std::string> VALID_SIGNATURES = {"IWSP", "IWCB"};
    FAISS_THROW_IF_NOT_MSG(VALID_SIGNATURES.count(foureccBufferString) != 0, "Index format is incorrect.\n");

    CopyDataForLoadSQ(&dims, data, sizeof(int), offset, dataLen);
    CopyDataForLoadSQ(&dim2, data, sizeof(int), offset, dataLen);
    CopyDataForLoadSQ(&numLists, data, sizeof(int), offset, dataLen);
    CopyDataForLoadSQ(&filterable, data, sizeof(bool), offset, dataLen);
    CopyDataForLoadSQ(&handleBatch, data, sizeof(int), offset, dataLen);
    CopyDataForLoadSQ(&nprobe, data, sizeof(int), offset, dataLen);
    CopyDataForLoadSQ(&searchListSize, data, sizeof(int), offset, dataLen);
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig(devices, resourceSize);
    ivfspsqConfig.filterable = filterable;
    ivfspsqConfig.handleBatch = handleBatch;
    ivfspsqConfig.nprobe = nprobe;
    ivfspsqConfig.searchListSize = searchListSize;
    auto instance = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(dims, dim2, numLists, numLists,
                        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    return instance;
}


AscendIndexIVFSPSQ::AscendIndexIVFSPSQ(int dims, int dims2, int k, int nlist,
    faiss::ScalarQuantizer::QuantizerType qType,
    faiss::MetricType metric, bool encodeResidual,
    AscendIndexIVFSPSQConfig config)
    : AscendIndex(dims, metric, config),
      impl_(std::make_shared<AscendIndexIVFSPSQImpl>(dims, dims2, k, nlist,
        this, qType, metric, encodeResidual, config))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSPSQ::~AscendIndexIVFSPSQ() {}


void AscendIndexIVFSPSQ::reset()
{
    impl_->reset();
}

void AscendIndexIVFSPSQ::train(idx_t n, const float *x)
{
    impl_->train(n, x);
}

void AscendIndexIVFSPSQ::search_with_masks(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    impl_->search_with_masks(n, x, k, distances, labels, mask);
}

void AscendIndexIVFSPSQ::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    impl_->search_with_filter(n, x, k, distances, labels, filters);
}


void AscendIndexIVFSPSQ::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, float *l1distances, const void *filters) const
{
    impl_->search_with_filter(n, x, k, distances, labels, l1distances, filters);
}

void AscendIndexIVFSPSQ::setNumProbes(int nprobes)
{
    impl_->setNumProbes(nprobes);
}

void AscendIndexIVFSPSQ::trainCodeBook(const AscendIndexCodeBookTrainerConfig &codeBookTrainerConfig) const
{
    impl_->trainCodeBook(codeBookTrainerConfig);
}

void AscendIndexIVFSPSQ::addCodeBook(int n, int dim, const float *x, idx_t *offset)
{
    impl_->addCodeBook(n, dim, x, offset);
}

void AscendIndexIVFSPSQ::addCodeBook(const AscendIndexIVFSPSQ &loadedIndex)
{
    const AscendIndexIVFSPSQImpl& loadedIndexImpl = *(loadedIndex.impl_);
    impl_->addCodeBook(loadedIndexImpl);
}

void AscendIndexIVFSPSQ::add(idx_t n, const float* feature,
    const idx_t* ids)
{
    impl_->add(n, feature, ids);
}

void AscendIndexIVFSPSQ::loadAllData(const char *dataPath)
{
    impl_->loadAllData(dataPath);
}

void AscendIndexIVFSPSQ::loadAllData(const char *dataPath, const AscendIndexIVFSPSQ &loadedIndex)
{
    const AscendIndexIVFSPSQImpl& loadedIndexImpl = *(loadedIndex.impl_);
    impl_->loadAllData(dataPath, loadedIndexImpl);
}

void AscendIndexIVFSPSQ::loadAllData(const uint8_t* data, size_t dataLen)
{
    // 对输入进行基本的空指针和长度校验
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "data is not nullptr.");
    FAISS_THROW_IF_NOT_MSG(dataLen != 0, "dataLen must be non-zero.");
    impl_->loadAllData(data, dataLen);
}

void AscendIndexIVFSPSQ::loadAllData(const uint8_t* data, size_t dataLen, const AscendIndexIVFSPSQ &loadedIndex)
{
    // 对输入进行基本的空指针和长度校验
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "data is not nullptr.");
    FAISS_THROW_IF_NOT_MSG(dataLen != 0, "dataLen must be non-zero.");
    const AscendIndexIVFSPSQImpl& loadedIndexImpl = *(loadedIndex.impl_);
    impl_->loadAllData(data, dataLen, loadedIndexImpl);
}

void AscendIndexIVFSPSQ::loadCodeBookOnly(const uint8_t* data, size_t dataLen)
{
    // 对输入进行基本的空指针和长度校验
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "data is not nullptr.");
    FAISS_THROW_IF_NOT_MSG(dataLen != 0, "dataLen must be non-zero.");
    impl_->loadCodeBookOnly(data, dataLen);
}

void AscendIndexIVFSPSQ::saveAllData(const char *dataPath)
{
    impl_->saveAllData(dataPath);
}

void AscendIndexIVFSPSQ::saveAllData(uint8_t*& data, size_t& dataLen) const
{
    FAISS_THROW_IF_NOT_MSG(data == nullptr, "data to store serialized index must be a nullptr.\n");
    impl_->saveAllData(data, dataLen);
}

std::shared_ptr<AscendIndexIVFSPSQ> AscendIndexIVFSPSQ::createAndLoadData(const uint8_t* data, size_t dataLen,
    const std::vector<int> &devices, int64_t resourceSize)
{
    auto instance = createIVFSPSQInstanceFromData(data, dataLen, devices, resourceSize);
    FAISS_THROW_IF_NOT_MSG(instance != nullptr, "create index instance from data failed.\n");
    bool isCodeBookOnlyIndex = AscendIndexIVFSPSQ::checkCodeBookOnlyIndex(data, dataLen);
    if (isCodeBookOnlyIndex) {
        instance->loadCodeBookOnly(data, dataLen);
    } else {
        instance->loadAllData(data, dataLen);
    }
    return instance;
}

std::shared_ptr<AscendIndexIVFSPSQ> AscendIndexIVFSPSQ::createAndLoadData(const uint8_t* data, size_t dataLen,
    const std::vector<int> &devices, int64_t resourceSize, const AscendIndexIVFSPSQ &loadedIndex)
{
    auto instance = createIVFSPSQInstanceFromData(data, dataLen, devices, resourceSize);
    FAISS_THROW_IF_NOT_MSG(instance != nullptr, "create index instance from data failed.\n");
    bool isCodeBookOnlyIndex = AscendIndexIVFSPSQ::checkCodeBookOnlyIndex(data, dataLen);
    if (isCodeBookOnlyIndex) {
        // 如果一个仅有码本的索引仍要与一个填充好的索引进行码本共享，那data内的数据已无用，可以直接调用addCodeBook共享码本功能
        instance->addCodeBook(loadedIndex);
    } else {
        instance->loadAllData(data, dataLen, loadedIndex);
    }
    return instance;
}

void AscendIndexIVFSPSQ::SearchMultiIndex(std::vector<AscendIndex *> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    Search(indexes, n, x, k, distances, labels, merged);
}

void AscendIndexIVFSPSQ::SearchWithFilterMultiIndex(std::vector<AscendIndex *> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    SearchWithFilter(indexes, n, x, k, distances, labels, filters, merged);
}

void AscendIndexIVFSPSQ::SearchWithFilterMultiIndex(std::vector<AscendIndex *> indexes,
    idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, void *filters[], bool merged)
{
    SearchWithFilter(indexes, n, x, k, distances, labels, filters, merged);
}

size_t AscendIndexIVFSPSQ::remove_ids(const std::vector<idx_t> sel)
{
    return impl_->remove_ids(sel);
}

size_t AscendIndexIVFSPSQ::remove_ids(const idx_t minRange, const idx_t maxRange)
{
    return impl_->remove_ids(minRange, maxRange);
}

void AscendIndexIVFSPSQ::addCodeBook(const std::string &path)
{
    return impl_->addCodeBook(path);
}

int AscendIndexIVFSPSQ::getDims() const
{
    return impl_->getDims();
}

int AscendIndexIVFSPSQ::getDims2() const
{
    return impl_->getDims2();
}

int AscendIndexIVFSPSQ::getNumList() const
{
    return impl_->getNumList();
}

bool AscendIndexIVFSPSQ::getFilterable() const
{
    return impl_->getFilterable();
}

faiss::MetricType AscendIndexIVFSPSQ::getMetric() const
{
    return faiss::MetricType::METRIC_L2;
}

faiss::ScalarQuantizer::QuantizerType AscendIndexIVFSPSQ::getQuantizerType() const
{
    return faiss::ScalarQuantizer::QuantizerType::QT_8bit;
}

bool AscendIndexIVFSPSQ::checkCodeBookOnlyIndex(const uint8_t *data, size_t dataLen)
{
    // 对输入进行基本的空指针和长度校验
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "data is not nullptr.");
    FAISS_THROW_IF_NOT_MSG(dataLen != 0, "dataLen must be non-zero.");

    size_t offset = 0;
    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'C', 'B'}; // 如果序列化的索引仅有码本，头魔术字应该为IWCB
    char foureccBuffer[MAGIC_NUMBER_LEN];
    CopyDataForLoadSQ(foureccBuffer, data, sizeof(foureccBuffer), offset, dataLen);
    for (int i = 0; i < MAGIC_NUMBER_LEN; i++) {
        if (foureccBuffer[i] != fourcc[i]) {
            return false;
        }
    }
    return true;
}

void AscendIndexIVFSPSQ::addFinish()
{
    impl_->addFinish();
}

} // ascend
} // faiss