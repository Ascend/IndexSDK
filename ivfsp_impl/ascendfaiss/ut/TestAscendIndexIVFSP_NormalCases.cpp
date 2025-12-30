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


#include "Utils.h"

/**
 * 码本训练测试用例1： train后（trainAndAdd接口) + add底库 + 检索
 */
TEST(AscendIndexIVFSPSQ, train_codebook_then_add_search)
{
    std::vector<float> bdata = GenRandData(g_dim, g_nonzeroNum * g_nlist);

    /* 创建索引1 */
    auto index1 = CreateIndex();
    auto config = CreateCodeBookTrainConfig(bdata.data(), bdata.size());
    index1->trainCodeBook(config);

    // 添加底库，使用add接口
    index1->verbose = true;
    auto ids = CreateIotaVector<int64_t>(g_ntotal);
    index1->add(g_ntotal, bdata.data(), ids.data());
    std::cout << "finish add" << std::endl;

    // 检索
    index1->setNumProbes(g_nProbe);
    TestSearch(index1.get(), bdata, g_dim, g_handleBatch, g_k);
}

/**
 * 码本训练测试用例2： 使用磁盘上数据训练 + train后落盘
 */
TEST(AscendIndexIVFSPSQ, train_codebook_from_disk_data_then_save_codebook_to_disk)
{
    std::vector<float> bdata = GenRandData(g_dim, g_nonzeroNum * g_nlist);
    std::ofstream outFile("./learn.bin", std::ios::binary);
    outFile.write(reinterpret_cast<char *>(bdata.data()), sizeof(float) * bdata.size());

    /* 创建索引1 */
    auto index1 = CreateIndex();
    int device = 1;
    float sampleRatio = 0.5;
    auto config = CreateCodeBookTrainConfig("./learn.bin", ".", device, sampleRatio);
    index1->trainCodeBook(config);

    // 清除临时文件
    std::remove("./learn.bin");
    std::remove("./codebook_128_32_256.bin");
}

/**
 * 端到端基础功能测试1，包括
 * 1) 通过内存指针addCodeBook；
 * 2）add底库；
 * 3）正常search；
 * 4）searchWithFilter
 * 5）searchWithFilter + l1距离
 * 6）remove功能
 * 7）reset功能
 */
TEST(AscendIndexIVFSPSQ, add_base_then_search_then_delete)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);

    /* 创建索引1 */
    auto index1 = CreateIndex();

    /* 在添加任何底库前尝试remove，返回0 */
    std::vector<int64_t> removeVec {11, 12};
    EXPECT_EQ(index1->remove_ids(removeVec), 0);

    // 添加码本
    auto offsetIndex1 = CreateIotaVector<faiss::idx_t>(g_nlist);
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    index1->addCodeBook(g_nlist * g_nonzeroNum, g_dim, codeBookDataIndex1.data(), offsetIndex1.data());

    // 添加底库，使用add接口
    index1->verbose = true;
    auto ids = CreateIotaVector<int64_t>(g_ntotal);
    index1->add(g_ntotal, bdata.data(), ids.data());
    std::cout << "finish add" << std::endl;

    // 检索
    index1->setNumProbes(g_nProbe);
    TestSearch(index1.get(), bdata, g_dim, g_handleBatch, g_k);

    // SearchWithFilter
    auto searchCID = CreateIotaVector<int>(K_MAX_CAMERA_NUM);
    TestSearchWithFilter(index1.get(), bdata, g_dim, g_handleBatch, g_k, g_timeStamp, searchCID);

    // SearchWithFilter with l1distance
    TestSearchWithFilter(index1.get(), bdata, g_dim, g_handleBatch, g_k, g_timeStamp, searchCID, true);

    // 调用remove_ids接口
    std::vector<int64_t> emptyVec;
    EXPECT_EQ(index1->remove_ids(emptyVec), 0);
    EXPECT_EQ(index1->remove_ids(20, 30), 10);
    EXPECT_EQ(index1->remove_ids(removeVec), 2);

    // 调用reset接口
    index1->reset();
}

/**
 * 端到端基础功能测试2，包括
 * 1）通过磁盘上路径addCodeBook；
 * 2) add底库；
 * 3）searchWithMask;
 */
TEST(AscendIndexIVFSPSQ, add_base_then_train_then_search_with_mask)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);

    /* 创建索引1 */
    auto index1 = CreateIndex();

    // 添加码本 （先对码本进行落盘，直接从磁盘上读取）
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    std::string codebookPath = "./codebook.bin";

    // 码本需要固定的metadata格式
    std::vector<char> magicNumber = {'C', 'D', 'B', 'K'};
    std::vector<uint8_t> version = {1, 0, 0};
    const int blankDataSize = 64;
    WriteCodeBookWithMeta(codeBookDataIndex1, magicNumber, version, g_dim, g_nonzeroNum,
        g_nlist, blankDataSize, codebookPath);
    index1->addCodeBook(codebookPath);

    // 添加底库
    index1->verbose = true;
    auto ids = CreateIotaVector<int64_t>(g_ntotal);
    index1->add(g_ntotal, bdata.data(), nullptr); // ids参数为空指针
    std::cout << "finish add" << std::endl;

    // SearchWithMask (this function is not implemented, so we expect it to throw)
    try {
        index1->setNumProbes(g_nProbe);
        TestSearch(index1.get(), bdata, g_dim, g_handleBatch, g_k, true);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // 调用train接口
    index1->train(g_ntotal, bdata.data());

    // 删除生成的临时索引文件
    std::remove(codebookPath.c_str());
}

/**
 * 测试序列化/反序列接口
 */
TEST(AscendIndexIVFSPSQ, add_base_then_serialize_then_deserialize)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);

    /* 创建索引1 */
    auto index1 = CreateIndex();

    // 添加码本
    auto offsetIndex1 = CreateIotaVector<faiss::idx_t>(g_nlist);
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    index1->addCodeBook(g_nlist * g_nonzeroNum, g_dim, codeBookDataIndex1.data(), offsetIndex1.data());

    // 序列化仅有码本的索引
    size_t dataLengthCBOnly = 0;
    uint8_t *dataPtrCBOnly = nullptr;
    index1->saveAllData(dataPtrCBOnly, dataLengthCBOnly);

    // 添加底库
    index1->verbose = true;
    auto ids = CreateIotaVector<int64_t>(g_ntotal);
    index1->add(g_ntotal, bdata.data(), ids.data());
    std::cout << "finish add_with_ids" << std::endl;

    // 序列化索引
    size_t dataLength = 0;
    uint8_t *dataPtr = nullptr;
    index1->saveAllData(dataPtr, dataLength);

    // 反序列化索引
    auto index2 = faiss::ascendSearch::AscendIndexIVFSPSQ::createAndLoadData(dataPtr, dataLength,
        {0}, 2LLU * 1024 * 1024 * 1024);

    // 反序列化索引,与索引1共享码本
    auto index3 = faiss::ascendSearch::AscendIndexIVFSPSQ::createAndLoadData(dataPtr, dataLength,
        {0}, 2LLU * 1024 * 1024 * 1024, *index1);

    // 创建一个索引实例，然后通过序列化的dataPtr调用loadAllData
    auto index4 = CreateIndex();
    index4->loadAllData(dataPtr, dataLength);
    
    // 创建一个索引实例，然后通过序列化的dataPtr调用loadAllData,并与index1共享码本
    auto index5 = CreateIndex();
    index5->loadAllData(dataPtr, dataLength, *index1);

    // 创建一个索引实例，仅加载码本
    auto index6 = CreateIndex();
    index6->loadCodeBookOnly(dataPtrCBOnly, dataLengthCBOnly);

    // 清理使用的资源
    if (dataPtr != nullptr) {
        delete[] dataPtr;
    }

    if (dataPtrCBOnly != nullptr) {
        delete[] dataPtrCBOnly;
    }
}

/**
 * 测试落盘读盘至磁盘内的逻辑
 */
TEST(AscendIndexIVFSPSQ, add_base_then_save_file_then_load)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);

    /* 创建索引1 */
    auto index1 = CreateIndex();

    // 添加码本
    auto offsetIndex1 = CreateIotaVector<faiss::idx_t>(g_nlist);
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    index1->addCodeBook(g_nlist * g_nonzeroNum, g_dim, codeBookDataIndex1.data(), offsetIndex1.data());

    // 添加底库
    index1->verbose = true;
    auto ids = CreateIotaVector<int64_t>(g_ntotal);
    index1->add(g_ntotal, bdata.data(), ids.data());
    std::cout << "finish add_with_ids" << std::endl;

    // 序列化完整索引
    std::string indexPath = "./index1.idx";
    index1->saveAllData(indexPath.c_str());

    // 创建一个新的索引，并加载index1的内容
    auto index2 = CreateIndex();
    index2->loadAllData(indexPath.c_str());

    // index2调用addCodeBook，但是与index1共享码本
    index2->addCodeBook(*index1);

    // 创建一个新的索引，并加载index1的内容，与索引1共享码本
    auto index3 = CreateIndex();
    index3->loadAllData(indexPath.c_str(), *index1);

    // 删除生成的临时索引文件
    std::remove(indexPath.c_str());
}

/**
 * 测试对外的getter类接口
 */
TEST(AscendIndexIVFSPSQ, call_getters)
{
    /* 创建索引1 */
    auto index1 = CreateIndex();

    EXPECT_EQ(index1->getDims(), g_dim);
    EXPECT_EQ(index1->getDims2(), g_nonzeroNum);
    EXPECT_EQ(index1->getNumList(), g_nlist);
    EXPECT_EQ(index1->getFilterable(), g_filterable);
    EXPECT_EQ(index1->getMetric(), faiss::MetricType::METRIC_L2);
    EXPECT_EQ(index1->getQuantizerType(), faiss::ScalarQuantizer::QuantizerType::QT_8bit);
}

/**
 * 测试MultiSearch相关接口(Search 和 SearchWithFilter)，以及各种它们的重载API形态
 */
TEST(AscendIndexIVFSPSQ, multi_search)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);
    auto offsetIndex1 = CreateIotaVector<faiss::idx_t>(g_nlist);
    std::vector<faiss::ascendSearch::AscendIndex *> indexes;
    std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;

    for (int i = 0; i < g_nShards; ++i) {
        auto index = CreateIndex();
        indexesSmartPtr.emplace_back(index);
        indexes.emplace_back(index.get());
        // 添加码本
        if (i == 0) {
            std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
            index->addCodeBook(g_nlist * g_nonzeroNum, g_dim, codeBookDataIndex1.data(), offsetIndex1.data());
        } else {
            if (dynamic_cast<faiss::ascendSearch::AscendIndexIVFSPSQ *>(indexes[0]) != nullptr) {
                index->addCodeBook(*dynamic_cast<faiss::ascendSearch::AscendIndexIVFSPSQ *>(indexes[0]));
            }
        }
        // 添加底库
        index->verbose = true;
        auto ids = CreateIotaVector<int64_t>(g_ntotal);
        index->add(g_ntotal, bdata.data(), ids.data());
    }

    std::vector<float> dist(g_k * g_handleBatch * g_nShards, 0);
    std::vector<int64_t> label(g_k * g_handleBatch * g_nShards, 0);

    // 开始MultiSearch，并merge
    Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    // 开始MultiSearch，但不merge
    Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), false);

    auto searchCID = CreateIotaVector<int>(K_MAX_CAMERA_NUM);
    IDFilter idFilters[g_handleBatch];
    void *pFilter = &idFilters[0];
    ConstructCidFilter(idFilters, g_handleBatch, searchCID, g_timeStamp);
    // 开始MultiSearchWithFilter，并merge
    SearchWithFilter(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), pFilter, true);
    // 开始MultiSearchWithFilter，但不merge
    SearchWithFilter(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), pFilter, false);

    // 使用SearchWithFilterMultiIndex和SearchMultiIndex接口
    indexesSmartPtr[0]->SearchMultiIndex(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    indexesSmartPtr[0]->SearchWithFilterMultiIndex(indexes, g_handleBatch, bdata.data(), g_k, dist.data(),
        label.data(), pFilter, true);
    std::vector<void *> filterArray;
    for (int i = 0; i < g_handleBatch; ++i) {
        filterArray.emplace_back(pFilter);
    }
    indexesSmartPtr[0]->SearchWithFilterMultiIndex(indexes, g_handleBatch, bdata.data(), g_k, dist.data(),
        label.data(), filterArray.data(), true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}