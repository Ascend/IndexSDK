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
 * 反序列化索引，但输入非法
 */
TEST(AscendIndexIVFSPSQ, createIVFSPSQInstanceFromData_invalid_input)
{
    size_t testDataLength = 100;
    std::vector<uint8_t> testDataPtr(testDataLength);

    // dataPtr为空指针
    try {
        uint8_t *dataPtr = nullptr;
        auto index = faiss::ascendSearch::AscendIndexIVFSPSQ::createAndLoadData(dataPtr,
            testDataLength, { g_device }, g_resourceSize);
        FAIL();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // dataLength为0
    try {
        auto index = faiss::ascendSearch::AscendIndexIVFSPSQ::createAndLoadData(testDataPtr.data(),
            0, { g_device }, g_resourceSize);
        FAIL();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // 魔术字错误
    try {
        auto index = faiss::ascendSearch::AscendIndexIVFSPSQ::createAndLoadData(testDataPtr.data(),
            testDataLength, { g_device }, g_resourceSize);
        FAIL();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

/**
 * 添加非法格式的码本, 测试用例1
 */
TEST(AscendIndexIVFSPSQ, addCodeBook_invalid_meta_test1)
{
    auto index1 = CreateIndex();
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    std::string codebookPath = "./codebook.bin";
    // 魔术字错误
    try {
        std::vector<char> magicNumber = {'C', 'D', 'B', 'C'};
        std::vector<uint8_t> version = {1, 0, 0};
        const int blankDataSize = 64;
        WriteCodeBookWithMeta(codeBookDataIndex1, magicNumber, version, g_dim, g_nonzeroNum,
            g_nlist, blankDataSize, codebookPath);
        index1->addCodeBook(codebookPath);
    } catch (const std::exception &e) {
        std::remove(codebookPath.c_str());
    }
    // version错误
    try {
        std::vector<char> magicNumber = {'C', 'D', 'B', 'K'};
        std::vector<uint8_t> version = {1, 1, 0};
        const int blankDataSize = 64;
        WriteCodeBookWithMeta(codeBookDataIndex1, magicNumber, version, g_dim, g_nonzeroNum,
            g_nlist, blankDataSize, codebookPath);
        index1->addCodeBook(codebookPath);
    } catch (const std::exception &e) {
        std::remove(codebookPath.c_str());
    }
}

/**
 * 添加非法格式的码本, 测试用例2
 */
TEST(AscendIndexIVFSPSQ, addCodeBook_invalid_meta_test2)
{
    auto index1 = CreateIndex();
    std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
    std::string codebookPath = "./codebook.bin";
    // 数据长度错误
    try {
        std::vector<char> magicNumber = {'C', 'D', 'B', 'K'};
        std::vector<uint8_t> version = {1, 0, 0};
        const int blankDataSize = 72; // blank数据长度错误
        WriteCodeBookWithMeta(codeBookDataIndex1, magicNumber, version, g_dim, g_nonzeroNum,
            g_nlist, blankDataSize, codebookPath);
        index1->addCodeBook(codebookPath);
    } catch (const std::exception &e) {
        std::remove(codebookPath.c_str());
    }
    // 某个参数错误
    try {
        std::vector<char> magicNumber = {'C', 'D', 'B', 'K'};
        std::vector<uint8_t> version = {1, 0, 0};
        const int blankDataSize = 64;
        WriteCodeBookWithMeta(codeBookDataIndex1, magicNumber, version, g_dim, g_nonzeroNum,
            g_nlist + 1, blankDataSize, codebookPath); // gnlist参数错误
        index1->addCodeBook(codebookPath);
    } catch (const std::exception &e) {
        std::remove(codebookPath.c_str());
    }
    // 码本路径错误
    try {
        index1->addCodeBook("./wrongCB.bin"); // 该路径不存在
    } catch (const std::exception &e) {}
    // 非寻常文件
    try {
        index1->addCodeBook("/dev/random");
    } catch (const std::exception &e) {}
}

/**
    测试创建索引时错误输入：测试用例1
 */
TEST(AscendIndexIVFSPSQ, createIndex_invalid_input_test1)
{
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    ivfspsqConfig.filterable = g_filterable;
    ivfspsqConfig.handleBatch = g_handleBatch;
    ivfspsqConfig.searchListSize = g_searchListSize;

    // dim错误
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim + 1, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // nonzeroNum错误
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum + 1, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // nlist范围非法
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist + 1,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // nlist 和 不等
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist + 1, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // quantizer类型错误
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_fp16, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // metric错误
    try {
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_INNER_PRODUCT,
            false, ivfspsqConfig);
    } catch (const std::exception &e) {}
}

/**
    测试创建索引时错误输入：测试用例2
 */
TEST(AscendIndexIVFSPSQ, createIndex_invalid_input_test2)
{
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);

    // 大dim场景下，nlist超过最大允许值
    try {
        int highDim = 768;
        int invalidNList = 4096;
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(highDim, g_nonzeroNum, invalidNList,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
    } catch (const std::exception &e) {}

    // device数组大小大于1
    try {
        faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({g_device, g_device + 1}, g_resourceSize);
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, invalidIvfspsqConfig);
    } catch (const std::exception &e) {}

    // device值非法
    try {
        int invalidDeviceId = 2048;
        faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({invalidDeviceId}, g_resourceSize);
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, invalidIvfspsqConfig);
    } catch (const std::exception &e) {}

    // nprobe值非法
    try {
        faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({g_device}, g_resourceSize);
        invalidIvfspsqConfig.nprobe = 99;
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, invalidIvfspsqConfig);
    } catch (const std::exception &e) {}

    // handleBatch值非法
    try {
        faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({g_device}, g_resourceSize);
        invalidIvfspsqConfig.handleBatch = 99;
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, invalidIvfspsqConfig);
    } catch (const std::exception &e) {}

    // searchListSize值非法
    try {
        faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({g_device}, g_resourceSize);
        invalidIvfspsqConfig.searchListSize = 99;
        auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
            g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, invalidIvfspsqConfig);
    } catch (const std::exception &e) {}
}

/**
 * MultiSearch场景下需要多个index共享一些参数，对该校验中的异常场景测试，测试用例1
 */
TEST(AscendIndexIVFSPSQ, multi_search_not_same_params_test1)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);
    std::vector<float> dist(g_k * g_handleBatch * g_nShards, 0);
    std::vector<int64_t> label(g_k * g_handleBatch * g_nShards, 0);
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> index;

    // 2个index的metrics类型不同
    try {
        std::vector<faiss::ascendSearch::AscendIndex *> indexes;
        std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;
        for (int i = 0; i < g_nShards; ++i) {
            if (i == 0) {
                index = CreateIndex();
            } else {
                index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
                    g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_INNER_PRODUCT, false,
                    ivfspsqConfig);
            }
            index->is_trained = true;
            indexesSmartPtr.emplace_back(index);
            indexes.emplace_back(index.get());
        }
        Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    } catch (const std::exception &e) {}

    // 2个index的SQ量化类型不同
    try {
        std::vector<faiss::ascendSearch::AscendIndex *> indexes;
        std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;
        for (int i = 0; i < g_nShards; ++i) {
            if (i == 0) {
                index = CreateIndex();
            } else {
                index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
                    g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform, faiss::METRIC_L2, false,
                    ivfspsqConfig);
            }
            index->is_trained = true;
            indexesSmartPtr.emplace_back(index);
            indexes.emplace_back(index.get());
        }
        Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    } catch (const std::exception &e) {}
}

/**
 * MultiSearch场景下需要多个index共享一些参数，对该校验中的异常场景测试，测试用例2
 */
TEST(AscendIndexIVFSPSQ, multi_search_not_same_params_test2)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);
    std::vector<float> dist(g_k * g_handleBatch * g_nShards, 0);
    std::vector<int64_t> label(g_k * g_handleBatch * g_nShards, 0);
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> index;

    // 2个index的deviceList不同
    try {
        std::vector<faiss::ascendSearch::AscendIndex *> indexes;
        std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;
        for (int i = 0; i < g_nShards; ++i) {
            if (i == 0) {
                index = CreateIndex();
            } else {
                faiss::ascendSearch::AscendIndexIVFSPSQConfig invalidIvfspsqConfig({g_device + 1}, g_resourceSize);
                index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(2 * g_dim, g_nonzeroNum, g_nlist,
                    g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false,
                    invalidIvfspsqConfig);
            }
            index->is_trained = true;
            indexesSmartPtr.emplace_back(index);
            indexes.emplace_back(index.get());
        }
        Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // 码本不同 (索引2不添加索引)
    try {
        std::vector<faiss::ascendSearch::AscendIndex *> indexes;
        std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;
        for (int i = 0; i < g_nShards; ++i) {
            index = CreateIndex();
            if (i == 0) {
                std::vector<float> codeBookDataIndex1 = GenRandCodeBook(g_dim, g_nonzeroNum, g_nlist);
                auto offsetIndex1 = CreateIotaVector<faiss::idx_t>(g_nlist);
                index->addCodeBook(g_nlist * g_nonzeroNum, g_dim, codeBookDataIndex1.data(), offsetIndex1.data());
            }
            index->is_trained = true;
            indexesSmartPtr.emplace_back(index);
            indexes.emplace_back(index.get());
        }
        Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

/**
 * MultiSearch场景下需要多个index共享一些参数，对该校验中的异常场景测试，测试用例3
 */
TEST(AscendIndexIVFSPSQ, multi_search_not_same_params_test3)
{
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);
    std::vector<float> dist(g_k * g_handleBatch * g_nShards, 0);
    std::vector<int64_t> label(g_k * g_handleBatch * g_nShards, 0);
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> index;

    // 2个index的维度不同
    try {
        std::vector<faiss::ascendSearch::AscendIndex *> indexes;
        std::vector<std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ>> indexesSmartPtr;
        for (int i = 0; i < g_nShards; ++i) {
            if (i == 0) {
                index = CreateIndex();
            } else {
                index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(2 * g_dim, g_nonzeroNum, g_nlist,
                    g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);
            }
            index->is_trained = true;
            indexesSmartPtr.emplace_back(index);
            indexes.emplace_back(index.get());
        }
        Search(indexes, g_handleBatch, bdata.data(), g_k, dist.data(), label.data(), true);
    } catch (const std::exception &e) {}
}

/**
 * 创建一个量化方式为SQ 8bit_Uniform的索引，然后调用train接口
 */
TEST(AscendIndexIVFSPSQ, create_8bit_uniform_index_then_train)
{
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
        g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform, faiss::METRIC_L2, false, ivfspsqConfig);

    // 调用train接口
    std::vector<float> bdata = GenRandData(g_dim, g_ntotal);
    index->train(g_ntotal, bdata.data());
}

/**
 * 端到端正常流程，但检索时参数非法，用例1
 */
TEST(AscendIndexIVFSPSQ, add_base_then_search_with_invalid_input_test1)
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

    std::vector<float> dist(g_ntotal);
    std::vector<int64_t> label(g_ntotal);

    // batchSize < 0
    try {
        int batchSize = -1;
        int k = 1;
        index1->search(batchSize, bdata.data(), k, dist.data(), label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // batchSize > 1e9 (MAX_N)
    try {
        int batchSize = 1e9 + 1;
        int k = 1;
        index1->search(batchSize, bdata.data(), k, dist.data(), label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // k < 0
    try {
        int batchSize = 64;
        int k = -1;
        index1->search(batchSize, bdata.data(), k, dist.data(), label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // k > 4096 (MAX_K)
    try {
        int batchSize = 64;
        int k = 5000;
        index1->search(batchSize, bdata.data(), k, dist.data(), label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

/**
 * 端到端正常流程，但检索时参数非法，用例2
 */
TEST(AscendIndexIVFSPSQ, add_base_then_search_with_invalid_input_test2)
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

    std::vector<float> dist(g_ntotal);
    std::vector<int64_t> label(g_ntotal);

    // dist空指针
    try {
        int batchSize = 64;
        int k = 1;
        index1->search(batchSize, bdata.data(), k, nullptr, label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // label空指针
    try {
        int batchSize = 64;
        int k = 1;
        index1->search(batchSize, bdata.data(), k, dist.data(), nullptr);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    // query空指针
    try {
        int batchSize = 64;
        int k = 1;
        index1->search(batchSize, nullptr, k, dist.data(), label.data());
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

/**
 * index调用非法输入的loadAllData
 */
TEST(AscendIndexIVFSPSQ, load_and_save_with_invalid_input)
{
    /* 创建索引1 */
    auto index1 = CreateIndex();
    std::vector<uint8_t> testDataPtr(g_ntotal);
    // 路径为空
    try {
        index1->loadAllData("");
    } catch (const std::exception &e) {}
    // 路径为空指针
    try {
        index1->loadAllData(nullptr);
    } catch (const std::exception &e) {}
    // 反序列化时，数据为空
    try {
        size_t dataLength = 1;
        index1->loadAllData(nullptr, dataLength);
    } catch (const std::exception &e) {}
    // 反序列化时，数据长度为0
    try {
        size_t dataLength = 0;
        index1->loadAllData(testDataPtr.data(), dataLength);
    } catch (const std::exception &e) {}
    // 反序列化，共享码本索引，错误输入
    auto index2 = CreateIndex();
    try {
        size_t dataLength = 1;
        index1->loadAllData(nullptr, dataLength, *index2);
    } catch (const std::exception &e) {}
    try {
        size_t dataLength = 0;
        index1->loadAllData(testDataPtr.data(), dataLength, *index2);
    } catch (const std::exception &e) {}
    // loadCodeBook only, 错误输入
    try {
        size_t dataLength = 1;
        index1->loadCodeBookOnly(nullptr, dataLength);
    } catch (const std::exception &e) {}
    try {
        size_t dataLength = 0;
        index1->loadCodeBookOnly(testDataPtr.data(), dataLength);
    } catch (const std::exception &e) {}
}