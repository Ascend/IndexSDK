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

// 需要生成aicpu算子+ivfsp_pycal算子(-d 256 -nonzero_num 64 -nlist 256 -batch_size 64 -code_num 32768)+ivfsp_model算子

#include <faiss/ascend/AscendIndexIVFSP.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>

namespace
{

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

using recallMap = std::unordered_map<int, float>;

// camera id 简写为 cid， 7位数， 取值范围0~127
const int32_t K_MAX_CAMERA_NUM = 128;
const int MASK_LEN = 8;
const int INDEX_NUM = 10;
const int TOPK = 100;
const int QUERY_NUM = 5306;  // queryNum应该小于等于使用的"查询数据"集的特征向量的实际条数
const int TIME = 20000;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
struct IDFilter
{
    IDFilter()
    {
        std::fill_n(cameraIdMask, K_MAX_CAMERA_NUM / MASK_LEN, static_cast<uint8_t>(0));
        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    // 一个IDFilter对象是可以涵盖处理所有cid in [0, 127] 共128个camera
    uint8_t cameraIdMask[K_MAX_CAMERA_NUM / MASK_LEN] = {0};
    uint32_t timeRange[2] = {0};
};

// batch即searchNum， 一条被检索的特征向量，传递一个IDFilter对象
// std::vector<int> &cids, 是一个固定的128元素的向量，其值从0到127
void ConstructCidFilter(IDFilter* idFilters, int batch, const std::vector<int>& cids,
                        const std::vector<uint32_t>& timestamps)
{
    for (int i = 0; i < batch; ++i)
    {
        for (auto current_cid : cids)
        {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
            idFilters[i].cameraIdMask[g] += (1 << k);
        }
        idFilters[i].timeRange[0] = timestamps[0];  // start
        idFilters[i].timeRange[1] = timestamps[1];  // end
    }
}

void ConstructCidFilter(IDFilter* idFilters, int staIdx, int batch, const std::vector<int>& cids,
                        const std::vector<uint32_t>& timestamps)
{
    for (int i = 0; i < batch; ++i)
    {
        for (auto current_cid : cids)
        {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
            idFilters[staIdx + i].cameraIdMask[g] += (1 << k);
        }
        idFilters[staIdx + i].timeRange[0] = timestamps[0];  // start
        idFilters[staIdx + i].timeRange[1] = timestamps[1];  // end
    }
}

void ComputeRecall(recallMap& recMap, int j)
{
    recMap[RECMAP_KEY_100]++;
    switch (j)
    {
        case 0:
            recMap[RECMAP_KEY_1]++;
            recMap[RECMAP_KEY_10]++;
            break;
        case 1 ... 9:  // case 1到9
            recMap[RECMAP_KEY_10]++;
            break;
        default:
            break;
    }
}

//  Calculate Recall
template <class T>
recallMap CalRecall(std::vector<T>& label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[RECMAP_KEY_1] = 0;
    Map[RECMAP_KEY_10] = 0;
    Map[RECMAP_KEY_100] = 0;
    if (shape <= 0)
    {
        std::cerr << "Error: Invalid shape value." << std::endl;
        return Map;
    }
    int k = label.size() / shape;

    for (int i = 0; i < shape; i++)
    {
        std::set<int> labelSet(label.begin() + i * k, label.begin() + i * k + k);

        for (int j = 0; j < k; j++)
        {
            if (gt[i * k] == label[i * k + j])
            {  // 被检索的query[i]这条向量， 如果在检索得到的topk条结果中有任意一条
               // label[i * k + j] 等于 gt[i * k], (topk 命中)， 则表示命中
                ComputeRecall(Map, j);
                break;
            }
        }
    }
    Map[RECMAP_KEY_1] = Map[RECMAP_KEY_1] / shape * 100;  // recMap[1]的百分比 这里的100代表的是百分比的计算因子
    Map[RECMAP_KEY_10] = Map[RECMAP_KEY_10] / shape * 100;  // recMap[10]的百分比 这里的100代表的是百分比的计算因子
    Map[RECMAP_KEY_100] = Map[RECMAP_KEY_100] / shape * 100;  // recMap[100]的百分比 这里的100代表的是百分比的计算因子
    return Map;
}

template <class T>
void PrintRecall(std::vector<T>& labels, int64_t* gt, int shape, int bs)
{
    std::cout << "-------------CalRecall-------------------" << std::endl;
    recallMap Map = CalRecall(labels, gt, shape);
    std::cout << "batchSize: " << bs << std::endl;
    std::cout << "recall 1@1: " << Map[RECMAP_KEY_1] << std::endl;
    std::cout << "recall 1@10: " << Map[RECMAP_KEY_10] << std::endl;
    std::cout << "recall 1@100: " << Map[RECMAP_KEY_100] << std::endl;
}

template <class T>
void PrintMultiRecall(std::vector<T>& labels, int64_t* gt, int bs, int batchNum)
{
    recallMap Map;
    Map[RECMAP_KEY_1] = 0;
    Map[RECMAP_KEY_10] = 0;
    Map[RECMAP_KEY_100] = 0;
    for (int batchIdx = 0; batchIdx < batchNum; batchIdx++)
    {
        for (int i = 0; i < INDEX_NUM; i++)
        {
            std::vector<T> subLabels(labels.cbegin() + batchIdx * bs * INDEX_NUM * TOPK + i * bs * TOPK,
                                     labels.cbegin() + batchIdx * bs * INDEX_NUM * TOPK + (i + 1) * bs * TOPK);
            recallMap subMap = CalRecall(subLabels, gt + batchIdx * bs * TOPK, bs);
            Map[RECMAP_KEY_1] += subMap[RECMAP_KEY_1];
            Map[RECMAP_KEY_10] += subMap[RECMAP_KEY_10];
            Map[RECMAP_KEY_100] += subMap[RECMAP_KEY_100];
        }
    }

    Map[RECMAP_KEY_1] = Map[RECMAP_KEY_1] / (batchNum * INDEX_NUM);
    Map[RECMAP_KEY_10] = Map[RECMAP_KEY_10] / (batchNum * INDEX_NUM);
    Map[RECMAP_KEY_100] = Map[RECMAP_KEY_100] / (batchNum * INDEX_NUM);

    std::cout << "-------------CalRecall-------------------" << std::endl;
    std::cout << "batchSize: " << bs << std::endl;
    std::cout << "recall 1@1: " << Map[RECMAP_KEY_1] << std::endl;
    std::cout << "recall 1@10: " << Map[RECMAP_KEY_10] << std::endl;
    std::cout << "recall 1@100: " << Map[RECMAP_KEY_100] << std::endl;
}

void InitData(std::vector<float>& data, std::vector<float>& qData, std::vector<int64_t>& gts, int dim, int ntotal)
{
    // 数据集（特征数据、查询数据、groundtruth数据）， 所在的目录。请根据实际情况填写。
    std::string baseInitPath = " ";
    // 特征底库数据
    std::string dataPath = baseInitPath + "base.bin";
    // 查询数据
    std::string queryPath = baseInitPath + "query.bin";
    // （groundtruth） 测试召回的比较数据
    std::string gtsPath = baseInitPath + "gt.bin";

    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read(reinterpret_cast<char*>(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // query data
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read(reinterpret_cast<char*>(qData.data()), sizeof(float) * QUERY_NUM * dim);
    queryFin.close();

    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read(reinterpret_cast<char*>(gts.data()), sizeof(int64_t) * QUERY_NUM * TOPK);
    gtsFin.close();
}

void SearchData(faiss::ascend::AscendIndexIVFSP& index, std::vector<int>& batches, std::vector<float>& qData, int dim,
                std::vector<int64_t>& gts)
{
    printf("-------------search-------------------\n");
    for (int batch : batches)
    {
        int loop = QUERY_NUM / batch;
        std::vector<float> dist(loop * TOPK * batch, 0);
        std::vector<faiss::idx_t> labels(loop * TOPK * batch, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loop; i++)
        {
            index.search(batch, qData.data() + i * batch * dim, TOPK, dist.data() + i * TOPK * batch,
                         labels.data() + i * TOPK * batch);
        }
        double te = GetMillisecs();
        // QPS = 并发线程数 * (1000 / 平均耗时ms)
        printf("search, TOPK: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n", TOPK, dim, batch,
               loop * batch, MILLI_SECOND * loop * batch / (te - ts));
        PrintRecall(labels, gts.data(), batch * loop, batch);
    }
}

void SearchFilter(faiss::ascend::AscendIndexIVFSP& index, std::vector<int>& batches, std::vector<float>& qData, int dim,
                  std::vector<int64_t>& gts)
{
    std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
    std::iota(search_cid.begin(), search_cid.end(), 0);
    std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(TIME)};

    printf("-------------Search with Filter-------------------\n");
    for (int batch : batches)
    {
        IDFilter idFilters[batch];
        void* pFilter = &idFilters[0];
        ConstructCidFilter(idFilters, batch, search_cid, search_time);
        int loop = QUERY_NUM / batch;
        std::vector<float> dist4filter(loop * TOPK * batch, 0);
        std::vector<faiss::idx_t> labels4filter(loop * TOPK * batch, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loop; i++)
        {
            index.search_with_filter(batch, qData.data() + i * batch * dim, TOPK, dist4filter.data() + i * TOPK * batch,
                                     labels4filter.data() + i * TOPK * batch, pFilter);
        }
        double te = GetMillisecs();
        // QPS = 并发线程数 * (1000 / 平均耗时ms)
        printf("search with filter, TOPK: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n", TOPK, dim, batch,
               loop * batch, MILLI_SECOND * loop * batch / (te - ts));
        PrintRecall(labels4filter, gts.data(), batch * loop, batch);
    }
}

void MultiSearch(std::vector<faiss::ascend::AscendIndex*>& indexes, std::vector<int>& batches,
                 std::vector<float>& qData, int dim, std::vector<int64_t>& gts)
{
    printf("-------------MultiSearch-------------------\n");
    for (size_t j = 0; j < batches.size(); j++)
    {
        int iloop = QUERY_NUM / batches[j];
        std::vector<float> dist(iloop * INDEX_NUM * TOPK * batches[j], 0);
        std::vector<faiss::idx_t> label(iloop * INDEX_NUM * TOPK * batches[j], 0);
        double ts = GetMillisecs();
        for (int iStep = 0; iStep < iloop; iStep++)
        {
            Search(indexes, batches[j], qData.data() + iStep * batches[j] * dim, TOPK,
                   dist.data() + iStep * INDEX_NUM * TOPK * batches[j],
                   label.data() + iStep * INDEX_NUM * TOPK * batches[j], false);
            // 每512次记录一次进度
            if (iStep * batches[j] % 512 == 0)
            {
                printf("istep:%d\n", iStep);
            }
        }
        double te = GetMillisecs();
        // QPS = 并发线程数 * (1000 / 平均耗时ms)
        printf(
            "multi search: true, index num: %d, TOPK: %d, dim: %d, batch size: %d,"
            "search num: %2d, QPS: %9.4f\n",
            INDEX_NUM, TOPK, dim, batches[j], iloop * batches[j], MILLI_SECOND * iloop * batches[j] / (te - ts));
        PrintMultiRecall(label, gts.data(), batches[j], iloop);
    }
}

void MultiSearchWithSameFilter(std::vector<faiss::ascend::AscendIndex*>& indexes, std::vector<int>& batches,
                               std::vector<float>& qData, int dim, std::vector<int64_t>& gts)
{
    std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
    std::iota(search_cid.begin(), search_cid.end(), 0);
    std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(TIME)};

    printf("-------------MultiSearchFilter for same filters-------------------\n");
    for (size_t j = 0; j < batches.size(); j++)
    {
        IDFilter idFilters[batches[j]];
        void* pFilters = &idFilters[0];
        ConstructCidFilter(idFilters, batches[j], search_cid, search_time);

        int iloop = QUERY_NUM / batches[j];
        std::vector<float> dist(iloop * INDEX_NUM * TOPK * batches[j], 0);
        std::vector<faiss::idx_t> label(iloop * INDEX_NUM * TOPK * batches[j], 0);
        double ts = GetMillisecs();
        for (int iStep = 0; iStep < iloop; iStep++)
        {
            SearchWithFilter(indexes, batches[j], qData.data() + iStep * batches[j] * dim, TOPK,
                             dist.data() + iStep * INDEX_NUM * TOPK * batches[j],
                             label.data() + iStep * INDEX_NUM * TOPK * batches[j], pFilters, false);
            // 每512次记录一次进度
            if (iStep * batches[j] % 512 == 0)
            {
                printf("istep:%d\n", iStep);
            }
        }
        double te = GetMillisecs();
        // QPS = 并发线程数 * (1000 / 平均耗时ms)
        printf(
            "multi search for same filters: true, index num: %d, TOPK: %d, dim: %d, batch size: %d, "
            "search num: %2d, QPS: %9.4f\n",
            INDEX_NUM, TOPK, dim, batches[j], iloop * batches[j], MILLI_SECOND * iloop * batches[j] / (te - ts));
        PrintMultiRecall(label, gts.data(), batches[j], iloop);
    }
}

void MultiSearchWithDifFilter(std::vector<faiss::ascend::AscendIndex*>& indexes, std::vector<int>& batches,
                              std::vector<float>& qData, int dim, std::vector<int64_t>& gts)
{
    printf("-------------MultiSearchFilter for different filters-------------------\n");
    std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
    std::iota(search_cid.begin(), search_cid.end(), 0);
    std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(TIME)};
    for (size_t j = 0; j < batches.size(); j++)
    {
        void* pFilters[batches[j]];
        IDFilter idFilters[INDEX_NUM * batches[j]];
        for (int queryIdx = 0; queryIdx < batches[j]; queryIdx++)
        {
            for (int indexIdx = 0; indexIdx < INDEX_NUM; indexIdx++)
            {
                ConstructCidFilter(idFilters, indexIdx + queryIdx * INDEX_NUM, 1, search_cid, search_time);
            }
            pFilters[queryIdx] = &idFilters[INDEX_NUM * queryIdx];
        }

        int iloop = QUERY_NUM / batches[j];
        std::vector<float> dist(iloop * INDEX_NUM * TOPK * batches[j], 0);
        std::vector<faiss::idx_t> label(iloop * INDEX_NUM * TOPK * batches[j], 0);
        double ts = GetMillisecs();
        for (int iStep = 0; iStep < iloop; iStep++)
        {
            SearchWithFilter(indexes, batches[j], qData.data() + iStep * batches[j] * dim, TOPK,
                             dist.data() + iStep * INDEX_NUM * TOPK * batches[j],
                             label.data() + iStep * INDEX_NUM * TOPK * batches[j], pFilters, false);
            // 每512次记录一次进度
            if (iStep * batches[j] % 512 == 0)
            {
                printf("istep:%d\n", iStep);
            }
        }
        double te = GetMillisecs();
        // QPS = 并发线程数 * (1000 / 平均耗时ms)
        printf(
            "multi search for different filters: true, index num: %d, TOPK: %d, dim: %d, "
            "batch size: %d, search num: %2d, QPS: %9.4f\n",
            INDEX_NUM, TOPK, dim, batches[j], iloop * batches[j], MILLI_SECOND * iloop * batches[j] / (te - ts));
        PrintMultiRecall(label, gts.data(), batches[j], iloop);
    }
}

void LoadAndSaveData(std::vector<faiss::ascend::AscendIndex*>& indexes, int ntotal, std::vector<float>& data)
{
    std::string baseLoadPath = " ";
    // ivfsp索引数据落盘保存到的路径indexPath
    std::string indexPath = baseLoadPath + "myivfsp_base_data.bin";
    struct stat indexPathStat;
    if (lstat(indexPath.c_str(), &indexPathStat) == 0)
    {
        remove(indexPath.c_str());
    }
    for (int i = 0; i < INDEX_NUM; ++i)
    {
        faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
        printf("add data index:%d\n", i);
        if (FILE* file = fopen(indexPath.c_str(), "r"))
        {
            fclose(file);
            index->loadAllData(indexPath.c_str());
            std::cout << "loadAllData from " << indexPath << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
        }
        else
        {
            index->add(ntotal, data.data());
            std::cout << "add" << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
            index->saveAllData(indexPath.c_str());
            std::cout << "saveAllData to " << indexPath << std::endl;
        }
    }
}

void RecallAndRecallFilter()
{
    // 数据集（特征数据、查询数据、groundtruth数据）、码本， 所在的目录。请根据实际情况填写。
    std::string basePath = " ";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";

    // 参数值 dim、nonzeroNum、nlist、searchListSize，应该和使用的codeBook
    // 码本保持一致，即和训练码本时指定的参数保持一致。
    int dim = 256;
    int nonzeroNum = 64;
    int nlist = 256;
    int handleBatch = 64;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64};
    int searchListSize = 32768;

    // nototal应该小于等于使用的"特征底库数据"集的特征向量的实际条数
    int ntotal = 2000000;

    // base data
    std::vector<float> data(dim * ntotal);
    // query data
    std::vector<float> qData(QUERY_NUM * dim);
    std::vector<int64_t> gts(QUERY_NUM * TOPK, 0);
    try
    {
        InitData(data, qData, gts, dim, ntotal);

        faiss::ascend::AscendIndexIVFSPConfig conf({0});
        conf.handleBatch = handleBatch;        // 和OM算子保持一致
        conf.nprobe = handleBatch;             // 64 32 128    16的倍数，且0 < nprobe <= nlist
        conf.searchListSize = searchListSize;  // 大于等于512 且为2的幂次。
        conf.filterable = true;

        faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, nlist, codeBookPath.c_str(),
                                              faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                                              faiss::MetricType::METRIC_L2, conf);
        index.setVerbose(true);
        index.add(ntotal, data.data());
        std::cout << "index.ntotal: " << index.ntotal << std::endl;

        std::vector<int> nprobeList = {handleBatch, handleBatch * 2, handleBatch / 2};
        for (int tmpNprobe : nprobeList)
        {
            printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
            index.setNumProbes(tmpNprobe);
            SearchData(index, batches, qData, dim, gts);
            SearchFilter(index, batches, qData, dim, gts);
        }
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
    }
}

void CreateMultiIndex(std::vector<faiss::ascend::AscendIndex*>& indexes, int dim,
                      faiss::ascend::AscendIndexIVFSPConfig& conf)
{
    int nonzeroNum = 64;
    int nlist = 256;
    std::string baseMultiPath = " ";
    // codeBook 码本
    std::string codeBookPath = baseMultiPath + "codebook.bin";
    for (int i = 0; i < INDEX_NUM; ++i)
    {
        faiss::ascend::AscendIndexIVFSP* index;
        if (i == 0)
        {
            index = new faiss::ascend::AscendIndexIVFSP(dim, nonzeroNum, nlist, codeBookPath.c_str(),
                                                        faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                                                        faiss::MetricType::METRIC_L2, conf);
        }
        else
        {
            index = new faiss::ascend::AscendIndexIVFSP(
                dim, nonzeroNum, nlist, *(faiss::ascend::AscendIndexIVFSP*)indexes[0],
                faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_L2, conf);
        }

        index->setVerbose(true);
        indexes.emplace_back(index);
        printf("create index:%d\n", i);
    }
}

void MultiSearchAndMultiSearchFilter()
{
    // 数据集（特征数据、查询数据、groundtruth数据）、码本， 所在的目录。请根据实际情况填写。

    // 参数值 dim、nlist、nonzeroNum、searchListSize，应该和使用的codeBook
    // 码本保持一致，即和训练码本时指定的参数保持一致。
    int dim = 256;

    int handleBatch = 64;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64};
    int searchListSize = 32768;

    // nototal应该小于等于使用的"特征底库数据"集的特征向量的实际条数
    int ntotal = 2000000;
    // base data
    std::vector<float> data(dim * ntotal);
    // query data
    std::vector<float> qData(QUERY_NUM * dim);
    // ground truth data
    std::vector<int64_t> gts(QUERY_NUM * TOPK, 0);
    InitData(data, qData, gts, dim, ntotal);

    int64_t resourceSize = 2 * static_cast<int64_t>(1024 * 1024 * 1024);
    faiss::ascend::AscendIndexIVFSPConfig conf({0}, resourceSize);
    conf.handleBatch = handleBatch;        // 和OM算子保持一致
    conf.nprobe = handleBatch;             // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.searchListSize = searchListSize;  // 大于等于512 且为2的幂次。
    conf.filterable = true;

    std::vector<faiss::ascend::AscendIndex*> indexes;
    try
    {
        CreateMultiIndex(indexes, dim, conf);

        LoadAndSaveData(indexes, ntotal, data);

        std::vector<int> nprobeList = {handleBatch, handleBatch * 2, handleBatch / 2};
        for (int tmpNprobe : nprobeList)
        {
            printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
            for (int i = 0; i < INDEX_NUM; ++i)
            {
                faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
                index->setNumProbes(tmpNprobe);
            }

            MultiSearch(indexes, batches, qData, dim, gts);
            MultiSearchWithSameFilter(indexes, batches, qData, dim, gts);
            MultiSearchWithDifFilter(indexes, batches, qData, dim, gts);
        }

        for (int i = 0; i < INDEX_NUM; ++i)
        {
            delete indexes[i];
        }
    }
    catch (std::exception& e)
    {
        for (int i = 0; i < INDEX_NUM; ++i)
        {
            delete indexes[i];
        }
        printf("%s\n", e.what());
    }
}

}  // namespace

int main(int argc, char** argv)
{
    RecallAndRecallFilter();
    MultiSearchAndMultiSearchFilter();
    return 0;
}
