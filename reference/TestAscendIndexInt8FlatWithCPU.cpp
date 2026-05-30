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

// 需要生成 aicpu算子+int8flat算子(-d 512)

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/index_io.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

const int K = 1;
const int SEARCH_NUM = 10;
const std::vector<int> DEVICE{0};
inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

void PrintSearch(const std::vector<float> &dist, const std::vector<faiss::idx_t> &label)
{
    printf("I=\n");
    for (int i = 0; i < SEARCH_NUM; i++)
    {
        for (int j = 0; j < K; j++)
        {
            printf("%5ld ", label[i * K + j]);
        }
        printf("\n");
    }
    printf("D=\n");
    for (int i = 0; i < SEARCH_NUM; i++)
    {
        for (int j = 0; j < K; j++)
        {
            printf("%5f ", dist[i * K + j]);
        }
        printf("\n");
    }
}

void Generate(size_t ntotal, std::vector<int8_t> &data, int seed = 5678)
{
    int maxValue = 255;
    int offset = 128;
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    data.resize(ntotal);
    for (size_t i = 0; i < ntotal; ++i)
    {
        data[i] = static_cast<int8_t>(maxValue * rCode(e) - offset);
    }
}

void CpuIndexAdd(faiss::IndexIDMap *cpuIDMap, int ntotal, std::vector<int8_t> &base, std::vector<faiss::idx_t> &ids)
{
    size_t len = base.size();
    uint8_t *codesUint8 = reinterpret_cast<uint8_t *>(base.data());
    auto cpuSQ = dynamic_cast<faiss::IndexScalarQuantizer *>(cpuIDMap->index);
    cpuSQ->ntotal += ntotal;
    cpuSQ->codes.insert(cpuSQ->codes.end(), codesUint8, codesUint8 + len);
    cpuIDMap->id_map.insert(cpuIDMap->id_map.end(), ids.begin(), ids.end());
}

size_t RemoveImpl(faiss::IndexIDMap *cpuIDMap, std::vector<size_t> &removes)
{
    auto cpuSQ = dynamic_cast<faiss::IndexScalarQuantizer *>(cpuIDMap->index);
    std::sort(removes.rbegin(), removes.rend());
    for (auto const &pos : removes)
    {
        size_t lastIdx = cpuIDMap->id_map.size() - 1;
        cpuIDMap->id_map[pos] = cpuIDMap->id_map[lastIdx];
        cpuIDMap->id_map.pop_back();

        size_t curCodeIdx = static_cast<size_t>(pos) * static_cast<size_t>(cpuIDMap->index->d);
        size_t lastCodeIdx = lastIdx * static_cast<size_t>(cpuIDMap->index->d);
        for (int i = 0; i < cpuIDMap->index->d; i++)
        {
            cpuSQ->codes[curCodeIdx + i] = cpuSQ->codes[lastCodeIdx + i];
        }
        cpuSQ->codes.resize(lastCodeIdx);
        cpuSQ->ntotal--;
    }
    return removes.size();
}

void GetRemoveIDS(faiss::IndexIDMap *cpuIDMap, const std::unordered_set<faiss::idx_t> &idSet,
                  std::vector<size_t> &removes)
{
    for (auto it = idSet.begin(); it != idSet.end(); it++)
    {
        for (size_t i = 0; i < cpuIDMap->id_map.size(); i++)
        {
            if (cpuIDMap->id_map[i] == *it)
            {
                removes.push_back(i);
                break;
            }
        }
    }
}

size_t CpuIndexRemove(faiss::IndexIDMap *cpuIDMap, const faiss::IDSelector &sel)
{
    std::vector<size_t> removes;
    if (auto rangeSel = dynamic_cast<const faiss::IDSelectorBatch *>(&sel))
    {
        GetRemoveIDS(cpuIDMap, rangeSel->set, removes);
    }
    else if (auto rangeSel = dynamic_cast<const faiss::IDSelectorRange *>(&sel))
    {
        std::unordered_set<faiss::idx_t> idSet;
        for (auto idx = rangeSel->imin; idx < rangeSel->imax; ++idx)
        {
            idSet.insert(idx);
        }
        GetRemoveIDS(cpuIDMap, idSet, removes);
    }
    else
    {
        printf("not support type\n");
        return 0;
    }
    return RemoveImpl(cpuIDMap, removes);
}

void CheckResult(faiss::ascend::AscendIndexInt8Flat &ascendIndex, faiss::ascend::AscendIndexInt8Flat *newAscendIndex)
{
    size_t baseSize1 = 0;
    std::vector<int8_t> codes1;
    std::vector<faiss::idx_t> idx1;
    for (auto const &deviceId : DEVICE)
    {
        std::vector<int8_t> codes;
        std::vector<faiss::idx_t> idx;
        ascendIndex.getBase(deviceId, codes);
        codes1.insert(codes1.end(), codes.begin(), codes.end());
        ascendIndex.getIdxMap(deviceId, idx);
        idx1.insert(idx1.end(), idx.begin(), idx.end());
        baseSize1 += ascendIndex.getBaseSize(deviceId);
    }

    size_t baseSize2 = 0;
    std::vector<int8_t> codes2;
    std::vector<faiss::idx_t> idx2;
    for (auto const &deviceId : DEVICE)
    {
        std::vector<int8_t> codes;
        std::vector<faiss::idx_t> idx;
        newAscendIndex->getBase(deviceId, codes);
        codes2.insert(codes2.end(), codes.begin(), codes.end());
        newAscendIndex->getIdxMap(deviceId, idx);
        idx2.insert(idx2.end(), idx.begin(), idx.end());
        baseSize2 += newAscendIndex->getBaseSize(deviceId);
    }
    printf("baseSize1:%ld, baseSize2:%ld\n", baseSize1, baseSize2);
    if (baseSize1 != baseSize2)
    {
        printf("baseSize not equal!!!!!: %ld vs %ld\n", baseSize1, baseSize2);
    }
    for (size_t i = 0; i < codes1.size(); i++)
    {
        if (codes1[i] != codes2[i])
        {
            printf("codes[%ld] not equal!!!!!: %d vs %d\n", i, codes1[i], codes2[i]);
        }
    }
    for (size_t i = 0; i < idx1.size(); i++)
    {
        if (idx1[i] != idx2[i])
        {
            printf("idx[%ld] not equal!!!!!: %ld vs %ld\n", i, idx1[i], idx2[i]);
        }
    }
}

void DelRangeAndSearch(faiss::ascend::AscendIndexInt8Flat &ascendIndex, faiss::IndexIDMap *cpuIDMap,
                       std::vector<int8_t> &base, std::vector<float> &dist, std::vector<faiss::idx_t> &label)
{
    int delRangeMin = 6;
    int delRangeMax = 10;
    faiss::IDSelectorRange del1(delRangeMin, delRangeMax);
    // ascend index delete
    auto removeCnt = ascendIndex.remove_ids(del1);
    printf("ascend delete count:%ld\n", removeCnt);
    // cpu index delete
    removeCnt = CpuIndexRemove(cpuIDMap, del1);
    printf("cpu delete count:%ld\n", removeCnt);

    ascendIndex.search(SEARCH_NUM, base.data(), K, dist.data(), label.data());
}

void AddAndSearch(faiss::ascend::AscendIndexInt8Flat &ascendIndex, faiss::IndexIDMap *cpuIDMap,
                  std::vector<float> &dist, std::vector<faiss::idx_t> &label, int dim)
{
    std::vector<int8_t> base2(static_cast<size_t>(dim) * SEARCH_NUM);
    int seed = 1234;
    Generate(base2.size(), base2, seed);
    std::vector<faiss::idx_t> ids2(SEARCH_NUM);
    std::iota(ids2.begin(), ids2.end(), 1);

    // ascend index添加10个底库
    ascendIndex.add_with_ids(SEARCH_NUM, base2.data(), ids2.data());

    // cpu index添加10个底库
    CpuIndexAdd(cpuIDMap, SEARCH_NUM, base2, ids2);

    // ascend index查找底库最后10个
    ascendIndex.search(SEARCH_NUM, base2.data(), K, dist.data(), label.data());
    std::vector<faiss::idx_t> delBatches = {1000001, 1000003, 1000005, 1000007, 1000009};
    faiss::IDSelectorBatch del2(delBatches.size(), delBatches.data());
    // ascend index delete
    auto removeCnt = ascendIndex.remove_ids(del2);
    printf("ascend delete count:%ld\n", removeCnt);
    // cpu index delete
    removeCnt = CpuIndexRemove(cpuIDMap, del2);
    printf("cpu delete count:%ld\n", removeCnt);

    ascendIndex.search(SEARCH_NUM, base2.data(), K, dist.data(), label.data());
}

void LoadAndSearch(faiss::ascend::AscendIndexInt8Flat &ascendIndex, faiss::IndexIDMap *cpuIDMap,
                   std::vector<int8_t> &base, std::vector<float> &dist, std::vector<faiss::idx_t> &label)
{
    double t1 = GetMillisecs();
    const char *fileName = "int8flat.faiss";
    faiss::write_index(cpuIDMap, fileName);
    double t2 = GetMillisecs();
    printf("save cpu index cost time:%f\n", t2 - t1);

    // 加载cpu index到ascend idnex
    faiss::IndexIDMap *newCpuIndex = dynamic_cast<faiss::IndexIDMap *>(faiss::read_index(fileName));
    faiss::ascend::AscendIndexInt8Flat *newAscendIndex = dynamic_cast<faiss::ascend::AscendIndexInt8Flat *>(
        faiss::ascend::index_int8_cpu_to_ascend(DEVICE, newCpuIndex));

    // 加载后第1次查找底库前10个
    newAscendIndex->search(SEARCH_NUM, base.data(), K, dist.data(), label.data());

    // 对比ascendIndex和newAscendIndex的codes和ids
    CheckResult(ascendIndex, newAscendIndex);
    delete newCpuIndex;
    delete newAscendIndex;
}

int main(int argc, char **argv)
{
    try
    {
        int64_t resourceSize = 2 * static_cast<int64_t>(1024 * 1024 * 1024);
        faiss::ascend::AscendIndexInt8FlatConfig conf(DEVICE, resourceSize);
        int dim = 512;
        faiss::idx_t ntotal = 1000000;
        faiss::MetricType metricType = faiss::MetricType::METRIC_INNER_PRODUCT;

        // 生成初始底库
        std::vector<int8_t> base(static_cast<size_t>(dim) * ntotal);
        Generate(base.size(), base);
        std::vector<faiss::idx_t> ids(ntotal);
        std::iota(ids.begin(), ids.end(), 1);

        // 创建ascend index
        faiss::ascend::AscendIndexInt8Flat ascendIndex(dim, metricType, conf);

        // 创建faiss index
        faiss::IndexScalarQuantizer *cpuSQ = new faiss::IndexScalarQuantizer();
        faiss::IndexIDMap *cpuIDMap = new faiss::IndexIDMap(cpuSQ);
        cpuIDMap->own_fields = true;
        cpuIDMap->index->reset();
        cpuIDMap->index->metric_type = metricType;
        cpuIDMap->index->d = dim;
        cpuIDMap->index->is_trained = true;

        /************************* 第1次添加1000000个底库 *************************/
        // ascend index添加1000000底库
        ascendIndex.add_with_ids(ntotal, base.data(), ids.data());

        // cpu index添加1000000底库
        CpuIndexAdd(cpuIDMap, ntotal, base, ids);

        // ascend index查找底库前10个
        std::vector<float> dist(SEARCH_NUM * K, 0);
        std::vector<faiss::idx_t> label(SEARCH_NUM * K, 0);
        ascendIndex.search(SEARCH_NUM, base.data(), K, dist.data(), label.data());

        printf("-------------------search 1-------------------\n");
        PrintSearch(dist, label);

        // 第1次删除 id [delRangeMin, delRangeMax)
        DelRangeAndSearch(ascendIndex, cpuIDMap, base, dist, label);

        printf("-------------------search 2-------------------\n");
        PrintSearch(dist, label);
        AddAndSearch(ascendIndex, cpuIDMap, dist, label, dim);
        /************************* 第2次添加10个底库 *************************/

        printf("-------------------search 3-------------------\n");
        PrintSearch(dist, label);
        LoadAndSearch(ascendIndex, cpuIDMap, base, dist, label);
        delete cpuSQ;
        delete cpuIDMap;
    }
    catch (std::exception &e)
    {
        printf("Exception caught:%s!\n", e.what());
    }

    return 0;
}
