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


#include <gtest/gtest.h>
#include <sys/time.h>

#include <faiss/ascend/AscendIndexIVFFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

TEST(TestAscendIndexIVFFlat, All)
{
    int dim = 128;
    int ntotal = 200000;
    int ncentroids = 2048;
    int nprobe = 64;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFFlat index(dim, ncentroids, faiss::METRIC_L2, conf);
    index.verbose = true;

    index.setNumProbes(nprobe);
    EXPECT_EQ(index.getNumProbes(), nprobe);

    index.train(ntotal, data.data());
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    ASSERT_EQ(index.ntotal, ntotal);
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    std::vector<std::vector<uint8_t>> codes(ncentroids, std::vector<uint8_t>());
    std::vector<std::vector<faiss::ascend::ascend_idx_t>> indices(ncentroids,
        std::vector<faiss::ascend::ascend_idx_t>());
    for (int i = 0; i < ncentroids; i++) {
        index.getListCodesAndIds(i, codes[i], indices[i]);
        ASSERT_EQ(codes[i].size(), indices[i].size() * index.d * 2);
    }

    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.reserveMemory(ntotal);
    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> tmpCode;
        std::vector<faiss::ascend::ascend_idx_t> tmpIndi;
        index.getListCodesAndIds(i, tmpCode, tmpIndi);
        ASSERT_EQ(tmpCode.size(), tmpIndi.size() * index.d * 2) << "failure index:" << i;
        ASSERT_EQ(tmpCode.size(), codes[i].size()) << "failure index:" << i;
        ASSERT_EQ(tmpIndi.size(), indices[i].size()) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpCode.data(), codes[i].data(), indices[i].size() * index.d * 2), 0) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpIndi.data(), indices[i].data(), indices[i].size() * sizeof(faiss::ascend::ascend_idx_t)),
            0) << "failure index:" << i;
    }

    {
        int n = 1;
        int k = 1000;
        for (int i = 3; i < 10; i++) {
            std::vector<float> dist(k, 0);
            std::vector<faiss::idx_t> label(k, 0);
            index.search(n, data.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);

            faiss::idx_t assign;
            index.assign(1, data.data() + i * dim, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexIVFFlat, CloneAscend2CPU)
{
    int dim = 128;
    int ntotal = 250000;
    int ncentroids = 2048;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    // ascend index
    faiss::ascend::AscendIndexIVFFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFFlat ascendIndex(dim, ncentroids, faiss::METRIC_L2, conf);
    ascendIndex.verbose = true;
    ascendIndex.setNumProbes(nprobe);
    EXPECT_EQ(ascendIndex.getNumProbes(), nprobe);

    ascendIndex.train(ntotal, data.data());

    // add data
    ascendIndex.reserveMemory(ntotal);
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./ivfflattest.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    faiss::IndexIVFFlat *cpuIvfFlat = dynamic_cast<faiss::IndexIVFFlat *>(cpuIndex);
    EXPECT_EQ(ascendIndex.metric_type, cpuIvfFlat->metric_type);
    EXPECT_EQ(ascendIndex.d, cpuIvfFlat->d);
    EXPECT_EQ(ascendIndex.ntotal, cpuIvfFlat->ntotal);
    EXPECT_EQ(ascendIndex.is_trained, cpuIvfFlat->is_trained);
    EXPECT_EQ(ascendIndex.getNumLists(), cpuIvfFlat->nlist);
    EXPECT_EQ(ascendIndex.getNumProbes(), cpuIvfFlat->nprobe);

    const faiss::InvertedLists *ivf = cpuIvfFlat->invlists;
    EXPECT_NE(ivf, nullptr);

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; ++i) {
        size_t listSize = ivf->list_size(i);
        size_t listSize1 = ascendIndex.getListLength(i);
        EXPECT_EQ(listSize, listSize1);

        const uint8_t *codeCpu = ivf->get_codes(i);
        const faiss::idx_t *idCpu = ivf->get_ids(i);

        std::vector<unsigned char> codes;
        std::vector<faiss::ascend::ascend_idx_t> indices;
        ascendIndex.getListCodesAndIds(i, codes, indices);

        std::vector<faiss::ascend::ascend_idx_t> indicesCpu(listSize, 0);
        transform(idCpu, idCpu + listSize, begin(indicesCpu),
            [](faiss::idx_t x) { return faiss::ascend::ascend_idx_t(x); });
        for (size_t i = 0; i < indices.size(); ++i) {
            float *pCpuData = ((float *)codeCpu) + i;
            uint16_t *pAscendData = ((uint16_t *)codes.data()) + i;
            float ascendData = (float)(faiss::ascend::fp16(*pAscendData));
            EXPECT_TRUE(ascendData - *pCpuData < 1e-6);
        }
        EXPECT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(faiss::ascend::ascend_idx_t)), 0);

        tmpTotal += listSize1;
    }
    EXPECT_EQ(tmpTotal, ntotal);

    for (int i = 0; i < 100; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        ascendIndex.search(1, data.data() + idx * dim, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete cpuIndex;
}

TEST(TestAscendIndexIVFFlat, CloneCPU2Ascend)
{
    int dim = 128;
    int ntotal = 250000;
    int ncentroids = 2048;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    const char *indexfilename = "./ivfflattest.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);
    faiss::IndexIVFFlat *cpuIndex = dynamic_cast<faiss::IndexIVFFlat *>(initIndex);

    // ascend index
    std::vector<int> devices = { 0, 1, 2, 3 };
    faiss::ascend::AscendIndexIVFFlat *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexIVFFlat *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    EXPECT_EQ(ascendIndex->d, dim);
    EXPECT_EQ(ascendIndex->ntotal, ntotal);
    EXPECT_EQ(ascendIndex->getNumLists(), ncentroids);
    EXPECT_EQ(ascendIndex->getNumProbes(), nprobe);
    EXPECT_EQ(ascendIndex->metric_type, cpuIndex->metric_type);
    EXPECT_EQ(ascendIndex->d, cpuIndex->d);
    EXPECT_EQ(ascendIndex->ntotal, cpuIndex->ntotal);
    EXPECT_EQ(ascendIndex->is_trained, cpuIndex->is_trained);
    EXPECT_EQ(ascendIndex->getNumLists(), cpuIndex->nlist);
    EXPECT_EQ(ascendIndex->getNumProbes(), cpuIndex->nprobe);

    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            int sizeCpuList = cpuIndex->get_list_size(i);
            int sizeAscendList = ascendIndex->getListLength(i);
            ASSERT_EQ(sizeCpuList, sizeAscendList) << "Failure idx:" << i;

            const uint8_t *codeCpu = cpuIndex->invlists->get_codes(i);
            const faiss::idx_t *idCpu = cpuIndex->invlists->get_ids(i);

            std::vector<unsigned char> codes;
            std::vector<faiss::ascend::ascend_idx_t> indices;
            ascendIndex->getListCodesAndIds(i, codes, indices);

            std::vector<faiss::ascend::ascend_idx_t> indicesCpu(sizeCpuList, 0);
            transform(idCpu, idCpu + sizeCpuList, begin(indicesCpu),
                [](faiss::idx_t x) { return faiss::ascend::ascend_idx_t(x); });

            for (size_t j = 0; j < indices.size(); ++j) {
                float *pCpuData = ((float *)codeCpu) + j;
                uint16_t *pAscendData = ((uint16_t *)codes.data()) + j;
                float ascendData = (float)(faiss::ascend::fp16(*pAscendData));
                EXPECT_TRUE(ascendData - *pCpuData < 1e-6);
            }
            ASSERT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(faiss::ascend::ascend_idx_t)),
                0) << "Failure idx:" << i;

            tmpTotal += sizeAscendList;
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < 100; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        ascendIndex->search(1, data.data() + idx * dim, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexIVFFlat, remove)
{
    int dim = 128;
    int ntotal = 200000;
    int ncentroids = 2048;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFFlat index(dim, ncentroids, faiss::METRIC_L2, conf);
    index.verbose = true;

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IDSelectorRange del(0, 2);
    int rmCnt = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<faiss::ascend::ascend_idx_t> ids;

        index.getListCodesAndIds(i, code, ids);
        for (size_t k = 0; k < ids.size(); k++) {
            rmCnt += del.is_member((int64_t)ids[k]) ? 1 : 0;
        }
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<faiss::ascend::ascend_idx_t> ids;
        index.getListCodesAndIds(i, code, ids);
        ASSERT_EQ(code.size(), ids.size() * index.d * 2);
        for (size_t k = 0; k < ids.size(); k++) {
            ASSERT_FALSE(del.is_member((int64_t)ids[k])) << "failure index:" << i;
        }
        tmpTotal += index.getListLength(i);
    }

    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));
}

TEST(TestAscendIndexIVFFlat, QPS)
{
    std::vector<int> nlists = { 2048 };
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128 };

    int dim = 128;
    int ntotal = 1000000;
    int nprobe = 64;
    int k = 100;
    int loopTimes = 100;

    std::vector<float> data(dim * ntotal);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = drand48();
    }

    int caseId = 0;
    faiss::ascend::AscendIndexIVFFlatConfig conf({ 0, 1, 2, 3 });
    for (auto nlist : nlists) {
        faiss::ascend::AscendIndexIVFFlat index(dim, nlist, faiss::METRIC_L2, conf);
        index.verbose = true;

        index.setNumProbes(nprobe);
        EXPECT_EQ(index.getNumProbes(), nprobe);

        index.train(ntotal, data.data());
        index.add(ntotal, data.data());
        ASSERT_EQ(index.ntotal, ntotal);

        int tmpTotal = 0;
        for (int i = 0; i < nlist; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);

        for (size_t n = 0; n < searchNum.size(); n++) {
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++) {
                index.search(searchNum[n], data.data(), k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            printf("case[%d]: base:%d, dim:%d, nlist:%d, search num:%d, QPS:%.4f\n", ++caseId, ntotal, dim, nlist,
                searchNum[n], 1000 * searchNum[n] * loopTimes / (te - ts));
        }
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}