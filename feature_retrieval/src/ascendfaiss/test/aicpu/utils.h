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


#ifndef AICPU_UTEST_UTILS_H
#define AICPU_UTEST_UTILS_H

#include <algorithm>
#include <random>
#include <vector>
#include <sys/time.h>
namespace aicpu {
namespace unittest {
class Utils {
public:
template<typename T>
static void SetRandomValue(T input[], uint64_t num, float min = 0.0, float max = 1.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(min, max);
    for (uint64_t i = 0; i < num; ++i) {
        input[i] = static_cast<T>(dis(gen));
    }
}

static int64_t GetOffset(std::vector<int64_t> &shapes, const std::initializer_list<int64_t> &dims)
{
    size_t dim = dims.size();
    if (dim > shapes.size()) {
        return -1;
    }
    std::vector<int64_t> strides(shapes.size());
    int64_t stride = 1;
    for (int64_t i = shapes.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shapes[i];
    }
    int64_t offset = 0;
    for (int i = 0; i < dim; i++) {
        offset += *(dims.begin() + i) * strides[i];
    }
    return offset;
}

template<class ForwardIterator>
static inline size_t Argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
static inline size_t Argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

template<typename T>
static void PrintData(T *data, size_t len, size_t lenWrap = 16)
{
    std::cout << std::endl << std::endl;
    size_t idx = 0;
    while (idx < len) {
        if (idx > 0 && idx % lenWrap == 0) {
            std::cout << std::endl;
        }
        std::cout << *(data + idx) << " ";
        ++idx;
    }
    std::cout << std::endl << std::endl;
}

template<typename DIST_T, typename IDX_T>
static bool VerifySearchResult(size_t n, int k, int blockSize, DIST_T *inDists, 
    DIST_T *dists, IDX_T *labels, DIST_T *gtd, IDX_T *gtl, bool returnOnError=false)
{
    bool result = true;
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            size_t offset = i * k + j;
            if (dists[offset] != gtd[offset]) {
                std::cout << "[ERROR]VerifySearchResult failed, query[" << i << "]top[" << j << "]: " << dists[offset]
                          << ", expect: " << gtd[offset] << std::endl;
                if (returnOnError) {
                    return false;
                }
                result = false;
                continue;
            }
            if (labels[offset] == gtl[offset]) {
                continue;
            }
            std::cout << "[WARNING]VerifySearchResult failed, query[" << i << "]top[" << j << "]: " << labels[offset]
                        << ", expect: " << gtl[offset] << std::endl;
            int64_t cmOffset = (labels[offset] / blockSize * n * blockSize + i * blockSize +
                labels[offset] % blockSize);
            int64_t gtOffset = (gtl[offset] / blockSize * n * blockSize + i * blockSize + gtl[offset] % blockSize);
            if (inDists[cmOffset] != inDists[gtOffset]) {
                std::cout << "[ERROR]VerifySearchResult failed, query[" << i << "]top[" << j << "]: "
                            << "inDists[" << labels[offset] << "]:" << inDists[labels[offset]]
                            << ", expect: inDists[" << gtl[offset] << "]:" << inDists[gtl[offset]] << std::endl;
                if (returnOnError) {
                    return false;
                }
                result = false;
            } else {
                std::cout << "label not equal, but dist equal, query[" << i << "]top[" << j << "] indist["
                          << labels[offset] << "]: " << inDists[cmOffset] << " = indist["
                          << gtl[offset] << "]:" << inDists[gtOffset] << std::endl;
            }
        }
    }
    return result;
}

static double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

static void InitGTDistAndLabel(int64_t asc, std::vector<float16_t> &gtdists, std::vector<int64_t> &gtlabels)
{
    uint16_t *outdists = reinterpret_cast<uint16_t *>(gtdists.data());
    int64_t *outlabels = gtlabels.data();
    // Set initial outlabels value -1
    std::fill_n(outlabels, gtlabels.size(), 0xffffffffffffffff);
    if (asc != 0) {
        std::fill_n(outdists, gtdists.size(), 0x7bff);  // 65504.0
    } else {
        std::fill_n(outdists, gtdists.size(), 0xfbff);  // -65504.0
    }
}
};
} // namespace unittest
} // namespace aicpu

#endif